"""Regression checks for the pre-Colab review findings."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import os
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from members.mrcyrilgoud import model_mlp as m

MEMBER = Path(__file__).resolve().parent


def tagged_cell(notebook, tag):
    cells = json.loads((MEMBER / notebook).read_text())['cells']
    matches = [cell for cell in cells if tag in cell.get('metadata', {}).get('tags', [])]
    if len(matches) != 1:
        raise AssertionError(f'{notebook}: expected one {tag} cell, found {len(matches)}')
    return ''.join(matches[0]['source'])


class ReviewRegressions(unittest.TestCase):
    def setup_prefix(self, notebook):
        setup = tagged_cell(notebook, 'setup')
        # Exercise the actual bootstrap before dependency installation / GPU checks.
        return setup[setup.index('import os, sys'):setup.index('# The shared evaluator reads')]

    def test_environment_values_are_injected_without_echoing_secrets(self):
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            with self.subTest(notebook=name):
                tree = ast.parse(tagged_cell(name, 'environment'))
                values = {'KAGGLE_API_TOKEN': None, 'CASMI_STAGE': 'debug', 'EXTRA_VALUE': 'example'}
                assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'ENV_VALUES' for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                assignments[0].value = ast.parse(repr(values), mode='eval').body
                code = compile(ast.fix_missing_locations(tree), '<environment-cell>', 'exec')
                output = io.StringIO()
                with patch.dict(os.environ, {}, clear=True), patch('getpass.getpass', return_value='test-secret') as prompt:
                    with redirect_stdout(output):
                        exec(code, {})
                    self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'test-secret')
                    self.assertEqual(os.environ['CASMI_STAGE'], 'debug')
                    self.assertEqual(os.environ['EXTRA_VALUE'], 'example')
                    prompt.assert_called_once()
                    prompt.reset_mock()
                    exec(code, {})  # Existing credentials should not prompt again.
                    prompt.assert_not_called()
                self.assertNotIn('test-secret', output.getvalue())

    def test_literal_environment_values_control_configuration(self):
        values = {
            'KAGGLE_API_TOKEN': 'literal-test-secret',
            'CASMI_DRIVE_MOUNT': '/content/drive',
            'CASMI_PROJECT': '/content/drive/MyDrive/alternate-project',
            'CASMI_OUTPUT': '/content/drive/MyDrive/alternate-project/results',
            'CASMI_STAGE': 'final',
            'CASMI_RESUME': 'TRUE',
        }
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            with self.subTest(notebook=name):
                tree = ast.parse(tagged_cell(name, 'environment'))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'ENV_VALUES'
                            for target in node.targets):
                        node.value = ast.parse(repr(values), mode='eval').body
                output = io.StringIO()
                with patch.dict(os.environ, {}, clear=True), patch('getpass.getpass', side_effect=AssertionError('Unexpected prompt')):
                    env = {}
                    with redirect_stdout(output):
                        exec(compile(ast.fix_missing_locations(tree), '<environment-cell>', 'exec'), env)
                        exec(tagged_cell(name, 'config'), env)
                    self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'literal-test-secret')
                    self.assertEqual(env['DRIVE_MOUNT'], Path(values['CASMI_DRIVE_MOUNT']))
                    self.assertEqual(env['PROJECT'], Path(values['CASMI_PROJECT']))
                    self.assertEqual(env['REPO'], env['PROJECT'] / 'repo')
                    self.assertEqual(env['TRAIN'], env['REPO'] / 'data/raw/train.parquet')
                    self.assertEqual(env['OUTPUT'], Path(values['CASMI_OUTPUT']))
                    self.assertEqual(env['STAGE'], 'final')
                    self.assertTrue(env['RESUME'])
                    os.environ['CASMI_RESUME'] = 'false'
                    exec(tagged_cell(name, 'config'), env)
                    self.assertFalse(env['RESUME'])
                self.assertNotIn('literal-test-secret', output.getvalue())

    def test_notebooks_keep_only_single_line_titles_and_no_colab_secrets(self):
        for path in sorted(MEMBER.glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text())
                markdown = [cell for cell in notebook['cells'] if cell['cell_type'] == 'markdown']
                self.assertEqual(len(markdown), 1)
                title = ''.join(markdown[0]['source']).strip().splitlines()
                self.assertEqual(len(title), 1)
                self.assertTrue(title[0].startswith('# '))
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb', '03_kaggle_submission.ipynb'):
            with self.subTest(notebook=name):
                self.assertNotIn('userdata', tagged_cell(name, 'setup'))
        with patch.dict(os.environ, {}, clear=True), patch('getpass.getpass', side_effect=AssertionError('Unexpected prompt')):
            exec(tagged_cell('03_kaggle_submission.ipynb', 'environment'), {})

    def test_notebooks_configure_all_project_paths_on_drive(self):
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            with self.subTest(notebook=name):
                nb = json.loads((MEMBER / name).read_text())
                env = {}
                exec(tagged_cell(name, 'config'), env)
                for key in ('PROJECT', 'REPO', 'TRAIN', 'OUTPUT'):
                    self.assertIn(env['DRIVE_MOUNT'], env[key].parents)
                self.assertEqual(env['TRAIN'], env['REPO'] / 'data/raw/train.parquet')
                setup = tagged_cell(name, 'setup')
                self.assertNotIn('copytree', setup)
                self.assertNotIn('symlink_to', setup)

    def test_fresh_colab_setup_uses_drive_without_local_copy(self):
        script = """
import json, sys, types, shutil, tempfile, os
from pathlib import Path
from unittest.mock import patch
import torch
import pandas as pd
payload = json.loads(sys.stdin.read())
colab = types.ModuleType('google.colab')
colab.drive = types.SimpleNamespace(mount=lambda path: None)
google = types.ModuleType('google')
google.colab = colab
sys.modules['google'] = google
sys.modules['google.colab'] = colab
mount = Path(payload['mount'])
project = mount / 'MyDrive/Data266/EnvedaGroupProject'
repo = project / 'repo'
source = Path(payload['source'])
shutil.copytree(source / 'src', repo / 'src')
shutil.copytree(source / 'members/mrcyrilgoud', repo / 'members/mrcyrilgoud', ignore=shutil.ignore_patterns('__pycache__'))
train = repo / 'data/raw/train.parquet'
train.parent.mkdir(parents=True)
sys.path.insert(0, str(repo))
from members.mrcyrilgoud.kaggle_data import REQUIRED_COLUMNS
for name, columns in REQUIRED_COLUMNS.items():
    frame = pd.DataFrame({c: ['fixture'] for c in columns})
    if name.endswith('.parquet'):
        frame.to_parquet(train.parent / name)
    else:
        frame.to_csv(train.parent / name, index=False)
env = dict(Path=Path, STAGE='smoke', DRIVE_MOUNT=mount, PROJECT=project,
           REPO=repo, TRAIN=train, OUTPUT=project / 'cyril')
with patch('subprocess.check_call'), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'get_device_name', return_value='mock GPU'):
    exec(payload['setup'], env)
    exec(payload['setup'], env)  # same-source rerun is safe
import src.data.dataset
assert Path(src.data.dataset.__file__).is_relative_to(repo)
assert Path.cwd().resolve() == repo.resolve()
assert not train.is_symlink()
assert Path(tempfile.gettempdir()).is_relative_to(project)
for key in ('PIP_CACHE_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'KAGGLEHUB_CACHE'):
    assert Path(os.environ[key]).is_relative_to(project)
assert not (mount.parent / 'local').exists()
"""
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as tmp:
                setup = tagged_cell(name, 'setup')
                payload = dict(setup=setup, mount=str(Path(tmp) / 'drive'), source=str(MEMBER.parents[1]))
                result = subprocess.run([sys.executable, '-c', script], input=json.dumps(payload),
                    text=True, capture_output=True, cwd=tmp)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_drive_source_change_requires_restart(self):
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as tmp:
                mount = Path(tmp) / 'drive'
                project = mount / 'MyDrive/Data266/EnvedaGroupProject'
                repo = project / 'repo'
                p = repo / 'members/mrcyrilgoud/model_mlp.py'
                p.parent.mkdir(parents=True)
                p.write_text('# original\n')
                env = dict(Path=Path, DRIVE_MOUNT=mount, PROJECT=project, REPO=repo,
                           TRAIN=repo / 'data/raw/train.parquet', OUTPUT=project / 'cyril')
                bootstrap = self.setup_prefix(name)
                exec(bootstrap, env)
                exec(bootstrap, env)
                p.write_text('# updated\n')
                with self.assertRaisesRegex(RuntimeError, 'restart|Restart'):
                    exec(bootstrap, env)

    def test_imported_module_refuses_to_hash_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model_mlp.py'
            p.write_bytes((MEMBER / 'model_mlp.py').read_bytes())
            spec = importlib.util.spec_from_file_location('regression_model', p)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.implementation_hashes()
            p.write_text(p.read_text() + '\n# Changed after import\n')
            with self.assertRaisesRegex(RuntimeError, 'restart|Restart'):
                module.implementation_hashes()

    def test_handoff_uses_recorded_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m.save_json(root / 'E103/selection.json', {'model': 'E101', 'alpha': .1})
            metrics = dict(mrr=.8, top1=.7, top10=.9, top25=1.)
            for folder in ('baseline', 'E101/evaluation', 'E102/evaluation', 'E103/0.1'):
                m.save_json(root / folder / 'metrics.json', metrics)
            for exp in ('E101', 'E102'):
                m.save_json(root / exp / 'training.json', dict(data={'train_rows': 25}, epochs=2,
                    best_epoch=1, hardware='cpu', elapsed_seconds=1, checkpoint_sha256='fixture',
                    config={'lr': .002, 'batch_size': 32}))
                (root / exp / 'training_curve.png').write_bytes(b'fixture')
            (root / 'E103/blend_curve.png').write_bytes(b'fixture')
            with patch.object(m, 'plot_training'):
                report = (m.export_handoff(root) / 'results.md').read_text()
            self.assertIn('lr=0.002, batch=32', report)
            self.assertNotIn('batch=256', report)


if __name__ == '__main__':
    unittest.main()
