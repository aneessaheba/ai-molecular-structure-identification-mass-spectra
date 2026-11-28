"""CPU checks using synthetic spectra, never competition-result evidence.

Run: python -m unittest members.mrcyrilgoud.test_model_mlp -v
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from members.mrcyrilgoud import model_mlp as m
from src.data.spectra import formula_mass, PROTON
from src.features.featurize import morgan_fps, tanimoto


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.spectrum = dict(mz=np.array([0., 1., 82., 99.8, 100., 100.2]),
            intensity=np.ones(6, np.float32), precursor_mz=100., adduct='[M+H]+',
            collision_energy=None)

    def test_features_and_loss_boundaries(self):
        self.assertEqual(m.features(self.spectrum, 'spectrum').shape, (1000,))
        x = m.features(self.spectrum, 'loss_metadata')
        self.assertEqual(x.shape, (1512,))
        self.assertEqual(x[1000], 2.)  # loss zero and +0.2, excludes -0.2
        self.assertEqual(x[-2], 0.)
        edge = dict(self.spectrum, precursor_mz=600., mz=np.array([100., 100.1, 600.1]), intensity=np.ones(3))
        self.assertEqual(m.features(edge, 'loss_metadata')[1000:1500].sum(), 1.)

    def test_training_resume_scoring_and_submission(self):
        from src.data.build_reference import build
        from src.evaluation import validate
        from src.models.base import CosineScorer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            # Unique real structures with enough groups for the 5% validation split.
            for smiles in ['C' * n + 'O' for n in range(2, 27)]:
                mol = Chem.MolFromSmiles(smiles)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                rows.append(dict(ingest_lib='gnps', normalized_smiles=smiles,
                    inchikey14=Chem.MolToInchiKey(mol).split('-')[0], molecular_formula=formula,
                    ionization_mode='positive', adduct='[M+H]+', precursor_error_ppm=0.,
                    precursor_mz=formula_mass(formula) + PROTON, collision_energy_ev=[20.],
                    ms2_mzs=[10., 20., 30.], ms2_normalized_intensities=[1., .5, .2]))
            rows.append(dict(rows[0], ingest_lib='enveda-np-examples'))
            train = root / 'train.parquet'
            pd.DataFrame(rows).to_parquet(train)
            cache = root / 'cache'
            manifest = m.prepare_cache(train, cache)
            self.assertEqual(manifest, m.prepare_cache(train, cache))
            identities = pd.read_csv(cache / 'rows.csv')
            self.assertNotIn(rows[0]['inchikey14'], identities.inchikey14.tolist())
            self.assertFalse(set(identities[identities.is_val].inchikey14) & set(identities[~identities.is_val].inchikey14))
            runs = root / 'runs'
            # Compare uninterrupted and interrupted training, including dropout/shuffle RNG.
            m.train_model(cache, runs / 'whole', 'spectrum', epochs=2, batch_size=32, device='cpu')
            m.train_model(cache, runs / 'E101', 'spectrum', epochs=1, batch_size=32, device='cpu')
            net = m.train_model(cache, runs / 'E101', 'spectrum', epochs=2, batch_size=32, device='cpu', resume=True)
            whole = torch.load(runs / 'whole/last.pt', weights_only=False)
            resumed = torch.load(runs / 'E101/last.pt', weights_only=False)
            self.assertEqual(whole['history'], resumed['history'])
            for k in whole['state_dict']:
                torch.testing.assert_close(whole['state_dict'][k], resumed['state_dict'][k], rtol=0, atol=0)
            self.assertLess(resumed['history'][-1]['train'], resumed['history'][0]['train'])
            best = torch.load(runs / 'E101/best.pt', weights_only=True)
            for k, value in net.state_dict().items():
                torch.testing.assert_close(value, best['state_dict'][k])
            scorer = m.FingerprintMLP(runs / 'E101/best.pt')
            candidates = pd.DataFrame({'smiles': ['CCO', 'invalid smiles', 'CCCCO'], 'cosine': [.2, .8, .5]}, index=[8, 3, 2])
            spectra = [self.spectrum, dict(self.spectrum, intensity=self.spectrum['intensity'] * .1)]
            with torch.inference_mode():
                expected = scorer.net(torch.tensor(np.stack([m.features(s, 'spectrum') for s in spectra]))).sigmoid().mean(0).numpy()
            np.testing.assert_allclose(scorer.score(spectra, candidates), tanimoto(expected, morgan_fps(candidates.smiles.tolist())))
            scores = scorer.score(spectra, candidates)
            np.testing.assert_allclose(scorer.score(spectra, candidates.iloc[::-1]), scores[::-1])
            self.assertEqual(scores[1], 0.)
            self.assertEqual(scorer.score([], candidates.iloc[:0]).shape, (0,))
            np.testing.assert_allclose(m.BlendScorer(scorer, .3).score(spectra, candidates), .3 * scores + .7 * candidates.cosine)
            # Exercise richer input training and scorer loading too.
            m.train_model(cache, runs / 'E102', 'loss_metadata', epochs=1, device='cpu')
            self.assertTrue(np.isfinite(m.FingerprintMLP(runs / 'E102/best.pt').score(spectra, candidates)).all())
            ref = root / 'reference'
            build(str(train), str(ref))
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                Path('data/raw').mkdir(parents=True)
                Path('data/raw/train.parquet').symlink_to(train)
                for exp in ['E101', 'E102']:
                    model = m.FingerprintMLP(runs / exp / 'best.pt')
                    result = m.evaluate_saved(model, runs / exp / 'evaluation', runs / exp / 'best.pt', ref_dir=str(ref), ppm=17)
                    self.assertEqual(result['mrr'], 1.)
                m.evaluate_saved(CosineScorer(), runs / 'baseline', ref_dir=str(ref), ppm=17)
                curve = m.evaluate_blends(runs, ref)
                self.assertEqual(len(curve), 6)
                blend_metrics = json.loads((runs / 'E103/0.1/metrics.json').read_text())
                self.assertEqual(blend_metrics['evaluation_protocol']['ppm'], 17)
                selection = json.loads((runs / 'E103/selection.json').read_text())
                self.assertEqual(selection['model'], 'E101')
                self.assertEqual(selection['alpha'], .1)
                package = m.package_weights(runs, root / 'upload')
                self.assertTrue((package / 'checksums.json').exists())
                # A changed losing model must invalidate both selection and packaging.
                losing = runs / 'E102/best.pt'
                original_checkpoint = losing.read_bytes()
                losing.write_bytes(original_checkpoint + b'changed')
                with self.assertRaisesRegex(ValueError, 'stale|checkpoint'):
                    m.evaluate_blends(runs, ref)
                with self.assertRaisesRegex(ValueError, 'stale|checkpoint'):
                    m.package_weights(runs, root / 'stale-upload')
                losing.write_bytes(original_checkpoint)
                # Matching checkpoints from different data splits are not comparable.
                original_metrics = (runs / 'E102/evaluation/metrics.json').read_bytes()
                wrong = json.loads(original_metrics)
                wrong['split_id'] = 'different split'
                m.save_json(runs / 'E102/evaluation/metrics.json', wrong)
                with self.assertRaisesRegex(ValueError, 'split|identity'):
                    m.evaluate_blends(runs, ref)
                (runs / 'E102/evaluation/metrics.json').write_bytes(original_metrics)
                # Reject evaluations made with another candidate/validation protocol.
                wrong = json.loads(original_metrics)
                wrong['evaluation_protocol'] = {'ppm': 999}
                m.save_json(runs / 'E102/evaluation/metrics.json', wrong)
                with self.assertRaisesRegex(ValueError, 'protocol|identity'):
                    m.evaluate_blends(runs, ref)
                (runs / 'E102/evaluation/metrics.json').write_bytes(original_metrics)
                # Even reevaluated metrics require reselecting the blend before upload.
                revised = json.loads(original_metrics)
                revised['mrr'] = 0.5
                m.save_json(runs / 'E102/evaluation/metrics.json', revised)
                with self.assertRaisesRegex(ValueError, 'stale'):
                    m.package_weights(runs, root / 'reselected-upload')
                (runs / 'E102/evaluation/metrics.json').write_bytes(original_metrics)
                handoff = m.export_handoff(runs)
                self.assertIn('E103', (handoff / 'results.md').read_text())
                m.plot_training(runs / 'E101')
                test = pd.DataFrame([dict(rows[0], molecule_id='test1')])
                test.to_parquet(root / 'test.parquet')
                sub = validate.make_submission(scorer, str(root / 'test.parquet'), str(ref), str(root / 'submission.csv'))
                self.assertEqual(sub.molecule_id.tolist(), ['test1'])
                self.assertEqual(sub.smiles.tolist(), ['CCO'])
                # Execute the actual offline notebook cells against packaged weights and
                # synthetic data; only redirect Kaggle's output directory to this temp root.
                notebook = json.loads((old_cwd / 'members/mrcyrilgoud/03_kaggle_submission.ipynb').read_text())
                env = dict(Path=Path, WEIGHTS=package, SOURCE=old_cwd,
                           TEST_PATH=root / 'test.parquet', WHEELS=None, EXPERIMENT='E103')
                (root / 'working').mkdir()
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code' or set(cell.get('metadata', {}).get('tags', [])) & {'environment', 'config'}:
                        continue
                    source = ''.join(cell['source']).replace('/kaggle/working', str(root / 'working'))
                    exec(compile(source, '<offline-notebook>', 'exec'), env)
                self.assertTrue((root / 'working/submission.csv').exists())
                # Force validation to worsen to exercise patience and best restoration.
                real_loss = torch.nn.BCEWithLogitsLoss()
                validation_losses = iter([0.1, 0.2])
                def controlled_loss(logits, target):
                    return real_loss(logits, target) if torch.is_grad_enabled() else torch.tensor(next(validation_losses))
                with patch.object(m.nn, 'BCEWithLogitsLoss', return_value=controlled_loss):
                    restored = m.train_model(cache, runs / 'early', 'spectrum', epochs=5,
                                             batch_size=32, patience=1, device='cpu')
                early_best = torch.load(runs / 'early/best.pt', weights_only=True)
                self.assertEqual(early_best['epoch'], 1)
                self.assertEqual(len(json.loads((runs / 'early/history.json').read_text())), 2)
                for key, value in restored.state_dict().items():
                    torch.testing.assert_close(value, early_best['state_dict'][key])
            finally:
                os.chdir(old_cwd)
                validate._CACHE.clear()
            # Fail closed when a completed cache is corrupted.
            with open(cache / 'target_idx.npy', 'ab') as f:
                f.write(b'corruption')
            with self.assertRaises(ValueError):
                m.prepare_cache(train, cache)


if __name__ == '__main__':
    unittest.main()
