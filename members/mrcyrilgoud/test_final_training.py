"""Checks for final training setup and baseline comparability."""
import copy
import json
import os
from pathlib import Path
import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from members.mrcyrilgoud.baseline_audit import (
    audit_baseline, compare_rankings, final_training_check, probe_reference_order,
    require_same_baseline, summarize,
)
from members.mrcyrilgoud.model_mlp import file_hash, save_json
from src.models.library_search import query_molecule


class FinalTrainingTests(unittest.TestCase):
    def test_audit_reads_real_parquet_and_preserves_a_previous_context(self):
        from src.data import build_reference, spectra
        from src.evaluation import validate
        row = dict(ingest_lib='reference', normalized_smiles='CCO', inchikey14='truth',
            molecular_formula='C2H6O', ionization_mode='positive', adduct='[M+H]+',
            precursor_mz=spectra.formula_mass('C2H6O') + spectra.PROTON,
            ms2_mzs=[10., 20., 30.], ms2_normalized_intensities=[1., .5, .2])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, ref, out = root / 'train.parquet', root / 'ref', root / 'audit'
            pd.DataFrame([row, dict(row, ingest_lib='enveda-np-examples')]).to_parquet(train)
            build_reference.build(str(train), str(ref))
            save_json(ref / 'source.json', dict(train_sha256=file_hash(train),
                builder_sha256=file_hash(inspect.getfile(build_reference)),
                cleaner_sha256=file_hash(inspect.getfile(spectra))))
            try:
                result = audit_baseline(train, ref, out, expected_molecules=1)
                self.assertEqual(result['shared']['mrr'], 1.)
                self.assertEqual(result['spectra'], 1)
                self.assertEqual(result, audit_baseline(train, ref, out, expected_molecules=1))
                probe = probe_reference_order(train, ref, out)
                self.assertEqual(probe['default'], probe['stable'])
                self.assertEqual(probe['changed_molecules'], 0)
                original = (out / 'audit.json').read_bytes()
                with self.assertRaisesRegex(ValueError, 'Expected 250'):
                    audit_baseline(train, ref, out)
                self.assertEqual((out / 'audit.json').read_bytes(), original)
            finally:
                validate._CACHE.clear()

    def test_ranking_comparison_keeps_legacy_order_and_shared_tie_break(self):
        queries = {'truth': []}
        candidates = {'truth': pd.DataFrame(dict(
            inchikey14=['truth', 'other'], cosine=[0.5000, 0.5001], n_spectra=[1000, 1]))}
        per = compare_rankings(queries, candidates)
        self.assertEqual(summarize(per, 'legacy')['mrr'], 1.)
        self.assertEqual(summarize(per, 'shared')['mrr'], .5)

    def test_first_fifty_reference_cutoff_depends_on_order(self):
        class Library:
            prior = np.zeros(1)
            mode = ['positive'] * 51

            def candidates(self, mass, ppm):
                return np.array([0])

            def spectra_of(self, s):
                return self.order

            def peaks(self, j):
                return np.array([100. if j == 50 else 10.]), np.ones(1)

        lib = Library()
        query = dict(mz=np.array([100.]), intensity=np.ones(1),
                     precursor_mz=101.007276, adduct='[M+H]+', ionization_mode='positive')
        lib.order = list(range(51))
        self.assertEqual(query_molecule(lib, [query])[0][1], 0.)
        lib.order = list(reversed(lib.order))
        self.assertEqual(query_molecule(lib, [query])[0][1], 1.)

    def test_changed_baseline_context_requires_separate_run(self):
        context = dict(train_sha256='train', reference_sha256={'peaks': 'ref'},
            source_sha256={'scorer': 'source'}, candidate_sha256='candidates',
            versions={'numpy': 'version'}, machine='machine', shared={'mrr': .8},
            molecules=250, ppm=15, k=25)
        require_same_baseline(context, copy.deepcopy(context))
        for key in context:
            with self.subTest(key=key):
                changed = copy.deepcopy(context)
                changed[key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'Baseline context changed'):
                    require_same_baseline(context, changed)

    def test_final_rejects_sampled_cache(self):
        manifest = dict(settings=dict(sample=None, max_per_structure=3, val_frac=.05),
                        train_rows=190, val_rows=10, structures=100)
        final_training_check(manifest, 'final')
        manifest['settings']['sample'] = 2000
        with self.assertRaisesRegex(ValueError, 'all eligible'):
            final_training_check(manifest, 'final')
        final_training_check(manifest, 'smoke')

    def test_notebooks_default_to_final_and_preserve_environment_overrides(self):
        member = Path(__file__).resolve().parent
        for name in ('01_fingerprint_mlp.ipynb', '02_losses_metadata_mlp.ipynb'):
            nb = json.loads((member / name).read_text())
            cells = {t: c for c in nb['cells'] for t in c.get('metadata', {}).get('tags', [])}
            config = ''.join(cells['config']['source'])
            with self.subTest(name=name), patch.dict(os.environ, {}, clear=True):
                namespace = {}
                exec(config, namespace)
                self.assertEqual(namespace['STAGE'], 'final')
                self.assertIsNone(namespace['SAMPLES'][namespace['STAGE']])
                self.assertEqual(namespace['EPOCHS'][namespace['STAGE']], 30)
                self.assertIsNone(namespace['RESUME'])
                self.assertEqual(namespace['MODEL_CONFIG'], dict(batch_size=256, lr=.001, patience=3, seed=0))
                os.environ.update(CASMI_STAGE='smoke', CASMI_RESUME='false')
                exec(config, namespace)
                self.assertEqual(namespace['STAGE'], 'smoke')
                self.assertFalse(namespace['RESUME'])
                tree_source = ''.join(cells['environment']['source'])
                self.assertNotIn('KGAT_', tree_source)
                self.assertTrue(all(not c.get('outputs') for c in nb['cells']))


if __name__ == '__main__':
    unittest.main()
