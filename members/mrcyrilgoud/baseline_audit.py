"""Compare the original baseline ranking with the shared scorer on identical queries."""
import importlib.metadata
import inspect
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd

from members.mrcyrilgoud.model_mlp import file_hash, save_json


def summarize(per, prefix):
    ranks = per[prefix + '_rank']
    return dict(mrr=float(per[prefix + '_rr'].mean()), top1=float((ranks == 1).mean()),
                top10=float((ranks <= 10).mean()), top25=float(ranks.notna().mean()))


def compare_rankings(queries, candidates):
    from src.evaluation.validate import rank
    from src.models.base import CosineScorer
    rows = []
    for molecule, spectra in queries.items():
        c = candidates[molecule]
        # candidate_table preserves query_molecule's original cosine + frequency order.
        orders = {'legacy': c, 'shared': rank(CosineScorer(), spectra, c)}
        row = dict(molecule=molecule, n_candidates=len(c))
        for label, ordered in orders.items():
            keys = ordered.inchikey14.tolist()[:25]
            pos = keys.index(molecule) + 1 if molecule in keys else None
            row[label + '_rank'] = pos
            row[label + '_rr'] = 1.0 / pos if pos else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def audit_baseline(train_path, ref_dir, out_dir, expected_molecules=250, train_sha256=None):
    """Reload queries explicitly: the shared evaluator's cache omits train_path from its key."""
    from src.evaluation import validate
    from src.models import base, library_search
    from src.data import build_reference, spectra
    train_path, ref_dir, out_dir = Path(train_path), Path(ref_dir), Path(out_dir)
    print('Baseline audit: loading validation and reference...', flush=True)
    validate._CACHE.clear()
    queries, candidates = validate.load_validation(str(ref_dir), train_path=str(train_path), ppm=15)
    if len(queries) != expected_molecules:
        raise ValueError(f'Expected {expected_molecules} validation molecules, got {len(queries)}')
    per = compare_rankings(queries, candidates)
    shared, legacy = summarize(per, 'shared'), summarize(per, 'legacy')
    changed = per[per.shared_rr != per.legacy_rr]
    reference_files = ('ref_meta.parquet', 'structures.parquet', 'ref_peaks.npz')
    manifest = json.loads((ref_dir / 'source.json').read_text())
    # Notebooks may pass the digest they just computed while checking source.json.
    print('Baseline audit: verifying data and reference identities...', flush=True)
    train_digest = train_sha256 or file_hash(train_path)
    if manifest['train_sha256'] != train_digest:
        raise ValueError('Reference was built from a different training parquet')
    if (manifest['builder_sha256'] != file_hash(inspect.getfile(build_reference))
            or manifest['cleaner_sha256'] != file_hash(inspect.getfile(spectra))):
        raise ValueError('Reference preprocessing source has changed')
    explained = abs(legacy['mrr'] - 0.8789) < 0.00005
    report = dict(
        molecules=len(per), spectra=sum(map(len, queries.values())), ppm=15, k=25,
        shared=shared, legacy=legacy, changed_molecules=len(changed),
        ranking_delta=shared['mrr'] - legacy['mrr'], historical_e002_mrr=0.8789,
        legacy_minus_historical=legacy['mrr'] - 0.8789,
        conclusion=('Original ranking reproduces E002; ranking policy explains the difference.'
                    if explained else 'Ranking policy comparison measured; the remaining historical '
                    'difference is not established without the original environment/reference.'),
        protocol=dict(validation_library=validate.VAL_LIB, reference_excludes=[validate.VAL_LIB],
                      max_ref_per_structure=50,
                      shared_ranking='cosine, then n_spectra for ties',
                      legacy_ranking='cosine + 0.001 * log1p(n_spectra)'),
        train_sha256=train_digest, reference_source=manifest,
        reference_sha256={name: file_hash(ref_dir / name) for name in reference_files},
        source_sha256={module.__name__: file_hash(inspect.getfile(module))
                       for module in (validate, base, library_search, build_reference, spectra)},
        # Equal-key reference sorting and the first-50 cutoff can change the actual
        # spectra compared across NumPy/platform versions. Bind comparisons to scores.
        candidate_sha256=hashlib.sha256(''.join(
            molecule + candidates[molecule].to_csv(index=False, float_format='%.17g')
            for molecule in queries).encode()).hexdigest(),
        machine=platform.machine(),
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'pyarrow')})
    previous_path = out_dir / 'audit.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        require_same_baseline(previous, report)
    save_json(out_dir / 'audit.json', report)
    per.to_csv(out_dir / 'per_molecule.csv', index=False)
    changed.to_csv(out_dir / 'changed_molecules.csv', index=False)
    print(f"Shared cosine MRR@25: {shared['mrr']:.6f}; original ranking: {legacy['mrr']:.6f}; "
          f"changed molecules: {len(changed)}")
    print(report['conclusion'])
    return report


def require_same_baseline(previous, current):
    """Never mix E101 and E102 results made with different reference rankings."""
    for key in ('train_sha256', 'reference_sha256', 'source_sha256', 'candidate_sha256',
                'versions', 'machine', 'shared', 'molecules', 'ppm', 'k'):
        if previous.get(key) != current.get(key):
            raise ValueError(f'Baseline context changed ({key}). Use the original environment '
                             'or a new CASMI_OUTPUT and rerun both models there.')


def probe_reference_order(train_path, ref_dir, out_dir):
    """Measure stable reference grouping without changing the team's shared code."""
    from src.evaluation import validate
    from src.models.library_search import ReferenceLibrary, candidate_table
    validate._CACHE.clear()
    queries, original = validate.load_validation(str(ref_dir), train_path=str(train_path))
    lib = ReferenceLibrary(str(ref_dir), exclude_libs=[validate.VAL_LIB])
    meta = pd.read_parquet(Path(ref_dir) / 'ref_meta.parquet')
    meta = meta[meta.ingest_lib != validate.VAL_LIB]
    meta = meta.sort_values('struct_idx', kind='stable').reset_index(drop=True)
    # Structure counts and group boundaries are unchanged; only within-group order changes.
    lib.meta = meta
    lib.mode = meta.ionization_mode.values
    lib.pstart, lib.plen = meta.peak_start.values, meta.n_peaks.values
    stable = {mol: candidate_table(lib, spectra) for mol, spectra in queries.items()}
    original_per = compare_rankings(queries, original)
    stable_per = compare_rankings(queries, stable)
    per = original_per[['molecule', 'shared_rr']].rename(columns={'shared_rr': 'default_rr'})
    per['stable_rr'] = stable_per.shared_rr.values
    result = dict(default=summarize(original_per, 'shared'),
                  stable=summarize(stable_per, 'shared'),
                  changed_molecules=int((per.default_rr != per.stable_rr).sum()),
                  note='Sensitivity diagnostic only. Final model evaluation uses the shared default ranking.')
    save_json(Path(out_dir) / 'reference_order.json', result)
    per.to_csv(Path(out_dir) / 'reference_order_per_molecule.csv', index=False)
    print(result)
    return result


def final_training_check(manifest, stage):
    settings = manifest['settings']
    if stage == 'final' and settings['sample'] is not None:
        raise ValueError('Final training requires all eligible spectra (sample=None)')
    if settings['max_per_structure'] != 3 or settings['val_frac'] != 0.05:
        raise ValueError('Use the agreed three-spectra cap and 5% structure split')
    if manifest['train_rows'] <= 0 or manifest['val_rows'] <= 0:
        raise ValueError('Both training and early-stopping rows are required')
    print(f"{stage}: {manifest['train_rows']:,} training rows, {manifest['val_rows']:,} "
          f"early-stopping rows, {manifest['structures']:,} structures")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--order-sensitivity', action='store_true')
    args = ap.parse_args()
    audit_baseline(args.train, args.ref, args.out)
    if args.order_sensitivity:
        probe_reference_order(args.train, args.ref, args.out)
