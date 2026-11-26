"""Cyril's spectrum-to-fingerprint models. Run notebooks from the repository root."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.features.featurize import bin_spectrum, metadata_vector, morgan_fps, tanimoto
from src.models.base import Scorer

DIMS = {"spectrum": 1000, "loss_metadata": 1512}
FEATURE_SETTINGS = {"version": 1, "bits": 2048, "radius": 2, "top_n": 64,
                    "min_rel": 0.01, "bin_size": 1, "max_mz": 1000, "max_loss": 500}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _implementation_files():
    from src.data import dataset, spectra
    from src.features import featurize
    return {"model_mlp": __file__, **{module.__name__: inspect.getfile(module)
            for module in (dataset, spectra, featurize)}}


# Capture once at import, rather than certifying newly overwritten source files.
_LOADED_SOURCE_HASHES = {name: file_hash(path) for name, path in _implementation_files().items()}


def implementation_hashes():
    current = {name: file_hash(path) for name, path in _implementation_files().items()}
    if current != _LOADED_SOURCE_HASHES:
        raise RuntimeError('Source changed since import. Restart the runtime before continuing.')
    return dict(_LOADED_SOURCE_HASHES)


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def features(spectrum, variant):
    if variant not in DIMS:
        raise ValueError(f'Unknown variant: {variant}')
    mz = np.asarray(spectrum['mz'])
    intensity = np.asarray(spectrum['intensity'])
    x = bin_spectrum(mz, intensity)
    if variant == 'loss_metadata':
        losses = spectrum['precursor_mz'] - mz
        keep = (losses >= 0) & (losses < 500)
        x = np.concatenate([x, bin_spectrum(losses[keep], intensity[keep], max_mz=500),
                            metadata_vector(spectrum['adduct'], spectrum['collision_energy'],
                                            spectrum['precursor_mz'])])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite spectrum features')
    return x.astype(np.float32)


def prepare_cache(train_path, cache_dir, sample=None, seed=0):
    """Load once; both variants share exact rows, targets, and structure splits."""
    from src.data.dataset import load_training, split_by_structure, validation_keys
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Content hash catches changed input even when its path and size stay unchanged.
    settings = dict(FEATURE_SETTINGS, source_sha256=file_hash(train_path), sample=sample,
                    seed=seed, max_per_structure=3, val_frac=0.05,
                    implementation=implementation_hashes())
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            raise ValueError('Cache settings changed; select a new cache directory.')
        for name, digest in manifest['files'].items():
            if not (root / name).is_file() or file_hash(root / name) != digest:
                raise ValueError(f'Incomplete or corrupted cache: {name}')
        return manifest
    df = load_training(train_path=str(train_path), max_per_structure=3, sample=sample, seed=seed)
    tr, va = split_by_structure(df, val_frac=0.05, seed=seed)
    train_keys, val_keys = set(tr.inchikey14), set(va.inchikey14)
    forbidden = validation_keys(str(train_path))
    assert train_keys and val_keys, 'Use at least 20 unique structures.'
    assert not train_keys & val_keys
    assert not (train_keys | val_keys) & forbidden
    is_val = df.inchikey14.isin(val_keys).to_numpy()
    np.save(root / 'train_idx.npy', np.flatnonzero(~is_val))
    np.save(root / 'val_idx.npy', np.flatnonzero(is_val))
    codes, unique_smiles = pd.factorize(df.smiles, sort=True)
    if (codes < 0).any():
        raise ValueError('Missing SMILES in training rows')
    np.save(root / 'target_idx.npy', codes)
    target = np.lib.format.open_memmap(root / 'targets.npy', mode='w+', dtype='uint8',
                                      shape=(len(unique_smiles), 2048))
    for start in range(0, len(unique_smiles), 2048):
        target[start:start + 2048] = morgan_fps(unique_smiles[start:start + 2048].tolist())
    target.flush()
    # Row content digests distinguish repeated acquisitions with identical metadata.
    row_ids = []
    for values in df.itertuples(index=False):
        row = values._asdict()
        h = hashlib.sha256()
        h.update(str([row[k] for k in ('inchikey14', 'smiles', 'adduct', 'precursor_mz',
                                       'collision_energy', 'ingest_lib')]).encode())
        h.update(np.asarray(row['mz'], dtype=np.float32).tobytes())
        h.update(np.asarray(row['intensity'], dtype=np.float32).tobytes())
        row_ids.append(h.hexdigest())
    pd.DataFrame({'row_id': row_ids, 'inchikey14': df.inchikey14, 'smiles': df.smiles,
                  'is_val': is_val}).to_csv(root / 'rows.csv', index=False)
    for variant, dim in DIMS.items():
        x = np.lib.format.open_memmap(root / f'{variant}.npy', mode='w+', dtype='float32',
                                     shape=(len(df), dim))
        for start in range(0, len(df), 2048):
            x[start:start + 2048] = np.stack([features(r, variant)
                for r in df.iloc[start:start + 2048].to_dict('records')])
        x.flush()
        del x
    files = ['train_idx.npy', 'val_idx.npy', 'target_idx.npy', 'targets.npy', 'rows.csv',
             'spectrum.npy', 'loss_metadata.npy']
    manifest = {'settings': settings, 'rows': len(df), 'structures': int(df.inchikey14.nunique()),
                'train_rows': int((~is_val).sum()), 'val_rows': int(is_val.sum()),
                'files': {name: file_hash(root / name) for name in files}}
    manifest['split_id'] = hashlib.sha256((''.join(row_ids) + str(sorted(val_keys))).encode()).hexdigest()
    save_json(manifest_path, manifest)  # Written last: interrupted caches are rebuilt.
    return manifest


class CachedSpectra(Dataset):
    def __init__(self, cache_dir, variant, split):
        root = Path(cache_dir)
        self.x = np.load(root / f'{variant}.npy', mmap_mode='r')
        self.y = np.load(root / 'targets.npy', mmap_mode='r')
        self.target_idx = np.load(root / 'target_idx.npy', mmap_mode='r')
        self.indices = np.load(root / f'{split}_idx.npy')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return self.x[row].copy(), self.y[self.target_idx[row]].astype(np.float32)


def network(variant):
    return nn.Sequential(nn.Linear(DIMS[variant], 1024), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(1024, 2048))


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def train_model(cache_dir, run_dir, variant, epochs=30, batch_size=256, lr=1e-3,
                patience=3, seed=0, device=None, resume=False):
    """Resume only trusted, locally generated last.pt (contains Python/NumPy RNG state)."""
    seed_all(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(cache_dir) / 'manifest.json').read_text())
    config = dict(variant=variant, batch_size=batch_size, lr=lr, patience=patience, seed=seed,
                  implementation=implementation_hashes(),
                  split_id=manifest['split_id'], cache_sha256=file_hash(Path(cache_dir) / 'manifest.json'))
    net = network(variant).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loaders = {s: DataLoader(CachedSpectra(cache_dir, variant, s), batch_size=batch_size,
                            shuffle=(s == 'train'), num_workers=0) for s in ('train', 'val')}
    start, best, bad, history, elapsed = 0, float('inf'), 0, [], 0.0
    if resume:
        state = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        if state['config'] != config or not (root / 'best.pt').exists():
            raise ValueError('Resume requires matching configuration and best.pt')
        net.load_state_dict(state['state_dict'])
        opt.load_state_dict(state['optimizer'])
        start, best, bad = state['epoch'], state['best_loss'], state['bad_epochs']
        history, elapsed = state['history'], state['elapsed_seconds']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'].cpu())
        if torch.cuda.is_available() and state['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
    elif (root / 'last.pt').exists():
        raise FileExistsError('Run exists; set resume=True or use a new directory.')
    criterion = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for epoch in range(start, epochs):
        if bad >= patience:
            break
        losses = {}
        for split, loader in loaders.items():
            net.train(split == 'train')
            total = 0.0
            with torch.set_grad_enabled(split == 'train'):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if split == 'train':
                        opt.zero_grad()
                    loss = criterion(net(x), y)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    if split == 'train':
                        loss.backward()
                        if not all(torch.isfinite(p.grad).all() for p in net.parameters()):
                            raise ValueError('Nonfinite gradients')
                        opt.step()
                    total += loss.item() * len(x)
            losses[split] = total / len(loader.dataset)
        history.append(dict(epoch=epoch + 1, **losses))
        improved = losses['val'] < best
        best, bad = (losses['val'], 0) if improved else (best, bad + 1)
        inference = dict(state_dict=net.state_dict(), variant=variant,
                         feature_settings=FEATURE_SETTINGS, config=config, epoch=epoch + 1)
        if improved:
            atomic_save(inference, root / 'best.pt')
        state = dict(inference, optimizer=opt.state_dict(), best_loss=best, bad_epochs=bad,
                     history=history, elapsed_seconds=elapsed + time.time() - t0,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(state, root / 'last.pt')
        save_json(root / 'history.json', history)
        print(f"epoch {epoch + 1}: train={losses['train']:.5f}, val={losses['val']:.5f}", flush=True)
    best_state = torch.load(root / 'best.pt', map_location=device, weights_only=True)
    net.load_state_dict(best_state['state_dict'])
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas', 'rdkit', 'pyarrow')}
    save_json(root / 'training.json', dict(config=config, epochs=len(history), best_epoch=best_state['epoch'],
        best_val_loss=best, elapsed_seconds=elapsed + time.time() - t0, versions=versions,
        hardware=torch.cuda.get_device_name() if str(device).startswith('cuda') else str(device),
        data={k: manifest[k] for k in ('rows', 'structures', 'train_rows', 'val_rows')},
        checkpoint_sha256=file_hash(root / 'best.pt')))
    return net.eval()


class FingerprintMLP(Scorer):
    name = 'fp-mlp'

    def __init__(self, weights_path, device='cpu'):
        self.device = device
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        if checkpoint['feature_settings'] != FEATURE_SETTINGS:
            raise ValueError('Checkpoint preprocessing does not match this module')
        if checkpoint['config']['implementation'] != implementation_hashes():
            raise ValueError('Checkpoint source differs; use the exact training module and shared source')
        self.variant = checkpoint['variant']
        self.name = 'fp-mlp-' + self.variant
        self.net = network(self.variant).to(device)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.eval()

    def score(self, spectra, candidates):
        if len(candidates) == 0:
            return np.empty(0, dtype=np.float32)
        if not spectra:
            raise ValueError('At least one spectrum is required')
        x = np.stack([features(s, self.variant) for s in spectra])
        with torch.inference_mode():
            prob = self.net(torch.from_numpy(x).to(self.device)).sigmoid().mean(0).cpu().numpy()
        # Shared Morgan helper caches fingerprint bytes per SMILES.
        result = tanimoto(prob, morgan_fps(candidates.smiles.tolist()))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite candidate scores')
        return result


class BlendScorer(Scorer):
    def __init__(self, model, alpha):
        self.model, self.alpha = model, float(alpha)
        self.name = f'{model.name}-blend-{alpha:g}'

    def score(self, spectra, candidates):
        return self.alpha * self.model.score(spectra, candidates) + (1 - self.alpha) * candidates.cosine.to_numpy()


def reference_identity(ref_dir):
    root = Path(ref_dir)
    marker = root / 'source.json'
    if marker.exists():
        return {'source_manifest': file_hash(marker)}
    return {name: file_hash(root / name) for name in
            ('ref_meta.parquet', 'ref_peaks.npz', 'structures.parquet')}


def evaluate_saved(scorer, out_dir, checkpoint=None, **kwargs):
    from src.evaluation.validate import evaluate
    implementation_hashes()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = evaluate(scorer, **kwargs)
    per = result.pop('per_molecule')
    per.to_csv(root / 'per_molecule.csv', index=False)
    result['evaluation_protocol'] = dict(
        reference=reference_identity(kwargs.get('ref_dir', 'data/processed')),
        ppm=kwargs.get('ppm', 15), hide_true=kwargs.get('hide_true', False),
        limit=kwargs.get('limit'), k=25,
        molecules_sha256=hashlib.sha256(json.dumps(per.molecule.tolist()).encode()).hexdigest())
    if checkpoint is not None:
        result['checkpoint_sha256'] = file_hash(checkpoint)
        config = torch.load(checkpoint, map_location='cpu', weights_only=True)['config']
        result['split_id'] = config['split_id']
        result['cache_sha256'] = config['cache_sha256']
    save_json(root / 'metrics.json', result)
    return result


def validated_evaluations(run_root, ref_dir=None):
    """Validate both comparisons before choosing a winner or packaging any weights."""
    root = Path(run_root)
    metrics = {exp: json.loads((root / exp / 'evaluation/metrics.json').read_text())
               for exp in ('E101', 'E102')}
    baseline = json.loads((root / 'baseline/metrics.json').read_text())
    protocol = baseline.get('evaluation_protocol')
    if not protocol or protocol.get('limit') is not None or protocol.get('hide_true'):
        raise ValueError('Missing or partial evaluation protocol; reevaluate all models and baseline')
    if ref_dir is not None and protocol.get('reference') != reference_identity(ref_dir):
        raise ValueError('Reference identity changed; reevaluate all models and baseline')
    for exp, result in metrics.items():
        checkpoint = root / exp / 'best.pt'
        if result.get('checkpoint_sha256') != file_hash(checkpoint):
            raise ValueError(f'{exp}: stale checkpoint evaluation; reevaluate before continuing')
        config = torch.load(checkpoint, map_location='cpu', weights_only=True)['config']
        if (result.get('split_id') != config['split_id']
                or result.get('cache_sha256') != config['cache_sha256']):
            raise ValueError(f'{exp}: evaluation split/cache identity does not match checkpoint')
        if result.get('evaluation_protocol') != protocol:
            raise ValueError(f'{exp}: evaluation protocol does not match baseline')
    if any(metrics['E101'][key] != metrics['E102'][key] for key in ('split_id', 'cache_sha256')):
        raise ValueError('Model split/cache identities differ; compare models trained on the same rows')
    return metrics, baseline


def evaluation_hashes(run_root):
    root = Path(run_root)
    return {exp: file_hash(root / exp / ('metrics.json' if exp == 'baseline' else 'evaluation/metrics.json'))
            for exp in ('baseline', 'E101', 'E102')}


def plot_training(run_dir):
    import matplotlib.pyplot as plt
    root = Path(run_dir)
    history = pd.read_json(root / 'history.json')
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.train, label='Training')
    ax.plot(history.epoch, history.val, label='Structure-held-out validation')
    ax.set(xlabel='Epoch', ylabel='BCE loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'training_curve.png', dpi=160)
    return fig


def evaluate_blends(run_root, ref_dir):
    import matplotlib.pyplot as plt
    root = Path(run_root)
    metrics, baseline = validated_evaluations(root, ref_dir)
    chosen = max(('E101', 'E102'), key=lambda e: (metrics[e]['mrr'], e == 'E101'))
    checkpoint = root / chosen / 'best.pt'
    model = FingerprintMLP(checkpoint)
    rows = [{'alpha': 0.0, 'mrr': baseline['mrr']}, {'alpha': 1.0, 'mrr': metrics[chosen]['mrr']}]
    for alpha in (0.1, 0.2, 0.3, 0.5):
        result = evaluate_saved(BlendScorer(model, alpha), root / 'E103' / str(alpha), checkpoint,
                                ref_dir=str(ref_dir), ppm=baseline['evaluation_protocol']['ppm'])
        rows.append({'alpha': alpha, 'mrr': result['mrr']})
    curve = pd.DataFrame(rows).sort_values('alpha')
    curve.to_csv(root / 'E103/blend_curve.csv', index=False)
    best = sorted(rows[2:], key=lambda r: (-r['mrr'], r['alpha']))[0]
    save_json(root / 'E103/selection.json', dict(model=chosen, alpha=best['alpha'], mrr=best['mrr'],
        checkpoint_sha256=file_hash(checkpoint),
        checkpoint_hashes={exp: metrics[exp]['checkpoint_sha256'] for exp in ('E101', 'E102')},
        evaluation_hashes=evaluation_hashes(root), selection='Selected on shared validation; not an independent test'))
    fig, ax = plt.subplots()
    ax.plot(curve.alpha, curve.mrr, marker='o')
    ax.set(xlabel='Fingerprint weight (alpha)', ylabel='Validation MRR@25', title=f'Validation-selected blend: {chosen}')
    fig.tight_layout()
    fig.savefig(root / 'E103/blend_curve.png', dpi=160)
    return curve


def package_weights(run_root, destination):
    """Create a local upload folder; uploading/submitting remains an explicit Kaggle action."""
    implementation_hashes()
    root, dest = Path(run_root), Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / 'E103/selection.json').read_text())
    metrics, _ = validated_evaluations(root)
    if (selection.get('checkpoint_hashes') != {exp: metrics[exp]['checkpoint_sha256'] for exp in ('E101', 'E102')}
            or selection.get('evaluation_hashes') != evaluation_hashes(root)
            or selection['checkpoint_sha256'] != metrics[selection['model']]['checkpoint_sha256']):
        raise ValueError('Blend selection is stale; rerun blend evaluation')
    for exp in ('E101', 'E102'):
        shutil.copy2(root / exp / 'best.pt', dest / f'{exp}.pt')
        shutil.copy2(root / exp / 'training.json', dest / f'{exp}_training.json')
    shutil.copy2(__file__, dest / 'model_mlp.py')
    save_json(dest / 'selection.json', selection)
    save_json(dest / 'checksums.json', {p.name: file_hash(p) for p in dest.iterdir() if p.is_file() and p.name != 'checksums.json'})
    return dest


def export_handoff(run_root):
    """Write measured local results and log rows for review; never invent Kaggle scores."""
    root = Path(run_root)
    destination = root / 'handoff'
    destination.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / 'E103/selection.json').read_text())
    lines = ['# Cyril: measured local results', '',
             'Kaggle scores/screenshots must be added after actual submissions.', '',
             '| Experiment | MRR@25 | Top-1 | Top-10 | Top-25 |',
             '|---|---|---|---|---|']
    log_rows = []
    for exp, description in [('baseline', 'Library cosine'), ('E101', 'Binned-spectrum MLP'),
                             ('E102', 'MLP with neutral losses and metadata'), ('E103', 'Validation-selected blend')]:
        folder = root / exp / str(selection['alpha']) if exp == 'E103' else root / exp / 'evaluation'
        if exp == 'baseline':
            folder = root / 'baseline'
        result = json.loads((folder / 'metrics.json').read_text())
        lines.append(f"| {exp} | {result['mrr']:.6f} | {result['top1']:.4f} | {result['top10']:.4f} | {result['top25']:.4f} |")
        if exp in ('E101', 'E102'):
            training = json.loads((root / exp / 'training.json').read_text())
            lines.extend(['', f'## {exp}: {description}', '',
                f"Training: {training['data']['train_rows']} spectra, {training['epochs']} epochs; "
                f"best epoch {training['best_epoch']}; {training['hardware']}; "
                f"{training['elapsed_seconds']:.0f} seconds. Adam lr={training['config']['lr']:g}, "
                f"batch={training['config']['batch_size']}, "
                'BCE loss; structure-disjoint 5% early stopping.',
                f"Checkpoint SHA256: `{training['checkpoint_sha256']}`."])
            # Keep comparison table contiguous; detailed paragraphs are moved below later.
            plot_training(root / exp)
            shutil.copy2(root / exp / 'training_curve.png', destination / f'{exp}_training.png')
        if exp != 'baseline':
            log_rows.append(f"| {exp} | DATE | mrcyrilgoud | {description} | "
                            f"{'Substructure evidence improves ranking' if exp != 'E103' else 'Combine learned and library evidence'} "
                            f"| {result['mrr']:.6f} | | Local validation only; interpretation pending | members/mrcyrilgoud/ |")
    # Reorder the table before model details for Markdown rendering.
    table = [line for line in lines if line.startswith('|')]
    details = [line for line in lines[8:] if not line.startswith('|')]
    lines = lines[:4] + table + details
    lines += ['', f"E103 uses {selection['model']} with alpha={selection['alpha']}; selected on shared validation.",
              '', 'These scorers cannot recover structures absent from the candidate pool. '
              'Compare with the measured baseline before explaining any failure; '
              'validation overlap favors spectral library search.', '',
              '## Experiment-log rows', '', *log_rows]
    (destination / 'results.md').write_text('\n'.join(lines) + '\n')
    shutil.copy2(root / 'E103/blend_curve.png', destination / 'blend_curve.png')
    return destination
