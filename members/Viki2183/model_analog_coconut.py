"""Analog-search and COCONUT candidate-expansion scorers for Viki2183.

The implementation deliberately keeps generated artifacts outside the member
folder.  Reference matrices are written below ``data/processed`` and the
COCONUT table below ``data/external``; both locations are ignored by the
repository's ``.gitignore``.

Run from the repository root, for example::

    python -m members.Viki2183.model_analog_coconut smoke
    python -m members.Viki2183.model_analog_coconut build-coconut --input path/to/coconut.csv
    python -m members.Viki2183.model_analog_coconut evaluate --model analog
    python -m members.Viki2183.model_analog_coconut evaluate --model coconut --hide-true

The scorer never loads validation spectra into its analog library.  It uses
the existing ``ReferenceLibrary`` with ``enveda-np-examples`` excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.spectra import neutral_mass
from src.evaluation.validate import evaluate
from src.features.featurize import bin_spectrum, morgan_fp, tanimoto
from src.models.base import Scorer
from src.models.library_search import ReferenceLibrary


DEFAULT_REF_DIR = Path("data/processed")
DEFAULT_MATRIX_DIR = DEFAULT_REF_DIR / "analog_search"
DEFAULT_COCONUT_PATH = Path("data/external/coconut/coconut.parquet")
VAL_LIB = "enveda-np-examples"
BIN_SIZE = 1.0
MAX_MZ = 1000.0
TOP_K = 20
MASS_PPM = 15.0


def _safe_mode_name(mode: object) -> str:
    """Return a stable filename component for an ionisation-mode value."""

    value = str(mode) if mode is not None else "unknown"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "unknown"


def _normalised_sparse_row(mz: np.ndarray, intensity: np.ndarray):
    """Create one sparse, L2-normalised 1 m/z row."""

    from scipy.sparse import csr_matrix

    vector = bin_spectrum(mz, intensity, max_mz=MAX_MZ, bin_size=BIN_SIZE)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return csr_matrix(vector.reshape(1, -1), dtype=np.float32)


def build_sparse_reference_matrix(
    lib: ReferenceLibrary,
    mode: object,
    output_dir: Path = DEFAULT_MATRIX_DIR,
):
    """Build and save a sparse reference matrix for one ionisation mode.

    The returned row ids index ``lib.meta``.  Keeping this mapping beside the
    matrix makes the matrix reusable while preserving the library's structure
    and peak offsets exactly.
    """

    from scipy.sparse import csr_matrix, save_npz, vstack

    mode_value = str(mode)
    row_ids = np.flatnonzero(lib.meta.ionization_mode.astype(str).values == mode_value)
    rows = []
    for j in row_ids:
        mz, intensity = lib.peaks(int(j))
        rows.append(_normalised_sparse_row(mz, intensity))

    matrix = vstack(rows, format="csr") if rows else csr_matrix((0, int(MAX_MZ / BIN_SIZE)), dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reference_{_safe_mode_name(mode_value)}_{int(BIN_SIZE)}mz"
    save_npz(output_dir / f"{stem}.npz", matrix)
    np.save(output_dir / f"{stem}_rows.npy", row_ids.astype(np.int64))
    metadata = {
        "mode": mode_value,
        "n_rows": int(matrix.shape[0]),
        "n_bins": int(matrix.shape[1]),
        "bin_size": BIN_SIZE,
        "max_mz": MAX_MZ,
        "l2_normalized": True,
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, row_ids


def load_or_build_reference_matrices(
    lib: ReferenceLibrary,
    output_dir: Path = DEFAULT_MATRIX_DIR,
):
    """Load cached mode matrices, building any missing matrix locally."""

    from scipy.sparse import load_npz

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = {}
    modes = sorted({str(x) for x in lib.meta.ionization_mode.dropna().unique()})
    for mode in modes:
        stem = f"reference_{_safe_mode_name(mode)}_{int(BIN_SIZE)}mz"
        matrix_path = output_dir / f"{stem}.npz"
        rows_path = output_dir / f"{stem}_rows.npy"
        if matrix_path.exists() and rows_path.exists():
            matrix = load_npz(matrix_path).tocsr()
            row_ids = np.load(rows_path)
        else:
            matrix, row_ids = build_sparse_reference_matrix(lib, mode, output_dir)
        if matrix.shape[0] != len(row_ids):
            raise ValueError(f"Cached matrix row count does not match row map for mode {mode!r}")
        matrices[mode] = (matrix, row_ids.astype(np.int64, copy=False))
    return matrices


def _top_analog_rows(matrix, query_vector: np.ndarray, top_k: int):
    """Return positive top-k sparse similarities and local row positions."""

    scores = (matrix @ query_vector.T).toarray().ravel()
    if not len(scores):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    k = min(int(top_k), len(scores))
    if k < len(scores):
        local = np.argpartition(-scores, k - 1)[:k]
    else:
        local = np.arange(len(scores))
    local = local[scores[local] > 0.0]
    order = np.argsort(-scores[local], kind="stable")
    local = local[order]
    return local.astype(np.int64), scores[local].astype(np.float32)


class AnalogSearchScorer(Scorer):
    """Rank candidates by fingerprint similarity to spectral analogs.

    For each query spectrum, the scorer searches the top 20 reference spectra
    in the same ionisation mode without applying a precursor-mass filter.  The
    final candidate score is the similarity-weighted mean of Morgan-fingerprint
    Tanimoto similarities to those analog structures.
    """

    name = "analog-search"

    def __init__(
        self,
        ref_dir: str | Path = DEFAULT_REF_DIR,
        matrix_dir: str | Path = DEFAULT_MATRIX_DIR,
        top_k: int = TOP_K,
        baseline_weight: float = 0.0,
        coconut_path: str | Path | None = None,
    ):
        if not 0.0 <= baseline_weight <= 1.0:
            raise ValueError("baseline_weight must be between 0 and 1")
        self.ref_dir = Path(ref_dir)
        self.matrix_dir = Path(matrix_dir)
        self.top_k = int(top_k)
        self.baseline_weight = float(baseline_weight)
        self.coconut_path = Path(coconut_path) if coconut_path else None
        self.lib = ReferenceLibrary(self.ref_dir, exclude_libs=[VAL_LIB])
        self.matrices = load_or_build_reference_matrices(self.lib, self.matrix_dir)
        self._coconut = self._load_coconut(self.coconut_path) if self.coconut_path else None
        if self._coconut is not None:
            self.name = "analog-coconut"

    @staticmethod
    def _load_coconut(path: Optional[Path]):
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(
                f"COCONUT candidate table not found: {path}. "
                "Run the build-coconut command with the approved CSV first."
            )
        table = pd.read_parquet(path, columns=["smiles", "inchikey14", "mass"])
        table = table.dropna(subset=["smiles", "mass"]).sort_values("mass").reset_index(drop=True)
        return table

    def score(self, spectra, candidates):
        """Return one similarity-weighted Morgan-Tanimoto score per candidate."""

        n_candidates = len(candidates)
        if n_candidates == 0:
            return np.empty(0, dtype=float)
        candidate_smiles = candidates["smiles"].fillna("").astype(str).tolist()
        candidate_fps = np.stack([morgan_fp(s) for s in candidate_smiles])
        numerator = np.zeros(n_candidates, dtype=np.float64)
        denominator = 0.0

        for spectrum in spectra:
            mode = str(spectrum.get("ionization_mode"))
            matrix_and_rows = self.matrices.get(mode)
            if matrix_and_rows is None:
                continue
            matrix, row_ids = matrix_and_rows
            query = _normalised_sparse_row(spectrum["mz"], spectrum["intensity"])
            local_rows, similarities = _top_analog_rows(query, matrix, self.top_k)
            if not len(local_rows):
                continue
            analog_structs = self.lib.meta.iloc[row_ids[local_rows]].struct_idx.to_numpy(dtype=np.int64)
            analog_smiles = self.lib.structs.iloc[analog_structs].smiles.tolist()
            for similarity, analog_smiles_one in zip(similarities, analog_smiles):
                numerator += float(similarity) * tanimoto(morgan_fp(analog_smiles_one), candidate_fps)
                denominator += float(similarity)

        analog_scores = numerator / denominator if denominator > 0.0 else np.zeros(n_candidates)
        if self.baseline_weight and "cosine" in candidates:
            baseline = candidates["cosine"].to_numpy(dtype=float)
            analog_scores = (1.0 - self.baseline_weight) * analog_scores + self.baseline_weight * baseline
        return analog_scores

    def extra_candidates(self, spectra):
        """Return COCONUT SMILES within the task's ±15 ppm neutral-mass window."""

        if self._coconut is None or self._coconut.empty:
            return []
        masses = [neutral_mass(s["precursor_mz"], s["adduct"]) for s in spectra]
        masses = [float(m) for m in masses if np.isfinite(m)]
        if not masses:
            return []
        target = float(np.median(masses))
        tolerance = target * MASS_PPM * 1e-6
        values = self._coconut["mass"].to_numpy(dtype=float)
        lo = int(np.searchsorted(values, target - tolerance, side="left"))
        hi = int(np.searchsorted(values, target + tolerance, side="right"))
        return self._coconut.iloc[lo:hi]["smiles"].astype(str).tolist()


def _find_smiles_column(columns: Iterable[str]) -> str:
    preferred = ("canonical_smiles", "smiles", "SMILES", "CanonicalSMILES")
    for name in preferred:
        if name in columns:
            return name
    raise ValueError(f"Could not find a SMILES column; available columns: {list(columns)}")


def prepare_coconut(input_path: str | Path, output_path: str | Path = DEFAULT_COCONUT_PATH):
    """Convert an approved COCONUT CSV into the local sorted candidate table."""

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    source = Path(input_path)
    output = Path(output_path)
    source_table = pd.read_csv(source)
    smiles_column = _find_smiles_column(source_table.columns)
    rows = []
    for smiles in source_table[smiles_column].dropna().astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        if not key:
            continue
        rows.append({
            "smiles": smiles,
            "inchikey14": key.split("-")[0],
            "mass": float(Descriptors.ExactMolWt(mol)),
        })
    table = pd.DataFrame(rows, columns=["smiles", "inchikey14", "mass"])
    table = table.drop_duplicates("inchikey14").sort_values("mass").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    return table


def coconut_coverage(table_path: str | Path, train_path: str | Path = "data/raw/train.parquet"):
    """Report exact validation-molecule coverage without modifying any data."""

    import pyarrow.parquet as pq

    coconut = pd.read_parquet(table_path, columns=["inchikey14"])
    validation = pq.read_table(train_path, columns=["ingest_lib", "inchikey14"], filters=[("ingest_lib", "=", VAL_LIB)]).to_pandas()
    keys = set(validation.inchikey14.dropna())
    covered = keys.intersection(set(coconut.inchikey14.dropna()))
    return {"validation_molecules": len(keys), "covered": len(covered), "fraction": len(covered) / len(keys) if keys else 0.0}


def _run_smoke():
    """Dependency-light smoke checks for pure helper behavior."""

    assert _safe_mode_name("positive mode") == "positive_mode"
    assert _safe_mode_name(None) == "unknown"
    assert len(bin_spectrum([1.2, 2.2], [1.0, 2.0])) == 1000
    print("[smoke] helper checks passed")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    prep = sub.add_parser("build-coconut")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", default=str(DEFAULT_COCONUT_PATH))

    coverage = sub.add_parser("coverage")
    coverage.add_argument("--table", default=str(DEFAULT_COCONUT_PATH))
    coverage.add_argument("--train", default="data/raw/train.parquet")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", choices=("analog", "coconut"), default="analog")
    ev.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    ev.add_argument("--matrix-dir", default=str(DEFAULT_MATRIX_DIR))
    ev.add_argument("--coconut", default=str(DEFAULT_COCONUT_PATH))
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--hide-true", action="store_true")
    ev.add_argument("--baseline-weight", type=float, default=0.0)

    args = parser.parse_args(argv)
    if args.command == "smoke":
        _run_smoke()
        return
    if args.command == "build-coconut":
        table = prepare_coconut(args.input, args.output)
        print(f"[coconut] wrote {len(table)} unique molecules to {args.output}")
        return
    if args.command == "coverage":
        result = coconut_coverage(args.table, args.train)
        percentage = 100.0 * result["fraction"]
        print(f"covered molecules: {result['covered']}")
        print(f"total validation molecules: {result['validation_molecules']}")
        print(f"percentage: {percentage:.6f}%")
        return
    coconut_path = args.coconut if args.model == "coconut" else None
    scorer = AnalogSearchScorer(
        ref_dir=args.ref_dir,
        matrix_dir=args.matrix_dir,
        baseline_weight=args.baseline_weight,
        coconut_path=coconut_path,
    )
    result = evaluate(scorer, ref_dir=args.ref_dir, hide_true=args.hide_true, limit=args.limit)
    print({key: result[key] for key in ("model", "mrr", "top1", "top10", "top25")})


if __name__ == "__main__":
    main()
