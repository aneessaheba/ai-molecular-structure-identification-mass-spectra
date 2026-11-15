"""Library-search tuning (Anees, Model 2): one-at-a-time sweep around the baseline settings.

Search settings (same reference): m/z tolerance, aggregation over a molecule's spectra,
cross-mode matching, number of reference spectra per candidate.
Cleaning settings (reference rebuilt each time): intensity floor, top-N peaks, intensity power.

Same validation protocol as src/evaluation/validate.py (enveda-np-examples held out).
Writes experiments/sweeps/library_search_sweep.csv and results/figures/sweep_library_search.png.

Usage: python -m scripts.sweep_library_search [--skip-rebuild]
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.build_reference import build
from src.evaluation.metric import mrr_from_keys
from src.models.library_search import ReferenceLibrary, prepare_queries, query_molecule

VAL_LIB = "enveda-np-examples"
TRAIN = "data/raw/train.parquet"
BASE_CLEAN = dict(min_rel=0.01, top_n=64, power=0.5)
BASE_SEARCH = dict(tol=0.01, agg="mean", cross_mode=False, max_ref_per_struct=50)
ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
           "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


def score(lib, queries, **search):
    keys = lib.structs.inchikey14.values
    preds, truth = [], []
    for mol, spectra in queries.items():
        ranked = query_molecule(lib, spectra, ppm=15, **search)
        preds.append([keys[s] for s, _ in ranked[:25]])
        truth.append(mol)
    top1 = np.mean([p[:1] == [t] for p, t in zip(preds, truth)])
    return mrr_from_keys(preds, truth), top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rebuild", action="store_true", help="only sweep search settings")
    args = ap.parse_args()

    val = pq.read_table(TRAIN, filters=[("ingest_lib", "=", VAL_LIB)]).to_pandas()
    val = val[val.adduct.isin(ADDUCTS)]
    val["molecule_id"] = val.inchikey14

    rows = []

    def record(group, setting, value, clean, search, lib, t0):
        queries = prepare_queries(val, **clean)
        mrr, top1 = score(lib, queries, **search)
        rows.append(dict(group=group, setting=setting, value=str(value), mrr=round(mrr, 4),
                         top1=round(top1, 4), **{f"clean_{k}": v for k, v in clean.items()},
                         **{f"search_{k}": v for k, v in search.items()}))
        print(f"{group:8s} {setting:20s} {str(value):8s} MRR={mrr:.4f} top1={top1:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # --- search settings on the baseline reference ---
    lib = ReferenceLibrary("data/processed", exclude_libs=[VAL_LIB])
    t0 = time.time()
    record("baseline", "-", "-", BASE_CLEAN, BASE_SEARCH, lib, t0)
    for setting, values in [("tol", [0.002, 0.005, 0.02, 0.05]),
                            ("agg", ["max"]),
                            ("cross_mode", [True]),
                            ("max_ref_per_struct", [10, 200])]:
        for v in values:
            record("search", setting, v, BASE_CLEAN, {**BASE_SEARCH, setting: v}, lib, time.time())

    # --- cleaning settings (rebuild the reference for each) ---
    if not args.skip_rebuild:
        tmp = "data/processed/sweep_tmp"
        for setting, values in [("min_rel", [0.0, 0.001, 0.02]),
                                ("top_n", [32, 128]),
                                ("power", [0.25, 1.0])]:
            for v in values:
                clean = {**BASE_CLEAN, setting: v}
                t0 = time.time()
                build(TRAIN, tmp, **clean)
                lib_v = ReferenceLibrary(tmp, exclude_libs=[VAL_LIB])
                record("clean", setting, v, clean, BASE_SEARCH, lib_v, t0)
                del lib_v
        shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs("experiments/sweeps", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("experiments/sweeps/library_search_sweep.csv", index=False)
    print(df[["group", "setting", "value", "mrr", "top1"]].to_string(index=False))


def plot(csv="experiments/sweeps/library_search_sweep.csv", out="results/figures/sweep_library_search.png"):
    """Bar chart of every sweep setting vs the baseline (run: python -c 'from scripts.sweep_library_search import plot; plot()')."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv)
    base = df.loc[df.group == "baseline", "mrr"].iloc[0]
    df = df[df.group != "baseline"].copy()
    df["label"] = df.setting + " = " + df.value.astype(str)
    df.loc[df.group == "combined", "label"] = df.setting + (" " + df.value.astype(str)).where(
        ~df.setting.str.contains("cross_mode"), "")
    colors = {"search": "#4C72B0", "clean": "#55A868", "combined": "#DD8452"}
    fig, ax = plt.subplots(figsize=(9, 7))
    y = range(len(df))
    ax.barh(list(y), df.mrr - base, left=base, color=[colors[g] for g in df.group])
    ax.axvline(base, color="k", lw=1)
    ax.text(base, -0.9, f" baseline {base:.3f}", fontsize=9, va="center")
    for i, (m, g) in enumerate(zip(df.mrr, df.group)):
        ax.text(max(m, base) + 0.002, i, f"{m:.3f}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df.label, fontsize=9)
    ax.set_xlabel("validation MRR@25 (enveda-np-examples held out)")
    ax.set_title("Library-search tuning: one setting at a time, then combined")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()], loc="lower left")
    ax.set_xlim(right=max(df.mrr.max(), base) + 0.015)
    ax.set_ylim(len(df) - 0.5, -1.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
    plot()
