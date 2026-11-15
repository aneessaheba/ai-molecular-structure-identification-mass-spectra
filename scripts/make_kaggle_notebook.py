"""Bundle the library-search baseline into a self-contained Kaggle notebook (no internet needed).

Usage: python -m scripts.make_kaggle_notebook --out kaggle/library_search_baseline
Then:  kaggle kernels push -p kaggle/library_search_baseline
"""
import argparse
import json
import os
import re

MODULES = ["src/data/spectra.py", "src/data/build_reference.py", "src/models/library_search.py"]

MAIN = '''
import glob, os, time
import pandas as pd

t0 = time.time()
test_path = glob.glob("/kaggle/input/**/test.parquet", recursive=True)[0]
data_dir = os.path.dirname(test_path)
print("data dir:", data_dir)

build(os.path.join(data_dir, "train.parquet"), "/kaggle/working/ref")
lib = ReferenceLibrary("/kaggle/working/ref")
test = pd.read_parquet(test_path)
queries = prepare_queries(test)
preds = predict(lib, queries, ppm=PPM)
sub =to_submission(preds, test.molecule_id.drop_duplicates().tolist())
sub.to_csv("/kaggle/working/submission.csv", index=False)
for f in ["ref_meta.parquet", "ref_peaks.npz", "structures.parquet"]:
    os.remove(os.path.join("/kaggle/working/ref", f))
print(sub.shape, f"{time.time() - t0:.0f}s")
sub.head()
'''


def strip(src):
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)
    src = re.sub(r"^from src\..*\n", "", src, flags=re.M)
    src = re.sub(r'\nif __name__ == "__main__":\n(    .*\n|\n)*', "\n", src)
    return src.strip() + "\n"


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle/library_search_baseline")
    ap.add_argument("--user", default="aneessahebaguddi")
    ap.add_argument("--slug", default="casmi26-library-search-baseline")
    ap.add_argument("--ppm", type=float, default=15)
    args = ap.parse_args()

    cells = [cell("markdown",
                  "# CASMI 2026: library-search baseline\n\n"
                  "Precursor-mass candidate retrieval over the training structures, ranked by "
                  "greedy cosine similarity between test spectra and each candidate's reference "
                  "spectra. Generated from the team repo by `scripts/make_kaggle_notebook.py`.")]
    for path in MODULES:
        cells.append(cell("markdown", f"## `{path}`"))
        cells.append(cell("code", strip(open(path).read())))
    cells.append(cell("markdown", "## Run"))
    cells.append(cell("code", f"PPM = {args.ppm}\n" + MAIN.strip() + "\n"))

    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                       "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "notebook.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "kernel_sources": [],
    }
    with open(os.path.join(args.out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
