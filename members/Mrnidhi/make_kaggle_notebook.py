"""Build the Kaggle submission notebook for Model 1 from the model file.

The notebook follows kaggle/TEMPLATE_submission.ipynb. The model cell is copied from
model_entropy.py by this script, so the notebook can never drift from the tested code.

Run from the repo root:
    python -m members.Mrnidhi.make_kaggle_notebook
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).parent
OUT = HERE / "kaggle_model1_entropy.ipynb"

INTRO = """# CASMI 2026: Model 1, entropy similarity (E501)

Srinidhi, team GROUP_PROGRAMMING_01_a. Scores each candidate by spectral entropy similarity
against all of its reference spectra in the same ionization mode.

**Inputs to add:** the competition data and the team dataset `casmi-team-src`.
No weights and no extra packages are needed. **Settings:** Internet OFF, no GPU.
"""

SETUP = '''import glob
import os
import sys
import time

t0 = time.time()

# The two folders can be overridden for a dry run outside Kaggle.
INPUT = os.environ.get("CASMI_INPUT", "/kaggle/input")
WORK = os.environ.get("CASMI_WORK", "/kaggle/working")

# Team code (src/) from the casmi-team-src dataset
BASE = glob.glob(f"{INPUT}/**/src/models/base.py", recursive=True)[0]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(BASE))))

from src.data.build_reference import build
from src.evaluation.validate import make_submission

TEST = glob.glob(f"{INPUT}/**/test.parquet", recursive=True)[0]
DATA = os.path.dirname(TEST)
REF = os.path.join(WORK, "ref")
build(os.path.join(DATA, "train.parquet"), REF)   # ~3-5 min
print(f"reference built in {time.time() - t0:.0f}s")
'''

MODEL_NOTE = """## The model
Copied from `members/Mrnidhi/model_entropy.py` by `make_kaggle_notebook.py`.
On Kaggle the full library is used (`exclude_libs=()`); locally the validation library is left out.
"""

SUBMIT = '''sub = make_submission(model, TEST, REF, os.path.join(WORK, "submission.csv"))
for f in os.listdir(REF):          # keep only submission.csv in the output
    os.remove(os.path.join(REF, f))
os.rmdir(REF)

guesses = sub.smiles.str.split(";").map(len)
assert sub.molecule_id.is_unique and not sub.isna().any().any() and guesses.max() <= 25
print(sub.shape, f"guesses per molecule: {guesses.min()} to {guesses.max()}", f"total {time.time() - t0:.0f}s")
sub.head()
'''


def model_source() -> str:
    """model_entropy.py without its command-line block."""
    source = (HERE / "model_entropy.py").read_text()
    body, marker, _ = source.partition('\n\nif __name__ == "__main__":')
    if not marker:
        raise ValueError("model_entropy.py has no __main__ block to cut at")
    return body.rstrip() + "\n\n\nmodel = EntropyScorer(ref_dir=REF, exclude_libs=())\nprint(model.name)\n"


def main() -> None:
    notebook = new_notebook(cells=[
        new_markdown_cell(INTRO), new_code_cell(SETUP), new_markdown_cell(MODEL_NOTE),
        new_code_cell(model_source()), new_code_cell(SUBMIT),
    ])
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for i, cell in enumerate(notebook.cells):      # fixed ids, so re-running gives the same file
        cell["id"] = f"cell-{i}"
    nbformat.write(notebook, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
