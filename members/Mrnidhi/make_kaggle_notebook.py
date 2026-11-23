"""Build the Kaggle submission notebooks for Model 1 and Model 2 from the model files.

The notebooks follow kaggle/TEMPLATE_submission.ipynb. The model cells are copied from
model_entropy.py and model_ranker.py by this script, so a notebook can never drift from
the tested code.

Run from the repo root:
    python -m members.Mrnidhi.make_kaggle_notebook
"""
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).parent
OUT = HERE / "kaggle_model1_entropy.ipynb"
OUT_RANKER = HERE / "kaggle_model2_ranker.ipynb"

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

INTRO_RANKER = """# CASMI 2026: Model 2, LightGBM ranker (E502)

Srinidhi, team GROUP_PROGRAMMING_01_a. Describes each candidate with 15 features and ranks
the candidates with a trained LightGBM LambdaRank model.

**Inputs to add:** the competition data, the team dataset `casmi-team-src` and my dataset
`casmi-srinidhi-weights`, which holds `ranker.txt`. **Settings:** Internet OFF, no GPU.
"""

WEIGHTS_CELL = '''# The trained ranker from the casmi-srinidhi-weights dataset
WEIGHTS = glob.glob(f"{INPUT}/**/ranker.txt", recursive=True)[0]

# lightgbm is normally part of the Kaggle image. If it is missing, install it from a wheel
# placed next to ranker.txt, because pip cannot download anything with the internet off.
try:
    import lightgbm
except ImportError:
    os.system(f"pip install --no-index --find-links {os.path.dirname(WEIGHTS)} lightgbm -q")
    import lightgbm
print("ranker:", WEIGHTS, "| lightgbm", lightgbm.__version__)
'''

MODEL_NOTE_RANKER = """## The model
Two cells, copied by `make_kaggle_notebook.py`: first `members/Mrnidhi/model_entropy.py`
(Model 1, whose score is one of the features), then `members/Mrnidhi/model_ranker.py`.
On Kaggle the full library is used (`exclude_libs=()`); locally the validation library is left out.
"""

ENTROPY_IMPORT = re.compile(r"from members\.Mrnidhi\.model_entropy import \([^)]*\)\n\n")   # and its blank line
LOCAL_WEIGHTS = 'MODEL_PATH = Path(__file__).parent / "weights" / "ranker.txt"'


def module_body(filename: str) -> str:
    """A model file without its command-line block."""
    source = (HERE / filename).read_text()
    body, marker, _ = source.partition('\n\nif __name__ == "__main__":')
    if not marker:
        raise ValueError(f"{filename} has no __main__ block to cut at")
    return body.rstrip() + "\n"


def model_source() -> str:
    """The Model 1 cell: model_entropy.py and the line that creates the model."""
    return module_body("model_entropy.py") + "\n\nmodel = EntropyScorer(ref_dir=REF, exclude_libs=())\nprint(model.name)\n"


def ranker_source() -> str:
    """The Model 2 cell: model_ranker.py changed in the two places a notebook needs.

    The import of Model 1 goes, because the cell above already defines those names. The
    weights path points at the Kaggle dataset, because a notebook has no __file__.
    """
    body, imports = ENTROPY_IMPORT.subn("", module_body("model_ranker.py"))
    if imports != 1 or body.count(LOCAL_WEIGHTS) != 1:
        raise ValueError("model_ranker.py no longer has the import or the weights path this script edits")
    body = body.replace(LOCAL_WEIGHTS, "MODEL_PATH = Path(WEIGHTS)")
    return body + "\n\nmodel = RankerScorer(ref_dir=REF, exclude_libs=())\nprint(model.name)\n"


def write_notebook(cells: list, out: Path) -> None:
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for i, cell in enumerate(notebook.cells):      # fixed ids, so re-running gives the same file
        cell["id"] = f"cell-{i}"
    nbformat.write(notebook, out)
    print(f"wrote {out}")


def main() -> None:
    write_notebook([
        new_markdown_cell(INTRO), new_code_cell(SETUP), new_markdown_cell(MODEL_NOTE),
        new_code_cell(model_source()), new_code_cell(SUBMIT),
    ], OUT)
    write_notebook([
        new_markdown_cell(INTRO_RANKER), new_code_cell(SETUP), new_code_cell(WEIGHTS_CELL),
        new_markdown_cell(MODEL_NOTE_RANKER), new_code_cell(module_body("model_entropy.py")),
        new_code_cell(ranker_source()), new_code_cell(SUBMIT),
    ], OUT_RANKER)


if __name__ == "__main__":
    main()
