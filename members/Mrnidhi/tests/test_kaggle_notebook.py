"""Tests for the Kaggle notebooks built by make_kaggle_notebook.py.

A Kaggle notebook cannot import from members/, and it has no __file__. These tests check
that the model cells respect both, and that the notebooks on disk match the model files.

Run from the repo root:
    python -m pytest members/Mrnidhi/tests/test_kaggle_notebook.py -q
"""
from pathlib import Path

import nbformat
import pytest

from members.Mrnidhi import make_kaggle_notebook as builder


def code_cells(path: Path) -> list:
    notebook = nbformat.read(path, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


@pytest.mark.parametrize("source", [builder.model_source(), builder.ranker_source()])
def test_model_cells_run_without_the_repo(source: str) -> None:
    compile(source, "model cell", "exec")              # valid Python
    assert "from members" not in source and "import members" not in source
    assert "__file__" not in source


def test_ranker_cell_reads_the_weights_from_the_kaggle_dataset() -> None:
    source = builder.ranker_source()
    assert "MODEL_PATH = Path(WEIGHTS)" in source
    assert source.rstrip().endswith("print(model.name)")
    assert "model = RankerScorer(ref_dir=REF, exclude_libs=())" in source      # the full library on Kaggle


def test_model_1_notebook_is_up_to_date() -> None:
    assert code_cells(builder.OUT) == [builder.SETUP, builder.model_source(), builder.SUBMIT]


def test_model_2_notebook_is_up_to_date() -> None:
    assert code_cells(builder.OUT_RANKER) == [
        builder.SETUP, builder.WEIGHTS_CELL, builder.module_body("model_entropy.py"),
        builder.ranker_source(), builder.SUBMIT,
    ]


def test_notebooks_carry_no_saved_output() -> None:
    for path in (builder.OUT, builder.OUT_RANKER):
        notebook = nbformat.read(path, as_version=4)
        assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
