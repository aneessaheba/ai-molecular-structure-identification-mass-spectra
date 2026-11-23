import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs the downloaded data and the reference library")


@pytest.fixture(scope="session")
def needs_data() -> None:
    """Skip a test when the competition data or the reference library is not on this machine."""
    for path in ("data/processed/ref_meta.parquet", "data/raw/train.parquet", "data/raw/test.parquet"):
        if not os.path.exists(path):
            pytest.skip(f"{path} is missing; run the setup steps in members/README.md first")
