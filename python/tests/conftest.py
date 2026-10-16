from pathlib import Path

import numpy as np
import pytest
import pyvai


@pytest.fixture(scope="session", autouse=True)
def vai_engine():
    pyvai.init_vai_engine()
    yield


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Repo `data/` directory (weights, images) for built-in model tests."""
    return Path(__file__).resolve().parents[2] / "data"


@pytest.fixture
def rng():
    """Seeded numpy RNG for reproducible dummy test data."""
    return np.random.default_rng(seed=0)
