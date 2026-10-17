import numpy as np
import pytest
import pyvai


@pytest.fixture(scope="session", autouse=True)
def vai_engine():
    """Initialize up front so eva's GPU banner prints once, before any test."""
    pyvai.init_vai_engine()
    yield


@pytest.fixture
def rng():
    """Seeded numpy RNG for reproducible dummy test data."""
    return np.random.default_rng(seed=0)
