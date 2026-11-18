import os

os.environ["ESSOS_PROGRESS"] = "0"  # default progress bars off in tests

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Each xdist worker would otherwise keep every module's executables in memory."""
    yield
    jax.clear_caches()
