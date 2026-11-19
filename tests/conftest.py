import os

os.environ["ESSOS_PROGRESS"] = "0"  # default progress bars off in tests

import jax
import pytest

import essos

essos.configure_jax()  # float64 and one CPU device per core, as importing essos did before


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Each xdist worker would otherwise keep every module's executables in memory."""
    yield
    jax.clear_caches()
