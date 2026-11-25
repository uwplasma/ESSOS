import os

os.environ["ESSOS_PROGRESS"] = "0"  # default progress bars off in tests

import jax
import pytest


def pytest_collection_modifyitems(config, items):
    """`pytest -m gpu` asks for a GPU: skip those tests, with a reason, where there is none.

    In the ordinary CPU run the `gpu` tests are cheap and run with the rest of the suite.
    """
    if (config.getoption("markexpr") or "").strip() != "gpu" or jax.default_backend() == "gpu":
        return
    skip = pytest.mark.skip(reason=f"pytest -m gpu without a GPU (backend: {jax.default_backend()})")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Each xdist worker would otherwise keep every module's executables in memory."""
    yield
    jax.clear_caches()
