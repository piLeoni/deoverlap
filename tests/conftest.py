import importlib

import pytest

# The package re-exports the `deoverlap` function under the submodule's name.
engine_module = importlib.import_module("deoverlap.deoverlap")


@pytest.fixture(autouse=True, params=["python", "rust"])
def engine(request, monkeypatch):
    """Run every test against both engines."""
    if request.param == "rust" and engine_module._core is None:
        pytest.skip("Rust core not built (run `maturin develop`)")
    monkeypatch.setenv("DEOVERLAP_ENGINE", request.param)
    return request.param
