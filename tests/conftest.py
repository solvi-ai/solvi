import pytest


def pytest_collection_modifyitems(config, items):
    """Tests marked `stand` run only when asked for: `pytest -m stand`."""
    if "stand" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="the task stand runs with -m stand")
    for item in items:
        if item.get_closest_marker("stand"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_deprecation_warnings():
    """An old name warns once per process; each test sees its first use again."""
    from solvi import _deprecate
    _deprecate._seen.clear()
    yield
