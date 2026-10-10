import pytest


def pytest_collection_modifyitems(config, items):
    """Tests marked `stand` or `slow` run only when asked for: `pytest -m stand`, `pytest -m slow`."""
    asked = config.getoption("-m") or ""
    for marker in ("stand", "slow"):
        if marker in asked:
            continue
        skip = pytest.mark.skip(reason=f"runs with -m {marker}")
        for item in items:
            if item.get_closest_marker(marker):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_deprecation_warnings():
    """An old name warns once per process; each test sees its first use again."""
    from solvi import _deprecate
    _deprecate._seen.clear()
    yield
