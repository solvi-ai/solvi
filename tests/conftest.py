import pytest


@pytest.fixture(autouse=True)
def _fresh_deprecation_warnings():
    """An old name warns once per process; each test sees its first use again."""
    from solvi import _deprecate
    _deprecate._seen.clear()
    yield
