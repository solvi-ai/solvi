import importlib
import warnings


def test_importing_the_capabilities_submodule_keeps_the_function():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importlib.import_module("solvi.decide.capabilities")
    importlib.import_module("solvi.core.deciders.capabilities")
    from solvi.core import deciders
    assert callable(deciders.capabilities)
