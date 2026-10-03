"""The 1.0 layout (LAYOUT.md): every 0.9 module path still imports for one release — as the same module, with a
SolviDeprecationWarning naming the new path — and `solvi migrate` rewrites code to the new paths from the same table."""
import importlib
import subprocess
import sys
import textwrap
import warnings

import pytest

import solvi
from solvi import _deprecate, _migrate

OLD = sorted(_deprecate.old_paths())


def _fresh(name):
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        if sys.modules[k].__name__ != k:                     # an alias entry, not the module's own
            del sys.modules[k]


def test_every_moved_module_is_in_the_table_and_its_new_path_exists():
    assert OLD, "no module moved"
    for old, new in _deprecate.old_paths().items():
        assert importlib.import_module(new).__name__ == new, (old, new)


@pytest.mark.parametrize("old", OLD)
def test_an_old_path_imports_the_same_module_with_a_warning(old):
    new = _deprecate.old_paths()[old]
    _fresh(old)
    with pytest.warns(solvi.SolviDeprecationWarning, match=rf"{old} moved in 1\.0: use {new}; the old path is removed in 1\.1"):
        mod = importlib.import_module(old)
    assert mod is importlib.import_module(new)
    assert mod.__name__ == new and mod.__spec__.name == new          # the module keeps its own identity


def test_the_warning_points_at_the_import():
    _fresh("solvi.typed")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import solvi.typed  # noqa: F401
    assert w and w[0].filename == __file__


def test_an_old_module_as_an_attribute_of_the_package():
    _fresh("solvi.provenance")
    if "provenance" in vars(solvi):
        delattr(solvi, "provenance")
    with pytest.warns(solvi.SolviDeprecationWarning, match="solvi.provenance moved"):
        from solvi import provenance
    from solvi.core import provenance as new
    assert provenance is new
    with pytest.raises(AttributeError):
        solvi.no_such_module_anywhere  # noqa: B018


def test_a_private_name_of_the_0_9_core_module():
    from solvi.core import catalog
    with pytest.warns(solvi.SolviDeprecationWarning, match="use solvi.core.catalog._group_func"):
        from solvi.core import _group_func
    assert _group_func is catalog._group_func
    with pytest.raises(ImportError):
        from solvi.core import no_such_name  # noqa: F401


def test_a_stored_module_qualname_of_0_9_loads_without_a_warning():
    """schema._span_type_name stores "module:qualname"; a 0.9 store names the 0.9 module."""
    from solvi.core.schema import _span_type, _span_type_name
    from solvi.core.types import FactTypeError
    _fresh("solvi.typed")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _span_type("solvi.typed:FactTypeError") is FactTypeError
    assert _span_type(_span_type_name(FactTypeError)) is FactTypeError


def test_a_fingerprint_names_the_0_9_module():
    from solvi.core.provenance import fp_module
    assert fp_module("solvi.core.types", "Maybe") == "solvi.typed"
    assert fp_module("solvi.core.catalog", "Quote") == "solvi.core"
    assert fp_module("myapp.parts", "Order") == "myapp.parts"


SOURCE = '''\
import solvi.typed as t
from solvi import Catalog, provenance, i18n as lang
from solvi import (System,   # a comment
                   costs)
from solvi.runtime import Trace
from solvi.core import Quote


def patch(monkeypatch):
    monkeypatch.setattr("solvi.runtime.now_ms", lambda: 0)
    return solvi.schema.dump, "solvi.typed:Maybe", solvi.systemone
'''

EXPECTED = '''\
import solvi.core.types as t
from solvi import Catalog
from solvi.core import provenance, _i18n as lang
from solvi import System
from solvi.core import costs
from solvi.core.runtime import Trace
from solvi.core import Quote


def patch(monkeypatch):
    monkeypatch.setattr("solvi.core.runtime.now_ms", lambda: 0)
    return solvi.core.schema.dump, "solvi.core.types:Maybe", solvi.systemone
'''


def test_migrate_rewrites_imports_dotted_paths_and_strings():
    assert _migrate.rewrite(SOURCE) == EXPECTED
    assert _migrate.rewrite(EXPECTED) == EXPECTED                       # idempotent


def test_migrate_names_that_left_the_package():
    out = _migrate.rewrite("from solvi import Catalog, Shadow as S\n", top_names={"Shadow": "solvi.core.store.diff"})
    assert out == "from solvi import Catalog\nfrom solvi.core.store.diff import Shadow as S\n"


def test_the_migrated_code_imports_without_a_warning(tmp_path):
    code = textwrap.dedent('''\
        import solvi.typed
        from solvi import provenance
        from solvi.runtime import Trace
        print(solvi.typed.Maybe, provenance.digest, Trace)
    ''')
    f = tmp_path / "app.py"
    f.write_text(code)
    run = [sys.executable, "-c", "import runpy, sys, warnings, solvi; warnings.simplefilter('error', "
           "solvi.SolviDeprecationWarning); runpy.run_path(sys.argv[1])", str(f)]
    assert subprocess.run(run, capture_output=True).returncode != 0
    check = subprocess.run([sys.executable, "-m", "solvi", "migrate", str(tmp_path), "--check"], capture_output=True,
                           text=True)
    assert check.returncode == 1 and "app.py" in check.stdout and f.read_text() == code
    done = subprocess.run([sys.executable, "-m", "solvi", "migrate", str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0 and "rewritten" in done.stdout
    assert subprocess.run(run, capture_output=True).returncode == 0
    again = subprocess.run([sys.executable, "-m", "solvi", "migrate", str(tmp_path), "--check"], capture_output=True,
                           text=True)
    assert again.returncode == 0 and "nothing to migrate" in again.stdout


def test_migrate_leaves_a_file_that_does_not_parse(tmp_path, capsys):
    f = tmp_path / "odd.py"
    f.write_text("from solvi import (provenance\n")
    assert _migrate.migrate(str(tmp_path)) == []
    assert f.read_text() == "from solvi import (provenance\n" and "does not parse" in capsys.readouterr().out


def test_the_changelog_migration_table_is_generated_from_the_table():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root / "tools" / "migration_table.py"), "--check"], capture_output=True,
                         text=True)
    assert run.returncode == 0, run.stdout
