"""The 1.0 layout (LAYOUT.md): every 0.9 module path still imports for one release — as the same module, with a
SolviDeprecationWarning naming the new path — and `solvi migrate` rewrites code to the new paths from the same table."""
import importlib
import subprocess
import sys
import textwrap
import warnings

import pytest

import solvi
from solvi import _deprecate
from solvi.cli import _migrate

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
    parent = old.rpartition(".")[0]
    if parent in _deprecate.old_paths():          # an old parent (solvi.agents) imports first with its own warning:
        with warnings.catch_warnings():           # loaded here, so only this module's warning is under test
            warnings.simplefilter("ignore")
            importlib.import_module(parent)
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
    return solvi.schema.dump, "solvi.typed:Maybe", solvi.show, solvi.systemone
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
    return solvi.core.schema.dump, "solvi.core.types:Maybe", solvi.show, solvi.core.deciders.systemone
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


# --- solvi.experimental: what a decision made with an experimental piece records
def _store_system(tmp_path, **kw):
    from solvi import Answer, Catalog, Question, System
    from solvi.core.store import JSONLStorage
    cat = Catalog()

    @cat.fn
    def m(n: int) -> int:
        return n

    @cat.rule("ok")
    def ok(m: int) -> str:
        return "yes" if m > 1 else "no"
    return System(cat, [Question("ok", "OK?", Answer.choice(["yes", "no"]))], storage=JSONLStorage(tmp_path / "d.jsonl"),
                  **kw)


def test_a_decision_made_with_an_experimental_piece_says_so_in_its_record(tmp_path):
    from solvi.core.store.sysreport import render, system_report
    from solvi.experimental import mark
    s = _store_system(tmp_path)
    plain = s.ask({"n": 2})
    assert plain.experimental == [] and s.storage.record(plain.stored_id).get("meta") is None
    mark(s, "learning")
    res = s.ask({"n": 2})
    assert res.experimental == ["learning"]
    assert s.storage.record(res.stored_id)["meta"] == {"experimental": ["learning"]}
    mine = s.storage.save(s.ask({"n": 3}, store=False), meta={"run": 7})
    assert s.storage.record(mine)["meta"] == {"run": 7, "experimental": ["learning"]}
    assert s.storage.verify()["ok"]
    rep = system_report(s.storage)
    assert rep.to_dict()["period"]["experimental"] == {"learning": 2}
    assert "Made with experimental pieces" in render(rep.to_dict()) and "learning (2 decisions)" in render(rep.to_dict())


def test_an_experimental_part_is_found_by_its_module(tmp_path):
    from solvi.core.system import experimental_of
    s = _store_system(tmp_path)

    class Adapter:                                         # stands for an adapter class defined in an experimental module
        pass
    Adapter.__module__ = "solvi.experimental.lora"
    part = s.catalog.parts["m"]
    part.func.lora = Adapter()
    try:
        assert experimental_of(s) == ["lora"]
    finally:
        del part.func.lora


def test_the_hook_command_does_not_warn(tmp_path):
    run = subprocess.run([sys.executable, "-W", "error::UserWarning", "-m", "solvi", "hook", "--help"],
                         capture_output=True, text=True)
    assert run.returncode == 0 and "ExperimentalWarning" not in run.stderr


def test_migrate_a_module_imported_from_an_old_package():
    assert _migrate.rewrite("from solvi.agents import mcp\nfrom solvi.decide import part, DecideModel\n") == (
        "from solvi.experimental import mcp\nfrom solvi.core.deciders import DecideModel\n"
        "from solvi.core.deciders import part\n")


def test_migrate_keeps_a_noqa_on_every_line_it_splits():
    out = _migrate.rewrite("from solvi import Catalog, provenance  # noqa: E402\n")
    assert out == "from solvi import Catalog  # noqa: E402\nfrom solvi.core import provenance  # noqa: E402\n"


# --- the high level
def test_decision_system_is_the_old_auto_system_and_build_does_not_compile():
    from solvi.solutions import decisions
    with pytest.warns(solvi.SolviDeprecationWarning, match="AutoSystem was renamed in 1.0"):
        assert decisions.AutoSystem is decisions.DecisionSystem
    assert solvi.build is decisions.build
    with pytest.raises(TypeError, match="compiled first"):
        decisions._slow_path(object(), None, [], False, "writer", None)


def test_the_model_providers(monkeypatch):
    from solvi import models
    from solvi.core.deciders import DecideModel
    assert models.DecideModel is DecideModel
    m = models.llm("http://127.0.0.1:9/v1", "some-model")
    assert callable(getattr(m, "decision", None))
    seen = []
    monkeypatch.setattr(models, "load", lambda spec, backend="auto", api_key=None: seen.append((spec, backend)) or "m")
    assert models.decider("solvi-base") == "m" and models.decider("./ckpt", backend="onnx") == "m"
    assert seen == [("solvi-ai/solvi-base", "auto"), ("./ckpt", "onnx")]
    with pytest.warns(solvi.SolviDeprecationWarning, match="solvi.models.cmd_models moved in 1.0"):
        from solvi.cli import _models
        assert models.cmd_models is _models.cmd_models
