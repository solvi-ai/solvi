"""Every module says what is public (__all__), and the API reference shows that and nothing else (mkdocs `filters:
public`); no library module imports from the command line (solvi.cli)."""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "solvi"


def test_every_module_declares_its_public_names():
    missing = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
               if p.name not in ("__main__.py", "_deprecate.py")
               and not any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in n.targets)
                           for n in ast.parse(p.read_text()).body)]
    assert missing == []
    assert "filters: public" in (SRC.parents[1] / "mkdocs.yml").read_text()


def test_no_library_module_imports_from_the_command_line():
    bad = []
    for p in SRC.rglob("*.py"):
        if p.name == "__main__.py" or p.parent.name == "cli":
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ImportFrom) and n.level and (n.module or "").split(".")[0] == "cli":
                bad.append(str(p.relative_to(SRC)))
    assert bad == []
