import pathlib
import re

import solvi


def test_version_matches_pyproject():
    text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert solvi.__version__ == re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def test_readme_quickstart_prints_what_the_readme_says():
    """The README's quickstart runs as written and prints the output block that follows it."""
    import contextlib
    import io

    readme = (pathlib.Path(__file__).parents[1] / "README.md").read_text()
    section = readme.split("## Quickstart", 1)[1]
    code, printed = re.findall(r"```(?:python)?\n(.*?)```", section, re.S)[:2]
    import linecache
    name = "<README quickstart>"          # not "README.md": inspect.getsource would read the whole file at the snippet's lines
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, name, "exec"), {})
    assert out.getvalue().strip() == printed.strip()
