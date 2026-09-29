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
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md", "exec"), {})
    assert out.getvalue().strip() == printed.strip()
