import pathlib
import re

import solvi


def test_version_matches_pyproject():
    text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert solvi.__version__ == re.search(r'^version = "([^"]+)"', text, re.M).group(1)
