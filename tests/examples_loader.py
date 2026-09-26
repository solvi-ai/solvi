"""Import the runnable scripts in examples/ (their file names start with a digit) as modules for the tests."""
import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(stem):
    spec = importlib.util.spec_from_file_location(stem, EXAMPLES / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
