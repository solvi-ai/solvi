"""A measured number or claim in the package's own texts — a docstring, a warning, an error — names the script that
produces it (benchmarks/..., tests/...), or it is said in words without the number."""
import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "solvi"
SOURCE = re.compile(r"benchmarks/|tests/")
CLAIMS = [re.compile(p) for p in (
    r"\b[Mm]easured(:| on (?!the calibration examples)| with | no )",    # "Measured: ...", "measured on 4k texts"
    r"\b(in our (measurements|runs|tests)|our measurements|we measured|in our experience)\b",
    r"\d+(\.\d+)?\s?[x×] (faster|slower|smaller|larger|cheaper)",
    r"~\s?\d+(\.\d+)? (points|pts?)\b",
    r"\d+(\.\d+)?% vs \d",
    r"\bnot recommended\b",
)]


def _texts(tree):
    """Every string the module writes: constants (implicit concatenation is one constant) and the literal parts of an
    f-string, joined."""
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            yield node.lineno, "".join(v.value for v in node.values if isinstance(v, ast.Constant))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def test_no_measured_number_or_claim_in_the_packages_texts_without_its_script():
    bad = []
    for p in SRC.rglob("*.py"):
        for line, text in _texts(ast.parse(p.read_text())):
            for rx in CLAIMS:
                m = rx.search(text)
                if m and not SOURCE.search(text):
                    bad.append(f"{p.relative_to(SRC)}:{line}: {m.group(0)!r}")
    assert bad == []
