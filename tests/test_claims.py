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


def test_the_readme_and_docs_give_advice_not_measurements_without_a_script():
    # a paragraph that reports a measurement names its script (benchmarks/...) or a model card; these phrases were
    # left from measurements that are not in the repository
    root = SRC.parents[1]
    phrases = re.compile(r"in our measurements|we measured|what we measured|not recommend|several times slower|"
                         r"mostly Python's\s+start-up|with the old 0\.7 sets", re.I)
    bad = []
    # vs_llm.md is one measurement throughout, re-run by benchmarks/vs_llm/ (the page says so at the top)
    for p in [root / "README.md", *sorted(d for d in (root / "docs").glob("*.md") if d.name != "vs_llm.md")]:
        for i, para in enumerate(p.read_text().split("\n\n")):
            m = phrases.search(para)
            if m and not re.search(r"benchmarks/|model card|\bcard\)", para):
                bad.append(f"{p.name} paragraph {i}: {m.group(0)!r}")
    assert bad == []
