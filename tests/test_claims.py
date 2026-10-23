"""A measured number or claim in the package's own texts — a docstring, a warning, an error — names the script that
produces it (benchmarks/..., tests/...), or it is said in words without the number."""
import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "solvi"
SOURCE = re.compile(r"benchmarks/|tests/|not\s+in\s+the\s+repo")
CLAIMS = [re.compile(p) for p in (
    r"\b[Mm]easured(:| on (?!the calibration examples)| with | no )",    # "Measured: ...", "measured on 4k texts"
    r"\b(in our (measurements|runs|tests)|our measurements|we measured|in our experience)\b",
    r"\d+(\.\d+)?\s?[x×] (faster|slower|smaller|larger|cheaper)",
    r"~\s?\d+(\.\d+)? (points|pts?)\b",
    r"\d+(\.\d+)?% vs \d",
    r"\bnot recommended\b",
    r"\b[Mm]easured once\b",    # "measured once on ...": the 1.4 wording the patterns above missed
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


def test_a_measured_experimental_piece_without_a_script_says_where_its_numbers_come_from():
    # an experimental piece whose STATUS names no script ("script": None) may carry numbers measured once outside the
    # repository; every paragraph that reports such a measurement — in its section of docs/experimental.md or in a
    # page that links to that section — says that its script is not in the repository, or names one in benchmarks/.
    # The wording of 1.4 ("Checked once on", "Measured once on real streams") passed the phrase lists above.
    from solvi.experimental import STATUS
    root = SRC.parents[1]
    measured = re.compile(r"\b(measured|checked) once\b|\ba simulation of\b", re.I)
    # the last marker: "still missing: a benchmark script in benchmarks/" says that there is none
    said = re.compile(r"benchmarks/\w|not\s+in\s+the\s+repo|benchmark\s+script\s+in\s+benchmarks/", re.I)
    page = (root / "docs" / "experimental.md").read_text()
    sections = dict(re.findall(r"^## `([a-z_]+)`\n(.*?)(?=^## |\Z)", page, re.M | re.S))
    pages = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    bad = []
    for name, status in STATUS.items():
        if status.get("script"):
            continue
        paras = [("experimental.md", para) for para in sections.get(name, "").split("\n\n")]
        paras += [(p.name, para) for p in pages for para in p.read_text().split("\n\n")
                  if f"experimental.md#{name})" in para]
        for where, para in paras:
            m = measured.search(para)
            if m and not said.search(para):
                bad.append(f"{name}: {where}: {para.strip()[:80]!r}")
    assert bad == []
