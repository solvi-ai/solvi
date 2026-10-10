"""solvi.core.explain (stable since 1.1, graduated from solvi.experimental.counterfactual — measured on the credit
policy by the X1 run, its script benchmarks/counterfactual_credit.py): the smallest change of a decision's inputs that
changes its answer, with the cost of changing a text (enumerated) field a parameter (`text_cost=`) and a weight per
field (`costs=`); the old path re-exports the same objects with the experimental warning for one release."""
import importlib
import subprocess
import sys

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.explain import Change, Counterfactual, Counterfactuals, Rerun, search


def credit():
    cat = Catalog()

    @cat.rule("approve")
    def approve(income: float, job: str) -> bool:
        return income >= 10000 or job == "skilled"
    return System(cat, [Question("approve", "Approve?", Answer.yes_no())])


def refused():
    s = credit()
    res = s.ask({"income": 2000.0, "job": "unskilled"})
    assert res["approve"].answer in ("no", False)
    return res


JOBS = {"job": ["skilled", "unskilled"]}


def test_by_default_a_text_field_change_costs_one_and_wins_over_a_large_number_change():
    cf = search(refused(), "approve", domains=JOBS)
    assert isinstance(cf, Counterfactuals) and isinstance(cf.best, Counterfactual)
    assert cf.best.changes[0].fact == "job" and cf.best.changes[0].cost == 1.0
    income = next(c for c in cf.found if c.changes[0].fact == "income")
    assert income.changes[0].cost == pytest.approx(4.0) and isinstance(income.changes[0], Change)


def test_the_cost_of_a_text_field_change_is_a_parameter():
    cf = search(refused(), "approve", domains=JOBS, text_cost=5.0)
    assert cf.best.changes[0].fact == "income" and str(cf.best).startswith("yes if income ≥ 10000")
    job = next(c for c in cf.found if c.changes[0].fact == "job")
    assert job.changes[0].cost == 5.0


def test_a_weight_per_field_scales_its_changes():
    assert search(refused(), "approve", domains=JOBS, costs={"job": 10}).best.changes[0].fact == "income"
    cheap = search(refused(), "approve", domains=JOBS, costs={"income": 0.1})
    assert cheap.best.changes[0].fact == "income" and cheap.best.changes[0].cost == pytest.approx(0.4)
    with pytest.raises(ValueError, match="cost"):
        search(refused(), "approve", domains=JOBS, text_cost=-1)
    with pytest.raises(ValueError, match="cost"):
        search(refused(), "approve", domains=JOBS, costs={"income": 0})


def test_the_old_path_re_exports_the_same_objects_with_the_experimental_warning():
    import solvi.experimental
    from solvi.core import ExperimentalWarning, explain
    sys.modules.pop("solvi.experimental.counterfactual", None)
    vars(solvi.experimental).pop("counterfactual", None)
    with pytest.warns(ExperimentalWarning, match="graduated in 1.1.*solvi.core.explain"):
        old = importlib.import_module("solvi.experimental.counterfactual")
    for name in explain.__all__:
        assert getattr(old, name) is getattr(explain, name), name
    assert "counterfactual" not in solvi.experimental.STATUS and Rerun is explain.Rerun


def test_the_moved_module_keeps_its_fingerprint_name_and_imports_nothing_experimental():
    from solvi.core.provenance import fp_module
    assert fp_module("solvi.core.explain", "search") == "solvi.counterfactual"
    code = ("import sys, warnings\nwarnings.simplefilter('error')\nimport solvi.core.explain\n"
            "assert not [m for m in sys.modules if m.startswith('solvi.experimental.')]\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]


def test_the_docs_moved_with_it():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "guide.md").read_text()
    intro = guide.split("\n## ")[0]
    building = intro.split("**Building blocks** —")[1].split("**Experimental** —")[0]
    assert "Explaining a decision: the smallest change that flips it" in building
    assert "Counterfactual explanations (experimental)" not in guide
    assert "## `counterfactual`" not in (root / "docs" / "experimental.md").read_text()
    assert "solvi.core.explain: api/explain.md" in (root / "mkdocs.yml").read_text()
