import pytest

from solvi.experimental.experience._common import WorkBudget, BudgetStop
from solvi.experimental.experience.programs import Grammar
from solvi.experimental.experience.competences import ExternalExample, CompetenceRegistry


def row(x, split, *, expected=None, source="environment", mechanism="m1", task=None, event=None):
    return ExternalExample(
        event or f"{split}-{x}",
        task or f"t-{split}-{x}",
        source,
        mechanism,
        "s1",
        "domain",
        {"x": x},
        2 * x + 1 if expected is None else expected,
        split,
    )


def setup_registry():
    g = Grammar(("x",), operators=("add",), max_depth=3, max_nodes=7, max_candidates=100)
    r = CompetenceRegistry(g)
    b = WorkBudget(100000)
    c = r.propose([row(0, "proposal"), row(2, "proposal"), row(3, "proposal")], b)
    assert c is not None
    return g, r, b, c


def admitted():
    g, r, b, c = setup_registry()
    result = r.admit(c, [row(x, "admission") for x in (4, 5, 6, 7)], b)
    assert result.accepted
    return g, r, b, c


def predict(r, b, x=8, task="fresh", mechanism="m1"):
    return r.predict({"x": x}, task_id=task, scope="domain", mechanism=mechanism, schema="s1", budget=b)


def test_composite_generalizes_and_pays_every_node():
    g, r, b, c = admitted()
    assert c.program.nodes >= 5
    before = b.counts.get("program_node", 0)
    a = predict(r, b)
    assert a.status == "applied" and a.value == 17
    assert b.counts["program_node"] - before == c.program.nodes
    assert c.proposal_examples[0].inputs == {"x": 0}
    altered = c.proposal_examples[0].inputs
    altered["x"] = 77
    assert c.proposal_examples[0].inputs == {"x": 0}


def test_wrong_candidate_rejected_and_admission_consumed():
    g = Grammar(("x",), operators=("mul",))
    r, b = CompetenceRegistry(g), WorkBudget(100000)
    c = r.propose([row(0, "proposal", expected=0), row(1, "proposal", expected=1)], b)
    rows = [row(x, "admission", expected=x * x) for x in (2, 3, 4, 5)]
    result = r.admit(c, rows, b)
    assert not result.accepted and result.reason == "counterexample"
    assert r.admit(c, rows, b) == result
    c2 = r.propose([row(6, "proposal", expected=36), row(7, "proposal", expected=49)], b)
    with pytest.raises(ValueError, match="consumed"):
        r.admit(c2, rows, b)


def test_empty_and_external_only():
    _, r, b, c = setup_registry()
    assert not r.admit(c, [], b).accepted
    for bad in [row(9, "evaluation"), row(9, "proposal", source="prediction")]:
        with pytest.raises(ValueError):
            r.propose([bad], b)
    with pytest.raises(ValueError, match="collision"):
        r.propose([row(0, "proposal", expected=999)], b)
    with pytest.raises(ValueError, match="collision"):
        r.propose([row(9, "proposal", task="t-proposal-0")], b)


def test_counterexample_revokes_and_deduplicates():
    _, r, b, _ = admitted()
    a = predict(r, b)
    e = row(8, "evaluation", expected=999, task="fresh")
    assert r.observe(a.application_id, e, b) == (a.competence_id,)
    assert r.observe(a.application_id, e, b) == ()
    assert predict(r, b, task="next").status == "unknown"
    with pytest.raises(ValueError):
        r.observe(a.application_id, row(8, "evaluation", expected=0, task="fresh"), b)


def test_context_change_never_revives_old_version():
    _, r, b, _ = admitted()
    assert predict(r, b, mechanism="m2").status == "unknown"
    assert predict(r, b, task="back", mechanism="m1").status == "unknown"


def test_snapshot_budget_and_tampering():
    g, r, b, c = admitted()
    a = predict(r, b)
    state = r.snapshot()
    restored = CompetenceRegistry.restore(state, grammar=g)
    assert restored.snapshot() == state
    assert predict(restored, b, task="after").value == 17
    state["applications"][a.application_id]["prediction"] = 0
    with pytest.raises(ValueError):
        CompetenceRegistry.restore(state, grammar=g)
    tiny = WorkBudget(0)
    with pytest.raises(BudgetStop):
        predict(r, tiny, task="tiny")
    assert tiny.spent == 0


def test_application_not_reusable_as_learning_data():
    _, r, b, _ = admitted()
    a = predict(r, b)
    with pytest.raises(ValueError, match="learning data"):
        r.propose([row(8, "proposal", task="fresh")], b)
    assert r.observe(a.application_id, row(8, "evaluation", task="fresh"), b) == ()


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda s: s["tasks"].clear(),
        lambda s: next(iter(s["competences"].values())).update(status="invented"),
        lambda s: next(iter(s["admissions"].values()))["event_ids"].clear(),
        lambda s: next(iter(s["applications"].values())).update(competence_id="missing"),
    ],
)
def test_restore_rejects_logical_corruption_with_recomputed_digest(corrupt):
    from solvi.experimental.experience._common import digest

    g, r, b, _ = admitted()
    predict(r, b)
    state = r.snapshot()
    corrupt(state)
    state["digest"] = digest({k: v for k, v in state.items() if k != "digest"})
    with pytest.raises(ValueError):
        CompetenceRegistry.restore(state, grammar=g)


def test_derive_expands_and_cascades():
    from solvi.experimental.experience.programs import Program

    g = Grammar(("x",), operators=("add",), max_depth=4, max_nodes=9, max_candidates=100)
    r, b = CompetenceRegistry(g), WorkBudget(100000)
    a = r.propose([row(0, "proposal"), row(2, "proposal"), row(3, "proposal")], b)
    ar = r.admit(a, [row(x, "admission") for x in (4, 5, 6, 7)], b)
    program = Program(("add", a.program.ast, ("const", 1)))
    c = r.derive(program, [row(x, "proposal", expected=2 * x + 2) for x in (10, 11)], (ar.competence_id,), b)
    cr = r.admit(c, [row(x, "admission", expected=2 * x + 2) for x in (12, 13, 14, 15)], b)
    assert cr.accepted
    CompetenceRegistry.restore(r.snapshot(), grammar=g)
    app = predict(r, b)
    revoked = r.observe(app.application_id, row(8, "evaluation", expected=0, task="fresh"), b)
    assert set(revoked) == {ar.competence_id, cr.competence_id}
    CompetenceRegistry.restore(r.snapshot(), grammar=g)


def test_reservation_hook_precedes_program_and_survives_raise():
    g, r, b, c = setup_registry()
    nodes_before = b.counts["program_node"]
    saved = []

    def fail_after_save():
        saved.append(r.snapshot())
        assert b.counts["program_node"] == nodes_before
        raise RuntimeError("durable writer failed")

    with pytest.raises(RuntimeError, match="durable"):
        r.admit(c, [row(x, "admission") for x in (4, 5, 6, 7)], b, on_reserved=fail_after_save)
    restored = CompetenceRegistry.restore(saved[0], grammar=g)
    assert restored.snapshot()["admissions"][c.fingerprint]["result"]["reason"] == "interrupted"
    assert len(restored.snapshot()["events"]) == 7


def test_external_store_retraction_prevents_apply():
    from solvi.core.knowledge import KnowledgeStore

    store = KnowledgeStore()
    # The host's admission gate is deliberately permissive for this adapter test.
    store.gates = []
    g = Grammar(("x",), operators=("add",), max_depth=3, max_nodes=7, max_candidates=100)
    r, b = CompetenceRegistry(g, store=store), WorkBudget(100000)
    c = r.propose([row(0, "proposal"), row(2, "proposal"), row(3, "proposal")], b)
    # KnowledgeStore can hold skills pending promotion. Explicit host promotion
    # models a successful store gate without replacing the real store methods.
    original_add = store.add

    def admitted_add(*args, **kwargs):
        key = original_add(*args, **kwargs)
        if key is not None:
            store.promote(key)
        return key

    store.add = admitted_add
    result = r.admit(c, [row(x, "admission") for x in (4, 5, 6, 7)], b)
    assert result.accepted
    sid = r.snapshot()["competences"][result.competence_id]["store_id"]
    store.retract(sid, why="host withdrawal")
    assert predict(r, b).status == "unknown"
    assert r.snapshot()["competences"][result.competence_id]["status"] == "revoked"
