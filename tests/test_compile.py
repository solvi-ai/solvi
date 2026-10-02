"""solvi.compile: a specification split into clauses, catalog parts written by a model (a stand-in here) and accepted
only when two drafts agree on generated inputs, pass the tests derived from the specification and keep the module
contract; a revised specification recompiled part by part; the decisions a change moves, with the clauses why; versions
that replay old decisions with the catalog that made them; compiled hard checks as guard policies."""
import json

import pytest

from solvi import Answer, JSONLStorage, Question
from solvi.compile import (Inputs, Rejected, Spec, Versions, compile_spec, coverage, decision_diff, read_module,
                           recompile, split_clauses, to_guard)

POLICY = """# Shipping

| zone | fee |
|---|---|
| domestic | 5 |
| world | 20 |

- An order of 50 or more ships free.
- Orders to the world zone heavier than 30 kg are refused.
"""
QS = [Question("ship", "Shipped, and at what fee?", Answer.choice(["free", "paid", "refused"]))]
INPUTS = Inputs({"zone": ["domestic", "world"], "total": (0, 200), "weight": (0, 50)}, n=150, seed=3)

GOOD = '''
def zone_fee(zone):
    return {"domestic": 5, "world": 20}[zone]

def free_shipping(total):
    return total >= 50

def not_too_heavy(zone, weight):
    return not (zone == "world" and weight > 30) or Fail(f"{weight} kg to the world zone")

def ship(zone_fee, free_shipping):
    return "free" if free_shipping else "paid"

PARTS = {
    "zone_fee": {"kind": "fn", "clauses": ["c1", "c2"]},
    "free_shipping": {"kind": "fn", "clauses": ["c3"]},
    "not_too_heavy": {"kind": "check", "hard": True, "then": {"ship": "refused"}, "clauses": ["c4"]},
    "ship": {"kind": "rule", "question": "ship", "clauses": ["c3"]},
}
NOT_NORMATIVE = {}
'''
OFF_BY_ONE = GOOD.replace("total >= 50", "total > 50")
TEST_OK = [{"clause": "c3", "input": {"zone": "domestic", "total": 50, "weight": 1}, "expect": {"ship": "free"},
            "why": "50 or more"},
           {"clause": "c4", "input": {"zone": "world", "total": 10, "weight": 31}, "expect": {"ship": "refused"},
            "why": "over 30 kg"}]


class Writer:
    """A stand-in for solvi.generate's Generator: scripted replies by what is asked (tests, review, a draft and its
    round), every prompt kept."""
    model_id = "stand-in"

    def __init__(self, drafts, tests=TEST_OK, review=None):
        self.drafts, self.tests, self.review, self.prompts = drafts, tests, review, []

    def fingerprint(self):
        return "stand-in"

    def generate(self, messages, parse=None, temperature=None, seed=None, **kw):
        p = messages[-1]["content"]
        self.prompts.append((p, temperature))
        if p.startswith("# Write tests"):
            reply = "```json\n" + json.dumps(self.tests) + "\n```"
        elif p.startswith("# Re-check one test"):
            reply = "```json\n" + json.dumps(self.review or {"verdict": "keep"}) + "\n```"
        else:
            draft = 0 if not temperature else 1
            rnd = 1 + sum(1 for q, t in self.prompts[:-1] if (0 if not t else 1) == draft and q.startswith("# Task"))
            seq = self.drafts[draft]
            reply = "```python\n" + seq[min(rnd, len(seq)) - 1] + "\n```"

        class G:
            pass
        g = G()
        g.value, g.meta = (parse(reply) if parse else reply), {"text": reply}
        return g


def test_paragraphs_list_items_and_table_rows_are_clauses_and_headings_name_their_section():
    cl = split_clauses(POLICY)
    assert cl == [("Shipping", "zone: domestic; fee: 5"), ("Shipping", "zone: world; fee: 20"),
                  ("Shipping", "An order of 50 or more ships free."),
                  ("Shipping", "Orders to the world zone heavier than 30 kg are refused.")]
    s = Spec(POLICY)
    assert list(s.clauses) == ["c1", "c2", "c3", "c4"] and "[c3] An order of 50" in s.render()
    assert s.hash == Spec(POLICY).hash != Spec(POLICY.replace("50", "60")).hash


def test_a_revised_specification_keeps_the_ids_of_unchanged_clauses_and_names_the_changes():
    s = Spec(POLICY)
    r = s.revise(POLICY.replace("50 or more", "60 or more").replace("are refused.", "are refused.\n- Gift cards ship free."))
    assert r.changes == {"unchanged": ["c1", "c2", "c4"], "changed": ["c3"], "added": ["c5"], "removed": []}
    assert r.clauses["c5"].text == "Gift cards ship free."
    gone = s.revise(POLICY.replace("- Orders to the world zone heavier than 30 kg are refused.\n", ""))
    assert gone.changes["removed"] == ["c4"] and "c4" not in gone.clauses


def test_the_pool_holds_the_samples_the_drawn_inputs_and_boundary_values_around_the_numbers():
    inp = Inputs({"zone": ["domestic", "world"], "total": (0, 200)}, samples=[{"zone": "world", "total": 7}], n=5)
    pool = inp.pool([50])
    assert pool[0] == {"zone": "world", "total": 7} and len(pool) == 1 + 5 + 3
    assert [x["total"] for x in pool[-3:]] == [49, 50, 51]
    assert inp.pool([50]) == pool
    assert inp.validate({"zone": "mars", "total": 1}) == ["zone = 'mars' is not one of its values"]
    assert inp.validate({"zone": "world"}) == ["missing field total"]
    with pytest.raises(ValueError):
        Inputs(fields={"doc": str}).pool()


def test_the_module_contract_names_each_problem():
    s = Spec(POLICY)
    _, _, _, why = read_module("def f(x):\n    return 1\n", s, QS)
    assert "no PARTS dict" in why[0]
    bad = GOOD.replace('"clauses": ["c4"]', '"clauses": ["c9"]').replace('"then": {"ship": "refused"}',
                                                                         '"then": {"ship": "maybe"}')
    _, _, _, why = read_module(bad, s, QS)
    assert any("cites 'c9'" in w for w in why) and any("not one of" in w for w in why)
    parts, nn, _, why = read_module(GOOD, s, QS)
    assert why == [] and coverage(s, parts, nn) == []
    parts, nn, _, _ = read_module(GOOD.replace('"clauses": ["c4"]', '"clauses": ["c3"]'), s, QS)
    assert coverage(s, parts, nn) == ["c4"]
    assert read_module("import os\n", s, QS)[3][0].startswith("the sandbox refuses")


def test_two_drafts_that_agree_and_pass_the_tests_are_accepted_and_the_record_keeps_how():
    w = Writer([[GOOD], [GOOD]])
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert c.accepted and c.reason == "accepted in round 1"
    assert c.clauses_of("not_too_heavy") == ["c4"] and c.clauses_of("answer:ship") == ["c3"]
    s = c.system()
    assert s.ask({"zone": "world", "total": 60, "weight": 40})["ship"].answer == "refused"
    assert s.ask({"zone": "domestic", "total": 60, "weight": 40})["ship"].answer == "free"
    r = c.record
    assert r["spec_hash"] == Spec(POLICY).hash and len(r["tests"]) == 2
    assert [x["what"] for x in r["calls"]] == ["tests", "round 1 draft 0", "round 1 draft 1"]
    assert "[c4] Orders to the world zone" in r["calls"][1]["prompt"]
    assert r["rounds"][0]["agreement"]["disagree"] == 0


def test_a_disagreement_goes_back_to_both_writers_with_the_input_both_answers_and_the_clauses():
    w = Writer([[GOOD], [OFF_BY_ONE, GOOD]], tests=[TEST_OK[1]])
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert c.accepted and c.reason == "accepted in round 2"
    first = c.record["rounds"][0]
    assert first["agreement"]["disagree"] >= 1 and "disagreement" in first["drafts"][1]["caught"]
    rewrite = [p for p, t in w.prompts if t and "Your previous module" in p][0]
    assert "decides differently" in rewrite and '"total": 50' in rewrite and "[c3] An order of 50" in rewrite


def test_a_test_both_drafts_fail_is_reviewed_once_and_a_dropped_test_no_longer_counts():
    wrong = {"clause": "c3", "input": {"zone": "domestic", "total": 49, "weight": 1}, "expect": {"ship": "free"},
             "why": "a wrong test"}
    w = Writer([[GOOD], [GOOD]], tests=TEST_OK + [wrong], review={"verdict": "drop", "why": "49 is below 50"})
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert c.accepted
    assert c.record["reviews"]["t3"]["verdict"] == "drop" and c.record["reviews"]["t3"]["drafts_gave"] == [{"ship": "paid"}] * 2
    w = Writer([[GOOD], [GOOD]], tests=TEST_OK + [wrong], review={"verdict": "keep"})
    c = compile_spec(Spec(POLICY), QS, INPUTS, w, rounds=2)
    assert not c.accepted and "tests" in c.reason
    with pytest.raises(Rejected):
        c.system()


def test_a_draft_the_sandbox_refuses_is_rewritten_with_the_reasons():
    w = Writer([["import os\n" + GOOD, GOOD], [GOOD]])
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert c.accepted and c.reason == "accepted in round 2"
    assert any("import os is not allowed" in p for p, t in w.prompts if "Your previous module" in p)


def test_labelled_examples_and_a_reference_are_checks_too_when_given():
    ref = lambda x: {"ship": "refused" if x["zone"] == "world" and x["weight"] > 30 else  # noqa: E731
                     "free" if x["total"] >= 50 else "paid"}
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]), reference=ref,
                     examples=[({"zone": "domestic", "total": 50, "weight": 1}, {"ship": "free"})])
    assert c.accepted
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[OFF_BY_ONE], [OFF_BY_ONE]], tests=[TEST_OK[1]]), rounds=1,
                     examples=[({"zone": "domestic", "total": 50, "weight": 1}, {"ship": "free"})])
    assert not c.accepted and "labelled examples" in c.reason


V2 = POLICY.replace("50 or more", "80 or more")
PATCH = '''
def free_shipping(total):
    return total >= 80

PARTS = {"free_shipping": {"kind": "fn", "clauses": ["c3"]}}
'''


def test_a_recompile_replaces_only_the_parts_the_change_touches_and_the_rest_stays_byte_identical():
    c1 = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]))
    tests = [{"clause": "c3", "input": {"zone": "domestic", "total": 79, "weight": 1}, "expect": {"ship": "paid"},
              "why": "below 80"}]
    c2 = recompile(c1, c1.spec.revise(V2), INPUTS, Writer([[PATCH], [PATCH]], tests=tests))
    assert c2.accepted
    assert c2.changes["parts"] == {"added": [], "replaced": ["free_shipping"], "removed": [],
                                   "kept": ["zone_fee", "not_too_heavy", "ship"]}
    keep = 'def not_too_heavy(zone, weight):\n    return not (zone == "world" and weight > 30) or Fail(f"{weight} kg to the world zone")\n'
    assert keep in c2.source and "total >= 80" in c2.source and "total >= 50" not in c2.source
    s1, s2 = c1.system().fingerprint()["parts"], c2.system().fingerprint()["parts"]
    assert s1["not_too_heavy"] == s2["not_too_heavy"] and s1["free_shipping"] != s2["free_shipping"]
    assert c2.system().ask({"zone": "domestic", "total": 70, "weight": 1})["ship"].answer == "paid"


def test_a_patch_that_changes_a_part_without_citing_a_changed_clause_is_refused():
    c1 = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]))
    sneaky = PATCH + '''
def zone_fee(zone):
    return {"domestic": 0, "world": 20}[zone]

PARTS["zone_fee"] = {"kind": "fn", "clauses": ["c1"]}
'''.replace('PARTS["zone_fee"] = ', "PARTS.update({\"zone_fee\": ").replace('"clauses": ["c1"]}\n', '"clauses": ["c1"]}})\n')
    c2 = recompile(c1, c1.spec.revise(V2), INPUTS, Writer([[sneaky], [sneaky]], tests=[]), rounds=1, tests=0)
    assert not c2.accepted
    assert "zone_fee is added or replaced but cites no [changed] or [added] clause" in json.dumps(c2.record["rounds"])


def test_the_decision_diff_lists_the_decisions_a_change_moves_with_the_clauses_of_their_causes():
    c1 = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]))
    c2 = recompile(c1, c1.spec.revise(V2), INPUTS, Writer([[PATCH], [PATCH]], tests=[]), tests=0)
    xs = [{"zone": "domestic", "total": t, "weight": 1} for t in (10, 50, 79, 80, 120)]
    dd = decision_diff(c1, c2, inputs=xs)
    assert [(d["seq"], d["old"], d["new"]) for d in dd.changed] == [(1, "free", "paid"), (2, "free", "paid")]
    assert dd.changed[0]["causes"][0]["part"] == "free_shipping" and dd.changed[0]["causes"][0]["clauses"] == ["c3"]
    assert str(dd).startswith("2 of 5 decisions change")


def test_versions_replay_each_stored_decision_with_the_catalog_that_made_it(tmp_path):
    c1 = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]))
    c2 = recompile(c1, c1.spec.revise(V2), INPUTS, Writer([[PATCH], [PATCH]], tests=[]), tests=0)
    v = Versions(tmp_path / "versions")
    assert (v.add(c1, "v1"), v.add(c2, "v2")) == (1, 2) and len(v) == 2
    assert v.find(c1.fingerprint) == 1 and v.get(2).source == c2.source
    store = JSONLStorage(str(tmp_path / "d.jsonl"))
    s1 = v.system(1, storage=store)
    for t in (60, 70):
        s1.ask({"zone": "domestic", "total": t, "weight": 1})
    v.system(2, storage=store).ask({"zone": "domestic", "total": 90, "weight": 1})
    assert v.replay_all(store) == []
    assert len(store.replay_all(v.system(2))) == 2              # the v1 decisions do not recompute under v2
    with pytest.raises(Rejected):
        v.add(compile_spec(Spec(POLICY), QS, INPUTS, Writer([["x = 1"], ["x = 1"]]), rounds=1))


def test_compiled_hard_checks_become_guard_policies_with_the_reasons_of_the_check():
    from solvi.agents import Guard
    spec = Spec("- A refund is at most 100.")
    mod = '''
def refund_small(tool_arguments):
    return tool_arguments.get("amount", 0) <= 100 or Fail("a refund over 100")

def allow(refund_small):
    return True

PARTS = {"refund_small": {"kind": "check", "hard": True, "then": {"allow": "no"}, "clauses": ["c1"]},
         "allow": {"kind": "rule", "question": "allow", "clauses": []}}
'''
    q = [Question("allow", "Is the call allowed?", Answer.yes_no())]
    inp = Inputs({"tool_arguments": dict}, samples=[{"tool_arguments": {"amount": a}} for a in (50, 100, 101, 500)])
    c = compile_spec(spec, q, inp, Writer([[mod], [mod]], tests=[]), tests=0)
    assert c.accepted
    guard = Guard()

    @guard.tool(authorize=False)
    def refund(amount: float) -> str:
        """Refund an amount."""
        return "done"
    assert to_guard(c, guard, "refund") == ["refund_small"]
    assert guard.call({"name": "refund", "arguments": {"amount": 50}}, context="refund 50").outcome == "allow"
    d = guard.call({"name": "refund", "arguments": {"amount": 500}}, context="refund 500")
    assert d.outcome == "deny" and "A refund is at most 100." in " ".join(d.reasons)      # the clause it implements


def test_a_part_that_reads_a_name_no_input_or_part_gives_is_sent_back_before_it_runs():
    reads_app = GOOD.replace("def free_shipping(total):\n    return total >= 50",
                             "def free_shipping(app):\n    return app['total'] >= 50")
    w = Writer([[reads_app, GOOD], [GOOD]])
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert c.accepted and c.reason == "accepted in round 2"
    assert "part free_shipping reads app: neither an input field nor a fact set by a part" in \
        c.record["rounds"][0]["drafts"][0]["problems"][0]


def test_tests_written_as_json_with_comments_are_read_and_no_valid_test_means_no_acceptance():
    commented = [dict(t) for t in TEST_OK]

    class Commenting(Writer):
        def generate(self, messages, parse=None, temperature=None, seed=None, **kw):
            if messages[-1]["content"].startswith("# Write tests"):
                text = "```json\n" + json.dumps(commented, indent=1).replace('"total": 50,', '"total": 50,   // the boundary') \
                    .replace("}\n]", "},\n]") + "\n```"
                self.prompts.append((messages[-1]["content"], temperature))

                class G:
                    pass
                g = G()
                g.value, g.meta = parse(text), {"text": text}
                return g
            return super().generate(messages, parse, temperature, seed, **kw)
    c = compile_spec(Spec(POLICY), QS, INPUTS, Commenting([[GOOD], [GOOD]]))
    assert c.accepted and len(c.record["tests"]) == 2
    bad = [{"clause": "c3", "input": {"zone": "moon", "total": 1, "weight": 1}, "expect": {"ship": "free"}}]
    w = Writer([[GOOD], [GOOD]], tests=bad)
    c = compile_spec(Spec(POLICY), QS, INPUTS, w)
    assert not c.accepted and c.reason.startswith("not accepted: no valid tests")
    assert [x["what"] for x in c.record["calls"]] == ["tests", "tests again"]


def test_a_draft_that_cannot_answer_some_inputs_is_not_accepted_even_when_both_drafts_agree():
    fragile = GOOD.replace('return {"domestic": 5, "world": 20}[zone]', 'return {"domestic": 5}[zone]').replace(
        '"zone_fee": {"kind": "fn", "clauses": ["c1", "c2"]}', '"zone_fee": {"kind": "fn", "clauses": ["c1", "c2"]}')
    fragile = fragile.replace("def ship(zone_fee, free_shipping):", "def ship(zone_fee, free_shipping):  # reads zone_fee")
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[fragile], [fragile]]), rounds=1)
    assert not c.accepted and "abstained" in c.reason
