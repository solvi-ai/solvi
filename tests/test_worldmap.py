"""solvi.core.knowledge.worldmap: a map an agent builds by acting — claims with a status, a source and evidence, a hash-chained journal,
ways over what is known, what is left to check; an observation refutes a claim whoever made it."""
import json

import pytest

from solvi.core.knowledge.worldmap import WorldMap

SITE = {"home": {"Billing": "billing", "Settings": "settings", "Help": "help"},
        "billing": {"Payments": "payments", "Home": "home"}, "payments": {"Refunds": "refunds", "Billing": "billing"},
        "settings": {"Team": "team", "Home": "home"}, "team": {"Roles": "roles"}, "help": {"Home": "home"},
        "refunds": {"Home": "home"}, "roles": {"Home": "home"}}


def walk(m, target, hints=False, limit=40):
    """Go to `target`: the known way, else explore. With hints the site shows where a link leads (as hrefs do)."""
    s, steps = "home", 0
    while s != target and steps < limit:
        m.visit(s, steps)
        for a, to in SITE[s].items():
            m.see(s, a, to if hints else None, steps)
        a = m.next(s, {target}) or m.explore(s)
        if a is None:
            break
        m.arrive(s, a, SITE[s][a], steps)
        s, steps = SITE[s][a], steps + 1
    return steps, s == target


def test_the_map_is_built_by_acting_and_carried_to_the_next_task():
    m = WorldMap()
    first, ok = walk(m, "refunds")
    assert ok and first > 3 and m.stats()["confirmed"] >= 3                 # found by exploring
    again, ok = walk(m, "refunds")
    assert ok and again == 3                                                 # the known way: home → billing → payments → refunds
    assert m.path("home", {"refunds"}) == [("home", "Billing"), ("billing", "Payments"), ("payments", "Refunds")]
    assert m.claim("payments", "Refunds") == {"to": "refunds", "status": "confirmed", "source": "observed",
                                             "evidence": m.claim("payments", "Refunds")["evidence"], "taken": 2}
    assert m.next("refunds", {"refunds"}) is None and m.next("home", {"nowhere"}) is None
    fresh = WorldMap()
    assert walk(fresh, "roles", hints=True)[1] and walk(fresh, "roles", hints=True) == (3, True)
    seen = WorldMap().see("home", "Help", "help")                           # shown, not walked: an unchecked claim
    assert seen.claim("home", "Help")["status"] == "hypothesis" and seen.frontier() == [("home", "Help")]
    assert seen.unvisited() == ["help"] and seen.explore("home") == "Help" and fresh.verify() and m.verify()


def test_a_claim_is_refuted_by_observation_whoever_made_it():
    m = WorldMap()
    m.told("home", "Help", "refunds", quote="Help → Refunds (site map, 2019)")       # an outdated document
    m.human("home", "Settings", "roles", note="Anna: Settings opens Roles")           # and a person who is wrong
    assert m.next("home", {"refunds"}) == "Help" and m.distances({"refunds"})["home"][0] == 3     # believed, at a price
    assert m.next("home", {"refunds"}, confirmed_only=True) is None
    assert m.arrive("home", "Help", "help", step=1) is True                            # went there: it is the help page
    assert m.arrive("home", "Settings", "settings", step=2) is True
    assert m.claim("home", "Help")["to"] == "help" and m.claim("home", "Help")["source"] == "observed"
    refuted = [r for r in m.journal if r["op"] == "refute"]
    assert [(r["believed"], r["source"], r["observed"]) for r in refuted] == [("refunds", "told", "help"),
                                                                             ("roles", "human", "settings")]
    m.told("home", "Help", "refunds")                                                  # told again: the observation stands
    assert m.claim("home", "Help")["to"] == "help" and m.journal[-1]["op"] == "told_ignored" and m.stats()["refuted"] == 2
    assert m.arrive("home", "Help", "help") is False
    with pytest.raises(ValueError):
        m.told("home", "Help", "x", source="observed")


def test_journal_snapshot_and_file(tmp_path):
    m = WorldMap(tmp_path / "maps" / "site.json")
    walk(m, "roles", hints=True)
    snap = m.snapshot("home", {"roles"})
    assert snap["next"] == "Settings" and snap["cost"] == 3 and set(snap["actions"]) == {"Billing", "Help", "Settings"}
    assert snap["actions"]["Settings"] == {"to": "settings", "status": "confirmed", "source": "observed"}
    assert snap == json.loads(json.dumps(snap)) and snap["head"] == m.journal[-1]["hash"]
    m.save()
    back = WorldMap(tmp_path / "maps" / "site.json")
    assert back.edges == m.edges and back.verify() and back.next("home", {"roles"}) == "Settings"
    assert back.states["home"]["visits"] == m.states["home"]["visits"] and back.stats() == m.stats()
    back.arrive("home", "Billing", "billing")                                          # the chain goes on after loading
    assert back.verify() and len(back.journal) == len(m.journal) + 1
    m.journal[1]["to"] = "elsewhere"                                                   # an edited entry is found
    assert not m.verify()
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="not a solvi.worldmap v1 file"):
        WorldMap(tmp_path / "other.json")
    with pytest.raises(ValueError):
        WorldMap().save()
    empty = WorldMap()
    assert empty.explore("home") is None and empty.frontier() == [] and empty.snapshot("home")["actions"] == {}


@pytest.mark.parametrize("a, b, c", [(("room", 3), ("room", 4), ("room", 5)), (1, 2, 3), ("a", "b", "c"), (1.5, 2, "x")],
                         ids=["tuples", "integers", "strings", "mixed"])
def test_a_map_with_states_that_are_not_strings_is_the_same_after_save_and_load(tmp_path, a, b, c):
    """Tuple states used to work in memory and raise at save(), at the end of the task; integer states saved, and after
    load() the visit counts were gone and visited states were reported unvisited."""
    m = WorldMap(tmp_path / "m.json")
    m.visit(a, title="start").see(a, "next", b)
    m.arrive(a, "next", b)
    m.visit(b).see(b, ("go", 2), c)
    m.save()
    again = WorldMap(tmp_path / "m.json")
    assert again.states == m.states and again.edges == m.edges and again.verify()
    assert again.unvisited() == m.unvisited() == [c] and again.snapshot(a)["visits"] == 1
    assert again.next(a, {c}) == "next" and again.path(a, {c}) == [(a, "next"), (b, ("go", 2))]
    assert again.snapshot(a, {c}) == m.snapshot(a, {c}) and json.dumps(again.to_dict()) == json.dumps(m.to_dict())


def test_a_map_of_string_states_is_written_as_before_and_an_unusable_state_is_refused_when_reported(tmp_path):
    m = WorldMap()
    m.visit("home").see("home", "Billing", "billing")
    assert m.to_dict()["states"] == {"home": {"visits": 1, "facts": {}}}
    for bad in (["room", 3], {"room": 3}, ("room", [3]), object()):
        with pytest.raises(TypeError):
            m.visit(bad)
        with pytest.raises(TypeError):
            m.arrive("home", "Billing", bad)
    assert m.stats()["states"] == 1 and m.claim("home", "Billing")["to"] == "billing"


def test_a_saved_map_is_loaded_from_its_journal_so_an_edited_edge_changes_nothing(tmp_path):
    """verify() re-hashed only the journal and load() took edges and states from the file unchecked: an edge edited to
    point elsewhere passed verify() and next() followed it."""
    m = WorldMap(tmp_path / "m.json")
    walk(m, "refunds", hints=True)
    m.human("home", "Reports", "audit", note="Anna")
    m.told("home", "Billing", "elsewhere")                        # ignored: the claim is confirmed
    m.save()
    data = json.loads((tmp_path / "m.json").read_text())
    for e in data["edges"]:
        if (e["state"], e["action"]) == ("home", "Billing"):
            e["to"] = "phishing"
    data["states"]["home"]["visits"] = 99
    (tmp_path / "m.json").write_text(json.dumps(data))
    again = WorldMap(tmp_path / "m.json")
    assert again.claim("home", "Billing")["to"] == "billing" and again.edges == m.edges and again.states == m.states
    assert again.verify() and again.next("home", {"refunds"}) == "Billing"
    again.edges[("home", "Billing")]["to"] = "phishing"           # the map in memory against its journal
    assert not again.verify()
    data["journal"][2]["to"] = "phishing"
    (tmp_path / "m.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash chain is broken"):
        WorldMap(tmp_path / "m.json")


def test_the_map_as_it_was_at_any_step_is_rebuilt_from_the_journal():
    m = WorldMap()
    m.visit("home").see("home", "Billing", "billing")
    n = len(m.journal)
    m.arrive("home", "Billing", "payments")                       # refutes the claim
    m.visit("payments")
    then = m.rebuild(upto=n)
    assert then.claim("home", "Billing") == {"to": "billing", "status": "hypothesis", "source": "seen",
                                             "evidence": [[None, None]], "taken": 0}
    now = m.rebuild()
    assert now.edges == m.edges and now.states == m.states and now.states["home"]["visits"] == 1
    assert [r["op"] for r in m.journal] == ["visit", "see", "refute", "confirm", "visit"]


def test_a_map_file_written_before_visits_were_journaled_still_loads(tmp_path):
    m = WorldMap()
    m._visits = False                                              # as the earlier version wrote it
    walk(m, "roles")
    d = m.to_dict()
    d.pop("visits_journaled")
    (tmp_path / "old.json").write_text(json.dumps(d))
    old = WorldMap(tmp_path / "old.json")
    assert old.states == m.states and old.edges == m.edges and old.verify() and not old._visits
