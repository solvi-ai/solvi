"""A map of an environment that an agent builds by acting — claims with provenance, not a given.

An agent that works in the same environment again and again — a site, an internal tool, a command line, a file tree —
re-discovers its structure on every task unless it keeps a map. A `WorldMap` is that map, written as the agent goes:

    from solvi.worldmap import WorldMap
    m = WorldMap("console.map.json")              # loaded when the file exists; m.save() writes it
    m.see(page, "Billing", to="/billing")         # an action on offer here; `to` when the environment shows it (a link)
    m.arrive(page, "Billing", "/billing")         # the action was taken: the claim is confirmed — or refuted
    m.next(page, {"/billing/refunds"})            # the action to take towards a target over what is known, or None
    m.explore(page)                               # ... or towards the nearest claim nobody has checked yet

Every edge is a claim "(state, action) leads to state" with a status — hypothesis, confirmed — a source — seen (the
environment showed where it leads), observed (the agent went), told (a document, a sign), human — and its evidence
(the steps, a quote). Whoever made a claim, the next observation can refute it: a person's correction or a line of an
outdated document is a hypothesis like any other, and `arrive` records the refutation with what was believed and by
whom. Every write is a journal entry in a hash chain (`verify()`), so the map as it was at any step can be rebuilt
(`rebuild(upto=n)`); a saved map is loaded by replaying its journal, so an edge edited in the file changes nothing.

What the map knows is what it was told through these calls: it does not know what a page or a directory is. The
adapter — list the actions of a state, take one — is yours. `snapshot(state, targets)` gives a decision the part of
the map it needs as a plain fact (the known way, what is unexplored here), so decisions that use it replay.

Where it helps: the gain is the map carried from one task to the next, in an environment that is deep and met again
(a command tree, a file tree, a documentation site several clicks deep) — later tasks take fewer steps than the first.
It does not make a first exploration shorter, it gives nothing where everything is one step away, and it does not tell
which state a task needs — only how to get to one that is named."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

SOURCES = ("seen", "observed", "told", "human")
HYPOTHESIS_COST = 3                  # a claim nobody has walked counts as this many confirmed steps when planning


def _name(x, what="a state"):
    """A state or an action as the map keeps it: a string, a number, or a tuple of those (("room", 3)) — what a JSON
    file gives back unchanged (a tuple is written as a list and read back as a tuple). Anything else is refused here,
    when it is reported, not at save()."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, tuple):
        return tuple(_name(y, what) for y in x)
    raise TypeError(f"{what} is a string, a number or a tuple of those, not {type(x).__name__} ({x!r})")


def _thaw(x):
    """A state or an action read from JSON: a list is the tuple it was written from."""
    return tuple(_thaw(y) for y in x) if isinstance(x, list) else x


class WorldMap:
    """See the module docstring. path: a JSON file the map is loaded from when it exists (save() writes it).
    hypothesis_cost: how many confirmed steps an unchecked claim with a destination counts as in `distances`.
    A state or an action is a string, a number or a tuple of those; save / load keep them as they are."""

    def __init__(self, path=None, hypothesis_cost=HYPOTHESIS_COST):
        self.file = Path(path) if path else None
        self.hypothesis_cost = int(hypothesis_cost)
        self.edges = {}                  # (state, action) → {"to", "status", "source", "evidence", "taken"}
        self.states = {}                 # state → {"visits", "facts"}
        self.journal, self._prev = [], ""
        self._visits = True              # every visit is a journal entry (False: a map loaded from a file written before)
        if self.file and self.file.exists():
            self.load(self.file)

    # --- the journal
    def _write(self, op, **data):
        from .core.runtime import vhash
        rec = {"n": len(self.journal), "op": op, **data, "prev": self._prev}
        rec["hash"] = vhash(rec)
        self._prev = rec["hash"]
        self.journal.append(rec)
        return rec

    def verify(self):
        """Is the journal's hash chain whole (nothing edited, removed or reordered), and is the map what the journal
        says — every claim (and, for a map whose visits are journaled, every state) the one its entries give?"""
        from .core.runtime import vhash
        prev = ""
        for r in self.journal:
            if r.get("prev") != prev or vhash({k: v for k, v in r.items() if k != "hash"}) != r.get("hash"):
                return False
            prev = r["hash"]
        try:
            m = self.rebuild()
        except ValueError:
            return False
        return m.edges == self.edges and (not self._visits or m.states == self.states)

    def rebuild(self, upto=None):
        """The map as its journal gives it — a new WorldMap made by applying the entries in order (upto: only the first
        `upto` of them: the map as it was at that step). ValueError when the entries do not give this journal back (an
        entry that could not have been written in its place)."""
        m = WorldMap(hypothesis_cost=self.hypothesis_cost)
        m._visits = self._visits
        entries = self.journal if upto is None else self.journal[:upto]
        for r in entries:
            op = r.get("op")
            st, a, to = _thaw(r.get("state")), _thaw(r.get("action")), _thaw(r.get("to"))
            if op == "visit":
                m.visit(st, r.get("step"), **(r.get("facts") or {}))
            elif op == "see":
                m.see(st, a, to, r.get("step"))
            elif op in ("told", "told_ignored"):      # told() writes the one or the other itself
                m.told(st, a, to, r.get("source"), r.get("quote"), r.get("step"))
            elif op == "confirm":
                m.arrive(st, a, to, r.get("step"))
            elif op != "refute":                      # a refutation is written by the arrive that follows it
                raise ValueError(f"journal entry {r.get('n')}: unknown operation {op!r}")
        if [x["hash"] for x in m.journal] != [x.get("hash") for x in entries]:
            raise ValueError("the journal does not replay: its entries do not give the same journal when applied in order")
        return m

    # --- what the agent reports
    def visit(self, state, step=None, **facts):
        """The agent is in this state (facts: what it noticed here — a title, a service; kept on the state)."""
        s = self.states.setdefault(_name(state), {"visits": 0, "facts": {}})
        s["visits"] += 1
        if facts:
            s["facts"].update(facts)
        if facts or self._visits:
            self._write("visit", state=state, step=step, facts=facts)
        return self

    def see(self, state, action, to=None, step=None):
        """An action is on offer in this state. `to`: where it leads when the environment shows that before acting (a
        link's address, a directory entry) — then a hypothesis with a destination, source "seen"; else the destination
        is unknown until someone takes it. A confirmed claim is not touched."""
        state, action, to = _name(state), _name(action, "an action"), _name(to)
        e = self.edges.get((state, action))
        if e is None or (e["status"] != "confirmed" and e["to"] is None and to is not None):
            self.edges[(state, action)] = {"to": to, "status": "hypothesis", "source": "seen", "evidence": [[step, None]],
                                           "taken": e["taken"] if e else 0}
            self._write("see", state=state, action=action, to=to, step=step)
        return self

    def told(self, state, action, to, source="told", quote=None, step=None):
        """Someone says where an action leads — a document, a sign (source="told"), a person (source="human"): a
        hypothesis with a destination and the quote. It replaces an unchecked claim; a confirmed one stands until an
        observation refutes it — use `arrive` for what was observed."""
        if source not in SOURCES or source == "observed":
            raise ValueError(f'source is one of "seen", "told", "human" (an observation is arrive()), not {source!r}')
        state, action, to = _name(state), _name(action, "an action"), _name(to)
        e = self.edges.get((state, action))
        if e is not None and e["status"] == "confirmed":
            self._write("told_ignored", state=state, action=action, to=to, source=source, quote=quote, step=step,
                        confirmed=e["to"])
            return self
        self.edges[(state, action)] = {"to": to, "status": "hypothesis", "source": source, "evidence": [[step, quote]],
                                       "taken": e["taken"] if e else 0}
        self._write("told", state=state, action=action, to=to, source=source, quote=quote, step=step)
        return self

    def human(self, state, action, to, note=None, step=None):
        """A person's correction: told(..., source="human")."""
        return self.told(state, action, to, source="human", quote=note, step=step)

    def arrive(self, state, action, to, step=None):
        """The action was taken in `state` and led to `to`: the claim is confirmed. → True when this refuted what was
        believed (another destination, whoever claimed it); the refutation is in the journal."""
        state, action, to = _name(state), _name(action, "an action"), _name(to)
        e = self.edges.setdefault((state, action), {"to": None, "status": "hypothesis", "source": "seen", "evidence": [],
                                                    "taken": 0})
        refuted = e["to"] is not None and e["to"] != to
        if refuted:
            self._write("refute", state=state, action=action, believed=e["to"], source=e["source"], observed=to, step=step)
        e.update(to=to, status="confirmed", source="observed", taken=e["taken"] + 1)
        e["evidence"].append([step, None])
        self._write("confirm", state=state, action=action, to=to, step=step)
        return refuted

    # --- what the map answers
    def claim(self, state, action):
        """The claim about an action in a state, or None."""
        return self.edges.get((state, action))

    def distances(self, targets, confirmed_only=False):
        """state → (cost to the nearest target, the first action to take): over confirmed claims (1 each) and, unless
        confirmed_only, unchecked claims with a destination (hypothesis_cost each). Targets map to (0, None)."""
        rev = {}
        for (s, a), e in sorted(self.edges.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            if e["to"] is None or (confirmed_only and e["status"] != "confirmed"):
                continue
            rev.setdefault(e["to"], []).append((s, a, 1 if e["status"] == "confirmed" else self.hypothesis_cost))
        dist = {t: (0, None) for t in targets}
        todo = deque(sorted(dist, key=str))
        while todo:
            t = todo.popleft()
            for s, a, c in rev.get(t, ()):
                d = dist[t][0] + c
                if s not in dist or d < dist[s][0]:
                    dist[s] = (d, a)
                    todo.append(s)
        return dist

    def next(self, state, targets, confirmed_only=False):
        """The action to take in `state` on the cheapest known way to a target; None when no way is known (or the
        state is a target)."""
        d = self.distances(set(targets), confirmed_only).get(state)
        return None if d is None else d[1]

    def path(self, state, targets, confirmed_only=False):
        """The actions of the cheapest known way from `state` to a target → [(state, action)], [] when none."""
        dist, out, seen = self.distances(set(targets), confirmed_only), [], set()
        while state in dist and dist[state][1] is not None and state not in seen:
            seen.add(state)
            a = dist[state][1]
            out.append((state, a))
            state = self.edges[(state, a)]["to"]
        return out

    def frontier(self):
        """What acting can still teach: the (state, action) pairs never taken — unknown destinations and claims nobody
        has checked."""
        return [k for k, e in self.edges.items() if e["taken"] == 0]

    def unvisited(self):
        """States some claim leads to that the agent has not been in."""
        return sorted({e["to"] for e in self.edges.values() if e["to"] is not None and e["to"] not in self.states}, key=str)

    def explore(self, state):
        """The action to take in `state` towards the nearest untaken action (an untaken action here first); None when
        the map has nothing left to check from here."""
        here = sorted((a for (s, a), e in self.edges.items() if s == state and e["taken"] == 0), key=str)
        if here:
            return here[0]
        return self.next(state, {s for s, _ in self.frontier()})

    def snapshot(self, state, targets=()):
        """The part of the map a decision needs, as a plain fact: the known way to a target, what is on offer here and
        what of it is unchecked. Give it to a decision as a given fact; the decision then replays."""
        d = self.distances(set(targets)).get(state) if targets else None
        here = {str(a): {"to": e["to"], "status": e["status"], "source": e["source"]}
                for (s, a), e in sorted(self.edges.items(), key=lambda kv: str(kv[0][1])) if s == state}
        return {"state": state, "next": None if d is None else d[1], "cost": None if d is None else d[0],
                "actions": here, "unchecked": sorted(a for a, e in here.items() if e["status"] != "confirmed"),
                "visits": self.states.get(state, {}).get("visits", 0), "head": self._prev}

    def stats(self):
        es = list(self.edges.values())
        return {"states": len(self.states), "claims": len(es), "confirmed": sum(e["status"] == "confirmed" for e in es),
                "unchecked": sum(e["taken"] == 0 for e in es), "refuted": sum(r["op"] == "refute" for r in self.journal),
                "journal": len(self.journal)}

    # --- keeping it
    def to_dict(self):
        """The map as JSON data. States that are all strings are written as an object (as before); otherwise as a list
        of {"state", "visits", "facts"}, since a JSON object's keys are strings only."""
        states = self.states if all(isinstance(k, str) for k in self.states) else \
            [{"state": k, **v} for k, v in self.states.items()]
        return {"format": "solvi.worldmap v1", "hypothesis_cost": self.hypothesis_cost, "visits_journaled": self._visits,
                "states": states,
                "edges": [{"state": s, "action": a, **e} for (s, a), e in self.edges.items()], "journal": self.journal}

    def save(self, path=None):
        p = Path(path) if path else self.file
        if p is None:
            raise ValueError("no path to save the map to")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return p

    def load(self, path):
        """Read a saved map. The journal is the stored truth: its hash chain is checked and the claims are rebuilt from
        it — the file's own `edges` are not trusted (an edge edited there changes nothing) — and so are the states of a
        map whose visits are journaled; a file written before that keeps its `states` as stored. ValueError when the
        journal's chain is broken or its entries do not replay."""
        from .core.runtime import vhash
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "solvi.worldmap v1":
            raise ValueError(f"{path} is not a solvi.worldmap v1 file")
        self.hypothesis_cost = int(data.get("hypothesis_cost", self.hypothesis_cost))
        self._visits = bool(data.get("visits_journaled", False))
        self.journal = data["journal"]
        prev = ""
        for r in self.journal:
            if r.get("prev") != prev or vhash({k: v for k, v in r.items() if k != "hash"}) != r.get("hash"):
                raise ValueError(f"{path}: the journal's hash chain is broken at entry {r.get('n')} (edited, removed or "
                                 "reordered)")
            prev = r["hash"]
        m = self.rebuild()
        self.edges = m.edges
        if self._visits:
            self.states = m.states
        else:
            st = data["states"]
            self.states = dict(st) if isinstance(st, dict) else \
                {_thaw(x["state"]): {"visits": x["visits"], "facts": x["facts"]} for x in st}
        self._prev = prev
        return self


__all__ = ["WorldMap"]
