"""An agent's memory as an input of its decisions: what was tried, what did not help, what worked before.

Part of the knowledge memory (solvi.core.knowledge): a finished episode is kept in a KnowledgeStore as an "episode"
item (`Episode.record(store, outcome)`) — a record and evidence, indexed by the decisions it stored; the store never
answers from episodes (an episodic answerer under a per-decision guarantee was measured and left out: on two intent
streams it never certified a threshold).

A decision in solvi depends on its recorded input and nothing else — that is what makes it replay. An agent that
takes many steps keeps state between them (what it tried, where it has been), and when that state lives in the
harness the decisions no longer replay, the model does not see what was already tried and offers it again, and every
agent writes its own loop detection. Here the memory is plain data:

    from solvi.core.knowledge.episodes import Episode, EpisodeView
    ep = Episode("ticket 4411")
    ep.note("act", "restart the router")              # an event (kind, key)
    ep.progress("the customer confirmed the fix")      # something moved: the counts "since progress" start again
    res = system.ask({"message": text, "episode": ep.snapshot()})     # the snapshot is a given fact: in the trace

    @cat.check(hard=True, then={"action": "handoff"})
    def not_going_in_circles(episode):                 # a catalog part reads it like any fact, and stays a pure function
        return not EpisodeView(episode).stalled(8)

`EpisodeView` (what catalog parts use) reads a snapshot: counts of events since the last progress and in total, the
facts board, the recent events, and the loop detectors — `repeated` (the same action again without progress),
`ping_pong` (A → B → A → B ...), `stalled` (many steps without progress), `revisits` (the same state again), and
`looping` (stalled and one of the first two: single detectors fire on honest repetition, the combination rarely does).
A detector is a signal: answer it softly — close that option for a while, escalate — not by stopping the agent.

`Chooser` is the step these pieces make together: a closed list of actions, a decider proposes one, a validator turns
down what is not in the list or was already done without progress (plus your own check), a rule answers when the
proposal is turned down or the model escalates. Everything a part reads is in the input, so `Chooser.replay()`
re-checks every stored decision.

`LongMemory` carries outcomes across episodes: what led to the goal in a context and what was a dead end, with decay,
as scores that are given to the decision as a fact (not a hidden bias).

What to expect: a model that sees only the last message proposes again what has already failed; with the episode in
its input and repeats turned down it stops doing that, and with the state in the input instead of the harness the
stored decisions replay. The memory makes a model-driven agent sound and auditable, it does not make the model better
than rules: where a hand-written script or a runbook exists, it can do as well or better. Where the bottleneck is skill
rather than memory, it changes nothing."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


def _key(k):
    return k if isinstance(k, str) else json.dumps(k, sort_keys=True, ensure_ascii=False, default=str)


class EpisodeView:
    """A snapshot of an episode, read-only: what catalog parts use (`EpisodeView(episode)` on the given fact)."""

    def __init__(self, snapshot):
        s = snapshot or {}
        self.n = s.get("n", 0)
        self.since_progress = s.get("since_progress", 0)
        self.facts = s.get("facts", {})
        self._since = s.get("counts_since_progress", {})
        self._total = s.get("counts_total", {})
        self.recent = s.get("recent", [])

    def count(self, kind, key=None, since="progress"):
        """How many events of this kind (with this key) since the last progress (since="total": in the whole episode)."""
        k = (self._since if since == "progress" else self._total).get(kind, {})
        return sum(k.values()) if key is None else k.get(_key(key), 0)

    def seen(self, kind, key, since="progress"):
        return self.count(kind, key, since) > 0

    def get(self, name, default=None):
        """A fact from the board (Episode.set)."""
        return self.facts.get(name, default)

    def last(self, kind=None, k=5):
        """The last k recent events [n, kind, key] (of one kind)."""
        return [e for e in self.recent if kind is None or e[1] == kind][-k:]

    # --- loop detectors
    def repeated(self, kind, key, limit=2, since="progress"):
        """This action was already taken `limit` times without progress."""
        return self.count(kind, key, since) >= limit

    def ping_pong(self, kind="place", round_trips=3):
        """A → B → A → ... for at least `round_trips` round trips in a row since the last progress → {A, B}, else an
        empty set. Several events in a row with the same key count as one stay."""
        seq = [e[2] for e in self.recent if e[1] == kind and e[0] > self.n - self.since_progress]
        stays = [p for i, p in enumerate(seq) if i == 0 or p != seq[i - 1]]
        if len(stays) < 2 * round_trips + 1:
            return set()
        tail = stays[-(2 * round_trips + 1):]
        a, b = tail[0], tail[1]
        return {a, b} if a != b and all(p == (a if i % 2 == 0 else b) for i, p in enumerate(tail)) else set()

    def stalled(self, steps):
        """No progress for at least `steps` events."""
        return self.since_progress >= steps

    def revisits(self, key, kind="state"):
        """How many times this state (`key`, an event of `kind`) was already met since the last progress. The key is
        required: count(kind) counts every event of a kind."""
        return self.count(kind, key)

    def looping(self, stalled=150, kind="place", round_trips=3, repeat=("act", None, 8)):
        """Stalled AND (a ping-pong OR an action repeated): the combination that told real loops from honest repetition
        (training, walking back to heal). repeat: (kind, key, limit) — key None: any action of that kind."""
        if not self.stalled(stalled):
            return False
        if self.ping_pong(kind, round_trips):
            return True
        rk, key, limit = repeat
        counts = self._since.get(rk, {})
        return (counts.get(_key(key), 0) if key is not None else max(counts.values(), default=0)) >= limit


class Episode(EpisodeView):
    """The memory of one episode: events with their numbers, counts since the last progress and in total, a board of
    facts. `snapshot()` is what a decision is given; `keep`: the recent events a snapshot carries."""

    def __init__(self, name="episode", keep=40):
        self.name, self.keep = name, int(keep)
        self.n = 0
        self.progress_at = 0
        self.facts = {}
        self.events = []                              # (n, kind, key)
        self._since = collections.defaultdict(collections.Counter)
        self._total = collections.defaultdict(collections.Counter)
        self.progress_log = []                        # [n, reason]

    @property
    def since_progress(self):
        return self.n - self.progress_at

    @property
    def recent(self):
        return [list(e[:3]) for e in self.events[-self.keep:]]

    def note(self, kind, key):
        """Record an event: what was done, where the agent is, what was seen (its key; a value worth deciding on is a
        fact: set(name, value))."""
        self.n += 1
        k = _key(key)
        self.events.append((self.n, kind, k))
        self._since[kind][k] += 1
        self._total[kind][k] += 1
        return self

    def set(self, name, value):
        """Put a fact on the board (overwrites)."""
        self.facts[name] = value
        return self

    def progress(self, reason=""):
        """Something moved: the counts since the last progress start again (dead ends before it no longer count). Say
        what counts as progress explicitly — a sub-goal reached, an event flag — not "anything changed": a wrong action
        changes the page too, and then it erases the memory of itself."""
        self.progress_at = self.n
        self._since = collections.defaultdict(collections.Counter)
        self.progress_log.append([self.n, reason])
        return self

    def snapshot(self):
        """The episode as a decision's input: plain JSON data, the same for the same history."""
        return {"name": self.name, "n": self.n, "since_progress": self.since_progress,
                "facts": json.loads(json.dumps(self.facts, sort_keys=True, default=str)),
                "counts_since_progress": {k: dict(sorted(v.items())) for k, v in sorted(self._since.items())},
                "counts_total": {k: dict(sorted(v.items())) for k, v in sorted(self._total.items())},
                "recent": self.recent}

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()[:16]

    def record(self, store, outcome, *, stored_ids=(), by="environment", source="outcome", scope=None):
        """Keep the finished episode as an "episode" item of a KnowledgeStore: its name, its outcome (what really
        happened — the source is "outcome" unless a person judged it), the stored decisions it made and the digest of
        its last snapshot. A record, raw material and evidence; the store never answers from it. → the item id."""
        return store.add("episode", {"name": self.name, "outcome": outcome, "stored_ids": [str(i) for i in stored_ids],
                                     "steps": self.n, "digest": self.digest()},
                         scope, source=source, by=by, evidence=[str(i) for i in stored_ids] or None)


class Chooser:
    """One step of an agent as a solvi decision: choose an action from a closed list, with the episode in the input.

        chooser = Chooser(model, storage=store)
        action, who, info = chooser.choose("next step", "What should support do next?", {"restart": "restart the router",
                                           "replace": "send a new router"}, context=dialogue, rule="restart", episode=ep)

    options: {option: action} (the option is what the model reads, the action what the episode counts). The producers
    of the choice: the model (rejected when its option is not in the list, when its action was already taken
    `repeat_limit` times without progress, when `check(option, question, episode)` says no, or when it escalates), then
    `rule` — the option a rule would take (one of the options: another value raises ValueError). who: "model" | "rule" |
    "only" (one option: no decision) | "abstain" (the model's option was rejected and there is no rule: the action is
    None — the caller decides, e.g. asks a person); info: the option and the producers tried. A single option is
    returned without a decision."""

    def __init__(self, model, storage=None, min_confidence=0.5, check=None, repeat_limit=1, use_act=None):
        self.model, self.storage, self.use_act = model, storage, use_act
        self.min_confidence, self.repeat_limit, self.check = min_confidence, int(repeat_limit), check
        self.catalog = self._catalog()
        self.asked = 0

    def _catalog(self):
        from ..catalog import Catalog
        cat, m, me = Catalog(), self.model, self

        def allowed(v, question, episode):
            if v not in question["actions"]:
                return False
            if EpisodeView(episode).repeated("act", question["actions"][v], limit=me.repeat_limit):
                return False                          # done before, and nothing moved
            return bool(me.check(v, question, episode)) if me.check else True

        @cat.fn(provides="pick", model=m, cost=100, validate=allowed)
        def pick_by_model(question, episode):
            """The model chooses among the options given in the input."""
            part = m.decision(question["name"], question["task"], "state", list(question["actions"]),
                              min_confidence=me.min_confidence, use_act=me.use_act)
            return part(state=question["context"])

        @cat.fn(provides="pick", cost=1000)
        def pick_by_rule(question):
            """The option the rule would take."""
            return question["rule"]

        @cat.rule("choice")
        def choice(pick):
            return pick

        return cat

    def system(self, options):
        from ..catalog import Answer, Question
        from ..system import System
        return System(self.catalog, [Question("choice", "Choose", Answer.choice(list(options)))], storage=self.storage)

    def choose(self, name, task, options, context="", rule=None, episode=None):
        actions = dict(options) if isinstance(options, dict) else {o: o for o in options}
        if not actions:
            raise ValueError("nothing to choose from")
        if rule is not None and rule not in actions:  # it would be chosen as an action of None
            raise ValueError(f"rule {rule!r} is not one of the options ({', '.join(map(repr, actions))})")
        if len(actions) == 1:
            v = next(iter(actions))
            return actions[v], "only", {"choice": v}
        q = {"name": name, "task": task, "actions": actions, "rule": rule, "context": context}
        snap = episode.snapshot() if isinstance(episode, Episode) else (episode or {})
        res = self.system(actions).ask({"question": q, "episode": snap})
        self.asked += 1
        r = res["choice"]
        pick = next((x for x in res.trace.records if x.name == "pick"), None)
        who = {"pick_by_model": "model", "pick_by_rule": "rule"}.get(getattr(pick, "producer", None) or "", "abstain")
        v = r.answer if r.status != "abstain" else rule
        return actions.get(v), who, {"choice": v, "tried": getattr(pick, "tried", None), "response": res}

    def replay(self):
        """Re-check every decision this chooser stored → (replayed, total, the first mismatches)."""
        if self.storage is None:
            return None
        bad, n = [], 0
        for st in self.storage.iter("ask"):
            init = (st.data.get("response") or {}).get("trace", {}).get("init") or {}
            if "question" not in init or "actions" not in (init.get("question") or {}):
                continue                              # another system's record in the same store
            n += 1
            r = st.response()
            rep = r.trace.replay(self.system(init["question"]["actions"]), r.flow)
            if not rep["ok"]:
                bad.append(rep["mismatches"][:2])
        return n - len(bad), n, bad[:3]


class LongMemory:
    """Outcomes across episodes: in this context, what led to the goal (+) and what was a dead end (−).

        lm = LongMemory("memory.json", decay=0.8)
        lm.begin("ticket 4411")                                   # a new episode: older scores decay
        lm.record(("customer", "c17"), "restart the router", +1, why="solved")
        lm.scores(("customer", "c17"))                            # {"restart the router": 1.0} → a fact of the decision
        lm.save()

    A score is the sum of the outcomes recorded for (context, key), each decayed once per episode since, within ±cap.
    Labels come from outcomes — progress, a dead end — never from what the model answered. Every item keeps the episode
    it was last recorded in and why. A key that is not a string (a tuple, a dict: ("tool", "ping")) is kept as its JSON
    text, like an Episode event's key, and `scores` gives it back in that form. The scores are given to a decision as a fact, so the trace shows what the memory
    said; nothing is applied behind the decision's back."""

    def __init__(self, path=None, decay=0.8, cap=5.0):
        self.path = Path(path) if path else None
        self.decay, self.cap = float(decay), float(cap)
        self.data = {"episodes": 0, "items": {}}      # context → key → {"score", "n", "last", "why"}
        self.episode = None
        if self.path and self.path.exists():
            self.data = self._read(self.path)

    @staticmethod
    def _read(path):
        """The memory file at path, checked: a JSON file of another kind is refused here (ValueError), not later as a
        KeyError deep in `scores` or `begin`."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"{path} is not a LongMemory file: it is not JSON ({e})") from None
        why = None
        if not isinstance(data, dict) or not isinstance(data.get("items"), dict) \
                or type(data.get("episodes")) is not int:
            why = 'it has no "episodes" count and "items" table'
        else:
            for ctx, keys in data["items"].items():
                bad = next((k for k, v in keys.items() if not isinstance(v, dict) or type(v.get("n")) is not int
                            or type(v.get("score")) not in (int, float)), None) if isinstance(keys, dict) else ctx
                if bad is not None:
                    why = f"item {bad!r} of context {ctx!r} has no numeric score and count"
                    break
        if why:
            raise ValueError(f"{path} is not a LongMemory file: {why} (written by LongMemory.save: "
                             '{"episodes": n, "items": {context: {key: {"score", "n", "last", "why"}}}})')
        return data

    @staticmethod
    def _ctx(c):
        return c if isinstance(c, str) else "|".join(map(str, c))

    def begin(self, name):
        """A new episode: every score decays."""
        self.data["episodes"] += 1
        self.episode = f"{self.data['episodes']}:{name}"
        for keys in self.data["items"].values():
            for v in keys.values():
                v["score"] = round(v["score"] * self.decay, 4)
        return self

    def record(self, context, key, outcome, why=""):
        it = self.data["items"].setdefault(self._ctx(context), {}).setdefault(_key(key), {"score": 0.0, "n": 0})
        it["score"] = round(max(-self.cap, min(self.cap, it["score"] + outcome)), 4)
        it["n"] += 1
        it["last"], it["why"] = self.episode, why
        return self

    def scores(self, context):
        """{key: score} of the context, without zeros, keys sorted — what a decision is given."""
        return {k: v["score"] for k, v in sorted(self.data["items"].get(self._ctx(context), {}).items()) if v["score"]}

    def forget(self, context=None):
        """Remove a context's items (None: everything) → how many were removed."""
        if context is None:
            n = sum(len(v) for v in self.data["items"].values())
            self.data["items"] = {}
            return n
        return len(self.data["items"].pop(self._ctx(context), {}))

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        return self.path

    def stats(self):
        items = [v for keys in self.data["items"].values() for v in keys.values()]
        return {"episodes": self.data["episodes"], "contexts": len(self.data["items"]), "items": len(items),
                "positive": sum(v["score"] > 0 for v in items), "negative": sum(v["score"] < 0 for v in items)}


__all__ = ["Chooser", "Episode", "EpisodeView", "LongMemory"]
