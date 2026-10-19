"""A multi-law (factored) transition model (experimental; use it for accuracy, not to learn from fewer transitions):
several law sets that predict one step together, each over its own channel, with one joint error budget.

    from solvi.experimental.laws.factored import Channel, Row, FactoredModel

    model = FactoredModel([Channel("self", g_self, base=lambda s, a: "stay"),
                           Channel("enter", g_enter, base=lambda s, a: "pass"),
                           Channel("gone", g_gone, base=lambda s, a: "stays")], extract=my_rows, delta=0.1)
    model.new_group(world_id)                          # a world, a level, a session: what must not judge its own laws
    st = model.step(pre, action, post)                 # my_rows(model, pre, action, post) -> [Row, ...]; each learned
    model.predict("enter", occupant_features)          # the composer (the caller's turn skeleton) reads the channels
    model.report()["bound"]                            # the joint promise: the sum of the per-channel deltas

What is the caller's. The channels (names, grammars, base predictors, effect encodings `apply` / `effects`), the atoms,
the row extraction (which situations of one transition are clean enough to be a row of which channel, and what the
row is labelled with — the credit assignment), the provenance of each row (`Row.depends`: the laws of other channels
whose predictions the extraction or the label relied on), the composer that turns channel answers into a next state,
the probe pool and `ask` for admissions, and the group boundaries (`new_group`). Nothing here knows a grid or a turn
order.

What the model does. One `LawSet` per channel (the channel name is its action slot) with one held-out `Buffer`
(every group; a candidate is never tested on the group it was proposed from) and one local `Buffer` of the current
group (the logged rows `admit` judges with), dedup within a group, the joint δ, provenance voiding, a `version` for a
planner's caches, a report, knowledge items and compiled modules per channel.

The guarantee.
  Per channel c: as `LawSet` with sized acceptance at delta = δ_c — P[the held-out test accepts, over every candidate
  ever tested on c's buffer, a law no better than its null] ≤ δ_c, assuming the held-out rows are independent draws,
  the candidates were not chosen by looking at them, and the rows' labels are right.
  Jointly: P[any such false law in any channel] ≤ Σ_c δ_c (the union bound; no independence between channels is
  needed, the channels may share transitions). share="equal" (the default) sets δ_c = δ / C, so the model promises
  `delta`; a dict {channel: δ_c} shares it unevenly, and `report()["bound"]` prints the sum (warned above 0.5).
  Not covered: laws admitted by probes (`admit`: identifiability, no δ — counted per channel in the report), the
  `min_precision` filter (a law wrong on up to 1 − min_precision of its rows passes by design), and labels that depend
  on other channels' laws. For the last the model gives provenance, not a bound: when a law named in a row's
  `depends` leaves the active set (retired, suspended, withdrawn), that row is voided in every buffer of its channel
  (`Buffer.void`): it leaves every test and its count leaves the running precision of the law it was counted against;
  N never decreases. A law accepted on voided rows is flagged `stale` in the report (no automatic re-test; it falls by
  its running precision as before). While the named law stands, a channel's bound is conditional on it.

The action slot (action=, opt-in). A channel's name is its LawSet's action slot, so the caller's own action is an
atom of the channel's grammar like any other, and a channel law may leave it out — generalise over actions the data
never varied (measured on MiniGrid: `front: key fits -> door open`, right for toggle, wrong for every other action).
FactoredModel(action="act") names the feature of the action atoms: in every channel every candidate law must name an
action value (an atom of that feature), unless the channel is declared Channel(..., action_free=True) (a channel whose
effect does not depend on the action). A channel without an atom of that feature is refused at construction. The
candidates dropped are counted (`report()[c]["action_dropped"]`); fewer candidates also make N smaller.

Dedup (dedup="group", the default): a row whose key — `Row.key`, default (channel, state, observed) — was already
logged in the current group is dropped (no new draw); the same row in another group counts again. dedup=False logs
every row. Rows of one transition are not independent draws even within a channel: the report prints rows per
transition per channel so a caller sees a channel that logs many rows per step.

Measured on the push-world toy (benchmarks/laws_factored_toy.py: 1.000 next-state accuracy, 0 false laws, 5 seeds), a
null simulation, and MiniGrid (3 environments, 5 fresh seeds: 0.999 against 0.984 for one joint LawSet, 0.94× and
1.05× its transitions against a bar of 0.6×, false laws on states the environments never reach in both); see the tests
(tests/test_laws_factored.py)."""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from ._core import Admission, Buffer, Grammar, Law, LawSet, Transition, Unknown, _key

__all__ = ["Channel", "Extractor", "FactoredModel", "FactoredStep", "Row", "Unknown"]


@dataclass(frozen=True)
class Row:
    """One row of one channel, made by the caller's extraction. state: what the channel's atoms (and its `apply`)
    read; observed: what was observed for it; depends: the laws of other channels whose predictions this row's
    extraction or label relied on — `Law` objects or (channel, law id) pairs; key: the dedup key (default (channel,
    state, observed))."""
    channel: str
    state: Any
    observed: Any
    depends: tuple = ()
    key: Any = None

    def dedup_key(self):
        return self.key if self.key is not None else (self.channel, _key(self.state), _key(self.observed))


class Extractor(Protocol):
    """The caller's credit assignment: `rows(model, pre, action, post)` → the rows of one transition (or a plain
    function of the same arguments)."""
    def rows(self, model: "FactoredModel", pre, action, post) -> Iterable[Row]: ...


class Channel:
    """One channel: its name (the LawSet's action slot), grammar, base predictor `base(state, name)` and the LawSet
    options (apply, effects, proposer, min_support, min_precision, null, rivals, ...). delta is set by the model."""

    def __init__(self, name: str, grammar: Grammar, base: Callable | None = None, *, apply: Callable | None = None,
                 effects: Callable | None = None, proposer=None, action_free: bool = False, **lawset_kwargs):
        if "delta" in lawset_kwargs:
            raise ValueError("a channel's delta is set by FactoredModel(delta=, share=), not on the Channel")
        if "sized" in lawset_kwargs or "acceptance" in lawset_kwargs:
            raise ValueError("FactoredModel's joint bound needs the sized acceptance; sized= / acceptance= are not "
                             "options of a Channel")
        self.name, self.grammar, self.base = str(name), grammar, base
        self.action_free = bool(action_free)
        self.kwargs = dict(lawset_kwargs, apply=apply, effects=effects, proposer=proposer)


class _ActionBound:
    """The per-channel action slot: a proposer that keeps only candidates whose condition names a value of the
    caller's action feature (one atom of that feature) — a channel law never generalises over actions."""

    def __init__(self, inner, grammar: Grammar, feature: str):
        self.inner, self.grammar, self.feature = inner, grammar, feature
        self.name = getattr(inner, "name", "enumerate")
        self.dropped = 0

    def propose(self, lawset, state, action, observed):
        out = []
        for c in self.inner.propose(lawset, state, action, observed):
            if any(self.grammar.by_name[n].feature == self.feature for n in c.condition if n in self.grammar.by_name):
                out.append(c)
            else:
                self.dropped += 1
        return out


@dataclass
class FactoredStep:
    """What one transition (or one batch of rows) did, per channel: steps (the LawSet `Step` of each logged row),
    logged ((row, transition id) pairs), dropped (dedup), refused (rows whose `depends` named a law no longer active:
    not logged), unsettled (rows whose mismatch the test did not settle — what the caller may hand to `admit`), and
    voided (rows voided by the falls this step caused)."""
    steps: dict = field(default_factory=dict)
    logged: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    unsettled: list = field(default_factory=list)
    voided: int = 0

    @property
    def accepted(self) -> list:
        return [(ch, law) for ch, sts in self.steps.items() for st in sts for law in st.accepted]

    @property
    def retired(self) -> list:
        return [(ch, law) for ch, sts in self.steps.items() for st in sts for law in st.retired]

    @property
    def mismatches(self) -> int:
        return sum(st.mismatch for sts in self.steps.values() for st in sts)


class FactoredModel:
    """Several law sets predicting one step together (see the module docstring for the guarantee).

    channels: the `Channel`s (distinct names); extract: the caller's `Extractor` (optional: `log(rows)` takes rows
    directly); delta: the joint budget; share: "equal" (δ_c = δ / C) or {channel: δ_c} (the sum is the promise);
    dedup: "group" or False.

    laws[c] → LawSet, buffers[c] → the held-out Buffer, local[c] → the current group's Buffer."""

    def __init__(self, channels: Iterable[Channel], extract: Extractor | Callable | None = None, *,
                 delta: float = 0.1, share: str | dict = "equal", dedup: str | bool = "group",
                 action: str | None = None):
        chans = list(channels)
        if not chans:
            raise ValueError("a FactoredModel needs at least one channel")
        names = [c.name for c in chans]
        if len(set(names)) != len(names):
            raise ValueError(f"channel names must be distinct: {names}")
        if share == "equal":
            if not 0.0 < float(delta) < 1.0:
                raise ValueError("delta is a probability strictly between 0 and 1")
            deltas = {n: float(delta) / len(names) for n in names}
        elif isinstance(share, dict):
            if set(share) != set(names):
                raise ValueError(f"share names every channel exactly once: {sorted(share)} vs {sorted(names)}")
            deltas = {n: float(share[n]) for n in names}
            if not all(0.0 < d < 1.0 for d in deltas.values()):
                raise ValueError("each channel's delta is strictly between 0 and 1")
        else:
            raise ValueError(f"share is 'equal' or a dict {{channel: delta}}, not {share!r}")
        if dedup not in ("group", False):
            raise ValueError(f"dedup is 'group' or False, not {dedup!r}")
        self.deltas = deltas
        self.bound = sum(deltas.values())
        if self.bound > 0.5:
            warnings.warn(f"FactoredModel: the per-channel deltas sum to {self.bound:.3g} > 0.5 — the joint promise "
                          f"(the union bound) is that loose", UserWarning, stacklevel=2)
        self.names = names
        self.channels = {c.name: c for c in chans}
        self.extract = extract
        self.dedup = dedup
        self.laws: dict[str, LawSet] = {}
        self.buffers: dict[str, Buffer] = {}
        self.local: dict[str, Buffer] = {}
        self.action = action
        self._bound: dict = {}
        for c in chans:
            kw = {k: v for k, v in c.kwargs.items() if v is not None or k not in ("apply", "effects", "proposer")}
            if action is not None and not c.action_free:
                if not any(a.feature == action for a in c.grammar.atoms):
                    raise ValueError(f"channel {c.name!r}: its grammar has no atom of the action feature {action!r} "
                                     f"(add the action atoms, or declare the channel action_free=True)")
                from ._core import Enumerate
                kw["proposer"] = self._bound[c.name] = _ActionBound(kw.get("proposer") or Enumerate(), c.grammar,
                                                                    action)
            self.laws[c.name] = LawSet(c.grammar, base=c.base, delta=deltas[c.name], sized=True, **kw)
            self.buffers[c.name] = Buffer(c.grammar)
            self.local[c.name] = Buffer(c.grammar)
        self.group = None
        self._seq = 0
        self._keys: set = set()                         # dedup keys logged in the current group
        self._tried: set = set()                        # (channel, key, group) admissions attempted
        self._deps: dict = {}                           # (channel, law id) → [(row channel, transition id)]
        self._rowinfo: dict = {}                        # (channel, tid) → (dedup key, group)
        self._counted: dict = {}                        # (channel, tid) → (law id, hit) counted by `observe`
        self._void_ids: dict = {n: set() for n in names}
        self.counts = {n: Counter() for n in names}     # rows, dropped, refused, transitions, probe_admitted, ...
        self.transitions = 0
        self._memo: dict = {}
        self._memo_at = None

    # ------------------------------------------------------------------------------- state
    @property
    def version(self) -> int:
        """Goes up whenever any channel's laws change (accept, retire, suspend, reactivate, probe admission): the sum
        of the channels' `LawSet.version`s. Unchanged by predictions and dry runs; kept across pickling."""
        return sum(ls.version for ls in self.laws.values())

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_memo"], st["_memo_at"] = {}, None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)

    def new_group(self, group) -> None:
        """Start a group (a world, a level, a session): fresh local buffers and dedup set."""
        self.group = group
        self.local = {n: Buffer(self.channels[n].grammar) for n in self.names}
        self._keys = set()

    def _check(self, channel):
        if channel not in self.laws:
            raise KeyError(f"no channel {channel!r} (channels: {self.names})")

    # ------------------------------------------------------------------------------- prediction
    def applied(self, channel: str, state) -> Law | None:
        """The law of `channel` that speaks in `state` (None: the base) — what a caller names in `Row.depends`."""
        self._check(channel)
        return self.laws[channel].applied(state, channel)

    def predict(self, channel: str, state):
        """The channel's prediction for `state` (its most specific law, else its base); memoised per `version`
        when the state is hashable."""
        self._check(channel)
        v = self.version
        if v != self._memo_at:
            self._memo, self._memo_at = {}, v
        try:
            k = (channel, state)
            hash(k)
        except TypeError:
            return self.laws[channel].predict(state, channel, disagree="first")
        if k not in self._memo:
            self._memo[k] = self.laws[channel].predict(state, channel, disagree="first")
        return self._memo[k]

    # ------------------------------------------------------------------------------- learning
    def _dep_keys(self, depends) -> list:
        out = []
        for d in depends:
            if isinstance(d, Law):
                ch = next((n for n, ls in self.laws.items() if any(x is d for x in ls.all() + ls.suspended)), None)
                if ch is None:
                    raise ValueError(f"depends names {d.id}, which is in no channel of this model")
                out.append((ch, d.id))
            else:
                ch, lid = d
                self._check(ch)
                out.append((str(ch), str(lid)))
        return out

    def _is_active(self, ch, lid) -> bool:
        return any(x.id == lid for x in self.laws[ch].laws)

    def step(self, pre, action, post, *, group=None) -> FactoredStep:
        """Extract the transition's rows with the caller's extractor (from the model as it is before this transition)
        and `log` them."""
        if self.extract is None:
            raise ValueError("no extract= was given: pass rows to log() instead")
        fn = getattr(self.extract, "rows", None) or self.extract
        rows = list(fn(self, pre, action, post))
        self.transitions += 1
        return self.log(rows, group=group, _transition=True)

    def log(self, rows: Iterable[Row], *, group=None, _transition: bool = False) -> FactoredStep:
        """Learn from rows, in order: dedup → for each row its channel's `LawSet.step` (predicted before it is learned
        from; a mismatch tested on the channel's buffer without this group) → remembered in the held-out and the local
        buffer. Laws that leave the active set void the rows that depend on them."""
        if group is not None and group != self.group:
            self.new_group(group)
        if self.group is None:
            raise ValueError("call new_group(group) first (or pass group=): a law must not be judged on its own group")
        g = self.group
        out = FactoredStep()
        seen = set()
        for row in rows:
            ch = row.channel
            self._check(ch)
            cnt = self.counts[ch]
            deps = self._dep_keys(row.depends)
            if any(not self._is_active(c, lid) for c, lid in deps):
                cnt["refused"] += 1
                out.refused[ch] = out.refused.get(ch, 0) + 1
                continue
            key = (ch, _key(row.dedup_key()))
            if self.dedup == "group":
                if key in self._keys:
                    cnt["dropped"] += 1
                    out.dropped[ch] = out.dropped.get(ch, 0) + 1
                    continue
                self._keys.add(key)
            ls = self.laws[ch]
            self._seq += 1
            t = Transition(row.state, ch, row.observed, group=g, id=f"row-{self._seq}")
            law = ls.applied(row.state, ch)
            if law is not None and not (law.born is not None and _key(g) == _key(law.born)):
                self._counted[(ch, t.id)] = (law.id, int(ls.agrees(ls.apply(law.effect, row.state), row.observed)))
            st = ls.step(t, self.buffers[ch], remember=True)
            self.local[ch].add(t)
            self._rowinfo[(ch, t.id)] = (key, g)
            for d in deps:
                self._deps.setdefault(d, []).append((ch, t.id))
            cnt["rows"] += 1
            seen.add(ch)
            out.steps.setdefault(ch, []).append(st)
            out.logged.setdefault(ch, []).append((row, t.id))
            if st.mismatch and not st.accepted:
                out.unsettled.append(row)
            out.voided += self._sync()
        if _transition:
            for ch in seen:
                self.counts[ch]["transitions"] += 1
        return out

    def _sync(self) -> int:
        """Void the dependents of every law named in a row's `depends` that is no longer active in its channel."""
        n = 0
        for ch, lid in list(self._deps):
            if not self._is_active(ch, lid):
                n += self._void_dependents(ch, lid)
        return n

    def void_dependents(self, channel: str, law: Law | str) -> int:
        """Void every row whose `depends` named this law (normally automatic when it leaves the active set) → the
        rows newly voided."""
        self._check(channel)
        return self._void_dependents(channel, law.id if isinstance(law, Law) else str(law))

    def _void_dependents(self, ch, lid) -> int:
        n = 0
        for rch, tid in self._deps.pop((ch, lid), []):
            if tid in self._void_ids[rch]:
                continue
            self._void_ids[rch].add(tid)
            self.buffers[rch].void([tid])
            self.local[rch].void([tid])
            n += 1
            self.counts[rch]["voided"] += 1
            got = self._counted.pop((rch, tid), None)
            if got is not None:                          # the row leaves the running precision it was counted in
                ls = self.laws[rch]
                law = next((x for x in ls.laws + ls.suspended if x.id == got[0]), None)
                if law is not None and law.n > 0:
                    law.n -= 1
                    law.hits_after = max(0, law.hits_after - got[1])
            info = self._rowinfo.get((rch, tid))
            if info is not None and _key(info[1]) == _key(self.group):
                self._keys.discard(info[0])              # the situation may be logged again, re-labelled
        return n

    def retire(self, channel: str, law: Law, *, why: str | None = None) -> int:
        """Retire a law of `channel` by hand → the rows voided because they depended on it."""
        self._check(channel)
        self.laws[channel].retire(law, why=why)
        return self._sync()

    # ------------------------------------------------------------------------------- admission by probes
    def admit(self, channel: str, row: Row, ask: Callable[[Any], Any], probes: Iterable[Any], *, budget: int = 8,
              **kw) -> Admission | None:
        """Admit a law for a mismatched row by asking the world (`LawSet.admit_by_probes`) with the library's guards:
        the current group's logged rows refute candidates (buffer=local[channel]), siblings="undetermined" (the pool
        cannot tell two conditions of one length apart → refuse), coverage="varied" over the channel's held-out buffer
        (an atom that varied elsewhere but not on the pool → refuse). ask(state) → the observation, or `Unknown`;
        probes: channel states that may be asked. One attempt per (channel, row key, group); None when already tried, when the row is no mismatch, or
        when nothing is proposed. Probe-admitted laws are outside δ (counted in the report). No held-out check: rows of
        other groups that contradict the admitted law do not refuse it here (that check is not shipped; ROADMAP, "Laws:
        `admit_by_probes(..., held_out=buffer)`"); the held-out test can retire it later. kw overrides admit_by_probes
        options."""
        self._check(channel)
        if row.channel != channel:
            raise ValueError(f"row of channel {row.channel!r} given to admit({channel!r}, ...)")
        ls = self.laws[channel]
        tk = (channel, _key(row.dedup_key()), _key(self.group))
        if tk in self._tried or ls.agrees(ls.predict(row.state, channel, disagree="first"), row.observed):
            return None
        self._tried.add(tk)
        cands = ls.mismatch(row.state, channel, row.observed)
        if not cands:
            return None
        opts = dict(budget=budget, buffer=self.local[channel], siblings="undetermined", coverage="varied",
                    coverage_on=[self.buffers[channel]])
        opts.update(kw)
        pool = [(s, channel) for s in probes]

        adm = ls.admit_by_probes(cands, lambda s, a: ask(s), probes=pool, **opts)
        if adm.status == "accepted":
            self.counts[channel]["probe_admitted"] += len(adm.laws or [adm.law])
        self._sync()
        return adm

    # ------------------------------------------------------------------------------- report and export
    def stale(self, channel: str) -> dict:
        """Active laws of `channel` accepted on rows since voided → {law id: how many of its source rows}."""
        v = self._void_ids[channel]
        out = {}
        for law in self.laws[channel].laws:
            k = sum(1 for s in law.sources if s in v)
            if k:
                out[law.id] = k
        return out

    def report(self) -> dict:
        """{channel: {laws, retired, suspended, tested (N), delta, null and need (the least support a law right on
        every row needs at this N, against the channel's least base null), rows, transitions, rows_per_transition,
        voided, dropped (dedup), refused (rows naming a fallen law), probe_admitted (outside δ), stale}, "bound": Σ δ_c,
        "probe_admitted": the total outside the bound}."""
        out = {}
        for ch in self.names:
            ls, buf, c = self.laws[ch], self.buffers[ch], self.counts[ch]
            act = ls.report(buf)["actions"].get(ch, {})
            tr = c["transitions"]
            out[ch] = {"laws": len(ls.laws), "retired": len(ls.retired), "suspended": len(ls.suspended),
                       "tested": buf.tested, "delta": self.deltas[ch], "null": act.get("null"), "need": act.get("need"),
                       "rows": c["rows"], "transitions": tr, "rows_per_transition": (c["rows"] / tr) if tr else None,
                       "voided": c["voided"], "dropped": c["dropped"], "refused": c["refused"],
                       "probe_admitted": c["probe_admitted"], "stale": self.stale(ch),
                       "action": ("free" if self.channels[ch].action_free else self.action) if self.action else None,
                       "action_dropped": self._bound[ch].dropped if ch in self._bound else 0}
        out["bound"] = self.bound
        out["probe_admitted"] = sum(self.counts[ch]["probe_admitted"] for ch in self.names)
        return out

    def all_laws(self) -> list[tuple[str, Law]]:
        """(channel, law) for every law ever accepted, channel by channel."""
        return [(ch, law) for ch in self.names for law in self.laws[ch].all()]

    def to_knowledge(self, store, scope: dict | None = None) -> dict:
        """Each channel's laws into a KnowledgeStore (`LawSet.to_knowledge`), the scope carrying "channel" →
        {channel: {law id: item id}}."""
        return {ch: self.laws[ch].to_knowledge(store, {**(scope or {}), "channel": ch}) for ch in self.names}

    def to_module(self) -> dict:
        """{channel: the module source of its active laws} (`LawSet.to_module`; check each with
        `check_compiled(source, model.buffers[channel], model.laws[channel])`)."""
        return {ch: self.laws[ch].to_module() for ch in self.names}
