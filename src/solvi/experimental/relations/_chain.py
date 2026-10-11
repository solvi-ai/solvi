"""Forward chaining to a fixed point over composition rules kept in a KnowledgeStore.

    from solvi.experimental.relations import CompositionTable, forward_chain, explain, pair_fact
    table.to_knowledge(store, {"domain": "kinship"})                    # the rules: one "rule" item per cell
    store.add("fact", pair_fact("ann", "bob", "father"), {"story": 1}, source="outcome")
    rep = forward_chain(store, scope={"story": 1}, rule_scope={"domain": "kinship"})
    rep.rounds, rep.facts_added, rep.disputes_opened                     # the fixed point and what it wrote
    explain(store, fact_id).cells                                        # the rules the derivation used

Facts are pairs: `pair_fact(x, y, relation)` = {"s": [x, y], "r": "relation", "o": relation} — the store's key is the
ordered pair, its value the relation, so one pair holds one relation at a time and two different derived relations of
one pair meet on one key. Rules are the composition cells `CompositionTable.to_knowledge` writes: {"kind":
"composition", "s": [r1, r2], "r": "compose", "o": r}.

Chaining. Semi-naive: each round joins the facts new in the previous round with every fact of the same scope; for
(x r1 y), (y r2 z) and a rule r1 ∘ r2 → r the fact (x r z) is proposed through `store.add` with derived_from = the two
facts and the rule, `inferred=True`, the source of the weaker premise (a derivation never outranks what it rests on),
evidence {"rule", "via": y, "depth": the round}. The store's key semantics do the rest: the same relation confirms (one
more justification); a different relation from an equal-rank derivation makes the pair "disputed" — the table's
«bracketings disagree → refuse» as a question for a person; a premise retracted or refuted takes what rests on it
along (JTMS); a fact derived through a disputed fact or a hypothesis rule is a hypothesis, made a fact by the store
once the dispute is answered or the rule promoted, without a re-run. Facts read: every present fact of the scope
(active, hypothesis, disputed), so both sides of a dispute carry their consequences — as with the table's pooled
bracketings — and a person's answer settles the consequences too. A justification that would rest on the fact it
justifies (a cycle in the story graph) is not written (`circular`): support stays well-founded, so a retraction is
exact. Terminates: no rule makes a new entity, and a pair holds at most |relations| items.

`explain(store, fact_id)` reads the derivation back: every justification of the fact, recursively down to the
original facts, and the rule cells used — what `CompositionTable.derive(chain).cells` gives for a chain."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

RELATION = "relation"                       # the "r" of a pair fact
FACT_STATUSES = ("active", "hypothesis", "disputed")
RULE_STATUSES = {"active": ("active",), "any": ("active", "hypothesis", "disputed")}


def pair_fact(x, y, relation, multi=False, family=None) -> dict:
    """The body of the fact «x is `relation` of … y» as forward_chain reads it: {"s": [x, y], "r": "relation",
    "o": relation} (the key is the ordered pair; x and y are strings or integers). multi=True: a pair may carry
    several relations — {"s": [x, y], "r": "relation/<family>", "o": relation}, one key per pair and family (family
    defaults to the relation itself; relations given one family, e.g. a relation and its complement, share a key and
    dispute)."""
    if multi:
        return {"s": [x, y], "r": f"{RELATION}/{family if family is not None else relation}", "o": relation}
    if family is not None:
        raise ValueError("family= is for multi=True pair facts")
    return {"s": [x, y], "r": RELATION, "o": relation}


def _pair(body):
    if isinstance(body, dict) and isinstance(body.get("r"), str) and isinstance(body.get("s"), list) \
            and len(body["s"]) == 2 and "o" in body \
            and (body["r"] == RELATION or body["r"].startswith(RELATION + "/")):
        return body["s"][0], body["s"][1], body["o"]
    return None


class ChainLimit(RuntimeError):
    """forward_chain proposed more new facts in one scope than `max_facts` (default |entities|² × |relations|, which
    no correct run exceeds)."""


@dataclass
class ChainReport:
    """What forward_chain did. rounds: rounds that proposed something (the depth of the fixed point); fixed_point:
    False when max_rounds stopped it first; facts_read / rules_read: what it started from; facts_proposed: new
    justifications written; facts_added: new fact items among them; refused: proposals a write gate refused (e.g. a
    derivation that would refute its own premise); circular: justifications not written because they would rest on
    the fact they justify; retracted: facts of an earlier call taken back (chain_rules: a negative-stratum fact whose
    condition ended); disputes_opened: new person questions; hypotheses: derived facts that are hypotheses at the
    end; closed_conflicts: subjects with several known positives of a declared functional relation (chain_rules: the
    closed world is not applied to them); per_round: [{"round", "proposed", "added", "refused", "circular"}];
    seconds: wall time."""
    rounds: int = 0
    fixed_point: bool = True
    facts_read: int = 0
    rules_read: int = 0
    facts_proposed: int = 0
    facts_added: int = 0
    refused: int = 0
    circular: int = 0
    retracted: int = 0
    disputes_opened: int = 0
    hypotheses: int = 0
    closed_conflicts: int = 0
    per_round: list = field(default_factory=list)
    seconds: float = 0.0


def _scope_key(scope):
    return json.dumps(scope, sort_keys=True, ensure_ascii=False)


def _live(it):
    cut = max(it.retracted, default=-1)
    return [a for a in it.asserts if a["t"] > cut]


def _rests_on(store, target, premises):
    """Does any of `premises` rest on `target` (is it, or are its live justifications, transitively)?"""
    seen, todo = set(), list(premises)
    while todo:
        i = todo.pop()
        if i == target:
            return True
        if i in seen:
            continue
        seen.add(i)
        it = store.items.get(i)
        if it is not None:
            for a in _live(it):
                todo.extend(a["just"])
    return False


def forward_chain(store, *, scope=None, rule_scope=None, rules="active", max_rounds=None, max_facts=None,
                  by="relations:forward_chain", multi=False, complements=None) -> ChainReport:
    """Derive the consequences of the pair facts in `store` under its composition rules, to a fixed point.

    scope: the facts to chain over (a scope pattern, store.find's; None: every scope) — facts join only within one
    exact scope, and a derived fact is written in its premises' scope; rule_scope: where the rules are (default: the
    same pattern as scope); rules: "active" (promoted rules only) or "any" (hypothesis rules too: what they derive is a
    hypothesis, by the store's own premise rule); max_rounds: stop after this many rounds (the report says
    fixed_point=False; a later call resumes); max_facts: a hard stop on new facts per scope (default |entities|² ×
    |relations|; ChainLimit); multi: derived facts are multi pair facts (`pair_fact(..., multi=True)`: a pair carries
    one fact per relation, and only relations of one family — `complements`: relation → the relation whose key it
    shares — dispute). Facts of either form are read. Every write goes through store.add (the write gates).
    Idempotent: a second call writes nothing. → ChainReport."""
    if rules not in RULE_STATUSES:
        raise ValueError('rules is "active" or "any"')
    if max_rounds is not None and int(max_rounds) < 1:
        raise ValueError("max_rounds is at least 1")
    from ...core.knowledge.gates import RANK
    from ...core.knowledge.store import item_id, plain_json
    by_rank = {v: k for k, v in RANK.items()}
    comp = dict(complements or {})
    t0 = time.perf_counter()
    rep = ChainReport()
    questions0 = len(store.questions)

    table = defaultdict(list)                                # (r1, r2) → [(rule id, relation)]
    for rec in store.find(kind="rule", scope=scope if rule_scope is None else rule_scope, status=RULE_STATUSES[rules]):
        b = rec["body"]
        if isinstance(b, dict) and b.get("kind") == "composition" and b.get("r") == "compose" \
                and isinstance(b.get("s"), list) and len(b["s"]) == 2:
            table[(b["s"][0], b["s"][1])].append((rec["id"], b["o"]))
    rep.rules_read = sum(len(v) for v in table.values())
    outputs = {r for v in table.values() for _, r in v}

    groups = defaultdict(dict)                               # scope key → {fact id: (x, y, relation)}
    scopes = {}
    for rec in store.find(kind="fact", scope=scope, status=FACT_STATUSES):
        p = _pair(rec["body"])
        if p is not None:
            k = _scope_key(rec["scope"])
            groups[k][rec["id"]] = p
            scopes[k] = rec["scope"]
    rep.facts_read = sum(len(g) for g in groups.values())

    derived_ids = set()
    rounds_done = defaultdict(int)
    for k in sorted(groups):
        facts, sc = groups[k], scopes[k]
        entities = {e for x, y, _ in facts.values() for e in (x, y)}
        relations = {r for _, _, r in facts.values()} | outputs
        cap = len(entities) ** 2 * max(1, len(relations)) if max_facts is None else int(max_facts)
        new_items = 0
        out_by, in_by = defaultdict(list), defaultdict(list)
        for fid, (x, y, r) in sorted(facts.items()):
            out_by[x].append(fid)
            in_by[y].append(fid)
        delta = set(facts)
        rnd = 0
        while delta:
            if max_rounds is not None and rnd >= int(max_rounds):
                rep.fixed_point = False
                break
            rnd += 1
            pairs = set()
            for f in delta:
                x, y, _ = facts[f]
                pairs.update((f, g) for g in out_by[y])
                pairs.update((e, f) for e in in_by[x])
            row = {"round": rnd, "proposed": 0, "added": 0, "refused": 0, "circular": 0}
            fresh, seen_fresh = [], set()
            with store.batch():
                for f1, f2 in sorted(pairs):
                    x, y, r1 = facts[f1]
                    _, z, r2 = facts[f2]
                    for rid, rel in table.get((r1, r2), ()):
                        body = plain_json(pair_fact(x, z, rel, multi=True, family=comp.get(rel, rel)) if multi
                                          else pair_fact(x, z, rel))
                        iid = item_id("fact", body, sc)
                        just = sorted({f1, f2, rid})
                        it = store.items.get(iid)
                        if it is not None:
                            if any(list(a["just"]) == just for a in _live(it)):
                                continue                    # this justification is already recorded
                            if _rests_on(store, iid, (f1, f2)):
                                row["circular"] += 1
                                continue
                        st1, st2 = store.state.get(f1), store.state.get(f2)
                        if st1 is None or st2 is None:
                            continue                        # a premise left the store during the chain
                        rank = min(st1["rank"], st2["rank"])
                        if rank not in by_rank:
                            continue
                        new = iid not in facts and iid not in seen_fresh   # not a usable fact before this proposal
                        got = store.add("fact", body, sc, source=by_rank[rank], by=by, derived_from=just,
                                        evidence={"rule": rid, "via": y, "depth": rnd}, inferred=True)
                        if got is None:
                            row["refused"] += 1
                            continue
                        row["proposed"] += 1
                        if new:
                            row["added"] += 1
                            new_items += 1
                            if new_items > cap:
                                raise ChainLimit(f"more than max_facts={cap} new facts in scope {k}: the rules or the "
                                                 "facts are not what forward_chain expects (pair facts, composition "
                                                 "rules)")
                            fresh.append(iid)
                            seen_fresh.add(iid)
            delta = set()
            for iid in fresh:                                # read after the batch: the store has resolved them
                derived_ids.add(iid)
                if iid not in facts and store.status(iid) in FACT_STATUSES:
                    facts[iid] = _pair(store.items[iid].body)
                    x, y, _ = facts[iid]
                    out_by[x].append(iid)
                    in_by[y].append(iid)
                    delta.add(iid)
            if row["proposed"]:
                rep.per_round.append(row)
            rep.facts_proposed += row["proposed"]
            rep.facts_added += row["added"]
            rep.refused += row["refused"]
            rep.circular += row["circular"]
            if row["proposed"]:
                rounds_done[k] = rnd
    rep.rounds = max(rounds_done.values(), default=0)
    rep.per_round = _merge_rounds(rep.per_round)
    rep.disputes_opened = len(store.questions) - questions0
    rep.hypotheses = sum(store.status(i) == "hypothesis" for i in derived_ids)
    rep.seconds = time.perf_counter() - t0
    return rep


def _merge_rounds(rows):
    """Per-round counts summed over scopes (round n of every scope together)."""
    out = {}
    for r in rows:
        acc = out.setdefault(r["round"], {"round": r["round"], "proposed": 0, "added": 0, "refused": 0, "circular": 0})
        for k in ("proposed", "added", "refused", "circular"):
            acc[k] += r[k]
    return [out[n] for n in sorted(out)]


@dataclass
class Explanation:
    """What explain() returns. fact: the id explained; nodes: id → {"body", "status", "source", "original" (asserted
    without premises), "derivations": [{"rule", "cell": (r1, r2, relation), "premises": (left fact, right fact)}]},
    every fact the derivation reaches; facts: the original facts it rests on; rules: the rule ids used; cells: the
    rule cells used, as (r1, r2, relation), sorted. A guarded rule (`chain_rules`) gives "cell": (template, body
    text, head), "premises" in body order with the unary guards among them, "stratum" (1 positive, 2 negative) and
    "absent" (the negated relation of a negative-stratum fact, else None); a closed-world fact gives "cell":
    ("closed", "<relation> functional for <subject unary>", head), "premises" (the known positive, the declaration —
    a spec item, shown as a premise), "stratum" 3."""
    fact: str
    nodes: dict
    facts: tuple
    rules: tuple
    cells: tuple

    def tree(self, iid=None) -> dict:
        """One derivation as a nested dict (the first justification of each fact, in journal order): {"id", "body",
        "status", "rule": {"id", "cell"} or None, "premises": [subtree, subtree] or []}."""
        n = self.nodes[self.fact if iid is None else iid]
        i = self.fact if iid is None else iid
        if not n["derivations"]:
            return {"id": i, "body": n["body"], "status": n["status"], "rule": None, "premises": []}
        d = n["derivations"][0]
        return {"id": i, "body": n["body"], "status": n["status"], "rule": {"id": d["rule"], "cell": d["cell"]},
                "premises": [self.tree(p) for p in d["premises"]]}


def explain(store, fact_id) -> Explanation:
    """The derivation of a fact read back from the store: its live justifications (a rule and two facts; a premise
    retracted or refuted ends one), recursively down to the facts asserted without premises."""
    if fact_id not in store.items:
        raise KeyError(f"no item {fact_id!r}")
    nodes, todo = {}, [fact_id]
    while todo:
        i = todo.pop()
        if i in nodes:
            continue
        it = store.items[i]
        node = {"body": it.body, "status": store.status(i), "source": it.asserts[-1]["source"] if it.asserts else None,
                "original": False, "derivations": []}
        nodes[i] = node
        sv = it.body.get("s") if isinstance(it.body, dict) else None
        x, z = sv if isinstance(sv, list) and len(sv) == 2 else (None, None)
        for a in _live(it):
            if not a["just"]:
                node["original"] = True
                continue
            if any(store.status(p) in (None, "retracted", "refuted") for p in a["just"]):
                continue
            ev = a.get("evidence") or {}
            rid = ev.get("rule")
            via = ev.get("via")
            rule = store.items.get(rid)
            prem = [p for p in a["just"] if p != rid]
            if rule is not None and isinstance(rule.body, dict) and rule.body.get("kind") == "functional":
                rb = rule.body                                # the closed-world stratum: the positive, the declaration
                order = [p for p in ev.get("premises") or a["just"] if p in a["just"]] or list(a["just"])
                node["derivations"].append({"rule": rid, "cell": ("closed", f"{rb['s'][0]} functional for "
                                                                            f"{rb['s'][1]}", it.body.get("o")),
                                            "premises": tuple(order), "stratum": ev.get("stratum", 3),
                                            "absent": None})
                todo += order                                 # the declaration is a leaf of the derivation
                continue
            if rule is not None and isinstance(rule.body, dict) and rule.body.get("kind") == "guarded":
                rb = rule.body                                # a guarded rule: premises in body order, guards included
                order = [p for p in ev.get("premises") or prem if p in prem] or prem
                node["derivations"].append({"rule": rid, "cell": (rb["template"], rb["s"][0], rb["o"]),
                                            "premises": tuple(order), "stratum": ev.get("stratum", 1),
                                            "absent": ev.get("absent")})
                todo += order
                continue
            if rule is None or len(prem) not in (1, 2):
                continue
            if len(prem) == 1:                                # x == via == z: one fact used twice
                left = right = prem[0]
            else:
                p0, p1 = (store.items[p].body.get("s") for p in prem)
                left, right = (prem[0], prem[1]) if p0 == [x, via] and p1 == [via, z] else (prem[1], prem[0])
            rb = rule.body
            node["derivations"].append({"rule": rid, "cell": (rb["s"][0], rb["s"][1], rb["o"]),
                                        "premises": (left, right)})
            todo += [left, right]
    leaves = tuple(sorted(i for i, n in nodes.items() if n["original"]))
    rules = tuple(sorted({d["rule"] for n in nodes.values() for d in n["derivations"]}))
    cells = tuple(sorted({d["cell"] for n in nodes.values() for d in n["derivations"]}))
    return Explanation(fact_id, nodes, leaves, rules, cells)


__all__ = ["ChainLimit", "ChainReport", "Explanation", "RELATION", "explain", "forward_chain", "pair_fact"]
