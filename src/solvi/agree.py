"""Agreement of generated candidates under a key: K outputs (samples of one model, or one each from several models),
grouped by what you say makes two of them the same, the largest group chosen, its share as a signal.

    from solvi.agree import agree, consensus

    c = consensus(queries, key=lambda sql: digest(run(db, sql)))   # {"index", "value", "share", "groups", "keys", ...}

    cat.fn(writer.part("candidates", prompt, k=3))                 # three queries (solvi.generate)
    agree(cat, "sql", candidates="candidates", key=row_digest)     # facts: sql, sql_agreement, sql_tally

solvi.multi.Vote combines decisions over the same closed options; generated outputs have no options — two queries that
differ in text can return the same rows, two plans in different words can be the same order. `key(candidate)` says what
counts as the same: the digest of the rows a query returns, a normalized plan, a parsed number. The share of the
candidates in the chosen group is a plain number fact (`<name>_agreement`), so a rule, a learned head or a guarantee can
read it like any other signal; "all K agree" is the share 1.0.

Rules. A candidate that is None (its generation failed), whose key raises, or whose key is None does not vote; it still
counts in K, so a failed candidate lowers the share. `prefer(candidate)`: when any candidate with a key passes it, only
those vote (rows that are not empty before an empty result, say); otherwise all with a key vote. The largest group wins;
a tie goes to the group whose first candidate comes first (with K samples, the first is usually the greedy one). The
chosen value is the first candidate of that group. Nothing voted: no value — the fact `<name>` is missing (the part
raises with each candidate's reason) and the questions that need it abstain, while `<name>_tally` and
`<name>_agreement` (0.0) are still there for the checks and the feedback.

The record. `<name>_tally` is a plain dict in the trace — per candidate its key (or why it has none) and whether it
voted, the groups, the chosen index, the share — so the audit shows how the vote went and replay recomputes it from the
recorded candidates (the key function is re-run: keep it deterministic, or cache what it computes).

Not done here: agreement is not correctness — K samples of one model often agree on the same mistake; measure the share
against labels before you trust it, and put a guarantee on it rather than a hand-picked threshold."""
from __future__ import annotations

import inspect
import json

from .core import Claim


def _jsonable(k):
    """A key as the tally records it: JSON values as they are, anything else as its repr."""
    try:
        json.dumps(k)
        return k
    except (TypeError, ValueError):
        return repr(k)


def _call(f, cand, facts):
    """f(candidate, **the facts it names after its first parameter)."""
    names = list(inspect.signature(f).parameters)[1:]
    return f(cand, **{n: facts[n] for n in names if n in facts})


def _preferred(prefer, cand, facts):
    """prefer(candidate) as a bool; a prefer that raises does not prefer the candidate."""
    try:
        return bool(_call(prefer, cand, facts))
    except Exception:  # noqa: BLE001 — prefer raising: not preferred
        return False


def consensus(candidates, key, *, prefer=None, facts=None):
    """K candidates → the tally: {"index" (the chosen candidate, -1 when nothing voted), "value", "key", "share" (the
    chosen group's size / K), "k", "groups" ([[indexes]], largest first), "candidates" ([{"index", "key" or "why",
    "voted"}])}. key(candidate[, facts by name]) → anything hashable that says which candidates are the same; prefer: see
    the module docs. facts: the extra arguments key / prefer take by name (agree passes the catalog's facts)."""
    cands = list(candidates)
    if not cands:
        raise ValueError("consensus of no candidates")
    facts = facts or {}
    rows, keys = [], {}
    for i, c in enumerate(cands):
        if c is None:
            rows.append({"index": i, "key": None, "why": "no candidate (its generation failed)", "voted": False})
            continue
        try:
            k = _call(key, c, facts)
        except Exception as e:  # noqa: BLE001 — a key that cannot be computed: the candidate does not vote
            rows.append({"index": i, "key": None, "why": f"key failed: {type(e).__name__}: {str(e)[:120]}", "voted": False})
            continue
        if k is None:
            rows.append({"index": i, "key": None, "why": "no key", "voted": False})
            continue
        keys[i] = k
        rows.append({"index": i, "key": _jsonable(k), "voted": True})
    pool = list(keys)
    if prefer is not None and pool:
        good = [i for i in pool if _preferred(prefer, cands[i], facts)]
        if good:
            for i in pool:
                if i not in good:
                    rows[i]["voted"] = False
                    rows[i]["why"] = "not preferred"
            pool = good
    groups = []
    for i in pool:
        for g in groups:
            if keys[g[0]] == keys[i]:
                g.append(i)
                break
        else:
            groups.append([i])
    groups.sort(key=lambda g: (-len(g), g[0]))
    if not groups:
        return {"index": -1, "value": None, "key": None, "share": 0.0, "k": len(cands), "groups": [], "candidates": rows}
    best = groups[0][0]
    share = sum(1 for i in keys if keys[i] == keys[best]) / len(cands)
    return {"index": best, "value": cands[best], "key": _jsonable(keys[best]), "share": share, "k": len(cands),
            "groups": groups, "candidates": rows}


def agree(cat, name, candidates, key, *, prefer=None, share=None, tally=None):
    """Register in `cat` the agreement of the candidates in the fact `candidates` (a list) under `key` → three facts:
    `<tally>` (default `<name>_tally`: the record, see consensus), `<name>` (the chosen candidate; missing when nothing
    voted) and `<share>` (default `<name>_agreement`: the chosen group's share of all K, 0.0 when nothing voted).
    key / prefer: functions of a candidate and, by name, of other facts (`def key(sql, db_path)`), which become inputs
    of the tally part. → the names (tally, name, share)."""
    tally = tally or f"{name}_tally"
    share = share or f"{name}_agreement"
    extra = []
    for f in (key, prefer):
        if f is not None:
            extra += [n for n in list(inspect.signature(f).parameters)[1:] if n not in extra and n != candidates]
    params = [inspect.Parameter(candidates, inspect.Parameter.POSITIONAL_OR_KEYWORD)] + \
             [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in extra]

    def tally_part(**facts):
        cands = facts[candidates]
        if not isinstance(cands, (list, tuple)):
            raise TypeError(f"{candidates} is not a list of candidates but a {type(cands).__name__}")
        t = consensus(cands, key, prefer=prefer, facts=facts)
        t = dict(t)
        t.pop("value")                                 # the chosen value is the fact `name`; the tally holds indexes
        return t
    tally_part.__name__ = tally_part.__qualname__ = tally
    tally_part.__signature__ = inspect.Signature(params)
    tally_part.__doc__ = f"The agreement of the candidates in {candidates}: keys, groups, the chosen one, the share."
    cat.fn(tally_part)

    def chosen(**facts):
        t, cands = facts[tally], facts[candidates]
        if t["index"] < 0:
            why = "; ".join(f"{r['index']}: {r.get('why')}" for r in t["candidates"])
            raise ValueError(f"no candidate has a key ({why})")
        return Claim(cands[t["index"]], extra={"agreement": {"index": t["index"], "share": t["share"]}})
    chosen.__name__ = chosen.__qualname__ = name
    chosen.__signature__ = inspect.Signature([inspect.Parameter(tally, inspect.Parameter.POSITIONAL_OR_KEYWORD),
                                              inspect.Parameter(candidates, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    chosen.__doc__ = f"The candidate of {candidates} in the largest group of equal keys."
    cat.fn(chosen)

    def share_part(**facts):
        return float(facts[tally]["share"])
    share_part.__name__ = share_part.__qualname__ = share
    share_part.__signature__ = inspect.Signature([inspect.Parameter(tally, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    share_part.__doc__ = f"The share of the candidates in {candidates} that agree with {name} (1.0: all of them)."
    share_part.__annotations__ = {"return": float}
    cat.fn(share_part)
    return tally, name, share
