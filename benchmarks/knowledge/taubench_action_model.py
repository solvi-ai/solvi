"""The action model on τ-bench retail: solvi.core.knowledge.ConservativeActionModel learned from environment probes.

What it measures. τ-bench retail's tools are run (no LLM, $0) on probe calls around every gold in-scope step of the
train tasks: the gold call, the call on perturbed states (another order status, a gift card at 0), with perturbed
arguments (another order, payment method or item, an unknown id, duplicates, an empty list, another reason, an empty
address field) and 6 scripted random calls per in-scope tool. The library's ConservativeActionModel learns from these
transitions over a Vocabulary of typed predicates (generic templates over the tools' JSON schema: existence, status,
lengths, enums, pair features of two conditions on the same argument, money comparisons) and predicts accept / refuse
/ unknown on the same probes around the test tasks. Reported: refusal precision and recall over answered held-out
transitions, strict recall (an abstention counted as a miss), the abstention share, the replay of the training
transitions, unexplained training refusals per tool, the status effect (the touched order's new status, and whether the
user record changed) predicted exactly — a narrower effect than full entity diffs — and two ablation rows: the
vocabulary without pair features and without money comparisons.

The vocabulary was written by someone who had read the tools' code: the result says what a conservative learner
recovers given a vocabulary that can express the environment's checks, not that it discovers them.

Data: a local clone of https://github.com/sierra-research/tau-bench (MIT): `git clone
https://github.com/sierra-research/tau-bench` — the tools, the database and the task lists are read from it (no
network). Train: the 500 train tasks less 100 reserved by a fixed draw (seed 20261003), probes seed 1; held-out: the 115
test tasks, probes seed 2.

    uv run python benchmarks/knowledge/taubench_action_model.py --taubench PATH/tau-bench [--quick]

→ benchmarks/knowledge/results/taubench_action_model.json (taubench_action_model_quick.json with --quick).
"""
from __future__ import annotations

import argparse
import collections
import copy
import itertools
import json
import random
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from solvi.core.knowledge import ConservativeActionModel, Vocabulary  # noqa: E402

T0 = time.time()
_last = [0.0]


def log(*a, force=False):
    now = time.time()
    if force or now - _last[0] >= 20:                    # heartbeat: at least every 20 s while working
        _last[0] = now
        print(f"[{now - T0:7.1f}s]", *a, flush=True)


INSCOPE = ["cancel_pending_order", "modify_pending_order_address", "modify_pending_order_payment",
           "modify_pending_order_items", "return_delivered_order_items", "exchange_delivered_order_items",
           "modify_user_address", "transfer_to_human_agents"]
ADDR = ["address1", "address2", "city", "state", "country", "zip"]
TOOLS = SCHEMA = DB = VI = PM_INDEX = ALL_ORDERS = ALL_USERS = DB_STATUSES = TASKS = None


def load_taubench(path):
    """Import the tools, the database and the tasks from a local clone (litellm stubbed: no model is called)."""
    global TOOLS, SCHEMA, DB, VI, PM_INDEX, ALL_ORDERS, ALL_USERS, DB_STATUSES, TASKS
    sys.path.insert(0, str(path))
    stub = types.ModuleType("litellm")
    stub.completion = None
    stub.provider_list = []
    sys.modules.setdefault("litellm", stub)
    from tau_bench.envs.retail.data import load_data
    from tau_bench.envs.retail.tasks_test import TASKS_TEST
    from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
    from tau_bench.envs.retail.tools import ALL_TOOLS
    TOOLS = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
    SCHEMA = {n: t.get_info()["function"]["parameters"] for n, t in TOOLS.items()}
    DB = load_data()                                     # pristine, never mutated
    VI = {vid: pid for pid, p in DB["products"].items() for vid in p["variants"]}
    PM_INDEX = {pid for u in DB["users"].values() for pid in u["payment_methods"]}
    ALL_ORDERS, ALL_USERS = sorted(DB["orders"]), sorted(DB["users"])
    DB_STATUSES = sorted({o["status"] for o in DB["orders"].values()})
    TASKS = {s: [{"user_id": t.user_id, "actions": [{"name": a.name, "kwargs": a.kwargs} for a in t.actions]}
                 for t in ts] for s, ts in (("train", TASKS_TRAIN), ("test", TASKS_TEST))}


# ------------------------------------------------------------------------------------------- state layers (no copies)
class Layer:
    """Read-only view: top dict over a base (dict or Layer)."""

    def __init__(self, base, top=None):
        self.base, self.top = base, ({} if top is None else top)

    def peek(self, k):
        if k in self.top:
            return self.top[k]
        return self.base.peek(k) if isinstance(self.base, Layer) else self.base.get(k)

    def __contains__(self, k):
        return k in self.top or k in self.base


class COW:
    """What a tool sees: reads copy the entity into the overlay; writes land there."""

    def __init__(self, layer):
        self.layer, self.ov = layer, {}

    def __contains__(self, k):
        return k in self.ov or k in self.layer

    def __getitem__(self, k):
        if k not in self.ov:
            v = self.layer.peek(k)
            if v is None:
                raise KeyError(k)
            self.ov[k] = copy.deepcopy(v)
        return self.ov[k]

    def get(self, k, d=None):
        return self[k] if k in self else d


class State:
    def __init__(self, orders=None, users=None, parent=None):
        po = parent.orders if parent else DB["orders"]
        pu = parent.users if parent else DB["users"]
        self.orders, self.users = Layer(po, orders), Layer(pu, users)

    def order(self, oid):
        return self.orders.peek(oid) if isinstance(oid, str) else None

    def user(self, uid):
        return self.users.peek(uid) if isinstance(uid, str) else None


def call(tool, args, st):
    """Run the benchmark's tool on a copy-on-write view of st → (refused, changed {(coll, id): (before, after)})."""
    data = {"orders": COW(st.orders), "users": COW(st.users), "products": COW(Layer(DB["products"]))}
    try:
        obs = TOOLS[tool].invoke(data=data, **args)
    except Exception as e:  # noqa: BLE001 — the benchmark's Env.step does exactly this
        obs = f"Error: {e}"
    changed = {}
    for coll, layer in (("orders", st.orders), ("users", st.users)):
        for k, v in data[coll].ov.items():
            b = layer.peek(k)
            if v != b:
                changed[(coll, k)] = (b, v)
    return obs.startswith("Error"), changed


def apply(st, changed):
    for (coll, k), (_, after) in changed.items():
        (st.orders if coll == "orders" else st.users).top[k] = copy.deepcopy(after)


# -------------------------------------------------------------------------------------------- the vocabulary's templates
def blen(x):
    if not isinstance(x, list):
        return None
    return str(len(x)) if len(x) < 4 else "4+"


def resolve_in_order(order, ids):
    """Each id an item of the order, counting multiplicity → [item dicts] or None."""
    if order is None or not isinstance(ids, list):
        return None
    left, out = list(order["items"]), []
    for i in ids:
        hit = next((x for x in left if x["item_id"] == i), None)
        if hit is None:
            return None
        left.remove(hit)
        out.append(hit)
    return out


def variant(vid):
    pid = VI.get(vid) if isinstance(vid, str) else None
    return DB["products"][pid]["variants"][vid] if pid else None


def bind(tool, args, st):
    props = SCHEMA[tool]["properties"]
    b = {"order": None, "user": None, "pm": None, "ph": None}
    if "order_id" in props:
        b["order"] = st.order(args.get("order_id"))
        if b["order"]:
            b["user"] = st.user(b["order"]["user_id"])
            b["ph"] = b["order"]["payment_history"]
    if "user_id" in props:
        b["user"] = st.user(args.get("user_id"))
    if "payment_method_id" in props and b["user"] is not None and isinstance(args.get("payment_method_id"), str):
        b["pm"] = b["user"]["payment_methods"].get(args["payment_method_id"])
    return b


def pairs_of(args, order):
    a, n = args.get("item_ids"), args.get("new_item_ids")
    old = resolve_in_order(order, a)
    if old is None or not isinstance(n, list) or len(old) != len(n) or not all(variant(x) for x in n):
        return None
    return list(zip(old, n))


def delta(args, order):
    pr = pairs_of(args, order)
    return None if pr is None else sum(variant(n)["price"] - o["price"] for o, n in pr)


def features(tool, args, st):
    """Typed predicates from generic templates over the schema → {name: value} (None = undefined). Pair features:
    every pair of conditions about the same argument."""
    props = SCHEMA[tool]["properties"]
    f, about = {}, collections.defaultdict(list)

    def put(name, val, tag):
        f[name] = val
        about[tag].append(name)

    b = bind(tool, args, st)
    order, user, pm, ph = b["order"], b["user"], b["pm"], b["ph"]
    pid = args.get("payment_method_id")
    if "order_id" in props:
        put("order:exists", order is not None, "order_id")
        put("order:status", order["status"] if order else None, "order_id")
        put("order:len(payment_history)", blen(ph), "order_id")
        put("order:len(items)", blen(order["items"]) if order else None, "order_id")
        put("order:payment_history[0].transaction_type", ph[0]["transaction_type"] if ph else None, "order_id")
        put("order:all(payment_history.transaction_type=payment)",
            all(t["transaction_type"] == "payment" for t in ph) if ph else None, "order_id")
        src0 = None
        if ph and user:
            p0 = user["payment_methods"].get(ph[0]["payment_method_id"])
            src0 = p0["source"] if p0 else "<not in user>"
        put("order:payment_history[0].source", src0, "order_id")
    if "user_id" in props:
        put("user:exists", user is not None, "user_id")
    for name, sch in props.items():
        if name in ("order_id", "user_id", "payment_method_id") or sch.get("type") != "string":
            continue
        v = args.get(name)
        if "enum" in sch:
            put(f"arg:{name}", (v if v in sch["enum"] else "<not in enum>") if isinstance(v, str) else None, name)
        else:
            put(f"arg:{name}:nonempty", bool(v.strip()) if isinstance(v, str) else None, name)
    if "payment_method_id" in props:
        put("pm:exists@user", (pm is not None) if user is not None else None, "pm")
        put("pm:exists@any", isinstance(pid, str) and pid in PM_INDEX, "pm")
        put("pm:source", pm["source"] if pm else None, "pm")
        put("pm:=payment_history[0].pm", (pid == ph[0]["payment_method_id"]) if ph else None, "pm")
        put("pm:in payment_history.pm", (pid in {t["payment_method_id"] for t in ph}) if ph else None, "pm")
    lists = [n for n in ("item_ids", "new_item_ids") if n in props]
    for L in lists:
        v = args.get(L)
        ok = isinstance(v, list) and all(isinstance(x, str) for x in v)
        put(f"{L}:len", blen(v) if ok else None, L)
        put(f"{L}:has_dup", (len(set(v)) < len(v)) if ok else None, L)
        put(f"{L}:all_in_order_items", (resolve_in_order(order, v) is not None) if ok and order else None, L)
        put(f"{L}:all_resolve_global", all(x in VI for x in v) if ok else None, L)
        vs = [variant(x) for x in v] if ok and all(x in VI for x in v) else None
        put(f"{L}:all_available", all(x["available"] for x in vs) if vs is not None else None, L)
        put(f"{L}:any_available", any(x["available"] for x in vs) if vs is not None else None, L)
    if len(lists) == 2:
        a, n = args.get("item_ids"), args.get("new_item_ids")
        put("len(item_ids)=len(new_item_ids)", (len(a) == len(n)) if isinstance(a, list) and isinstance(n, list) else None,
            "pair")
        pr = pairs_of(args, order)
        put("pair:all same product", all(VI[nv] == o["product_id"] for o, nv in pr) if pr is not None else None, "pair")
        put("pair:all new != old", all(nv != o["item_id"] for o, nv in pr) if pr is not None else None, "pair")
        put("pair:any new == old", any(nv == o["item_id"] for o, nv in pr) if pr is not None else None, "pair")
    terms = {}
    if "order_id" in props:
        terms["order.payment_history[0].amount"] = ph[0]["amount"] if ph else None
        terms["sum(order.payment_history.amount)"] = sum(t["amount"] for t in ph) if ph else None
        terms["sum(order.items.price)"] = sum(i["price"] for i in order["items"]) if order else None
    if len(lists) == 2:
        terms["DELTA"] = delta(args, order)
    if "payment_method_id" in props:
        bal = pm.get("balance") if pm else None
        for t, tv in list(terms.items()) + [("0", 0)]:
            put(f"pm.balance absent or >= {t}", None if (tv is None or pm is None) else (bal is None or bal >= tv - 1e-9),
                "pm")
            put(f"pm.balance present and >= {t}", None if (tv is None or pm is None) else (bal is not None and bal >= tv - 1e-9),
                "pm")
    names = list(terms)
    for x, y in itertools.permutations(names + ["0"], 2):
        xv = 0 if x == "0" else terms[x]
        yv = 0 if y == "0" else terms[y]
        put(f"{x} >= {y}", None if xv is None or yv is None else xv >= yv - 1e-9, "money")
        put(f"{x} > {y}", None if xv is None or yv is None else xv > yv + 1e-9, "money")
    for tag, ns in list(about.items()):
        if tag == "money":
            continue
        for x, y in itertools.combinations(ns, 2):
            f[f"({x}) & ({y})"] = (f[x], f[y])
    return f


_CACHE = [None, None, None, None]                         # (state, tool, args as JSON, features): the last call only


def cached(tool, args, st):
    key = json.dumps(args, sort_keys=True)
    c = _CACHE
    if c[0] is not st or c[1] != tool or c[2] != key:      # the state object itself is kept: no id() reuse
        c[:] = [st, tool, key, features(tool, args, st)]
    return c[3]


def vocabulary(drop=None):
    """One predicate per (tool, feature): "tool|feature" → features(tool, args, state)[feature]; drop(name) leaves a
    feature out (the ablations)."""
    preds, only = {}, {}
    for tool in INSCOPE:
        names = sorted(features(tool, {}, State()))          # the names follow from the schema only
        keep = [n for n in names if drop is None or not drop(n)]
        for n in keep:
            preds[f"{tool}|{n}"] = (lambda t, n: lambda state, args: cached(t, args, state)[n])(tool, n)
        only[tool] = [f"{tool}|{n}" for n in keep]
    return Vocabulary(preds, only=only)


def effect_of(tool, args, changed):
    """The status effect of an accepted call: the touched order's new status (None: no order changed) and whether a
    user record changed."""
    oid = args.get("order_id")
    after = changed.get(("orders", oid))
    return {"order_status": after[1]["status"] if after else None,
            "user_changed": any(c == "users" for c, _ in changed)}


# ------------------------------------------------------------------------------------------------------------ probes
def other_user(rng, uid):
    while True:
        u = rng.choice(ALL_USERS)
        if u != uid and DB["users"][u]["orders"]:
            return u


def probe_calls(tool, args, st, uid, rng, statuses):
    """The gold call, it on perturbed states, it with perturbed arguments, and 6 scripted random calls per tool."""
    out = [(tool, args, None, "gold")]
    props = SCHEMA[tool]["properties"]
    user = st.user(uid)
    order = st.order(args.get("order_id"))
    if order is not None:
        for s in statuses:
            if s != order["status"]:
                o2 = copy.deepcopy(order)
                o2["status"] = s
                out.append((tool, args, ({order["order_id"]: o2}, {}), "state:status"))
    pmid = args.get("payment_method_id")
    owner = st.user(order["user_id"]) if order else user
    if owner is not None and pmid in owner["payment_methods"] and owner["payment_methods"][pmid]["source"] == "gift_card":
        u2 = copy.deepcopy(owner)
        u2["payment_methods"][pmid]["balance"] = 0
        okey = order["user_id"] if order else uid
        out.append((tool, args, ({}, {okey: u2}), "state:gift0"))

    def var(**kw):
        a = dict(args)
        a.update(kw)
        return a

    my_orders = list(user["orders"]) if user else []
    if "order_id" in props:
        others = [o for o in my_orders if o != args.get("order_id")]
        for o in rng.sample(others, min(3, len(others))):
            out.append((tool, var(order_id=o), None, "arg:order:same user"))
        ou = st.user(other_user(rng, uid))
        out.append((tool, var(order_id=rng.choice(ou["orders"])), None, "arg:order:other user"))
        out.append((tool, var(order_id="#W0000000"), None, "arg:order:unknown"))
    if "payment_method_id" in props and owner is not None:
        for p in owner["payment_methods"]:
            if p != pmid:
                out.append((tool, var(payment_method_id=p), None, "arg:pm:same user"))
        ou = st.user(other_user(rng, uid))
        out.append((tool, var(payment_method_id=rng.choice(sorted(ou["payment_methods"]))), None, "arg:pm:other user"))
        out.append((tool, var(payment_method_id="credit_card_0000000"), None, "arg:pm:unknown"))
    if "item_ids" in props and isinstance(args.get("item_ids"), list) and order is not None:
        a, n = list(args["item_ids"]), list(args.get("new_item_ids", []))
        has_new = "new_item_ids" in props
        if len(a) > 1:
            keep = sorted(rng.sample(range(len(a)), rng.randint(1, len(a) - 1)))
            out.append((tool, var(item_ids=[a[i] for i in keep], **({"new_item_ids": [n[i] for i in keep if i < len(n)]}
                                                                    if has_new else {})), None, "arg:items:subset"))
        if a:
            oo = st.order(rng.choice(ALL_ORDERS))
            alien = rng.choice(oo["items"])["item_id"]
            out.append((tool, var(item_ids=[alien] + a[1:]), None, "arg:items:other order"))
            out.append((tool, var(item_ids=a + a[:1], **({"new_item_ids": n + n[:1]} if has_new else {})), None,
                        "arg:items:duplicate"))
        out.append((tool, var(item_ids=[], **({"new_item_ids": []} if has_new else {})), None, "arg:items:empty"))
    if "new_item_ids" in props and isinstance(args.get("new_item_ids"), list) and args["new_item_ids"] and order is not None:
        a, n = list(args["item_ids"]), list(args["new_item_ids"])
        old = resolve_in_order(order, a)
        k = rng.randrange(len(n))
        if old is not None and k < len(old):
            prod = DB["products"][old[k]["product_id"]]["variants"]
            v2 = rng.choice(sorted(prod))
            out.append((tool, var(new_item_ids=n[:k] + [v2] + n[k + 1:]), None, "arg:new:another variant"))
            unav = sorted(x for x, v in prod.items() if not v["available"])
            if unav:
                out.append((tool, var(new_item_ids=n[:k] + [rng.choice(unav)] + n[k + 1:]), None, "arg:new:unavailable"))
            out.append((tool, var(new_item_ids=n[:k] + [old[k]["item_id"]] + n[k + 1:]), None, "arg:new:same item"))
            other_p = rng.choice(sorted(p for p in DB["products"] if p != old[k]["product_id"]))
            out.append((tool, var(new_item_ids=n[:k] + [rng.choice(sorted(DB["products"][other_p]["variants"]))] + n[k + 1:]),
                        None, "arg:new:other product"))
        out.append((tool, var(new_item_ids=n[:-1]), None, "arg:new:one fewer"))
    if "reason" in props:
        for r in ("no longer needed", "ordered by mistake", "changed my mind"):
            if r != args.get("reason"):
                out.append((tool, var(reason=r), None, "arg:reason"))
    if "address1" in props:
        fld = rng.choice(ADDR[:1] + ADDR[2:])
        out.append((tool, var(**{fld: ""}), None, "arg:address:empty field"))
    for t in INSCOPE:
        for _ in range(6):
            a = random_args(t, st, uid, rng)
            if a is not None:
                out.append((t, a, None, "random"))
    return out


def random_args(tool, st, uid, rng):
    props = SCHEMA[tool]["properties"]
    user = st.user(uid)
    a = {}
    if "order_id" in props:
        r = rng.random()
        if r < 0.90 and user["orders"]:
            a["order_id"] = rng.choice(user["orders"])
        elif r < 0.97:
            a["order_id"] = rng.choice(st.user(other_user(rng, uid))["orders"])
        else:
            a["order_id"] = "#W0000000"
    order = st.order(a.get("order_id"))
    owner = st.user(order["user_id"]) if order else user
    if "user_id" in props:
        r = rng.random()
        a["user_id"] = uid if r < 0.90 else (other_user(rng, uid) if r < 0.98 else "nobody_0000")
    if "payment_method_id" in props:
        r = rng.random()
        if r < 0.85:
            a["payment_method_id"] = rng.choice(sorted(owner["payment_methods"]))
        elif r < 0.95:
            a["payment_method_id"] = rng.choice(sorted(st.user(other_user(rng, uid))["payment_methods"]))
        else:
            a["payment_method_id"] = "gift_card_0000000"
    if "item_ids" in props:
        items = [i["item_id"] for i in order["items"]] if order else []
        r = rng.random()
        if not items or r < 0.05:
            ids = []
        else:
            ids = rng.sample(items, rng.randint(1, len(items)))
            if r < 0.10:
                ids[0] = rng.choice(st.order(rng.choice(ALL_ORDERS))["items"])["item_id"]
            elif r < 0.15:
                ids.append(ids[0])
        a["item_ids"] = ids
        if "new_item_ids" in props:
            new = []
            for i in ids:
                pid = VI.get(i)
                r2 = rng.random()
                if pid is None or r2 < 0.05:
                    new.append(rng.choice(sorted(DB["products"][rng.choice(sorted(DB["products"]))]["variants"])))
                elif r2 < 0.10:
                    new.append(i)
                else:
                    new.append(rng.choice(sorted(DB["products"][pid]["variants"])))
            if new and rng.random() < 0.05:
                new = new[:-1]
            a["new_item_ids"] = new
    if "reason" in props:
        a["reason"] = rng.choice(["no longer needed", "ordered by mistake"]) if rng.random() < 0.9 else "changed my mind"
    if "address1" in props:
        src = st.user(other_user(rng, uid))["address"]
        for k in ADDR:
            a[k] = src[k]
        if rng.random() < 0.05:
            a[rng.choice(["address1", "city", "state", "country", "zip"])] = ""
    if "summary" in props:
        a["summary"] = "The customer asks for help." if rng.random() < 0.9 else ""
    return a


def transitions(split, task_ids, seed):
    """→ iterator of (tool, args, state, refused, changed) over the probes of every gold in-scope step."""
    rng = random.Random(seed)
    statuses = set(DB_STATUSES)
    n_steps = 0
    for ti in task_ids:
        t = TASKS[split][ti]
        st = State()
        for a in t["actions"]:
            if a["name"] in INSCOPE:
                n_steps += 1
                for tool, args, pert, _ in probe_calls(a["name"], a["kwargs"], st, t["user_id"], rng, sorted(statuses)):
                    pst = State(orders=pert[0], users=pert[1], parent=st) if pert else st
                    refused, changed = call(tool, args, pst)
                    yield tool, args, pst, refused, changed
            refused, changed = call(a["name"], a["kwargs"], st)          # the gold step itself, for real
            apply(st, changed)
            for (coll, _), (_, after) in changed.items():
                if coll == "orders":
                    statuses.add(after["status"])
        log(f"{split}: task {ti}, gold steps {n_steps}")


def splits():
    reserved = sorted(random.Random(20261003).sample(range(500), 100))   # kept out for a later stream experiment
    return [i for i in range(500) if i not in set(reserved)], reserved


# ------------------------------------------------------------------------------------------------------------ metrics
def prf(rows):
    """rows: (predicted, refused), predicted in accept / refuse / unknown."""
    ans = [(p, r) for p, r in rows if p != "unknown"]
    tp = sum(1 for p, r in ans if p == "refuse" and r)
    fp = sum(1 for p, r in ans if p == "refuse" and not r)
    fn = sum(1 for p, r in ans if p == "accept" and r)
    ref_all = sum(1 for _, r in rows if r)
    ab = len(rows) - len(ans)
    return {"n": len(rows), "refused": ref_all, "answered": len(ans), "abstain": ab,
            "abstain_share": ab / max(1, len(rows)), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "recall_strict": tp / ref_all if ref_all else None}


A3 = {"precision": 1.0, "recall": 1.0, "refused_heldout": 7626, "answered_refusals": 7602, "heldout": 11763,
      "abstain": 30, "abstain_share": 0.0026, "recall_strict": 0.997, "effects_exact": "4,116 / 4,116 (full entity diffs)",
      "ablations_abstain_share": {"no pair features": 0.026, "no money comparisons": 0.126, "no pair features, no money "
                                  "comparisons": 0.151}}
ABLATIONS = {"no pair features": lambda n: "&" in n,
             "no money comparisons": lambda n: ">=" in n or " > " in n,
             "no pair features, no money comparisons": lambda n: "&" in n or ">=" in n or " > " in n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--taubench", required=True, help="a local clone of github.com/sierra-research/tau-bench")
    ap.add_argument("--quick", action="store_true", help="25 train / 12 test tasks: a smoke run")
    a = ap.parse_args()
    if not (Path(a.taubench) / "tau_bench" / "envs" / "retail").is_dir():
        print(f"{a.taubench} is not a tau-bench clone: git clone https://github.com/sierra-research/tau-bench and pass "
              "its path to --taubench", file=sys.stderr)
        sys.exit(2)
    load_taubench(Path(a.taubench).resolve())
    learn, reserved = splits()
    fit_ids, test_ids = learn, list(range(len(TASKS["test"])))
    if a.quick:
        fit_ids, test_ids = fit_ids[:25], test_ids[:12]
    vocabs = {"full": vocabulary(), **{k: vocabulary(d) for k, d in ABLATIONS.items()}}
    models = {k: ConservativeActionModel(v) for k, v in vocabs.items()}
    train = []
    for tool, args, st, refused, changed in transitions("train", fit_ids, seed=1):
        eff = None if refused else effect_of(tool, args, changed)
        for k, m in models.items():
            m.observe(st, tool, args, not refused, eff)
            if k == "full":
                train.append((tool, m.vocabulary.features(st, tool, args), refused))
    log(f"train: {len(train)} transitions ({sum(r for *_, r in train)} refused) from {len(fit_ids)} tasks", force=True)
    full = models["full"]
    replay = collections.Counter()
    for tool, fv, refused in train:
        p = full._predict_fv(tool, fv).verdict
        replay["unknown" if p == "unknown" else ("right" if (p == "refuse") == refused else "wrong")] += 1
    unexplained = {t: full.summary(t)["unexplained"] for t in full.actions()}
    rows = {k: [] for k in models}
    by_tool = collections.defaultdict(list)
    eff = collections.Counter()
    for tool, args, st, refused, changed in transitions("test", test_ids, seed=2):
        for k, m in models.items():
            p = m.predict(st, tool, args)
            rows[k].append((p.verdict, refused))
            if k == "full":
                by_tool[tool].append((p.verdict, refused))
                if not refused and p.verdict == "accept":
                    if p.effects is None:
                        eff["unknown"] += 1
                    else:
                        eff["exact" if p.effects == effect_of(tool, args, changed) else "wrong"] += 1
    e_ans = eff["exact"] + eff["wrong"]
    res = {"what": "solvi.core.knowledge.ConservativeActionModel on τ-bench retail probes (no LLM)",
           "quick": a.quick, "train_tasks": len(fit_ids), "reserved_tasks": len(reserved), "test_tasks": len(test_ids),
           "train_transitions": len(train), "train_refused": sum(r for *_, r in train),
           "replay_training_transitions": dict(replay), "unexplained_training_refusals": unexplained,
           "heldout": prf(rows["full"]), "heldout_by_tool": {t: prf(v) for t, v in sorted(by_tool.items())},
           "status_effect": {"exact": eff["exact"], "wrong": eff["wrong"], "unknown": eff["unknown"],
                             "exact_share": eff["exact"] / e_ans if e_ans else None,
                             "note": "the touched order's new status and whether a user record changed — narrower "
                                     "than the research run's full entity diffs (amounts, refunds, item lists)"},
           "ablations_heldout": {k: {"unexplained_training_refusals": sum(models[k].summary(t)["unexplained"]
                                                                          for t in models[k].actions()),
                                     "heldout": prf(rows[k])} for k in ABLATIONS},
           "research_run_for_comparison": A3, "seconds": round(time.time() - T0, 1)}
    out = HERE / "results" / ("taubench_action_model_quick.json" if a.quick else "taubench_action_model.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    h = res["heldout"]
    log(f"held-out {h['n']} transitions, {h['refused']} refused: precision {h['precision']}, recall {h['recall']}, "
        f"strict recall {h['recall_strict']:.4f}, abstain {h['abstain']} ({h['abstain_share']:.2%}); status effect "
        f"{eff['exact']} / {e_ans} exact, {eff['unknown']} unknown; ablations "
        + ", ".join(f"{k}: {v['heldout']['abstain_share']:.1%} abstain, precision {v['heldout']['precision']}"
                    for k, v in res["ablations_heldout"].items()) + f" → {out}", force=True)


if __name__ == "__main__":
    main()
