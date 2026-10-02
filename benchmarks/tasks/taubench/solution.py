"""τ-bench retail with solvi: the baseline's agent with a solvi Guard between the model and the tools.

The model (`openai/gpt-oss-120b`, the policy as the system prompt, one action per turn) proposes a call; the guard
checks it against the policy's hard rules and only then makes it:

- grounding: the values of a call (an order id, a zip, an address) must stand in the conversation — the customer's
  words or a tool's result — not be made up;
- policies over the given fact `known` (what the tools returned so far): the customer is authenticated, the order is
  theirs and in the right status, the items are in the order, a new item is an available variant of the same product,
  the payment method is the customer's;
- `guard.require_confirmation`: a change goes ahead only when a message of the agent named its values and the
  customer's next message accepted it explicitly;
- `once=True` on every change; a step rule: a change the customer has just accepted is made before the next message.

A refused call goes back to the model as the tool's answer (`GuardDecision.feedback()`: the reasons and what to do
next), and every tool's description lists its checks (`guard.described`). Every proposal and every message to the
customer is a stored decision with its trace, hash-chained; at the end of a task every stored decision is replayed.

    python taubench/solution.py [--n 5] [--cache DIR]       # the 30 test tasks -> runs/solution.jsonl (resumes)
"""
import copy
import json
import re
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.llm import DATA, options, parse, read_jsonl  # noqa: E402
from harness import chat, run_task  # noqa: E402
from score import score  # noqa: E402

import numpy  # noqa: E402,F401 - imported once before the threads: a concurrent first import fails
from solvi.agents import Guard  # noqa: E402
from solvi.agents.confirm import accepts  # noqa: E402

MODEL = "openai/gpt-oss-120b"
ORDER_WRITES = ["cancel_pending_order", "modify_pending_order_address", "modify_pending_order_payment",
                "modify_pending_order_items", "return_delivered_order_items", "exchange_delivered_order_items"]
CHANGES = ORDER_WRITES + ["modify_user_address"]                       # the policy's "consequential actions"
PENDING_ONLY = ORDER_WRITES[:4]
DELIVERED_ONLY = ORDER_WRITES[4:]
ITEM_SWAPS = ["modify_pending_order_items", "exchange_delivered_order_items"]
WITH_PAYMENT = ["modify_pending_order_payment", "modify_pending_order_items", "return_delivered_order_items",
                "exchange_delivered_order_items"]
NEED_AUTH = CHANGES + ["get_user_details", "get_order_details", "get_product_details", "list_all_product_types"]
RESPOND = "respond"


# ---------------------------------------------------------------------------------------------- reading the conversation
def order_id_matcher(value, text):
    """An order id as the customer writes it: with or without '#', in any letter case."""
    core = re.escape(value.lstrip("#"))
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<![A-Za-z0-9])#?{core}(?![A-Za-z0-9])", text, re.I)] if core else []


def loose_matcher(value, text):
    """A street, a city: the words in any letter case, not inside a longer word."""
    v = re.escape(value.strip())
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<![A-Za-z0-9]){v}(?![A-Za-z0-9])", text, re.I)] if v else []


ASKS_TO_CONFIRM = re.compile(r"\bconfirm\b|\byes\b|\bproceed\b|\bgo ahead\b", re.I)    # the step rule's own (not the policy's)
NAMES_A_CHANGE = re.compile(r"#w\d{5,}|\baddress\b", re.I)
SPACES = re.compile(r"\s+")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def norm(text):
    """A message as the naming helpers read it: NFKC, one space, lower case (lengths may change: helpers say yes / no)."""
    return SPACES.sub(" ", unicodedata.normalize("NFKC", text).translate(DASHES)).lower()


def confirmation_pending(conversation, conversation_roles):
    """The customer's last message accepts (solvi's `accepts`) a message of the agent that asked to confirm a change."""
    t = [(r, conversation[a:b]) for a, b, r, *_ in conversation_roles if r in ("user", "assistant")]
    if len(t) < 2 or t[-1][0] != "user" or t[-2][0] != "assistant":
        return False
    return accepts(t[-1][1]) and bool(ASKS_TO_CONFIRM.search(t[-2][1]) and NAMES_A_CHANGE.search(t[-2][1]))


def names_payment(text, pm_id, state, order=None):
    """`text` is normalised (norm) in all names_* helpers."""
    pid = pm_id.lower()
    if pid in text or pid.rsplit("_", 1)[-1] in text:
        return True
    pms = (state.get("profile") or {}).get("payment_methods") or {}
    pm = pms.get(pm_id, {})
    kind = pm.get("source") or pid.rsplit("_", 1)[0]
    if pm.get("last_four") and pm["last_four"] in text:
        return True
    if order and order["payment_history"] and order["payment_history"][0]["payment_method_id"] == pm_id \
            and re.search(r"original (payment|method|card)|same (payment|method|card)", text):
        return True
    words = {"gift_card": ["gift card", "giftcard"], "paypal": ["paypal"], "credit_card": ["credit card"]}.get(kind, [])
    if pm.get("brand"):
        words = words + [pm["brand"].lower()]
    alike = [p for p in pms.values() if p.get("source") == kind]
    return any(w in text for w in words) and len(alike) <= 1


def names_item(text, item, order):
    if item["item_id"] in text:
        return True
    same_name = [i for i in order["items"] if i["name"] == item["name"]]
    if len(same_name) == 1:
        return item["name"].lower() in text
    return all(str(v).lower() in text for v in item["options"].values())


def names_new_item(text, new_id, variant, old):
    if new_id in text:
        return True
    changed = [str(v).lower() for k, v in variant["options"].items() if old["options"].get(k) != v]
    return bool(changed) and all(v in text for v in changed)


def whole(text):
    return [(0, len(text))]


def payment_named(value, text, facts):
    """A payment method in a message: its id, its last four digits, "original payment" (the order's), or its kind."""
    known = facts["known"]
    order = next((o for o in (known.get("orders") or {}).values()
                  if o.get("payment_history") and o["payment_history"][0]["payment_method_id"] == value), None)
    return whole(text) if names_payment(norm(text), value, known, order) else []


def order_with_item(known, item_id):
    return next((o for o in (known.get("orders") or {}).values() if any(i["item_id"] == item_id for i in o["items"])),
                None)


def item_named(value, text, facts):
    """An item of an order in a message: its id, its name (when unique in the order), or its options."""
    order = order_with_item(facts["known"], value)
    if order is None:
        return whole(text) if value in text else []
    item = next(i for i in order["items"] if i["item_id"] == value)
    return whole(text) if names_item(norm(text), item, order) else []


def new_item_named(value, text, facts):
    """A new variant in a message: its id, or the options in which it differs from the order's item of that product."""
    known = facts["known"]
    if value in text:
        return whole(text)
    for pid, prod in (known.get("products") or {}).items():
        variant = (prod.get("variants") or {}).get(value)
        if variant is None:
            continue
        olds = [i for o in (known.get("orders") or {}).values() for i in o["items"] if i["product_id"] == pid]
        return whole(text) if any(names_new_item(norm(text), value, variant, o) for o in olds) else []
    return []


def order_of(state, order_id):
    return (state.get("orders") or {}).get(order_id)


def old_items(state, order_id, item_ids):
    """The order's items for the given ids (None when one is not in the order as it was last seen)."""
    order = order_of(state, order_id)
    if not order:
        return None
    left, out = list(order["items"]), []
    for i in item_ids:
        hit = next((x for x in left if x["item_id"] == i), None)
        if hit is None:
            return None
        left.remove(hit)
        out.append(hit)
    return out


def variant_of(state, old, new_id):
    return (((state.get("products") or {}).get(old["product_id"]) or {}).get("variants") or {}).get(new_id)


# ------------------------------------------------------------------------------------------------------- the app's state
def absorb(state, name, kwargs, obs):
    """What the tools returned, kept as the given fact `known` of every later decision."""
    if obs.startswith("Error"):
        if name.startswith("find_user_id"):
            state["auth_failed"] = state.get("auth_failed", 0) + 1
        return
    if name.startswith("find_user_id"):
        state.setdefault("user_id", obs.strip())             # one customer per conversation: the first one found
        return
    try:
        data = json.loads(obs)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(data, dict):
        return
    if name in ("get_user_details", "modify_user_address"):
        if kwargs.get("user_id") == state.get("user_id"):
            state["profile"], state["profile_fresh"] = data, True
    elif name == "get_product_details":
        state.setdefault("products", {})[data["product_id"]] = data
    elif "order_id" in data and "status" in data:            # get_order_details and every order change return the order
        state.setdefault("orders", {})[data["order_id"]] = data
        if name != "get_order_details":
            state["profile_fresh"] = False                  # a change may have moved a gift card's balance


# ------------------------------------------------------------------------------------------------------------- the guard
HUMAN = re.compile(r"\b(human|representative|supervisor|manager|real person|live agent)\b", re.I)


def build_guard(hs, store, state):
    """A Guard over the harness session's tools → (guard, the tools' definitions for the model, each with its checks
    listed in the description)."""
    guard = Guard(storage=store, fact_names={"known": dict})
    tools = hs.tools
    policy = guard.policy                                    # the model reads the reasons through guard.described

    def runner(name):
        def run(**kwargs):
            obs = hs.call(name, **kwargs)
            absorb(state, name, kwargs, obs)
            if obs.startswith("Error"):
                raise RuntimeError(obs)                      # solvi records the call as failed (not "made")
            return obs
        run.__name__ = name
        return run

    grounded = {
        "find_user_id_by_email": {"email": loose_matcher},
        "find_user_id_by_name_zip": {"first_name": loose_matcher, "last_name": loose_matcher, "zip": "token"},
        "cancel_pending_order": {"order_id": order_id_matcher},
        "modify_pending_order_address": {"order_id": order_id_matcher, "address1": loose_matcher, "city": loose_matcher,
                                         "zip": "token"},
        "modify_pending_order_payment": {"order_id": order_id_matcher, "payment_method_id": "token"},
        "modify_pending_order_items": {"order_id": order_id_matcher, "item_ids": "token", "new_item_ids": "token",
                                       "payment_method_id": "token"},
        "return_delivered_order_items": {"order_id": order_id_matcher, "item_ids": "token", "payment_method_id": "token"},
        "exchange_delivered_order_items": {"order_id": order_id_matcher, "item_ids": "token", "new_item_ids": "token",
                                           "payment_method_id": "token"},
        "modify_user_address": {"address1": loose_matcher, "city": loose_matcher, "zip": "token"}}
    for info in tools:
        f = info["function"]
        n = f["name"]
        guard.tool(runner(n), name=n, schema=f["parameters"], description=f["description"], ground=grounded.get(n, ()),
                   ground_from=("user",) if n.startswith("find_user_id") else ("user", "tool"),
                   authorize=False, once=n in CHANGES)
    guard.declare(RESPOND, description="Send a message to the customer.", authorize=False,    # checked and recorded;
                  schema={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]})   # sent by the loop

    # --- who: authentication, one customer per conversation
    @policy(NEED_AUTH)
    def customer_authenticated(known: dict) -> bool:
        """The customer must be authenticated first: their user id located with find_user_id_by_email or find_user_id_by_name_zip (even if they gave a user id)."""
        return bool(known.get("user_id"))

    @policy(["get_user_details", "modify_user_address"])
    def own_profile_only(user_id: str, known: dict) -> bool:
        """The user id must be the authenticated customer's: requests about any other user are denied."""
        return not known.get("user_id") or user_id == known["user_id"]

    @policy(ORDER_WRITES)
    def order_belongs_to_customer(order_id: str, known: dict) -> bool:
        """The order must belong to the authenticated customer."""
        order = order_of(known, order_id)
        return not order or not known.get("user_id") or order["user_id"] == known["user_id"]

    @policy("transfer_to_human_agents")
    def transfer_only_when_out_of_scope(known: dict, user_request: str) -> bool:
        """Transfer only if the request cannot be handled with your tools: first authenticate the customer and look at their profile and orders with get_user_details (an order they cannot name is listed there)."""
        if not known.get("user_id"):                    # nobody to look up: the customer could not be found, or asks for a person
            return bool(known.get("auth_failed")) or bool(HUMAN.search(user_request or ""))
        return known.get("profile") is not None

    # --- what state the order is in (as the tools last returned it)
    @policy(ORDER_WRITES)
    def order_status_checked(order_id: str, known: dict) -> bool:
        """The order's status must be checked first: get_order_details for this order id (with the '#') earlier in the conversation."""
        return order_of(known, order_id) is not None

    @policy(PENDING_ONLY)
    def order_is_pending(order_id: str, known: dict) -> bool:
        """The order's status must be 'pending' (after its items were modified once it is 'pending (item modified)' and cannot be modified or cancelled again)."""
        order = order_of(known, order_id)
        return not order or order["status"] == "pending"

    @policy(DELIVERED_ONLY)
    def order_is_delivered(order_id: str, known: dict) -> bool:
        """The order's status must be 'delivered' (after a return or an exchange is requested, no further one)."""
        order = order_of(known, order_id)
        return not order or order["status"] == "delivered"

    @policy(["modify_pending_order_address", "modify_user_address"])
    def address_is_complete(address1: str, city: str, state: str, country: str, zip: str) -> bool:
        """The new address must be complete: street, city, state, country and zip, none of them empty."""
        return all(str(x).strip() for x in (address1, city, state, country, zip))

    # --- the items
    @policy(ITEM_SWAPS + ["return_delivered_order_items"])
    def items_are_in_the_order(order_id: str, item_ids: list, known: dict) -> bool:
        """Every item id must be an item of this order, as get_order_details returned it (an item id is not a product id)."""
        return order_of(known, order_id) is None or old_items(known, order_id, item_ids) is not None

    @policy(ITEM_SWAPS)
    def one_new_item_per_item(item_ids: list, new_item_ids: list) -> bool:
        """item_ids and new_item_ids must have the same length: one new item for each item changed."""
        return len(item_ids) == len(new_item_ids) and len(item_ids) > 0

    @policy(ITEM_SWAPS)
    def new_item_product_looked_up(order_id: str, item_ids: list, known: dict) -> bool:
        """The product of each item to be changed must be looked up with get_product_details (its product_id is in the order) before the new item is chosen."""
        old = old_items(known, order_id, item_ids)
        return old is None or all(o["product_id"] in (known.get("products") or {}) for o in old)

    @policy(ITEM_SWAPS)
    def new_item_same_product_and_available(order_id: str, item_ids: list, new_item_ids: list, known: dict) -> bool:
        """Each new item must be an available variant of the same product as the item it replaces, with different options (no change of product type; not the same item)."""
        old = old_items(known, order_id, item_ids)
        if old is None or len(old) != len(new_item_ids):
            return True
        for o, n in zip(old, new_item_ids):
            if o["product_id"] not in (known.get("products") or {}):
                continue
            v = variant_of(known, o, n)
            if v is None or not v.get("available") or n == o["item_id"]:
                return False
        return True

    # --- the payment method
    @policy(WITH_PAYMENT)
    def payment_method_is_the_customers(order_id: str, payment_method_id: str, known: dict) -> bool:
        """The payment method must be one of the customer's own (see get_user_details) or the order's original one."""
        order, prof = order_of(known, order_id), known.get("profile")
        if prof is not None:
            return payment_method_id in (prof.get("payment_methods") or {})
        return order is None or payment_method_id in [p["payment_method_id"] for p in order["payment_history"]]

    @policy("return_delivered_order_items")
    def refund_to_original_method_or_gift_card(order_id: str, payment_method_id: str, known: dict) -> bool:
        """The refund must go to the order's original payment method or to an existing gift card of the customer."""
        order = order_of(known, order_id)
        return order is None or payment_method_id.startswith("gift_card") \
            or payment_method_id == order["payment_history"][0]["payment_method_id"]

    @policy("modify_pending_order_payment")
    def new_payment_differs_and_covers(order_id: str, payment_method_id: str, known: dict) -> bool:
        """The new payment method must differ from the original one, and a gift card must have enough balance for the order's total."""
        order = order_of(known, order_id)
        if order is None or not order["payment_history"]:
            return True
        first = order["payment_history"][0]
        if first["payment_method_id"] == payment_method_id:
            return False
        pm = ((known.get("profile") or {}).get("payment_methods") or {}).get(payment_method_id)
        if pm and pm.get("source") == "gift_card" and known.get("profile_fresh"):
            return pm.get("balance", 0) >= first["amount"]
        return True

    @policy(ITEM_SWAPS)
    def gift_card_covers_price_difference(order_id: str, item_ids: list, new_item_ids: list, payment_method_id: str,
                                          known: dict) -> bool:
        """A gift card used for the price difference must have enough balance to cover it."""
        pm = ((known.get("profile") or {}).get("payment_methods") or {}).get(payment_method_id)
        old = old_items(known, order_id, item_ids)
        if not pm or pm.get("source") != "gift_card" or not known.get("profile_fresh") or old is None \
                or len(old) != len(new_item_ids):
            return True
        diff = 0.0
        for o, n in zip(old, new_item_ids):
            v = variant_of(known, o, n)
            if v is None:
                return True
            diff += v["price"] - o["price"]
        return pm.get("balance", 0) >= diff

    # --- the customer's explicit confirmation of the listed details: solvi's "the user confirmed this"
    guard.require_confirmation("cancel_pending_order", ["order_id", "reason"], match={"order_id": "id"})
    guard.require_confirmation("return_delivered_order_items", ["order_id", "item_ids", "payment_method_id"],
                               match={"order_id": "id", "item_ids": item_named, "payment_method_id": payment_named},
                               reads=["known"])
    guard.require_confirmation(ITEM_SWAPS, ["item_ids", "new_item_ids", "payment_method_id"],
                               match={"item_ids": item_named, "new_item_ids": new_item_named,
                                      "payment_method_id": payment_named}, reads=["known"])
    guard.require_confirmation("modify_pending_order_payment", ["payment_method_id"],
                               match={"payment_method_id": payment_named}, reads=["known"])
    guard.require_confirmation(["modify_pending_order_address", "modify_user_address"], ["address1", "zip"])

    # --- the confirmed call comes before the next message (a step rule, not a rule of the policy text)
    @policy([RESPOND, "transfer_to_human_agents"])
    def confirmed_change_is_made_first(conversation: str, conversation_roles: list, known: dict) -> bool:
        """A change the customer has just said yes to must be made first: make the confirmed tool call before any other message or a transfer (if a detail is still missing or nothing was confirmed, send your message again unchanged)."""
        turn = known.get("turn") or {}
        return bool(turn.get("changes_tried") or turn.get("nudged")) \
            or not confirmation_pending(conversation, conversation_roles)

    shown = []
    for info in tools:                                       # the model reads each tool's checks in its description
        f = dict(info["function"])
        lines = []
        if grounded.get(f["name"]):
            lines.append("The values of " + ", ".join(grounded[f["name"]]) + " must be written in the conversation ("
                         + ("by the customer" if f["name"].startswith("find_user_id") else "by the customer or in a tool's result")
                         + "), not made up.")
        if f["name"] in ITEM_SWAPS:
            lines.append("Before the customer confirms, remind them to confirm they have provided all items to be changed "
                         "(this call can be made only once per order).")
        if f["name"] == "modify_pending_order_items":
            lines.append("If the customer also wants the address or the payment method of this order changed, make that "
                         "change before this call: afterwards the order cannot be modified.")
        if f["name"] in CHANGES:
            lines.append("After the customer's yes, make this call at once, before any other message.")
        if f["name"] in guard.tools:
            f["description"] = guard.described(f["name"], f["description"])        # the policies' reasons, solvi's
            if lines:
                f["description"] += "\n" + "\n".join("- " + x for x in lines)
        shown.append({"type": "function", "function": f})
    return guard, shown


# ------------------------------------------------------------------------------------------------------------- the agent
def make_agent(logs, stores, model=MODEL, max_model_calls=70, max_blocks_in_a_row=6):
    def agent(hs):
        tid = f"retail_{hs.split}_{hs.index}"
        store = stores / f"{tid}.jsonl"
        for old in (store, store.with_name(store.name + ".head")):
            old.unlink(missing_ok=True)
        state = {"turn": {"changes_tried": 0, "nudged": False}}
        guard, tools = build_guard(hs, store, state)
        session = guard.session([("user", hs.first)], facts={"known": copy.deepcopy(state)})
        messages = [{"role": "system", "content": hs.wiki}, {"role": "user", "content": hs.first}]
        n_calls = blocks = 0
        try:
            while not hs.done and n_calls < max_model_calls and blocks < max_blocks_in_a_row:
                m = chat(model, messages, tag="taubench/solution", max_tokens=3000, reasoning="low", full=True, tools=tools)
                n_calls += 1
                calls = (m.get("tool_calls") or [])[:1]
                if calls:                                    # a tool call: checked, and made only when allowed
                    f = calls[0]["function"]
                    try:
                        kwargs = json.loads(f["arguments"] or "{}")
                    except json.JSONDecodeError:
                        kwargs = {}
                    session.facts["known"] = copy.deepcopy(state)
                    d = session.call({"name": f["name"], "arguments": kwargs, "id": calls[0]["id"]})
                    if f["name"] in CHANGES:
                        state["turn"]["changes_tried"] += 1
                    if d.executed:
                        obs, blocks = (d.error if d.error else d.result), 0
                    else:
                        obs, blocks = d.feedback()[0]["content"], blocks + 1       # the reasons and what to do next
                        if f["name"] == "transfer_to_human_agents":
                            state["turn"]["nudged"] = True
                        if blocks >= 3:
                            obs += " Do not repeat this call as it is: follow the reasons, or tell the customer what you can and cannot do."
                    messages += [{"role": "assistant", "content": m["content"] or "", "tool_calls": calls},
                                 {"role": "tool", "tool_call_id": calls[0]["id"], "name": f["name"], "content": str(obs)}]
                else:                                        # a message to the customer: checked by the step rule
                    text = m["content"] or ""
                    session.facts["known"] = copy.deepcopy(state)
                    d = session.check({"name": RESPOND, "arguments": {"content": text}})
                    if d.outcome != "allow":                 # the message is not sent; the model reads why
                        state["turn"]["nudged"] = True
                        blocks += 1
                        messages += d.feedback()
                        continue
                    obs = hs.say(text)
                    session.add("assistant", text)
                    session.add("user", obs)
                    state["turn"] = {"changes_tried": 0, "nudged": False}
                    blocks = 0
                    messages += [{"role": "assistant", "content": text}, {"role": "user", "content": obs}]
        finally:
            stored = list(guard.storage.iter())
            logs[tid] = {"decisions": len(stored), "blocked": sum(((r.meta or {}).get("guard") or {}).get("outcome") != "allow"
                                                                  for r in stored),
                         "replayed": len(stored) - len(guard.replay_all()), "chain_ok": guard.storage.verify()["ok"]}
    return agent


def main():
    p = options(__doc__)
    p.add_argument("--n", type=int, default=None, help="the first n of the 30 test tasks")
    p.add_argument("--out", default=None)
    a = parse(p)
    ids = json.loads((DATA / "taubench/prepared/eval_30_ids.json").read_text())[: a.n]
    out = Path(a.out or HERE / "runs" / "solution.jsonl")
    stores = out.with_suffix("")                             # runs/solution/<task>.jsonl: each task's stored decisions
    stores.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    logs, lock = {}, threading.Lock()

    def one(tid):
        row = run_task(int(tid.rsplit("_", 1)[1]), make_agent(logs, stores))
        row["guard"] = logs.get(tid)
        with lock:                                           # written as it goes: a killed run resumes
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(tid, "reward", row["reward"], "steps", row["steps"], row["guard"], row.get("agent_error", ""), flush=True)

    with ThreadPoolExecutor(5) as ex:
        list(ex.map(one, [i for i in ids if i not in done]))
    rows = read_jsonl(out)
    guard = {k: sum((r.get("guard") or {}).get(k, 0) for r in rows) for k in ("decisions", "blocked", "replayed")}
    print(json.dumps({**score(out), "guard": {**guard, "chains_ok": sum(bool((r.get("guard") or {}).get("chain_ok"))
                                                                         for r in rows)}}, indent=1))


if __name__ == "__main__":
    main()
