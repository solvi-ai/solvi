"""Did the user accept this very call? "Confirm before a consequential action" as a check of the guard.

A rule like "list the action's details and get the user's explicit yes before you change their order" is a relation
between three things: the call's arguments, a message of the assistant that proposed them, and the user's next message
that accepted it. `guard.require_confirmation` declares it for a tool:

    guard.require_confirmation(["cancel_order", "refund"])                       # every argument of the call
    guard.require_confirmation("change_address", arguments=["street", "zip"])    # only these must be named
    guard.require_confirmation("exchange_items", match={"item_ids": item_named}, reads=["known"])   # your matcher

A call of the tool is allowed by this check only when some message of the assistant names every required argument's
value (a string by the "nocase" matcher — case, Unicode spaces, typographic dashes and quotes aside —, an order id with
"id" if you say so, a number as a number token, a list item by item; or your matcher) and the user's next message
(tool outputs in between are skipped) accepts it explicitly. A value the user wrote in the accepting message itself
counts as proposed too ("yes, to my PayPal"). The proposal and the acceptance are quoted in the decision's evidence;
when the check fails, its reason says what was missing — no accepted proposal, or which values the accepted one does
not name.

What counts as an explicit acceptance (ACCEPT, REFUSE_START, RETRACT, RESERVE, `accepts`): a yes word or phrase in
English or Russian ("yes", "go ahead", "please proceed", "confirmed", "that's correct", "да", "подтверждаю",
"оформляйте", ...) that no negation shortly before it turns around ("not correct", "don't proceed", "не подтверждаю"),
in a message that does not open with a refusal ("no", "wait", "нет") and takes nothing back anywhere ("actually",
"instead", "hold on", "changed my mind", "вместо", "передумал"); the first sentence that says yes decides, and a
reservation in that sentence makes it conditional, so not an acceptance ("yes, but not the blue one", "да, но ...").
A reservation in a later sentence is about something else ("Yes, please proceed. But could I also get a coupon?" is
an acceptance). The text is read NFKC-normalised and case-folded. The patterns are deliberately narrow: "no, go
ahead with the other one" is not an acceptance; a user who accepts in other words ("let's roll") is asked again. It reads the user's words only: it does not judge whether the proposal was a good one (a wrong
choice the user approves is approved), and it cannot tell a user from someone typing as them."""
from __future__ import annotations

import inspect
import json
import re

# a yes word or phrase (over NFKC-normalised, case-folded text, typographic quotes read as "'"); a word that is often
# something else ("sure" in "I'm sure", "correct" in "to correct it") counts only at the start of a sentence
ACCEPT = (r"\b(yes|yeah|yep|yup|confirm|confirmed|confirming|go ahead|go for it|proceed|please do|do it|do that|"
          r"(that|it|this|everything)('s| is| looks| sounds) (all )?(right|correct|fine|good|perfect)|all (correct|good)|"
          r"sounds (right|good)|looks (right|correct|good)|agreed|approved?|let's do it|let's proceed)\b",
          r"^\W*((um|uh|oh|well|hmm)\W+)*(sure|ok|okay|alright|all right|of course|correct)\b",   # only first in a sentence
          r"(?<!\w)(да|ага|угу|конечно|давай|давайте|подтверждаю|подтверждаем|согласен|согласна|согласны|хорошо|ладно|"
          r"ок|окей|верно|правильно|продолжай|продолжайте|оформляй|оформляйте|делай|делайте|вперёд|вперед)(?!\w)")
# a message that opens with one of these does not accept
REFUSE_START = r"^\W*(no|nope|nah|not yet|wait|hold on|hang on|stop|нет|неа|подожди|подождите|погоди|погодите|стоп)\b"
# a retraction anywhere in the message: the user takes something back or changes it
RETRACT = (r"\b(instead|actually|wait|hold on|hang on|not yet|never ?mind|on second thought|changed? my mind)\b",
           r"(?<!\w)(вместо|подожди|подождите|погоди|погодите|стоп|передумал|передумала|не надо)(?!\w)")
# a reservation in the sentence of the yes: the yes is conditional ("yes, but not the blue one")
RESERVE = (r"\b(but|however|though|although|unless|except)\b", r"(?<!\w)(но|однако|хотя|если|кроме)(?!\w)")
_SENTENCE = re.compile(r"[^.!?;\n…]+")
# a negation within a few words before a yes word turns it around ("that's not correct", "don't proceed")
NEGATION = r"(?<!\w)(not|don't|dont|do not|never|isn't|is not|не|нет)(?!\w)"
_NEAR = 3                                                 # words between a negation and the yes word it negates


def _plain(text):
    from ..perturb import normalize
    from .guard import _TYPO
    return normalize(text, confusables=False)[0].casefold().translate(_TYPO)


def accepts(text) -> bool:
    """Is a user's message an explicit acceptance of what the assistant proposed? (see the module docs)"""
    if not isinstance(text, str) or not text.strip():
        return False
    t = _plain(text)
    if re.search(REFUSE_START, t) or any(re.search(p, t) for p in RETRACT):
        return False
    for sent in _SENTENCE.finditer(t):                    # the first sentence that says yes decides
        st = sent.group()
        for p in ACCEPT:
            for m in re.finditer(p, st):
                before = st[:m.start()].split()[-_NEAR:]
                if not any(re.fullmatch(NEGATION, w.strip(",:\"'()")) for w in before):
                    return not any(re.search(r, st) for r in RESERVE)
    return False


def accepted_proposals(conversation, conversation_roles, last=None) -> list:
    """The assistant's messages the user accepted → [(proposal index, acceptance index)] into conversation_roles, the
    latest first: each user message that `accepts`, paired with the assistant's message before it (tool and system
    messages in between are skipped; a user message in between breaks the pair). last: only the user's last N messages
    count as acceptances (None: all)."""
    roles = [(x[0], x[1], x[2]) for x in conversation_roles]
    users = [i for i, x in enumerate(roles) if x[2] == "user"]
    recent = set(users[-int(last):]) if last else None
    out, proposal = [], None
    for i, (s, e, r) in enumerate(roles):
        if r == "assistant":
            proposal = i
        elif r == "user":
            if proposal is not None and (recent is None or i in recent) and accepts(conversation[s:e]):
                out.append((proposal, i))
            proposal = None
    return out[::-1]


def _required(call_arguments, args):
    """The arguments the proposal must name → {argument: [values]}: `args`, else every argument whose value is text, a
    number or a list of them (not a bool, not empty)."""
    from decimal import Decimal
    from enum import Enum

    def nameable(v):
        return isinstance(v, (str, int, float, Decimal, Enum)) and not isinstance(v, bool) and not (
            isinstance(v, str) and not v.strip())
    out = {}
    for a, v in call_arguments.items():
        if args is not None and a not in args:
            continue
        items = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
        items = [x for x in items if nameable(x)]
        if items:
            out[a] = items
        elif args is not None and v not in (None, "", [], ()):
            out[a] = [v]                                  # named explicitly but not nameable: it cannot be found
    return out


def _matcher_text(m):
    from ..provenance import code_fingerprint
    return m if isinstance(m, str) else f"callable {getattr(m, '__qualname__', type(m).__name__)} {code_fingerprint(m)}"


def confirmation_fn(spec, matchers=None):
    """The computed fact `confirmation` of a tool's checks: (call_arguments, conversation, conversation_roles, and the
    facts `spec["reads"]` names) → {"confirmed", "proposal": [text, start, end] | None, "accepted": [...] | None,
    "found": {argument: [[text, start, end, role]]}, "missing": [...], "why"}."""
    matchers = dict(matchers or {})                       # argument → a callable matcher (its code is in `spec`)
    rules = json.loads(spec)
    reads = list(rules.get("reads") or ())

    def confirmation(call_arguments, conversation, conversation_roles, **facts):
        _ = spec
        from .guard import _occurrences
        need = _required(call_arguments, rules.get("args"))
        pairs = accepted_proposals(conversation, conversation_roles, rules.get("last"))
        if not pairs:
            return {"confirmed": False, "proposal": None, "accepted": None, "found": {}, "missing": sorted(need),
                    "why": "the user has not explicitly accepted any message of yours (\"yes\", \"go ahead\", "
                           "\"please proceed\")" + (f" in their last {rules['last']} messages" if rules.get("last") else "")}
        best = None
        for p, u in pairs:
            found, missing = {}, []
            for a, items in need.items():
                quotes = []
                for item in items:
                    hit = None
                    for i in (p, u):
                        s, e, role = conversation_roles[i][0], conversation_roles[i][1], conversation_roles[i][2]
                        text = conversation[s:e]
                        m = matchers.get(a, rules["match"].get(a, "nocase"))
                        if callable(m):
                            occ = m(item, text, {k: facts[k] for k in reads}) if _arity(m) >= 3 else m(item, text)
                            occ = [(int(x), int(y)) for x, y in (occ or ()) if 0 <= int(x) < int(y) <= len(text)]
                        else:
                            occ = _occurrences(item, text, m)
                        if occ:
                            x, y = occ[0]
                            hit = [text[x:y], s + x, s + y, role]
                            break
                    if hit is None:
                        missing.append(f"{a}={_short(item)}")
                    else:
                        quotes.append(hit)
                if quotes:
                    found[a] = quotes
            if best is None or len(missing) < len(best[3]):
                best = (p, u, found, missing)
            if not missing:
                break
        p, u, found, missing = best
        ps, pe = conversation_roles[p][0], conversation_roles[p][1]
        us, ue = conversation_roles[u][0], conversation_roles[u][1]
        out = {"confirmed": not missing, "proposal": [conversation[ps:pe], ps, pe],
               "accepted": [conversation[us:ue], us, ue], "found": found, "missing": missing}
        out["why"] = None if not missing else (
            "no message of yours that the user accepted names " + ", ".join(missing)
            + f" (the closest, accepted with {_short(conversation[us:ue], 40)}, does not)")
        return out
    params = [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
              for n in ("call_arguments", "conversation", "conversation_roles", *reads)]
    confirmation.__signature__ = inspect.Signature(params)
    return confirmation


def _arity(f):
    try:
        return len(inspect.signature(f).parameters)
    except (TypeError, ValueError):
        return 2


def _short(v, n=60):
    from .guard import _short as short
    return short(v, n)


def user_confirmed(confirmation) -> bool:
    """The user explicitly accepted a message of yours that names this call's values."""
    return bool(confirmation["confirmed"])


def confirm_spec(tool, arguments=None, match=None, reads=(), last=None):
    """A tool's confirmation rule → (spec JSON, {argument: callable matcher}) — checked against the tool's arguments."""
    from .guard import MATCHERS
    args = None if arguments is None else [arguments] if isinstance(arguments, str) else list(arguments)
    match = dict(match or {})
    known = set(tool.arguments)
    unknown = sorted(set(args or ()) - known) + sorted(set(match) - known)
    if unknown:
        raise ValueError(f"tool {tool.name}: require_confirmation names no argument: {unknown} (its arguments: "
                         f"{sorted(known)})")
    bad = {a: m for a, m in match.items() if not callable(m) and m not in MATCHERS}
    if bad:
        raise ValueError(f"tool {tool.name}: confirmation matchers are {', '.join(MATCHERS)} or a callable, not {bad}")
    if last is not None and (isinstance(last, bool) or not isinstance(last, int) or last < 1):
        raise ValueError(f"tool {tool.name}: last is a number of the user's last messages (1, 2, ...) or None")
    spec = {"args": args, "match": {a: _matcher_text(m) for a, m in match.items()}, "last": last,
            "reads": sorted(reads)}
    return json.dumps(spec, sort_keys=True), {a: m for a, m in match.items() if callable(m)}
