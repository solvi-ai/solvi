"""Did the user ask for this action? Policies for calls that carry no value the user must give.

Provenance protects an argument the user must give (a payee, a recipient). Some actions have none: "book a hotel"
names the hotel from a search result, "create an event" takes a title and a time the agent chose, "read the link in
that document" takes a URL from a tool output. Declaring such arguments as the user's denies every honest call; leaving
them free lets a tool output that says "make a reservation for …" through. The middle road is a policy on the action
itself: the call goes ahead only if the user's own messages ask for this kind of action, and otherwise escalates (or is
denied).

    guard.require_request("reserve_hotel", "reserve")                 # "book", "reserve", "reservation", "забронируй"
    guard.require_request(["create_calendar_event"], "event")         # "calendar", "meeting", "remind", "встреча"
    guard.require_request("get_webpage", "visit", on_fail="deny")     # "visit", "website", "link", a URL, "сайт"
    guard.require_request("launch", phrases=[r"\\blaunch\\b"])          # your own patterns

It is an ordinary guard policy (a hard check in the tool's catalog, recorded in the trace, fingerprinted with its
patterns) named `user_asked_to_<intent>`, reading `user_request` — the user's messages only, never tool outputs. It
says that the user asked for an action of this kind, not that they asked for this very call: pair it with
`injections="grounded"` or `"any"` on the tool and with value policies (a date range, a price cap) where that matters.
A user who pastes a text that asks for the action counts as asking (`scan_user=True` for that case). The patterns are
word patterns over the NFKC-normalised, case-folded text, in English and Russian; INTENTS lists them."""
from __future__ import annotations

import re

# intent → regular expressions (any one found in the user's messages means they asked for this kind of action)
INTENTS = {
    "reserve": (r"\b(book|books|booked|booking|bookings|reserve|reserves|reserved|reserving|reservations?)\b",
                r"(?<!\w)(за|пере)?брон\w*", r"(?<!\w)(за)?резерв\w*"),
    "event": (r"\b(calendar|events?|meetings?|appointments?|reminders?|remind|schedule|scheduled|scheduling)\b",
              r"(?<!\w)(календар|событи|встреч|напомин|запланир|расписани)\w*"),
    "visit": (r"\b(visit|visits|visiting|website|websites|web ?pages?|sites?|pages?|links?|urls?|articles?|browse|"
              r"browsing)\b", r"https?://", r"\bwww\.",
              r"(?<!\w)(зайди|зайдите|зайти|посети|посетите|посетить|сайт|страниц|ссылк|стать|перейди|перейдите)\w*"),
    "pay": (r"\b(pay|pays|paid|paying|payments?|transfers?|transferr?ed|transferring|wire|remit|refunds?|refunded)\b",
            r"\bsend (the |my |some )?money\b", r"(?<!\w)(оплат|заплат|плат[её]ж|перевед|переведи|перечисл)\w*"),
    "send": (r"\b(send|sends|sending|sent|e-?mails?|e-?mailed|mail|forward|forwarded|messages?|reply|replies|respond|"
             r"tell|notify|dm)\b", r"(?<!\w)(отправ|пошли|вышли|перешли|напиш|сообщ|ответ|письм)\w*"),
    "delete": (r"\b(delete|deletes|deleted|deleting|remove|removes|removed|removing|erase|erased|cancel|cancels|"
               r"cancell?ed|cancell?ing|purge|clean ?up)\b", r"(?<!\w)(удал|сотри|стери|отмен|очист)\w*"),
    "invite": (r"\b(invite|invites|invited|inviting|onboard)\b", r"\badd \S+ to\b", r"(?<!\w)(пригла|добав)\w*"),
    "post": (r"\b(post|posts|posted|posting|publish|published|publishing|upload|uploaded|uploading)\b",
             r"(?<!\w)(опубликуй|опубликовать|опубликуйте|публику|выложи|выложить|загрузи|загрузить)\w*"),
    "share": (r"\b(share|shares|shared|sharing)\b", r"(?<!\w)(подели|поделись|поделитесь|доступ)\w*"),
}


def said(text, patterns) -> bool:
    """Does the text match any of the patterns (NFKC-normalised, format characters dropped, case-folded)?"""
    if not isinstance(text, str) or not text:
        return False
    from ...core.deciders.perturb import normalize
    t = normalize(text, confusables=False)[0].casefold()
    return any(re.search(p, t) for p in patterns)


def intent_patterns(intent=None, phrases=None):
    """(name, patterns) of an intent — a key of INTENTS, a list of keys, and / or your own regular expressions."""
    keys = [] if intent is None else [intent] if isinstance(intent, str) else list(intent)
    unknown = [k for k in keys if k not in INTENTS]
    if unknown:
        raise ValueError(f"unknown intent(s) {unknown}: one of {', '.join(INTENTS)} (or phrases=[regex, ...])")
    pats = [p for k in keys for p in INTENTS[k]]
    own = [phrases] if isinstance(phrases, str) else list(phrases or ())
    for p in own:
        re.compile(p)
    pats += own
    if not pats:
        raise ValueError("require_request needs an intent (one of INTENTS) or phrases=[regex, ...]")
    name = "_or_".join(keys) + ("_or_" if keys and own else "") + ("phrases" if own else "")
    return re.sub(r"\W", "_", name), tuple(pats)


def request_policy(intent=None, phrases=None):
    """A guard policy `user_asked_to_<intent>(user_request) -> bool`: the user's messages ask for this kind of action."""
    name, pats = intent_patterns(intent, phrases)
    label = name.replace("_or_", " or ")

    def user_asked(user_request) -> bool:
        return said(user_request, pats)
    user_asked.__name__ = user_asked.__qualname__ = f"user_asked_to_{name}"
    user_asked.__doc__ = f"The user asked for this kind of action ({label}) in their own words."
    user_asked.__module__ = __name__
    return user_asked


__all__ = ["INTENTS", "request_policy"]
