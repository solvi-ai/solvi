"""Text in on real requests: routing to an entry point and reading its fields, measured on MASSIVE.

    uv run --with onnxruntime --with tokenizers python benchmarks/textin_massive.py [--n N] [--json out.json]

Data: MASSIVE 1.1 (Amazon, CC BY 4.0; FitzGerald et al. 2022), the en-US file — 16.5k spoken-style requests to a voice
assistant with a gold intent and gold slot spans ("wake me up at [time : five am] [date : this week]"). The file is read
from $MASSIVE_DIR (the folder with data/en-US.jsonl, or the release's tarball unpacked;
https://github.com/alexa/massive). The catalog was written from the train partition; everything is measured on the
test partition.

The catalog: eight entry points of a home assistant, each a question whose flow reads typed fields (a pydantic input
model with descriptions, enum synonyms, cue words and two patterns, as a developer would write them once):

    set_alarm          time, date                    alarm_set
    weather            place_name, date              weather_query
    add_event          event_name, date, time        calendar_set
    play_music         artist_name, music_genre      play_music
    book_train         place_name, date              transport_ticket
    change_lights      color_type, house_place       iot_hue_lightchange
    order_takeaway     food_type, business_name      takeaway_order
    send_email         person                        email_sendemail

Measured:
  routing   — TextIn.route with the decider over the eight entry points, on the test messages of those intents:
              accuracy, the share escalated (min_confidence 0.6, margin 0.1), accuracy of what is routed; and on messages
              of other intents (out of scope): how many are routed anyway (there is no "none of these" entry point).
  fields    — with the gold entry point given (routing apart), each field of each message: "right" (the quote equals a
              gold span of that slot, case and surrounding punctuation aside), "overlap" (overlaps one but differs),
              "wrong" (a span that overlaps none), "spurious" (read where the gold has no such slot), "missed" (the
              gold has it, nothing read), "absent" (neither). Separately for the CueExtractor (no model) and, when the
              decider has a pointer, for its span pointer (DeciderExtractor) and for the two in order (the pointer's
              span, else the cue extractor's). A field that is found but does not parse (a relative
              date such as "friday") is not read: it counts as missed, never as a value.
The decider: $SOLVI_DECIDE_MODEL, else solvi-ai/solvi-base from the local Hugging Face cache (never downloaded)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from solvi import Catalog, Question, System
from solvi.textin import CueExtractor, DeciderExtractor, TextIn

TODAY = dt.date(2026, 9, 28)
TIME_RX = (r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})"
           r"(?:(?::| )(?:\d{2}|thirty|fifteen|forty five|o'clock))?\s*(?:am|a\. m\.|pm|p\. m\.)|noon|midnight)")
GENRES = ["jazz", "rock", "pop", "classical", "country", "rap", "hip hop", "folk", "metal", "gospel", "blues", "disco",
          "techno", "latin", "reggae", "soul"]
COLORS = ["blue", "red", "green", "yellow", "orange", "pink", "white", "purple", "warm", "dark"]


class Request(BaseModel):
    """The fields a home assistant reads from a spoken request."""
    time: str = Field(description="The clock time", json_schema_extra={"pattern": TIME_RX, "cues": ["at", "for"]})
    date: dt.date = Field(description="The day", json_schema_extra={"cues": ["on", "for"]})
    place_name: str = Field(description="The city or place", json_schema_extra={"cues": ["weather", "ticket", "train",
                                                                                          "trip", "journey", "travel"]})
    event_name: str = Field(description="The event or meeting to put in the calendar",
                            json_schema_extra={"cues": ["remind", "reminder", "meeting", "schedule", "calendar"]})
    artist_name: str = Field(description="The artist or band", json_schema_extra={"cues": ["by", "songs", "music"]})
    music_genre: Literal[tuple(GENRES)] = Field(description="The genre of music",
                                                json_schema_extra={"synonyms": {"hip hop": ["hiphop", "hip-hop"],
                                                                                "classical": ["classic"]}})
    color_type: Literal[tuple(COLORS)] = Field(description="The colour of the light")
    house_place: str = Field(description="The room", json_schema_extra={"cues": ["in", "room", "lights"]})
    food_type: str = Field(description="The food to order", json_schema_extra={"cues": ["order", "want", "get"]})
    business_name: str = Field(description="The restaurant or shop", json_schema_extra={"cues": ["from"]})
    person: str = Field(description="The person to write to", json_schema_extra={"cues": ["email", "mail", "to"]})


ENTRY = {"set_alarm": ("alarm_set", "Set an alarm", ["time", "date"]),
         "weather": ("weather_query", "Tell the weather forecast", ["place_name", "date"]),
         "add_event": ("calendar_set", "Add an event or reminder to the calendar", ["event_name", "date", "time"]),
         "play_music": ("play_music", "Play music", ["artist_name", "music_genre"]),
         "book_train": ("transport_ticket", "Book a train or travel ticket", ["place_name", "date"]),
         "change_lights": ("iot_hue_lightchange", "Change the colour of the lights", ["color_type", "house_place"]),
         "order_takeaway": ("takeaway_order", "Order takeaway food", ["food_type", "business_name"]),
         "send_email": ("email_sendemail", "Send an email", ["person"])}
BY_INTENT = {v[0]: k for k, v in ENTRY.items()}


def catalog():
    """The eight entry points: each question's rule reads its fields (typed by the Request model)."""
    import inspect
    cat = Catalog()
    anns = Request.model_fields
    for name, (_, _, fields) in ENTRY.items():
        def rule(**kw):
            return "done"
        rule.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY,
                                                                  annotation=anns[f].annotation) for f in fields])
        rule.__annotations__ = {**{f: anns[f].annotation for f in fields}, "return": Literal["done"]}
        rule.__name__ = rule.__qualname__ = f"do_{name}"
        cat.rule(name)(rule)
    qs = [Question(n, text) for n, (_, text, _) in ENTRY.items()]
    return System(cat, qs, inputs=Request)


# --------------------------------------------------------------------------------------------------- data
def massive_file():
    d = os.environ.get("MASSIVE_DIR")
    if d:
        for p in (os.path.join(d, "data", "en-US.jsonl"), os.path.join(d, "en-US.jsonl")):
            if os.path.isfile(p):
                return p
    sys.exit("MASSIVE en-US.jsonl not found: set MASSIVE_DIR (https://github.com/alexa/massive, CC BY 4.0)")


SLOT = re.compile(r"\[(\w+) : ([^\]]*)\]")


def spans(annot):
    """annot_utt → (text, {slot: [(start, end)]})."""
    got, pos = defaultdict(list), 0
    text = ""
    for m in SLOT.finditer(annot):
        text += annot[pos:m.start()]
        s = len(text)
        text += m.group(2)
        got[m.group(1)].append((s, len(text)))
        pos = m.end()
    text += annot[pos:]
    return text, dict(got)


def load(partition="test"):
    rows = []
    with open(massive_file()) as f:
        for line in f:
            r = json.loads(line)
            if r["partition"] != partition:
                continue
            text, sl = spans(r["annot_utt"])
            if text != r["utt"]:
                continue
            rows.append({"id": r["id"], "text": text, "intent": r["intent"], "slots": sl})
    return rows


# --------------------------------------------------------------------------------------------------- measuring
def _norm(s):
    return re.sub(r"^[\W_]+|[\W_]+$", "", s.lower().strip())


def judge(quote, gold, text):
    """One field → right / overlap / wrong (another span) / spurious (the gold has no such slot) / missed / absent."""
    if quote is None:
        return "missed" if gold else "absent"
    if not gold:
        return "spurious"
    q = _norm(text[quote.start:quote.end])
    if any(q == _norm(text[a:b]) for a, b in gold):
        return "right"
    if any(quote.start < b and a < quote.end for a, b in gold):
        return "overlap"
    return "wrong"


def fields_eval(system, extractor, rows, label):
    tin = TextIn(system, None, extractor, today=TODAY)
    per = defaultdict(Counter)
    status = Counter()
    t0 = time.perf_counter()
    for r in rows:
        q = BY_INTENT[r["intent"]]
        read = tin.read(r["text"], question=q)
        for f, fr in read.fields.items():
            status[fr.status] += 1
            quote = fr.quote if fr.ok else None
            per[f][judge(quote, r["slots"].get(f), r["text"])] += 1
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
    tot = Counter()
    for c in per.values():
        tot.update(c)
    return {"extractor": label, "messages": len(rows), "ms_per_message": round(ms, 1), "total": dict(tot),
            "per_field": {f: dict(c) for f, c in sorted(per.items())}, "status": dict(status)}


def summary(c):
    present = sum(c.get(k, 0) for k in ("right", "overlap", "wrong", "missed"))       # the gold has the slot
    read = sum(c.get(k, 0) for k in ("right", "overlap", "wrong", "spurious"))
    bad = c.get("wrong", 0) + c.get("spurious", 0)
    return {"n": sum(c.values()), "gold": present, "read": read, "right": c.get("right", 0),
            "precision_exact": c.get("right", 0) / read if read else 0.0,
            "precision_overlap": (c.get("right", 0) + c.get("overlap", 0)) / read if read else 0.0,
            "recall_exact": c.get("right", 0) / present if present else 0.0,
            "wrong_share_of_read": bad / read if read else 0.0}


def routing_eval(system, decider, rows, out_rows):
    tin = TextIn(system, decider, today=TODAY)
    got = Counter()
    for r in rows:
        rt = tin.route(r["text"])
        gold = BY_INTENT[r["intent"]]
        if rt["question"] is None:
            got["escalated"] += 1
            got["escalated_top_right"] += max(rt["probs"], key=rt["probs"].get) == gold
        elif rt["question"] == gold:
            got["right"] += 1
        else:
            got["wrong"] += 1
    oos = Counter()
    for r in out_rows:
        rt = tin.route(r["text"])
        oos["routed" if rt["question"] is not None else "escalated"] += 1
    n, ans = len(rows), got["right"] + got["wrong"]
    top1 = got["right"] + got["escalated_top_right"]
    return {"messages": n, "top1_accuracy": top1 / n, "routed": ans / n, "accuracy_of_routed": got["right"] / ans if ans else 0,
            "wrong_routed_share": got["wrong"] / n, "escalated": got["escalated"] / n,
            "out_of_scope": {"messages": len(out_rows), "routed_anyway": oos["routed"] / max(1, len(out_rows))}}


def load_decider():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    from solvi.models import ModelError
    from solvi.models import load as load_model
    try:
        return load_model(os.path.expanduser(src) if src else "solvi-ai/solvi-base")
    except (ModelError, ImportError) as e:
        print(f"no decider ({e}): routing and the pointer are skipped", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="at most this many in-scope test messages (0: all)")
    ap.add_argument("--oos", type=int, default=300, help="out-of-scope test messages for routing")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args()
    test = load("test")
    rng = random.Random(0)
    ins = [r for r in test if r["intent"] in BY_INTENT]
    outs = [r for r in test if r["intent"] not in BY_INTENT]
    if a.n:
        ins = rng.sample(ins, min(a.n, len(ins)))
    outs = rng.sample(outs, min(a.oos, len(outs)))
    system = catalog()
    print(f"MASSIVE en-US test: {len(ins)} in-scope messages over {len(ENTRY)} entry points "
          f"({dict(Counter(BY_INTENT[r['intent']] for r in ins))}), {len(outs)} out of scope")
    res = {"data": "MASSIVE 1.1 en-US test (CC BY 4.0)", "in_scope": len(ins), "out_of_scope": len(outs)}
    res["fields"] = [fields_eval(system, CueExtractor(), ins, "CueExtractor")]
    decider = load_decider()
    if decider is not None:
        res["decider"] = decider.model_id
        t0 = time.perf_counter()
        res["routing"] = routing_eval(system, decider, ins, outs)
        res["routing"]["ms_per_message"] = round((time.perf_counter() - t0) * 1000 / (len(ins) + len(outs)), 1)
        if getattr(decider, "has_pointer", False):
            ptr = DeciderExtractor(decider)
            res["fields"].append(fields_eval(system, ptr, ins, f"pointer ({decider.model_id})"))
            res["fields"].append(fields_eval(system, [ptr, CueExtractor()], ins, "pointer, then CueExtractor"))
    for fe in res["fields"]:
        fe["summary"] = summary(fe["total"])
    if "routing" in res:
        r = res["routing"]
        print(f"\nrouting ({res['decider']}): top-1 {r['top1_accuracy']:.1%}; routed {r['routed']:.1%}, of them right "
              f"{r['accuracy_of_routed']:.1%}; routed to a wrong entry point {r['wrong_routed_share']:.1%}; escalated "
              f"{r['escalated']:.1%}; out of scope routed anyway {r['out_of_scope']['routed_anyway']:.1%}  "
              f"({r['ms_per_message']} ms/message)")
    for fe in res["fields"]:
        s = fe["summary"]
        print(f"\nfields — {fe['extractor']} ({fe['ms_per_message']} ms/message): read {s['read']} of {s['n']} field slots; "
              f"{s['gold']} gold slots; exact {s['precision_exact']:.1%} of what is read (overlapping "
              f"{s['precision_overlap']:.1%}), {s['recall_exact']:.1%} of the gold slots read exactly; wrong or spurious "
              f"{s['wrong_share_of_read']:.1%} of what is read")
        print(f"  {'field':14s} {'right':>6s} {'overlap':>8s} {'wrong':>6s} {'spurious':>9s} {'missed':>7s} {'absent':>7s}")
        for f, c in fe["per_field"].items():
            print(f"  {f:14s} {c.get('right', 0):6d} {c.get('overlap', 0):8d} {c.get('wrong', 0):6d} "
                  f"{c.get('spurious', 0):9d} {c.get('missed', 0):7d} {c.get('absent', 0):7d}")
        print(f"  statuses: {fe['status']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
