"""solvi documents: a Hugging Face Space for the solvi library.

Receipts and contracts go in; typed answers come out. Each answer has a confidence, a reason you can check, quotes at exact
character offsets and a hash-chained trace that is replayed after every run.

Models (Hugging Face ids or local directories written by LongSpanExtractor.save):
  SOLVI_BASE_MODEL      general extractor, fields by description   (default solvi-ai/extract-base)
  SOLVI_RECEIPTS_MODEL  receipts extractor                         (default solvi-ai/extract-receipts)
Local run from a clone:  PYTHONPATH=../../src python app.py
"""
from __future__ import annotations

import datetime as dt
import inspect
import os
import re
import threading

import gradio as gr
import pandas as pd

from samples import CONTRACT_FIELDS, CONTRACTS, RECEIPTS
from solvi import Answer, Catalog, Question, System

try:                                        # ZeroGPU on Hugging Face Spaces; a no-op decorator anywhere else
    import spaces

    GPU = spaces.GPU
except ImportError:
    spaces = None

    def GPU(fn=None, **_kw):
        return fn if callable(fn) else (lambda f: f)

BASE_ID = os.getenv("SOLVI_BASE_MODEL", "solvi-ai/extract-base")
RECEIPTS_ID = os.getenv("SOLVI_RECEIPTS_MODEL", "solvi-ai/extract-receipts")
ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))
GITHUB = "https://github.com/solvi-ai/solvi"
CUSTOM = "Your own text (paste below)"

# ---------------------------------------------------------------------------------------------------------- models
_models: dict = {}
_errors: dict = {}
_lock = threading.Lock()


def get_model(kind: str):
    """The extractor for "base" or "receipts": loaded once on first use, then cached."""
    with _lock:
        if kind not in _models:
            from solvi.extract_long import LongSpanExtractor

            mid = BASE_ID if kind == "base" else RECEIPTS_ID
            try:
                _models[kind] = LongSpanExtractor.load(mid, device="cuda" if ZERO_GPU else None)
                _errors.pop(kind, None)
            except Exception as e:  # noqa: BLE001
                _errors[kind] = f"{type(e).__name__}: {str(e)[:300]}"
                raise gr.Error(f"Could not load the model {mid!r} ({_errors[kind]}). If it is not published yet, set "
                               f"{'SOLVI_BASE_MODEL' if kind == 'base' else 'SOLVI_RECEIPTS_MODEL'} to a local directory "
                               "or another model id.") from e
        return _models[kind]


if ZERO_GPU:                                # ZeroGPU wants weights placed at startup; a failure is reported on first use
    for _k in ("receipts", "base"):
        try:
            get_model(_k)
        except Exception:  # noqa: BLE001
            pass


def device_note() -> str:
    if ZERO_GPU:
        return "Runs on ZeroGPU: a GPU is attached for each request (the first request may wait in the queue)."
    try:
        import torch

        if torch.cuda.is_available():
            return f"Runs on GPU ({torch.cuda.get_device_name(0)})."
    except Exception:  # noqa: BLE001
        pass
    return ("Runs on CPU, which is slow for ModernBERT-large: expect several seconds per receipt and a minute or more for a "
            "3-page contract with many fields. The first run also downloads and loads the model.")


# ------------------------------------------------------------------------------------------------ parsing helpers
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS.update({"mei": 5, "agu": 8, "agt": 8, "okt": 10, "des": 12})
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-five": 45,
                "forty five": 45, "forty": 40, "sixty": 60, "seventy-five": 75, "ninety": 90, "one hundred twenty": 120,
                "one hundred eighty": 180}
STATES = ["Delaware", "New York", "California", "other"]


def parse_amount(text: str) -> float:
    """The last number in an extracted amount: "RM 66.20" -> 66.2, "108,900" -> 108900, "Rp 74.500" -> 74500."""
    if not text:
        raise ValueError("the field was not found on the receipt")
    toks = re.findall(r"\d[\d.,]*", text)
    if not toks:
        raise ValueError(f"no number in {text!r}")
    t = toks[-1].rstrip(".,")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):              # 108,900 / 74.500: thousands separators
        return float(re.sub(r"[.,]", "", t))
    if "," in t and "." in t:                                   # 1,234.50 / 1.234,50: the last separator is the decimal one
        k = max(t.rfind(","), t.rfind("."))
        return float(re.sub(r"[.,]", "", t[:k]) + "." + t[k + 1:])
    return float(t.replace(",", "."))


def parse_date(text: str) -> dt.date:
    """Receipt dates: 24/02/2018, 15-01-2019, 2019-08-17, 03 Nov 2017, 5 Mei 19 (day first)."""
    if not text:
        raise ValueError("no date was found on the receipt")
    if m := re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text):
        y, mo, d = (int(x) for x in m.groups())
    elif m := re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", text):
        d, mo, y = (int(x) for x in m.groups())
    elif (m := re.search(r"(\d{1,2})[\s\-/.]*([A-Za-z]{3,9})[\s\-/.,]*(\d{2,4})", text)) and m.group(2)[:3].lower() in MONTHS:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], int(m.group(3))
    else:
        raise ValueError(f"cannot read a date from {text!r}")
    return dt.date(y if y > 99 else 2000 + y, mo, d)


def parse_notice_days(clause: str) -> int:
    """Notice period in days from a clause: "sixty (60) days" -> 60, "three (3) months" -> 90, "two weeks" -> 14."""
    low = clause.lower()
    m = re.search(r"(\d+)\s*\)?\s*(?:calendar\s+|business\s+)?(day|week|month)", low)
    if m:
        n, unit = int(m.group(1)), m.group(2)
    else:
        m = re.search(r"(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\s+(?:calendar\s+|business\s+)?(day|week|month)",
                      low)
        if not m:
            raise ValueError("no notice period in the extracted clause")
        n, unit = NUMBER_WORDS[m.group(1)], m.group(2)
    return n * {"day": 1, "week": 7, "month": 30}[unit]


def parse_today(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, (int, float)):
        return dt.date.fromtimestamp(value)
    if value:
        return dt.date.fromisoformat(str(value).strip()[:10])
    return dt.date.today()


# ---------------------------------------------------------------------------------------------- solvi: receipts
RECEIPT_FIELDS = {"company": "the name of the company that issued the receipt",
                  "date": "the date of the purchase",
                  "total": "the total amount paid",
                  "tax": "the tax amount",
                  "cash": "the cash amount given by the customer",
                  "change": "the change returned to the customer"}

RECEIPT_QUESTIONS = [
    Question("reimburse", "Reimbursable? (total within the limit, not older than N days)", Answer.yes_no(),
             requires=["not_too_old", "vendor_named"]),
    Question("weekend", "Bought on a weekend?", Answer.yes_no()),
    Question("change_correct", "Change correct? (cash - total = change)", Answer.yes_no()),
    Question("tax_shown", "Tax shown on the receipt?", Answer.yes_no()),
]


def receipt_catalog(extractor) -> Catalog:
    cat = Catalog()
    for name, description in RECEIPT_FIELDS.items():
        cat.extract(extractor.field(name, description))

    @cat.fn
    def amount(total):
        return parse_amount(total)

    @cat.fn
    def purchase_date(date):
        return parse_date(date)

    @cat.fn
    def purchase_weekday(date):
        return parse_date(date).strftime("%A")

    @cat.fn
    def cash_given(cash):
        return parse_amount(cash)

    @cat.fn
    def change_given(change):
        return parse_amount(change)

    @cat.fn
    def expected_change(cash_given, amount):
        return round(cash_given - amount, 2)

    @cat.check(hard=True, then={"reimburse": "no"})
    def not_too_old(purchase_date, today, max_age_days):
        """hard: a receipt older than max_age_days is never reimbursed, whatever the amount"""
        return (today - purchase_date).days <= max_age_days

    @cat.check(hard=True, then={"reimburse": "no"})
    def vendor_named(company):
        """hard: a receipt that does not name the issuing company is never reimbursed"""
        return company != ""

    @cat.check
    def cash_covers_total(cash_given, amount):
        return cash_given >= amount

    @cat.rule("reimburse")
    def reimburse(amount, limit):
        return amount <= limit

    @cat.rule("weekend")
    def weekend(purchase_weekday):
        return purchase_weekday in ("Saturday", "Sunday")

    @cat.rule("change_correct")
    def change_correct(expected_change, change_given):
        return abs(expected_change - change_given) < 0.01

    @cat.rule("tax_shown")
    def tax_shown(tax):
        return tax != ""

    return cat


# --------------------------------------------------------------------------------------------- solvi: contracts
CONTRACT_FIXED = {"governing_law": "the clause that says which state's or country's law governs the contract",
                  "termination_notice": "the notice period required to terminate the agreement",
                  "liability_cap": "a clause that caps or limits the amount of liability of a party",
                  "non_compete": "a clause that restricts a party from competing or operating in a business, market or area"}

CONTRACT_QUESTIONS = [
    Question("law", "Which law governs?", Answer.choice(STATES)),
    Question("notice_ok", "Termination notice at least 30 days?", Answer.yes_no()),
    Question("liability_capped", "Is liability capped?", Answer.yes_no()),
    Question("has_non_compete", "Is there a non-compete?", Answer.yes_no()),
]
STOPWORDS = {"the", "a", "an", "of", "that", "which", "who", "is", "are", "to", "in", "on", "for", "and", "or", "by", "with",
             "when", "where", "how", "what", "this", "any", "it", "its", "be"}
RESERVED = set(CONTRACT_FIXED) | {"doc", "notice_days", "law", "notice_ok", "liability_capped", "has_non_compete"}


def presence_rule(field: str):
    """A rule "is <field> present?" whose single argument is named after the field (solvi reads inputs from the signature)."""
    def rule(**kw):
        return kw[field] != ""
    rule.__name__ = f"{field}_present"
    rule.__signature__ = inspect.Signature([inspect.Parameter(field, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return rule


def contract_catalog(extractor, user_fields):
    """Fixed clauses + typed questions, plus one extractor and one "present?" question per user field."""
    cat = Catalog()
    for name, description in CONTRACT_FIXED.items():
        cat.extract(extractor.field(name, description))

    @cat.fn
    def notice_days(termination_notice):
        if not termination_notice:
            raise ValueError("no termination-notice clause found")
        return parse_notice_days(termination_notice)

    @cat.rule("law")
    def law(governing_law):
        if not governing_law:
            raise ValueError("no governing-law clause found")
        return next((s for s in STATES[:3] if s.lower() in governing_law.lower()), "other")

    @cat.rule("notice_ok")
    def notice_ok(notice_days):
        return notice_days >= 30

    @cat.rule("liability_capped")
    def liability_capped(liability_cap):
        return liability_cap != ""

    @cat.rule("has_non_compete")
    def has_non_compete(non_compete):
        return non_compete != ""

    questions = list(CONTRACT_QUESTIONS)
    for name, description in user_fields:
        cat.extract(extractor.field(name, description))
        q = f"{name}?"
        cat.rule(q)(presence_rule(name))
        questions.append(Question(q, f"Is {name} present?", Answer.yes_no()))
    return cat, questions


def clean_fields(rows) -> list:
    """Rows of the editable table -> [(identifier, description)]; empty rows dropped, names made unique identifiers."""
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        rows = rows.values.tolist()
    out, used = [], set(RESERVED)
    for row in rows:
        row = list(row) + ["", ""]
        name, desc = str(row[0] or "").strip(), str(row[1] or "").strip()
        if not desc or desc.lower() == "nan":
            continue
        if not name or name.lower() == "nan":
            name = "_".join([w for w in re.findall(r"\w+", desc.lower()) if w not in STOPWORDS][:3])
        base = re.sub(r"\W+", "_", name.lower()).strip("_")
        base = base or "field"
        if base[0].isdigit():
            base = "f_" + base
        ident, k = base, 2
        while ident in used:
            ident, k = f"{base}_{k}", k + 1
        used.add(ident)
        out.append((ident, desc))
    return out[:12]


# ---------------------------------------------------------------------------------------------- shared execution
def execute(cat, questions, init_state) -> dict:
    """ask + replay; returns plain data (ZeroGPU sends it back from a worker process, so nothing unpicklable)."""
    system = System(cat, questions)
    res = system.ask(init_state)
    rep = res.trace.replay(system)                 # with the System, replay also checks the stored answers
    answers = [dict(name=q.name, text=q.text, answer=res[q.name].answer, confidence=res[q.name].confidence,
                    status=res[q.name].status, why=res[q.name].why) for q in questions]
    extracts = [dict(name=r.name, value=r.value if isinstance(r.value, str) else None,
                     start=r.quote[0] if r.quote else 0, end=r.quote[1] if r.quote else 0,
                     confidence=r.confidence, error=r.error)
                for r in res.trace.records if r.kind == "extract"]
    return dict(answers=answers, extracts=extracts, state=res.state_text(), flow=str(res.flow),
                replay=dict(ok=rep["ok"], steps=rep["steps"], mismatches=[list(map(str, m)) for m in rep["mismatches"]]),
                ms=res.ms, hash=res.trace.records[-1].hash if res.trace.records else "")


@GPU(duration=60)
def run_receipt(text, today, limit, max_age_days):
    cat = receipt_catalog(get_model("receipts"))
    return execute(cat, RECEIPT_QUESTIONS, {"doc": text, "today": today, "limit": float(limit),
                                            "max_age_days": int(max_age_days)})


@GPU(duration=120)
def run_contract(text, user_fields):
    cat, questions = contract_catalog(get_model("base"), user_fields)
    return execute(cat, questions, {"doc": text})


# ---------------------------------------------------------------------------------------------------- rendering
PALETTE = ["#f59e0b", "#10b981", "#3b82f6", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16", "#f97316", "#6366f1",
           "#06b6d4", "#a855f7", "#eab308", "#22c55e", "#0ea5e9", "#d946ef"]


def highlight(text: str, extracts) -> list:
    """Text segments for gr.HighlightedText: every found field becomes a segment labeled with its name. Where fields overlap,
    the segment carries all of their names ("total + cash")."""
    found = [e for e in extracts if e["value"] and e["end"] > e["start"]]
    cuts = sorted({0, len(text)} | {min(max(x, 0), len(text)) for e in found for x in (e["start"], e["end"])})
    out = []
    for s, e in zip(cuts, cuts[1:]):
        names = [f["name"] for f in found if f["start"] <= s and e <= f["end"]]
        label = " + ".join(names) or None
        if out and out[-1][1] == label:
            out[-1] = (out[-1][0] + text[s:e], label)
        else:
            out.append((text[s:e], label))
    return out


def color_map(names) -> dict:
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def answers_table(answers) -> pd.DataFrame:
    rows = [[a["text"], "—" if a["answer"] is None else a["answer"], round(a["confidence"], 3), a["status"], a["why"]]
            for a in answers]
    return pd.DataFrame(rows, columns=["question", "answer", "confidence", "status", "why"])


def fields_table(extracts, descriptions) -> pd.DataFrame:
    rows = []
    for e in extracts:
        if e["error"] and e["value"] is None:
            rows.append([e["name"], descriptions.get(e["name"], ""), f"error: {e['error']}", 0.0, ""])
        elif e["value"]:
            v = e["value"] if len(e["value"]) <= 220 else e["value"][:217] + "..."
            rows.append([e["name"], descriptions.get(e["name"], ""), v, round(e["confidence"], 3), f"{e['start']}:{e['end']}"])
        else:
            rows.append([e["name"], descriptions.get(e["name"], ""), "absent", round(e["confidence"], 3), ""])
    return pd.DataFrame(rows, columns=["field", "description", "found value", "confidence", "offsets"])


def replay_md(out) -> str:
    rep = out["replay"]
    head = (f"**Trace replay: OK.** All {rep['steps']} steps were re-executed from the recorded input and matched the "
            f"hash-chained trace." if rep["ok"] else
            f"**Trace replay: {len(rep['mismatches'])} mismatch(es)** in {rep['steps']} steps: {rep['mismatches'][:3]}")
    return f"{head}  \nDecision time {out['ms']:.0f} ms (model included; repeated runs on the same text reuse cached predictions). Last record hash `{out['hash']}`."


# ------------------------------------------------------------------------------------------------------ handlers
def analyze_receipt(text, today, limit, max_age_days):
    if not text or not text.strip():
        raise gr.Error("Paste a receipt text or pick a sample first.")
    try:
        today = parse_today(today)
    except ValueError as e:
        raise gr.Error(f"Cannot read the date for today: {today!r}. Use YYYY-MM-DD.") from e
    out = run_receipt(text, today, limit, max_age_days)
    hl = gr.HighlightedText(value=highlight(text, out["extracts"]), color_map=color_map(list(RECEIPT_FIELDS)))
    return (answers_table(out["answers"]), hl, fields_table(out["extracts"], RECEIPT_FIELDS), out["state"], out["flow"],
            replay_md(out))


def analyze_contract(text, rows):
    if not text or not text.strip():
        raise gr.Error("Paste a contract or pick a sample first.")
    user_fields = clean_fields(rows)
    out = run_contract(text, user_fields)
    descriptions = dict(CONTRACT_FIXED) | dict(user_fields)
    by_q = {a["name"]: a for a in out["answers"]}
    fixed = [a for a in out["answers"] if a["name"] in {q.name for q in CONTRACT_QUESTIONS}]
    user_names = [n for n, _ in user_fields]
    ex_user = sorted((e for e in out["extracts"] if e["name"] in user_names), key=lambda e: user_names.index(e["name"]))
    ft = fields_table(ex_user, descriptions)
    ft["present?"] = [by_q.get(f"{e['name']}?", {}).get("answer") or "—" for e in ex_user]
    ft = ft[["field", "description", "present?", "found value", "confidence", "offsets"]]
    names = user_names + list(CONTRACT_FIXED)
    hl = gr.HighlightedText(value=highlight(text, out["extracts"]), color_map=color_map(names))
    return (ft, answers_table(fixed), fields_table([e for e in out["extracts"] if e["name"] in CONTRACT_FIXED], descriptions),
            hl, out["state"], out["flow"], replay_md(out))


CURRENCIES = {"RM": (1000, 10, 200), "IDR": (1_000_000, 5000, 250_000), "other": (10_000, 10, 200)}   # max, step, default


def limit_slider(currency, value=None):
    mx, step, default = CURRENCIES.get(currency, CURRENCIES["other"])
    return gr.update(value=default if value is None else value, maximum=mx, step=step,
                     label=f"Reimbursement limit ({currency}, the receipt's currency)")


def pick_receipt(name):
    if name not in RECEIPTS:
        return "", gr.update(), gr.update(), gr.update()
    s = RECEIPTS[name]
    return s["text"], s["today"], s["currency"], limit_slider(s["currency"], s["limit"])


def pick_currency(currency):
    return limit_slider(currency)


def pick_contract(name):
    return CONTRACTS.get(name, "")


# ------------------------------------------------------------------------------------------------------------- UI
INTRO = f"""# solvi documents
Typed answers from receipts and contracts with [solvi]({GITHUB}). A ModernBERT extractor finds each field and cites it at
exact character offsets (or says the field is absent). Plain Python rules and hard checks turn the fields into yes/no and
choice answers. Every run writes a hash-chained trace, and the app replays it to confirm each step.

<small>{device_note()} Models: `{RECEIPTS_ID}` (receipts), `{BASE_ID}` (contracts, fields by description).</small>
"""

RECEIPT_HELP = """Fields: **company, date, total, tax, cash, change** (receipts model). Questions come from a solvi catalog:
*reimbursable* is `total <= limit` under two **hard checks**: `not_too_old` (older than N days means "no", whatever the
amount) and `vendor_named` (no issuing company found means "no"),
*weekend* reads the weekday of the date, *change correct* computes `cash - total` and compares it with the printed change.
A question abstains when a field it needs is missing (for example, no cash on a card payment)."""

CONTRACT_HELP = """**Add your own fields**: write a name and a plain-English description in the table (add rows as you need);
the general extractor looks for each one across the whole contract, reading it in overlapping windows, and returns a quote or
"absent". Below are fixed typed questions built with solvi rules on four more clauses.

Fields close to what the model saw in training (common contract clauses) work best. For a new kind of field, treat the
result as a hint and check the highlighted quote. For production accuracy, fine-tune on about 100 labeled documents."""

COMPARE = f"""## What the models were measured on

Every number here comes from the models' published cards, as the solvi README quotes them.

- **extract-receipts** ([card](https://huggingface.co/solvi-ai/extract-receipts)): 97.3% on typed questions over CORD
  receipts (100 test receipts), with ECE 0.011 and 98.6% of questions answered at ≥ 99% precision.
- **extract-base** ([card](https://huggingface.co/solvi-ai/extract-base)), without labels of your task: CORD receipt
  fields it was never trained on, 46.5% and 67.9%; five never-trained CUAD clause types, 73.1%; Kleister-NDA and SROIE,
  never seen, 2–95% by field. With per-field thresholds from 40 labeled contracts it reached 85.5% on CUAD; fine-tuned
  on 25–100 SROIE receipts, 86–89%.
- Every answer is backed by a quote at stated offsets or a computed fact, or the system abstains.

## An honest note

The models in this demo are a starting point. They show how solvi works on your documents, but they are not tuned to
them. For production accuracy, label about 100 of your documents (the character span of each field) and fine-tune the
extractor with `LongSpanExtractor.fit` (the fine-tuned numbers above come from such task-specific training). solvi answers yes/no and
choice questions only; it does not generate free text.

## Links

- Code, docs and examples: [{GITHUB}]({GITHUB})
- API guide: [docs/guide.md]({GITHUB}/blob/main/docs/guide.md) · benchmarks: [docs/benchmarks.md]({GITHUB}/blob/main/docs/benchmarks.md)
- Models: [mxkuzn on Hugging Face](https://huggingface.co/solvi-ai)
- Install: `pip install "solvi[model]"`
"""

CSS = """
.doc textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; font-size: 13px !important; }
.hl { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
.mono textarea, .mono pre { font-size: 12px !important; }
"""

first_receipt = next(iter(RECEIPTS))
first_contract = next(iter(CONTRACTS))

with gr.Blocks(title="solvi documents") as demo:
    gr.Markdown(INTRO)
    with gr.Tabs():
        # ------------------------------------------------------------------ receipts
        with gr.Tab("Receipts"):
            gr.Markdown(RECEIPT_HELP)
            with gr.Row():
                with gr.Column(scale=5):
                    r_pick = gr.Dropdown(list(RECEIPTS) + [CUSTOM], value=first_receipt, label="Sample receipt (OCR text)")
                    r_text = gr.Textbox(RECEIPTS[first_receipt]["text"], lines=18, max_lines=40, label="Receipt text",
                                        elem_classes="doc")
                with gr.Column(scale=3):
                    s0 = RECEIPTS[first_receipt]
                    r_currency = gr.Radio(list(CURRENCIES), value=s0["currency"],
                                          label="Receipt currency (sets the range of the limit slider)")
                    r_limit = gr.Slider(0, CURRENCIES[s0["currency"]][0], value=s0["limit"],
                                        step=CURRENCIES[s0["currency"]][1],
                                        label=f"Reimbursement limit ({s0['currency']}, the receipt's currency)")
                    r_age = gr.Slider(1, 365, value=90, step=1, label="Hard check: not older than N days")
                    r_today = gr.DateTime(s0["today"], include_time=False, type="string", label="Today")
                    r_go = gr.Button("Check the receipt", variant="primary")
                    r_replay = gr.Markdown()
            r_answers = gr.Dataframe(label="Answers", interactive=False, wrap=True)
            r_hl = gr.HighlightedText(label="Receipt with extracted fields", show_legend=True, combine_adjacent=False,
                                      elem_classes="hl")
            r_fields = gr.Dataframe(label="Extracted fields", interactive=False, wrap=True)
            with gr.Accordion("computed_state and flow", open=False):
                r_state = gr.Code(label="computed_state (every fact, its value, quote offsets, confidence, errors)",
                                  language=None, elem_classes="mono")
                r_flow = gr.Code(label="Flow chosen by the strategist", language=None, elem_classes="mono")
            r_pick.change(pick_receipt, r_pick, [r_text, r_today, r_currency, r_limit])
            r_currency.input(pick_currency, r_currency, r_limit)
            r_go.click(analyze_receipt, [r_text, r_today, r_limit, r_age],
                       [r_answers, r_hl, r_fields, r_state, r_flow, r_replay])

        # ------------------------------------------------------------------ contracts
        with gr.Tab("Contracts: ask by description"):
            gr.Markdown(CONTRACT_HELP)
            with gr.Row():
                with gr.Column(scale=5):
                    c_pick = gr.Dropdown(list(CONTRACTS) + [CUSTOM], value=first_contract, label="Sample contract")
                    c_text = gr.Textbox(CONTRACTS[first_contract], lines=18, max_lines=40, label="Contract text",
                                        elem_classes="doc")
                with gr.Column(scale=4):
                    c_fields = gr.Dataframe(value=CONTRACT_FIELDS, headers=["name", "description"], type="array",
                                            interactive=True, wrap=True, column_widths=["35%", "65%"],
                                            label="Fields to find (edit, or add rows with your own descriptions)")
                    c_go = gr.Button("Read the contract", variant="primary")
                    c_replay = gr.Markdown()
            c_found = gr.Dataframe(label="Your fields", interactive=False, wrap=True)
            c_answers = gr.Dataframe(label="Typed questions (solvi rules)", interactive=False, wrap=True)
            c_fixed = gr.Dataframe(label="Clauses behind the typed questions", interactive=False, wrap=True)
            c_hl = gr.HighlightedText(label="Contract with found fields and clauses", show_legend=True,
                                      combine_adjacent=False, elem_classes="hl")
            with gr.Accordion("computed_state and flow", open=False):
                c_state = gr.Code(label="computed_state", language=None, elem_classes="mono")
                c_flow = gr.Code(label="Flow chosen by the strategist", language=None, elem_classes="mono")
            c_pick.change(pick_contract, c_pick, c_text)
            c_go.click(analyze_contract, [c_text, c_fields],
                       [c_found, c_answers, c_fixed, c_hl, c_state, c_flow, c_replay])

        # ------------------------------------------------------------------ compare
        with gr.Tab("Compare"):
            gr.Markdown(COMPARE)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="amber"),
        css=CSS,
    )                                       # host and port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT
