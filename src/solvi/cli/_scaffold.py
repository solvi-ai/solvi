"""`solvi init`: a new decision project — a small typed catalog, regression cases that pass, a README with the next steps,
a CI workflow and a .gitignore.

    solvi init [DIR] [--template support|refunds|minimal] [--with-model] [--force]

The catalog (catalog.py) has a computation, a hard check, a rule and — with --with-model — a question answered by a
decider (a keyword stand-in by default, so tests and CI need no model; SOLVI_DECIDE_MODEL names a real one; a
calibration file made with a real model is skipped, with a note, while the stand-in answers). cases.json
pins its answers for `solvi test`; `solvi check catalog.py:system` lints it; `solvi ask catalog.py:system example.json`
asks it once. Existing files are never overwritten without --force.
Exit status: 0 — written; 1 — a file exists, or DIR is a file (nothing is written); 2 — usage errors."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TEMPLATES = ("support", "refunds", "minimal")
MIN_SOLVI = "0.7"

# ------------------------------------------------------------------------------------------------ the templates
# Each template: the catalog's docstring, its imports, its body (a computation, a hard check, a rule, QUESTIONS), the
# model block (--with-model: a decider question and the keywords of the stand-in), an example state, the cases, and the
# labelled examples for `solvi calibrate`.

SUPPORT = {
    "title": "Support triage: a ticket → its priority and the team that handles it.",
    "imports": "import re\n",
    "body": '''
LEGAL = re.compile(r"\\b(lawyer|attorney|sue|suing|court|legal action)\\b", re.I)


# ---------- computations: argument names are the facts they read, the function name is the fact they set
@cat.fn
def sla_hours(tier: str) -> float:
    """How long a ticket of this tier may wait."""
    return 4.0 if tier == "vip" else 24.0


@cat.fn
def past_sla(hours_open: float, sla_hours: float) -> bool:
    return hours_open > sla_hours


@cat.fn
def legal_threat(message: str) -> bool:
    return LEGAL.search(message) is not None


# ---------- a hard check: when it fails, the answers in `then` are forced, whatever the rules or a model say
@cat.check(hard=True, then={"priority": "urgent", "route": "legal"})
def no_legal_threat(legal_threat: bool) -> bool:
    return not legal_threat


# ---------- rules: a question's answer from facts
@cat.rule("priority")
def priority(past_sla: bool, tier: str) -> str:
    if past_sla:
        return "high"
    return "normal" if tier == "vip" else "low"


QUESTIONS = [
    Question("priority", "How urgent is the ticket?", Answer.ordinal(["low", "normal", "high", "urgent"]),
             requires=["no_legal_threat"]),
]
''',
    "rule_only": '''

@cat.rule("route")
def route(message: str) -> str:
    text = message.lower()
    if any(w in text for w in ("charge", "refund", "invoice", "payment")):
        return "billing"
    if any(w in text for w in ("error", "crash", "bug", "down")):
        return "technical"
    return "general"


QUESTIONS.append(Question("route", "Which team should handle the ticket?", Answer.choice(ROUTES),
                          requires=["no_legal_threat"]))
''',
    "model": '''
# ---------- a question the model answers: it proposes one of the options; the hard check above still decides
route = model.decision("route", "Which team should handle this ticket?", "message", ROUTES, min_confidence=0.5)
CALIBRATION = HERE / "route.calib.json"           # written by: solvi calibrate catalog.py:system route labels.csv
if CALIBRATION.exists() and calibration_fits(CALIBRATION):
    route.load_calibration(CALIBRATION)           # refuses a calibration made with another model
QUESTIONS.append(route.question(cat, requires=["no_legal_threat"]))
''',
    "options": '''ROUTES = {"billing": "charges, refunds, invoices", "technical": "errors, bugs, outages",
          "general": "questions and anything else", "legal": "legal threats"}
''',
    "keywords": {"billing": ["charge", "refund", "invoice", "payment"], "technical": ["error", "crash", "bug", "down"],
                 "general": ["how do i", "question", "can i"], "legal": ["lawyer", "court"]},
    "model_question": "route",
    "model_fact": "message",
    "example": {"message": "I was charged twice this month, please refund the second payment.", "tier": "standard",
                "hours_open": 2},
    "cases": [
        {"name": "double charge, standard",
         "state": {"message": "I was charged twice this month, please refund the second payment.", "tier": "standard",
                   "hours_open": 2},
         "expected": {"priority": "low", "route": "billing"}},
        {"name": "VIP past the SLA",
         "state": {"message": "The app shows an error when I export a report.", "tier": "vip", "hours_open": 6},
         "expected": {"priority": "high", "route": "technical"}},
        {"name": "legal threat",
         "state": {"message": "Refund me now or my lawyer takes you to court.", "tier": "standard", "hours_open": 1},
         "expected": {"priority": "urgent", "route": "legal"},
         "status": {"priority": "forced", "route": "forced"},
         "note": "the hard check no_legal_threat forces both answers"},
    ],
    "model_pinned": {"legal threat"},
    "labels": [("I was charged twice, please refund one payment.", "billing"),
               ("Where is the invoice for March? The charge is wrong.", "billing"),
               ("Please refund my last payment.", "billing"),
               ("My card was charged but the order failed.", "billing"),
               ("The app shows an error on login.", "technical"),
               ("Export crashes every time, looks like a bug.", "technical"),
               ("The site is down for our whole team.", "technical"),
               ("I get error 500 when saving.", "technical"),
               ("How do I invite a colleague?", "general"),
               ("Quick question: can I change my username?", "general"),
               ("How do I export to CSV?", "general"),
               ("Can I use the API on the free plan?", "general")],
}

REFUNDS = {
    "title": "Refund requests: approve, send to a person, or reject.",
    "imports": "",
    "body": '''
WINDOW_DAYS = 30          # refunds within 30 days of purchase
AUTO_LIMIT = 200.0        # above this amount a person looks at it


# ---------- a computation: argument names are the facts it reads, the function name is the fact it sets
@cat.fn
def days_left(days_since_purchase: int) -> int:
    return WINDOW_DAYS - days_since_purchase


# ---------- hard checks: when one fails, the answers in `then` are forced, whatever the rule says
@cat.check(hard=True, then={"decision": "reject"})
def in_window(days_left: int) -> bool:
    return days_left >= 0


@cat.check(hard=True, then={"decision": "review"})
def not_serial_refunder(previous_refunds: int) -> bool:
    return previous_refunds < 3


# ---------- a rule: the question's answer from facts
@cat.rule("decision")
def decision(amount: float) -> str:
    return "approve" if amount <= AUTO_LIMIT else "review"


QUESTIONS = [
    Question("decision", "Refund, send to a person, or reject?", Answer.choice(["approve", "review", "reject"]),
             requires=["in_window", "not_serial_refunder"]),
]
''',
    "rule_only": "",
    "model": '''
# ---------- a question the model answers from the customer's own words (it proposes; a human reads the audit)
reason = model.decision("reason", "Why does the customer want a refund?", "message", REASONS, min_confidence=0.5)
CALIBRATION = HERE / "reason.calib.json"          # written by: solvi calibrate catalog.py:system reason labels.csv
if CALIBRATION.exists() and calibration_fits(CALIBRATION):
    reason.load_calibration(CALIBRATION)          # refuses a calibration made with another model
QUESTIONS.append(reason.question(cat))
''',
    "options": '''REASONS = {"damaged": "arrived broken or damaged", "late": "arrived late or never",
           "wrong_item": "not what was ordered or described", "changed_mind": "no longer wanted"}
''',
    "keywords": {"damaged": ["damaged", "broken", "cracked"], "late": ["late", "never arrived", "delayed"],
                 "wrong_item": ["wrong", "different", "not as described"],
                 "changed_mind": ["changed my mind", "no longer", "don't need"]},
    "model_question": "reason",
    "model_fact": "message",
    "example": {"amount": 80.0, "days_since_purchase": 10, "previous_refunds": 0},
    "cases": [
        {"name": "small, in the window", "state": {"amount": 80.0, "days_since_purchase": 10, "previous_refunds": 0},
         "expected": {"decision": "approve"}},
        {"name": "large amount", "state": {"amount": 950.0, "days_since_purchase": 3, "previous_refunds": 0},
         "expected": {"decision": "review"}},
        {"name": "past the window", "state": {"amount": 40.0, "days_since_purchase": 45, "previous_refunds": 0},
         "expected": {"decision": "reject"}, "status": {"decision": "forced"},
         "note": "the hard check in_window forces reject"},
        {"name": "serial refunder", "state": {"amount": 25.0, "days_since_purchase": 5, "previous_refunds": 4},
         "expected": {"decision": "review"}, "status": {"decision": "forced"}},
    ],
    "model_state": ["The parcel arrived damaged.", "It arrived two weeks late.", "The colour is wrong.",
                    "I changed my mind."],
    "model_pinned": set(),
    "labels": [("The mug arrived broken.", "damaged"), ("Screen was cracked in the box.", "damaged"),
               ("The lid is damaged.", "damaged"), ("It came broken, please refund.", "damaged"),
               ("The order never arrived.", "late"), ("Delivery was delayed by a month.", "late"),
               ("It arrived too late for the trip.", "late"), ("Parcel is late again.", "late"),
               ("You sent the wrong size.", "wrong_item"), ("The item is different from the photo.", "wrong_item"),
               ("I changed my mind about it.", "changed_mind"), ("I no longer need this.", "changed_mind")],
}

MINIMAL = {
    "title": "An expense check: approve an amount within a limit.",
    "imports": "",
    "body": '''

# ---------- a computation: argument names are the facts it reads, the function name is the fact it sets
@cat.fn
def headroom(limit: float, amount: float) -> float:
    return limit - amount


# ---------- a hard check: when it fails, "approve" is forced to "no", whatever the rule says
@cat.check(hard=True, then={"approve": "no"})
def within_limit(headroom: float) -> bool:
    return headroom >= 0


# ---------- a rule: the question's answer from facts
@cat.rule("approve")
def approve(amount: float) -> str:
    return "yes" if amount < 1000 else "no"


QUESTIONS = [Question("approve", "Approve the expense?", Answer.yes_no(), requires=["within_limit"])]
''',
    "rule_only": "",
    "model": '''
# ---------- a question the model answers from the employee's note
category = model.decision("category", "What kind of expense is it?", "note", CATEGORIES, min_confidence=0.5)
CALIBRATION = HERE / "category.calib.json"        # written by: solvi calibrate catalog.py:system category labels.csv
if CALIBRATION.exists() and calibration_fits(CALIBRATION):
    category.load_calibration(CALIBRATION)        # refuses a calibration made with another model
QUESTIONS.append(category.question(cat))
''',
    "options": '''CATEGORIES = {"travel": "flights, trains, hotels, taxis", "meals": "food and drinks",
              "equipment": "hardware and office supplies", "software": "licences and subscriptions"}
''',
    "keywords": {"travel": ["flight", "train", "hotel", "taxi"], "meals": ["lunch", "dinner", "coffee"],
                 "equipment": ["laptop", "monitor", "chair"], "software": ["licence", "license", "subscription"]},
    "model_question": "category",
    "model_fact": "note",
    "example": {"amount": 120.0, "limit": 500.0},
    "cases": [
        {"name": "within the limit", "state": {"amount": 120.0, "limit": 500.0}, "expected": {"approve": "yes"}},
        {"name": "over the limit", "state": {"amount": 700.0, "limit": 500.0}, "expected": {"approve": "no"},
         "status": {"approve": "forced"}, "note": "the hard check within_limit forces no"},
        {"name": "large amount", "state": {"amount": 1500.0, "limit": 5000.0}, "expected": {"approve": "no"}},
    ],
    "model_state": ["Taxi to the airport", "Team lunch with a client", "A second monitor"],
    "model_pinned": set(),
    "labels": [("Flight to Berlin", "travel"), ("Hotel, two nights", "travel"), ("Train ticket", "travel"),
               ("Taxi from the station", "travel"), ("Lunch with a client", "meals"), ("Team dinner", "meals"),
               ("Coffee for the workshop", "meals"), ("New laptop", "equipment"), ("Office chair", "equipment"),
               ("Monitor for home office", "equipment"), ("Design tool licence", "software"),
               ("Annual subscription to the IDE", "software")],
}

SPECS = {"support": SUPPORT, "refunds": REFUNDS, "minimal": MINIMAL}

STAND_IN = '''
# ---------- the model. By default a keyword stand-in, so tests and CI run without one; set SOLVI_DECIDE_MODEL to a real
# decider: a folder, a Hugging Face id you downloaded (solvi models pull solvi-ai/solvi-base; pip install "solvi[onnx]"),
# or systemone:URL#model. Its answers then change: pin them in cases.json only once you trust them.
KEYWORDS = <<KEYWORDS>>


class KeywordStandIn:
    """A stand-in decider: an option's logit is 2 × the number of its keywords in the text."""
    model_id = "stand-in/keywords"

    def fingerprint(self):
        return "keywords:" + json.dumps(KEYWORDS, sort_keys=True)

    def logits(self, items):
        out = []
        for it in items:
            text = it.text.lower()
            z = np.array([2.0 * sum(text.count(k) for k in KEYWORDS.get(o, ())) for o in it.options])
            out.append(np.stack([z, z], 1))       # one column per mode: choose one, several
        return out


def load_model():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    if src:
        from solvi.models import load
        return load(src)
    return DecideModel(KeywordStandIn(), meta={"format": "keyword stand-in", "temperature": 1.0})


def calibration_fits(path):
    """Is a calibration file for the model that runs now? A real model (SOLVI_DECIDE_MODEL) always loads it — and
    load_calibration refuses one made with another model. The keyword stand-in (tests, CI, a shell without the
    variable) loads only a calibration made with the stand-in: a real model's thresholds say nothing about it, so it
    runs without them, with a note."""
    if os.environ.get("SOLVI_DECIDE_MODEL"):
        return True
    made_with = json.loads(Path(path).read_text(encoding="utf-8")).get("models") or {}
    if KeywordStandIn.model_id in made_with:
        return True
    print(f"{Path(path).name}: calibrated with {', '.join(made_with) or 'another model'}; SOLVI_DECIDE_MODEL is not set, "
          "so the keyword stand-in runs without it", file=sys.stderr)
    return False


model = load_model()
'''


def catalog_py(template, with_model):
    t = SPECS[template]
    imports = t["imports"]
    if with_model:
        imports = "import json\nimport os\n" + imports + "import sys\nfrom pathlib import Path\n\nimport numpy as np\n"
    head = f'"""{t["title"]}\n\nMade by `solvi init --template {template}{" --with-model" if with_model else ""}`: a ' \
           "computation, a hard check and a rule" + (",\nand a question a model answers" if with_model else "") + """.

    solvi test .                                      the regression cases in cases.json
    solvi check catalog.py:system                     lint the catalog
    solvi ask catalog.py:system example.json --audit  ask once and see what each answer rests on
"""
    head += '"""\n'
    lines = [head, imports + ("\n" if imports else ""), "from solvi import Answer, Catalog, Question, System\n"]
    if with_model:
        lines.append("from solvi.core.deciders import DecideModel\n\nHERE = Path(__file__).parent\n")
    lines.append("\ncat = Catalog()\n")
    body = t["body"]
    if template == "support":
        body = body.replace("\nLEGAL =", "\n" + t["options"] + "LEGAL =")
    lines.append(body)
    if with_model:
        if template != "support":
            lines.append("\n" + t["options"])
        kw = "{\n" + "".join(f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n"
                             for k, v in t["keywords"].items()) + "}"
        lines.append(STAND_IN.replace("<<KEYWORDS>>", kw))
        lines.append(t["model"])
    else:
        lines.append(t["rule_only"])
    lines.append('''

def system():
    """The System: `solvi test`, `solvi check catalog.py:system`, `solvi ask`, `solvi serve catalog.py:system`."""
    return System(cat, QUESTIONS)
''')
    return "".join(lines)


def cases_json(template, with_model):
    t = SPECS[template]
    cases = json.loads(json.dumps(t["cases"]))
    extra = t.get("model_state")
    for i, c in enumerate(cases):
        if with_model and extra:                         # the text the model reads (support: the message it already has)
            c["state"][t["model_fact"]] = extra[i % len(extra)]
        if with_model and c["name"] not in t["model_pinned"]:
            c["expected"].pop(t["model_question"], None)
            (c.get("status") or {}).pop(t["model_question"], None)
    data = {"task": "catalog.py", "cases": cases}
    return _dump_cases(data)


def _dump_cases(data):
    out = ['{\n  "task": ' + json.dumps(data["task"]) + ',\n  "cases": [\n']
    out.append(",\n".join("    " + json.dumps(c, ensure_ascii=False) for c in data["cases"]))
    out.append("\n  ]\n}\n")
    return "".join(out)


def example_json(template, with_model):
    t = SPECS[template]
    st = dict(t["example"])
    if with_model and t.get("model_state"):
        st[t["model_fact"]] = t["model_state"][0]
    return json.dumps(st, ensure_ascii=False, indent=2) + "\n"


def labels_csv(template):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["text", "label"])
    w.writerows(SPECS[template]["labels"])
    return buf.getvalue()


def workflow_yml(workdir):
    return f"""# Decision regression tests and the catalog lint on every push and pull request.
# GitHub reads workflows from .github/workflows at the root of the repository: if this project is in a subfolder, move
# this file there (working-directory below already points to the project).
name: solvi
on: [push, pull_request]
jobs:
  decisions:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: {workdir}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "solvi>={MIN_SOLVI}"
      - run: solvi check catalog.py:system
      - run: solvi test .
"""


GITIGNORE = """__pycache__/
*.pyc
.venv/
# stored decisions (solvi ask --store, System(storage=...)): keep them out of git, or remove these lines
decisions.db
decisions.jsonl
"""


def readme_md(name, template, with_model):
    t = SPECS[template]
    q = t["model_question"]
    files = ["- `catalog.py` — the catalog: computations (`@cat.fn`), a hard check (`@cat.check(hard=True, then=...)`), "
             "rules (`@cat.rule`) and the questions" + (", one answered by a model" if with_model else "") + ";",
             "- `cases.json` — regression cases: the expected answers (and statuses) for fixed inputs;",
             "- `example.json` — an input for `solvi ask`;",
             "- `.github/workflows/solvi.yml` — CI: `solvi check` and `solvi test` on every push;"]
    if with_model:
        files.append("- `labels.csv` — labelled examples of the model's question, for `solvi models check` and "
                     "`solvi calibrate` (replace them with a few hundred from your own stream);")
    files.append("- `.gitignore`.")
    out = [f"# {name}", "", t["title"], "", f"Made by `solvi init --template {template}"
           + (" --with-model" if with_model else "") + "`: the model proposes, code decides, everything is in the trace.",
           "", "## Files", "", *files, "", "## Next steps", "", "```bash",
           f"pip install \"solvi{'[onnx]' if with_model else ''}>={MIN_SOLVI}\"",
           "solvi test .                                         # every case passes",
           "solvi check catalog.py:system --strict               # no lint findings",
           "solvi ask catalog.py:system example.json --audit     # one decision and what it rests on",
           "solvi ask catalog.py:system --state '" + json.dumps(json.loads(example_json(template, with_model)),
                                                                ensure_ascii=False) + "' --json",
           "```", "",
           ("1. Change the catalog: your own computations, checks and rules. A check that must never be overridden is "
            "`hard=True` with `then={question: answer}`, and the question lists it in `requires=`."),
           ("2. For every decision that matters, add a case to `cases.json` (`expected`, and `status` for a forced "
            "answer); `solvi test .` fails when an answer changes."),
           ("3. Keep decisions: `solvi ask ... --store decisions.db`, then `solvi report decisions.db`, `solvi verify "
            "decisions.db`, and `solvi diff decisions.db --system catalog.py:system` before you change a rule."),
           "4. Serve it: `pip install \"solvi[serve]\"`, `solvi serve catalog.py:system`."]
    if with_model:
        out += ["", "## The model", "",
                (f"The question `{q}` is answered by a decider. Without `SOLVI_DECIDE_MODEL` a keyword stand-in answers "
                 "(tests and CI need no model); with it, a real one:"), "", "```bash",
                "solvi models list                                    # published and downloaded deciders",
                "solvi models pull solvi-ai/solvi-base                # download it once",
                f"solvi models check solvi-ai/solvi-base --examples labels.csv --task \"{_task(template)}\"",
                "export SOLVI_DECIDE_MODEL=solvi-ai/solvi-base",
                f"solvi calibrate catalog.py:system {q} labels.csv --risk 0.1 --out {q}.calib.json",
                "```", "",
                (f"`solvi calibrate` sets when the model answers alone so that P(answered alone and wrong) ≤ 10% for inputs "
                 f"like the labelled ones, and writes `{q}.calib.json`; `catalog.py` loads it at start "
                 f"(`{q}.load_calibration(...)`), and refuses it when the model changed — calibrate again then. "
                 f"`{q}.calib.json` belongs to the model it was made with: where `SOLVI_DECIDE_MODEL` is not set (CI, a "
                 "new shell) the keyword stand-in answers without it and says so on stderr, so the tests keep passing; "
                 "commit the file, and set the variable wherever the real model should answer. Twelve "
                 "examples only show the mechanics: use a few hundred from your own stream. The model's answers are not "
                 "pinned in `cases.json` (except where a hard check forces them); pin them once you trust the model.")]
    return "\n".join(out) + "\n"


def _task(template):
    return {"support": "Which team should handle this ticket?", "refunds": "Why does the customer want a refund?",
            "minimal": "What kind of expense is it?"}[template]


def _workdir(target):
    """The project folder relative to the root of its git repository ("." when it is the root or not in one)."""
    p = target.resolve()
    for d in [p, *p.parents]:
        if (d / ".git").exists():
            rel = p.relative_to(d).as_posix()
            return rel or "."
    return "."


def files(target, template="support", with_model=False):
    """The files of a new project → {relative path: text}."""
    target = Path(target)
    out = {"catalog.py": catalog_py(template, with_model), "cases.json": cases_json(template, with_model),
           "example.json": example_json(template, with_model),
           "README.md": readme_md(target.resolve().name or "decisions", template, with_model),
           ".github/workflows/solvi.yml": workflow_yml(_workdir(target)), ".gitignore": GITIGNORE}
    if with_model:
        out["labels.csv"] = labels_csv(template)
    return out


def init(target=".", template="support", with_model=False, force=False):
    """Write a new project into `target` → the paths written. FileExistsError (nothing written; its first argument
    lists the files) when a file exists and force is False; NotADirectoryError when `target` is itself a file."""
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, not {template!r}")
    target = Path(target)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"{target} is a file, not a folder")
    fs = files(target, template, with_model)
    exist = [p for p in fs if (target / p).exists()]
    if exist and not force:
        raise FileExistsError(exist)
    written = []
    for rel, text in fs.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(p)
    return written


def cmd_init(a):
    """`solvi init` (see solvi.cli) → exit status."""
    try:
        written = init(a.dir, a.template, a.with_model, a.force)
    except NotADirectoryError:
        print(f"solvi init: {a.dir} is a file, not a folder; nothing written (name a folder)", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"solvi init: {a.dir} already has {', '.join(e.args[0])}; nothing written (--force overwrites)",
              file=sys.stderr)
        return 1
    for p in written:
        print(f"wrote {p}")
    where = "" if os.path.abspath(a.dir) == os.getcwd() else f"cd {a.dir} && "
    print(f"next: {where}solvi test . && solvi check catalog.py:system && solvi ask catalog.py:system example.json")
    return 0


def add_parser(sub):
    s = sub.add_parser("init", help="scaffold a decision project: a typed catalog, regression cases, README, CI workflow")
    s.add_argument("dir", nargs="?", default=".", help="the project folder (default: the current one; created if missing)")
    s.add_argument("--template", default="support", choices=TEMPLATES,
                   help="support (triage: priority and route), refunds (approve / review / reject) or minimal (an "
                        "expense within a limit); default support")
    s.add_argument("--with-model", action="store_true", help="add a question a decider answers (a keyword stand-in "
                                                            "until SOLVI_DECIDE_MODEL names a model) and labels.csv")
    s.add_argument("--force", action="store_true", help="overwrite files that exist")
    return s


__all__ = ["add_parser", "cmd_init", "files", "init", "SPECS", "TEMPLATES"]
