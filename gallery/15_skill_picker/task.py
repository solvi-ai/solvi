"""Skill picker for a coding agent, with an honest "none": a user's prompt -> exactly one of the agent's skills, no skill,
or a person decides.

The agent's installed skills (SKILLS: name + description) include look-alikes: pdf-extract and pdf-forms, deploy-staging
and deploy-production, db-migrate and db-backup, release-notes and commit-message. Three answers are kept apart:
"none" (a confident answer: the prompt needs no skill), a pick, and an abstention (not sure: a person decides). A
wrong pick is the failure the threshold is for. The producers of the pick, in order: a skill the user names ("/db-backup",
"use the slides skill"; cited, and never from an instruction-like passage in pasted text), then a decider's choice over
the skills and "none", with a threshold from act_guard (risk 5%) on labelled prompts, min_margin (two skills that tie
escalate, with the candidates that cannot be ruled out) and perturb (an answer that changes without an instruction-like
passage escalates), then a person. A hard check: a production deploy only when the user's own words say production.
The labelled prompts are SYNTHETIC (templates and a seed, below). The decider is a keyword stand-in (no model is
downloaded); the README shows solvi-large or any LLM in its place.
Try: "Deploy this branch" (staging and production tie), or "/slides make three slides about the Q3 numbers"."""
from __future__ import annotations

import random
import re
import zlib

import numpy as np

from solvi import Answer, Catalog, Question, Quote
from solvi.decide import DecideModel
from solvi.perturb import injection_spans

cat = Catalog()
RISK = 0.05

SKILLS = {
    "pdf-extract": "Read a PDF and pull out its text or tables.",
    "pdf-forms": "Fill in the fields of a PDF form and save the filled copy.",
    "spreadsheet": "Open, edit or create a spreadsheet (.xlsx, .csv): columns, formulas, charts.",
    "slides": "Make or edit a slide deck.",
    "deploy-staging": "Deploy the current branch to the staging environment.",
    "deploy-production": "Deploy a tagged release to production.",
    "db-migrate": "Write and run a database schema migration.",
    "db-backup": "Take or restore a database backup.",
    "release-notes": "Write release notes from the pull requests merged since the last tag.",
    "commit-message": "Write a commit message for the staged changes.",
}
NONE = "none"
OPTIONS = {**SKILLS, NONE: "no skill is needed: the agent answers or edits the code itself"}


# ---------- 1. a skill the user names (cited); a name inside an instruction-like passage does not count
@cat.extract(provides="picked", exact=True)
def named_skill(prompt):
    """"/db-backup ..." or "use the db-backup skill": the named skill, quoted; None (the next producer runs) otherwise"""
    spans = injection_spans(prompt)
    names = "|".join(re.escape(s) for s in sorted(SKILLS, key=len, reverse=True))
    for m in re.finditer(rf"(?<![\w/])/({names})(?![\w-])|\buse (?:the )?({names}) skill\b", prompt, re.IGNORECASE):
        g = 1 if m.group(1) else 2
        if not any(a <= m.start() < b for a, b in spans):
            return Quote(prompt[m.start(g):m.end(g)], m.start(g), m.end(g), "prompt")
    return None


# ---------- 2. the decider: a choice over the skills and "none"
KEYWORDS = {        # the keyword stand-in: per skill, what it is about (one point) and what is done with it (a second point)
    "pdf-extract": [r"\bpdf\b", r"extract|pull (?:out|the)|\btext\b|\btables?\b"],
    "pdf-forms": [r"\bpdf\b", r"\bforms?\b|fill (?:in|out)|\bfields?\b"],
    "spreadsheet": [r"spreadsheet|\.xlsx|\.csv|\bexcel\b|\bsheet\b", r"\bcolumns?\b|formula|\bchart\b|pivot"],
    "slides": [r"\bslides?\b|\bdeck\b|presentation", r"\bmake\b|\bturn\b|\bdraft\b|\bedit\b"],
    "deploy-staging": [r"\bdeploy\w*|\bship\b|roll out", r"\bstag(?:e|ing)\b|preview|test environment"],
    "deploy-production": [r"\bdeploy\w*|\bship\b|roll out", r"\bprod(?:uction)?\b|\blive\b|customers"],
    "db-migrate": [r"migrat\w*|schema", r"\bcolumns?\b|\bindex\b|\balter\b|\btables?\b"],
    "db-backup": [r"backup|back up|snapshot|\brestore\b", r"database|\bdb\b|postgres"],
    "release-notes": [r"release notes|changelog|what changed", r"since the last|merged|pull requests?|\btag\b"],
    "commit-message": [r"\bcommit\b", r"\bstaged\b|\bdiff\b|message"],
}


class KeywordSkillReader:
    """A keyword stand-in for a decider (no model): a skill's logit grows with its matching cue groups, "none" is sure
    when nothing matches and less so as the best skill matches more, plus deterministic noise. Like a naive model it
    reads every word of the prompt, pasted text included — the reason for perturb."""
    model_id = "stand-in/skill-reader"

    def fingerprint(self):
        return "stand-in-skill-reader-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            hits = {s: (2 if re.search(ps[1], low) else 1) if re.search(ps[0], low) else 0 for s, ps in KEYWORDS.items()}
            z = []
            for o in it.options:
                base = 4.0 - 2.0 * max(hits.values()) if o == NONE else 2.5 * hits.get(o, 0)
                z.append(base + (zlib.crc32((o + it.text).encode()) % 1000) / 1000 - 0.5)     # noise in [-0.5, 0.5]
            out.append(np.array(z))
        return out


reader = DecideModel(KeywordSkillReader(), meta={"format": "stand-in", "temperature": 1.0})

# ---------- SYNTHETIC labelled prompts for act_guard and the conformal candidates
TEMPLATES = {
    "pdf-extract": ["Pull the tables out of {f}.pdf", "Extract the text from the attached PDF", "Read {f}.pdf and extract the totals table"],
    "pdf-forms": ["Fill in the PDF form {f}.pdf with my details", "Complete the fields of this PDF form", "Fill out the tax form PDF"],
    "spreadsheet": ["Add a VAT column to {f}.xlsx", "Clean the date column in {f}.csv", "Make a chart from the sheet in {f}.xlsx"],
    "slides": ["Make five slides about {t}", "Turn these notes into a presentation deck", "Draft a slide deck on {t}"],
    "deploy-staging": ["Deploy this branch to staging", "Ship the fix to the staging environment", "Roll out the preview build to stage"],
    "deploy-production": ["Deploy release v{n} to production", "Roll out v{n} to prod", "Ship tag v{n} to production for customers"],
    "db-migrate": ["Write a migration that adds an index on orders.created_at", "Alter the users table schema: add a column",
                   "Create a schema migration for the new {t} table"],
    "db-backup": ["Back up the database before we start", "Restore last night's db snapshot", "Take a postgres backup"],
    "release-notes": ["Write the release notes since the last tag", "Draft the changelog from the merged pull requests",
                      "Summarise what changed since the last release as release notes"],
    "commit-message": ["Write a commit message for the staged changes", "Suggest a commit message for this diff",
                       "Commit the staged work with a good message"],
    NONE: ["What does KeyError 'user_id' mean?", "Rename x to total in utils.py", "Explain how the retry decorator works",
           "Why is test_login flaky?", "Add type hints to parse_date", "Speed up the {t} loop in reports.py"],
}
LOOKALIKES = [      # (prompt, label): words of a skill without its job; two of them are honestly ambiguous
    ("Why does my deploy script print a warning?", NONE), ("Fix the typo in the form validation code", NONE),
    ("Explain what a database migration is", NONE), ("Where is the table of contents generated?", NONE),
    ("Read the release notes of the new web framework and tell me if anything breaks for us", NONE),
    ("Extract the table from this PDF into a spreadsheet", "pdf-extract"),
    ("Extract the table from this PDF into a spreadsheet", "spreadsheet"),
]
FILES, TOPICS = ["invoice_0923", "q3_report", "lease", "timesheet"], ["the Q3 numbers", "onboarding", "billing", "search"]


def synthetic_prompts(n=400, seed=0):
    """SYNTHETIC [(prompt, label)]: clear prompts for each skill (72%), prompts that need no skill (14%), look-alikes (14%)"""
    rng, out = random.Random(f"skills:{seed}"), []
    for _ in range(n):
        r = rng.random()
        if r < 0.14:
            out.append(rng.choice(LOOKALIKES))
            continue
        label = NONE if r < 0.28 else rng.choice(list(SKILLS))
        text = rng.choice(TEMPLATES[label]).format(f=rng.choice(FILES), t=rng.choice(TOPICS), n=f"2.{rng.randint(0, 9)}.0")
        out.append((text if rng.random() < 0.5 else text[0].lower() + text[1:] + rng.choice([".", "", " please", " thanks"]),
                    label))
    return out


EXAMPLES = synthetic_prompts()
skill_decider = reader.decision("skill_decider", "Which of the agent's skills does this prompt need, if any?", "prompt",
                                OPTIONS, other=False, min_margin=0.15, perturb=2)
CALIBRATION = skill_decider.act_guard(EXAMPLES, max_risk=RISK)
CANDIDATES = skill_decider.conformal(EXAMPLES, coverage=0.97)
cat.fn(provides="picked")(skill_decider)


# ---------- 3. a person
@cat.fn(provides="picked")
def to_person(prompt):
    return "unsure"        # reached only when the decider escalated (unsure, a near tie, an instruction-like passage)


# ---------- hard check and answers
@cat.check(hard=True, then={"outcome": "ask a person"})
def production_deploy_asked_for(picked, prompt):
    """a production deploy only when the user's own words say production (instruction-like passages removed)"""
    if picked != "deploy-production":
        return True
    own = "".join(" " if any(a <= i < b for a, b in injection_spans(prompt)) else c for i, c in enumerate(prompt))
    return bool(re.search(r"\bprod(?:uction)?\b", own, re.IGNORECASE))


@cat.rule("skill")
def skill(picked):
    """a skill, or "none"; unsure → abstain (a person decides)"""
    return None if picked == "unsure" else picked


@cat.rule("outcome")
def outcome(picked):
    return "ask a person" if picked == "unsure" else "no skill" if picked == NONE else "use a skill"


QUESTIONS = [
    Question("skill", "Which skill should the agent use?", Answer.choice(list(OPTIONS)),
             checkpoints=["production_deploy_asked_for"]),
    Question("outcome", "Use a skill, answer without one, or ask a person?",
             Answer.choice(["use a skill", "no skill", "ask a person"]), checkpoints=["production_deploy_asked_for"]),
]
