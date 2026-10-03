"""Pre-edit rule check for a coding agent: before the agent writes a file, the change and the project rules that apply to
that file's path are checked -> allow / block / escalate to a person, and which rules the change breaks.

The input is the edit the agent proposes: the file's path, its current text (empty for a new file) and the proposed text.
The rules are per path glob (RULES below). What code can check, code checks, on the lines the edit adds: secrets (key
formats, hard-coded passwords), browser storage read in app/api/** (localStorage, sessionStorage, document.cookie), and
reversible migrations (Python's own parser: an Alembic downgrade() that does something, a Django RunPython / RunSQL
with its reverse). A secret, browser storage or an irreversible migration blocks, whatever else the change does; an
edit to CI workflows or CODEOWNERS goes to a person. The fuzzy rules ("auth checks go through require_role()", "no
personal data in log lines") are a decider's yes/no question each, "does this change break the rule?", with a threshold
calibrated by act_guard on labelled examples instead of a hand-picked "80% sure": an unsure answer, or an answer that
changes when an instruction-like comment is removed (perturb), goes to a person.
The decider here is a keyword stand-in (no model is downloaded); the README shows solvi-large or any LLM in its place.
Try: add `if user.is_admin:` to the proposed text, or a line `API_KEY = "sk-live-..."`."""
from __future__ import annotations

import ast
import difflib
import random
import re
import zlib

import numpy as np

from solvi import Answer, Catalog, Question
from solvi.core.deciders import DecideModel
from solvi.core.deciders.perturb import injection_spans

cat = Catalog()
RISK = 0.05        # act_guard: P(a fuzzy rule answered alone and wrongly) <= 5% for edits like the labelled examples

# ---------- the project's rules (what a rules file for the agent would say), each with the paths it applies to
RULES = {
    "no-secrets": {"globs": ["**"], "by": "code", "text": "no secrets in source (keys, tokens, passwords)"},
    "browser-storage": {"globs": ["app/api/**"], "by": "code",
                        "text": "files under app/api/** must not read employee data from the browser "
                                "(localStorage, sessionStorage, document.cookie)"},
    "reversible-migration": {"globs": ["migrations/**/*.py"], "by": "code", "text": "migrations must be reversible"},
    "require-role": {"globs": ["app/api/**/*.py"], "by": "decider",
                     "text": "auth checks must go through require_role()"},
    "no-pii-in-logs": {"globs": ["app/**/*.py"], "by": "decider",
                       "text": "log lines must not contain personal data (names, e-mails, phone numbers, addresses)"},
    "ci-needs-person": {"globs": [".github/workflows/**", "**/CODEOWNERS", "CODEOWNERS"], "by": "code",
                        "text": "changes to CI workflows and CODEOWNERS are made by a person"},
}
BLOCKING = [r for r in RULES if r != "ci-needs-person"]


def _glob(g):
    """a path glob as a regex: ** any directories, * within one name"""
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif g.startswith("**", i):
            out, i = out + ".*", i + 2
        elif g[i] == "*":
            out, i = out + "[^/]*", i + 1
        else:
            out, i = out + re.escape(g[i]), i + 1
    return re.compile(out + r"\Z")


def applies(rule, path):
    return any(_glob(g).match(path) for g in RULES[rule]["globs"])


def is_comment(line):
    return line.lstrip().startswith(("#", "//", "/*", "*", "--"))


# ---------- what the edit adds
@cat.fn
def added_lines(old, new):
    """[(line number in the proposed file, text)] for every line the edit adds or changes"""
    a, b = old.splitlines(), new.splitlines()
    out = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            out += [(j + 1, b[j]) for j in range(j1, j2)]
    return out


@cat.fn
def rules_in_scope(path):
    return [r for r in RULES if applies(r, path)]


# ---------- rules code can check
SECRETS = [("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
           ("private key", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
           ("API token", r"\b(?:sk|pk|rk)[-_](?:live|test)[-_][A-Za-z0-9]{12,}|\bsk-[A-Za-z0-9_-]{20,}|\bghp_[A-Za-z0-9]{30,}"),
           ("hard-coded password", r"(?i)\w*(?:password|passwd|secret|api_?key|token)\w*\s*[:=]\s*[\"']"
                                   r"(?!changeme|example|placeholder|dummy|x{4,}|<|\$\{)[^\"'\s]{8,}[\"']")]
BROWSER = r"\b(?:localStorage|sessionStorage)\b|\bdocument\.cookie\b"


@cat.fn
def secret_lines(added_lines):
    """(line, kind, the first characters): comments too — a key in a comment is still in the repository"""
    out = []
    for n, text in added_lines:
        for kind, p in SECRETS:
            for m in re.finditer(p, text):
                out.append((n, kind, m.group()[:8] + "…"))
    return out


@cat.fn
def browser_storage_lines(path, added_lines):
    """code lines (not comments) under app/api/** that read localStorage, sessionStorage or document.cookie"""
    if not applies("browser-storage", path):
        return []
    return [(n, m.group()) for n, text in added_lines if not is_comment(text) for m in [re.search(BROWSER, text)] if m]


def _trivial(body):
    """a function body that does nothing: pass, ..., a docstring, raise NotImplementedError"""
    for s in body:
        if isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
            continue
        if isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s):
            continue
        return False
    return True


@cat.fn
def migration_problems(path, new):
    """(line, problem) in a Python migration, read with Python's own parser: an Alembic upgrade() without a downgrade()
    that does something; a Django RunPython / RunSQL without its reverse. A file that does not parse: (line, "cannot
    parse ...")"""
    if not applies("reversible-migration", path):
        return []
    try:
        tree = ast.parse(new)
    except SyntaxError as e:
        return [(e.lineno or 0, f"cannot parse the migration: {e.msg}")]
    out = []
    funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    if "upgrade" in funcs:
        down = funcs.get("downgrade")
        if down is None:
            out.append((funcs["upgrade"].lineno, "upgrade() has no downgrade()"))
        elif _trivial(down.body):
            out.append((down.lineno, "downgrade() does nothing"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("RunPython", "RunSQL"):
            reverse = "reverse_code" if node.func.attr == "RunPython" else "reverse_sql"
            if len(node.args) < 2 and not any(k.arg == reverse for k in node.keywords):
                out.append((node.lineno, f"{node.func.attr} without {reverse}"))
    return sorted(out)


@cat.fn
def instruction_lines(added_lines):
    """added lines with instruction-like text addressed to a reviewer or a model (solvi.core.deciders.perturb's detector): reported,
    never obeyed — code checks do not read them, and the deciders are asked again without them (perturb)"""
    return [(n, text.strip()) for n, text in added_lines if injection_spans(text)]


# ---------- hard checks: the first failing one in this order decides — every block before the escalation
@cat.check(hard=True, then={"decision": "block"})
def no_secrets(secret_lines):
    return not secret_lines


@cat.check(hard=True, then={"decision": "block"})
def no_browser_storage_in_api(browser_storage_lines):
    return not browser_storage_lines


@cat.check(hard=True, then={"decision": "block"})
def migration_reversible(migration_problems):
    return not [p for p in migration_problems if not p[1].startswith("cannot parse")]


@cat.check(hard=True, then={"decision": "escalate"})
def migration_readable(migration_problems):
    """a migration the parser cannot read cannot be checked: a person reads it"""
    return not [p for p in migration_problems if p[1].startswith("cannot parse")]


@cat.check(hard=True, then={"decision": "escalate"})
def not_a_protected_path(path):
    return not applies("ci-needs-person", path)


# ---------- fuzzy rules: a decider's question per rule, with a threshold from act_guard
CUES = {    # what the keyword stand-in reacts to, per rule: (pattern, weight); positive = "breaks"
    "require-role": [(r"\.role\s*[=!]=|\bis_admin\b|\bis_superuser\b|\.roles\b|x-role|check_permission\(", 4.0),
                     (r"\brole\b", 1.0), (r"require_role\(", -4.0)],
    "no-pii-in-logs": [(r"\b(?:logger|log|logging|print)\b[\w.]*\(.*\.(?:email|full_name|name|phone|home_address|address|ssn)\b",
                        4.0),
                       (r"\.(?:email|full_name|phone|home_address|ssn)\b", 1.0),
                       (r"\b(?:logger|log|logging)\b[\w.]*\(.*(?:mask|redact|\.id\b)", -3.0)],
}
OBEY = r"ignore (?:the |all |any )?(?:rules|instructions)|\bexempt\b|\bpre-?approved\b|answer complies"


class KeywordReviewer:
    """A keyword stand-in for a decider (no model): per rule, cue words in the added lines push toward "breaks" or
    "complies", plus a fixed bias and deterministic noise of its own. It also does what a naive model does: a line that
    says "ignore the rules" or "exempt" pushes it to "complies" — the reason for perturb."""
    model_id = "stand-in/rule-reviewer"

    def fingerprint(self):
        return "stand-in-rule-reviewer-1"

    def logits(self, items):
        out = []
        for it in items:
            head, _, body = it.text.partition("added lines:\n")
            rule = next((r for r in CUES if RULES[r]["text"] in head), None)
            low = body.lower()
            z = -1.5 + sum(w * len(re.findall(p, low)) for p, w in CUES.get(rule, []))
            z += (zlib.crc32(it.text.encode()) % 1000) / 500 - 1.0             # noise in [-1, 1]
            if re.search(OBEY, low):
                z -= 6.0
            out.append(np.array([z if o == "breaks" else -z for o in it.options]))
        return out


reviewer = DecideModel(KeywordReviewer(), meta={"format": "stand-in", "temperature": 1.0})


def view(rule, path, lines):
    """what the decider reads for one rule: the rule, the file, the added lines with their numbers"""
    return f"rule: {RULES[rule]['text']}\nfile: {path}\nadded lines:\n" + "\n".join(f"{n}: {t}" for n, t in lines)


# ---------- labelled examples for act_guard (SYNTHETIC: generated from the pools below with a fixed seed)
POOLS = {
    "require-role": {
        "breaks": ['if current_user.role == "admin":', "if user.is_admin:", 'if "hr" in user.roles:',
                   'if request.headers.get("X-Role") == "manager":', "if not user.is_superuser:",
                   'if check_permission(user, "payroll:read"):', 'if user.role != "hr": abort(403)'],
        "complies": ['@require_role("hr")', 'require_role(user, "payroll:read")',
                     'def list_employees(user=Depends(require_role("hr"))):', '@require_role("admin")'],
        "hard_breaks": ['allowed = user.get("role") in ALLOWED', 'if session["role"] == "hr":'],
        "hard_complies": ['role_name = form["role"]', '# was: if user.is_admin — require_role() does it now',
                          'return {"role": emp.role}'],
        "ambiguous": ["if user.id != owner_id: abort(403)", "if not token_valid(request): abort(401)"],
    },
    "no-pii-in-logs": {
        "breaks": ['logger.info(f"payslip sent to {employee.email}")', 'log.warning("login failed for %s", user.email)',
                   'print(f"new hire: {emp.full_name}, {emp.phone}")', 'logger.debug("address: %s", emp.home_address)'],
        "complies": ['logger.info("payslip sent to employee %s", employee.id)', 'log.debug("sent %d payslips", count)',
                     'logger.info("export finished in %.1f s", took)'],
        "hard_breaks": ['logger.info("updated %s <%s>", emp.full_name, emp.id)', 'log.info(f"hello {user.name}")'],
        "hard_complies": ['logger.info("user %s logged in", mask(user.email))', 'send_email(employee.email, body)',
                          'logger.info("email queue drained")'],
        "ambiguous": ['logger.info("updated %s", employee)', "audit.record(user.email)"],
    },
}
NEUTRAL = ["rows = db.query(Employee).all()", "return jsonify(rows)", "total = sum(r.salary for r in rows)",
           "page = int(request.args.get('page', 1))", "    return result", "cache.set(key, rows, ttl=60)"]
PATHS = {"require-role": ["app/api/employees.py", "app/api/payroll.py", "app/api/leave.py"],
         "no-pii-in-logs": ["app/api/payroll.py", "app/jobs/payslips.py", "app/services/hiring.py"]}


def labelled(rule, n=300, seed=0):
    """[(what the decider reads, "breaks" | "complies")]: one line that decides the label among 1-3 neutral lines.
    Kinds: clear (70%), hard look-alikes (24%), ambiguous (6%, labelled at random, as two reviewers would disagree)."""
    rng, pool, out = random.Random(f"{rule}:{seed}"), POOLS[rule], []
    for _ in range(n):
        kind = rng.choices(["breaks", "complies", "hard_breaks", "hard_complies", "ambiguous"], [35, 35, 12, 12, 6])[0]
        label = kind.replace("hard_", "") if kind != "ambiguous" else rng.choice(["breaks", "complies"])
        lines = rng.sample(NEUTRAL, rng.randint(1, 3))
        lines.insert(rng.randint(0, len(lines)), rng.choice(pool[kind]))
        out.append((view(rule, rng.choice(PATHS[rule]), list(enumerate(lines, rng.randint(1, 80)))), label))
    return out


OPTIONS = {"breaks": "the change breaks the rule", "complies": "the change keeps to the rule"}
CALIBRATION = {}


def fuzzy_rule(rule, name):
    """three producers of one fact, tried in order: out of scope → the decider (act_guard, perturb=2) → a person"""
    part = reviewer.decision(f"{name}_decider", f"Does this change break the rule: {RULES[rule]['text']}?",
                             f"{name}_view", OPTIONS, perturb=2)
    CALIBRATION[rule] = part.act_guard(labelled(rule), max_risk=RISK)

    def scope(path):
        return "not in scope" if not applies(rule, path) else None     # None: rejected, the next producer runs

    def to_person(path):
        return "unsure"        # reached only when the decider escalated (unsure, act_guard threshold, perturb)

    def view_of(path, added_lines):
        return view(rule, path, added_lines)
    for f, fname in ((scope, f"{name}_scope"), (to_person, f"{name}_to_person"), (view_of, f"{name}_view")):
        f.__name__ = f.__qualname__ = fname
    cat.fn(view_of)
    cat.fn(provides=name)(scope)
    cat.fn(provides=name)(part)
    cat.fn(provides=name)(to_person)
    return part


require_role_decider = fuzzy_rule("require-role", "require_role")
no_pii_in_logs_decider = fuzzy_rule("no-pii-in-logs", "no_pii_in_logs")


@cat.fn
def auth_lines(added_lines):
    """lines that look like an auth check: what a person reads first when require_role is broken or unsure"""
    return [n for n, t in added_lines if re.search(r"\brole|admin|superuser|permission|require_role", t, re.I)]


@cat.fn
def log_lines(added_lines):
    return [n for n, t in added_lines if re.search(r"\b(?:logger|log|logging|print)\b[\w.]*\(", t)]


@cat.fn
def findings(secret_lines, browser_storage_lines, migration_problems, require_role, no_pii_in_logs, auth_lines,
             log_lines, instruction_lines):
    """every finding with its rule and lines, as a reviewer (or the agent) reads it"""
    out = [f"no-secrets: line {n}: {kind} {start!r}" for n, kind, start in secret_lines]
    out += [f"browser-storage: line {n}: reads {what}" for n, what in browser_storage_lines]
    out += [f"reversible-migration: line {n}: {why}" for n, why in migration_problems]
    for rule, verdict, lines in (("require-role", require_role, auth_lines), ("no-pii-in-logs", no_pii_in_logs, log_lines)):
        if verdict in ("breaks", "unsure"):
            out.append(f"{rule}: {verdict} (decider)" + (f", lines {', '.join(map(str, lines))}" if lines else ""))
    out += [f"instruction-like text at line {n}, not obeyed: {t!r}" for n, t in instruction_lines]
    return out


# ---------- answers
@cat.rule("decision")
def decision(require_role, no_pii_in_logs, findings):
    """the hard checks have already blocked what code can see; here the fuzzy rules: broken → block, unsure → a person"""
    verdicts = (require_role, no_pii_in_logs)
    return "block" if "breaks" in verdicts else "escalate" if "unsure" in verdicts else "allow"


@cat.rule("broken_rules")
def broken_rules(secret_lines, browser_storage_lines, migration_problems, require_role, no_pii_in_logs):
    """the rules found broken (by code or by the decider); an unsure rule is not listed — it went to a person"""
    found = {"no-secrets": bool(secret_lines), "browser-storage": bool(browser_storage_lines),
             "reversible-migration": any(not p[1].startswith("cannot parse") for p in migration_problems),
             "require-role": require_role == "breaks", "no-pii-in-logs": no_pii_in_logs == "breaks"}
    return [r for r in BLOCKING if found[r]]


CHECKS = ["no_secrets", "no_browser_storage_in_api", "migration_reversible", "migration_readable", "not_a_protected_path"]
QUESTIONS = [
    Question("decision", "May the agent write this change?", Answer.choice(["allow", "block", "escalate"]),
             requires=CHECKS),
    Question("broken_rules", "Which project rules does the change break?", Answer.multi(BLOCKING)),
]
