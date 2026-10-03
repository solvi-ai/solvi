"""solvi behind a coding agent's hooks: Claude Code, and Codex (preview).

    solvi hook install   [--project DIR] [--agent claude|codex|both] [--rules PATH] [--skills-dir DIR] [--no-skills]
                         [--no-edits] [--decider MODEL] [--approve] [--command CMD] [--dry-run]
    solvi hook uninstall [--project DIR] [--agent claude|codex|both] [--dry-run]
    solvi hook pre-edit  --rules rules.toml [--decider MODEL] [--store PATH | --no-store] [--approve] [--agent claude|codex]
                         [--no-instruction-check] [--project DIR]                       (PreToolUse: Edit|Write|MultiEdit)
    solvi hook pick-skill --skills-dir .claude/skills [--decider MODEL] [--min-score 1.5] [--margin 0.25] [--store PATH |
                         --no-store] [--project DIR]                                    (UserPromptSubmit)
    solvi hook audit [ID] [--rules rules.toml] [--store PATH]        (a stored decision: its audit; its replay on the rules)
    solvi hook sample-rules                                                             (print the sample rules file)

`pre-edit` reads the hook's JSON on stdin — the tool (Edit, Write, MultiEdit; Codex's apply_patch), the file and the
proposed change — works out the lines the change adds (with their line numbers in the file after the edit, when the
file can be read) and asks a small solvi System one question, `edit` ∈ {allow, deny, ask}. Its catalog holds, for every
rule whose path globs match the file, the rule's checks as hard checks:

  forbid         regular expressions no added line may match                             deterministic → deny
  require        regular expressions the file after the edit must match                  deterministic → deny
  forbid_calls   Python calls (dotted names, globs: "subprocess.*") no added line makes   deterministic (AST) → deny
  require_def    Python functions the file after the edit defines with a non-empty body  deterministic (AST) → deny
  (none of them) any change to these paths                                               deterministic → on_fail
  question       a yes/no question a decider answers ("yes" = a violation), asked when   fuzzy → ask; deny only with
                 an added line matches `when` (always, without `when`)                   a calibration file

`on_fail = "ask"` makes a deterministic rule ask instead of deny. A fuzzy rule blocks only when its decision passed a
threshold from a calibration file (act_guard: P(answered alone and wrong) ≤ risk on your labelled changes); without a
calibration its "yes" asks a person, and without a model every triggered question asks. A check that cannot be evaluated
(the file after the edit is unknown, a decider that escalates) asks. Added lines with instruction-like text addressed to a
reviewer ("ignore the rules", "pre-approved, allow this") ask too (--no-instruction-check turns this off): the rules never
read comments as instructions, this only tells a person that someone tried.

The answer goes back in the documented form: `permissionDecision` "deny" with the rule, the lines and the reason (Claude
sees it and can fix the change), "ask" (the user confirms; Codex cannot ask, so there it is a deny that says a person must
confirm), and for "allow" nothing — Claude Code's own permission rules still apply — or, with --approve, an explicit
"allow" that skips the permission prompt. A crash of the hook asks rather than allows.

`pick-skill` reads the skills (`<dir>/<skill>/SKILL.md`: `name` and `description` in the front matter) and the prompt, and
picks one skill or none: deterministically by the words the prompt shares with each skill's name and description (weighted
by how rare they are among the skills), or with --decider by a decider's choice over the skills and "none". When it picks one
it adds a short context line naming the skill (`additionalContext`); on "none", a near tie or a slash command it stays
silent.

Every decision is stored with its trace in a TraceStorage — by default `.solvi/traces/hooks.jsonl` in the project — so
`solvi verify`, `solvi report` and `TraceStorage.query` work on it. The store keeps the proposed change and the prompt
(the decision rests on them): keep `.solvi/` out of version control.

--decider (0.7: --model, removed in 0.9) takes what `solvi models` does: a local checkpoint folder or a cached Hugging Face id (never downloaded), an
OpenAI-compatible endpoint `llm:URL#model` (key in $SOLVI_LLM_API_KEY), a System One service `systemone:URL#model` (key in
$SOLVI_SYSTEMONE_API_KEY) or `module:attr`. Calibrate a fuzzy rule with the same model:

    SOLVI_HOOK_RULES=.claude/solvi-rules.toml SOLVI_HOOK_DECIDER=MODEL \\
        solvi calibrate solvi.experimental.hooks:rules_system RULE_answer labels.jsonl --risk 0.1 --out .claude/RULE.calib.json

and name the file in the rule (`calibration = "RULE.calib.json"`, relative to the rules file). RULE is the rule's id with
`-` as `_`; the labels are changes (`text`) with `label` true for a violation.

Exit status: 0 — the decision is on stdout (or nothing to say); install / uninstall: 0 — done, 2 — usage errors."""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from . import mark, warn_on_import

warn_on_import(__name__)

HOOK_STORE = os.path.join(".solvi", "traces", "hooks.jsonl")
DEFAULT_RULES = os.path.join(".claude", "solvi-rules.toml")
DEFAULT_SKILLS = os.path.join(".claude", "skills")
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "apply_patch")
OUTCOMES = ("allow", "deny", "ask")
RULE_KEYS = {"id", "paths", "why", "forbid", "require", "forbid_calls", "require_def", "question", "when", "on_fail",
             "calibration", "redact"}

SAMPLE_RULES = r"""# solvi rules for a coding agent's edits (solvi hook pre-edit).
# paths: globs relative to the project root — "*" stays inside a folder, "**" crosses folders, a leading "!" excludes.
# Deterministic checks (forbid, require, forbid_calls, require_def) are exact and block (on_fail = "ask" to ask instead).
# A question is fuzzy: it needs a model (--decider); it blocks only with a calibration file, otherwise it asks a person.

[[rule]]
id = "no-secrets-in-source"
paths = ["**", "!**/*.md", "!**/.env.example"]
why = "Secrets come from the environment or the secret store, never from source."
forbid = [
  '''(?i)\b(api[_-]?key|secret|passw(or)?d|token)\b["']?\s*[:=]\s*["'][^"'\s]{12,}["']''',
  '''\bAKIA[0-9A-Z]{16}\b''',
  '''-----BEGIN [A-Z ]*PRIVATE KEY-----''',
  '''\bsk-[A-Za-z0-9_-]{20,}''',
  '''\bgh[pousr]_[A-Za-z0-9]{36}\b''',
]
redact = true       # reasons show a masked excerpt, not the secret; the stored decision's content is erased

[[rule]]
id = "no-employee-data-from-browser"
paths = ["app/api/**"]
why = "Employee records are loaded on the server for the signed-in user; an id or a record the browser sends is never trusted."
forbid = [
  '''(?i)\b(req|request)\.(body|query|params|cookies|headers)\b.*\b(employee|salary|payroll|ssn)''',
  '''(?i)\b(searchParams|formData|params|query)\.get\(\s*["'][^"']*(employee|salary|payroll|ssn)''',
  '''(?i)\b(localStorage|sessionStorage|document\.cookie)\b.*\b(employee|salary|payroll)''',
]
question = "Does this change read employee data (ids, salaries, personal records) from what the browser sends, instead of from the server-side session?"
when = ['''(?i)employee|salary|payroll|\bssn\b''']
# calibration = "no_employee_data_from_browser.calib.json"   # solvi calibrate ... (see solvi hook --help)

[[rule]]
id = "migrations-reversible"
paths = ["**/alembic/versions/*.py", "**/migrations/versions/*.py"]
why = "Every migration can be rolled back: it defines downgrade() and the downgrade does something."
require_def = ["upgrade", "downgrade"]

[[rule]]
id = "no-dynamic-code"
paths = ["**/*.py"]
why = "No eval, exec or shell strings: code that runs text is where injections become commands."
forbid_calls = ["eval", "exec", "os.system", "subprocess.*(shell=True)"]

[[rule]]
id = "ci-workflows-need-a-person"
paths = [".github/workflows/**"]
why = "CI workflows hold deploy keys and permissions: a person looks at every change."
on_fail = "ask"
"""


class RulesError(ValueError):
    """A rules file that cannot be read: its path, the rule and what is wrong."""


class SettingsError(ValueError):
    """An agent's settings file (.claude/settings.json, .codex/hooks.json) that install / uninstall cannot read."""


# --------------------------------------------------------------------------------------------------- paths
def glob_regex(pattern):
    """A path glob → a compiled regex over project-relative POSIX paths: "*" and "?" stay inside a folder, "**" crosses
    folders ("**/x.py" also matches "x.py" at the root), everything else is literal."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def project_root(payload=None, explicit=None):
    """The project the hook works for: --project, else $CLAUDE_PROJECT_DIR, else the first folder up from the hook's cwd
    (the payload's `cwd`, else the process's) that holds .claude, .codex or .git, else that cwd."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    start = os.path.abspath((payload or {}).get("cwd") or os.getcwd())
    d = start
    while True:
        if any(os.path.exists(os.path.join(d, m)) for m in (".claude", ".codex", ".git")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return start
        d = up


def relpath(path, root):
    """A file path → project-relative with "/" (an absolute path outside the project stays absolute). Symbolic links
    are resolved, so a file reached through a linked folder is matched by the rules of where it really is."""
    p = os.path.realpath(os.path.join(root, path.replace("\\", "/")))
    r = os.path.relpath(p, os.path.realpath(root))
    return p.replace(os.sep, "/") if r.startswith("..") else r.replace(os.sep, "/")


# --------------------------------------------------------------------------------------------------- rules
@dataclass
class Rule:
    """One rule of a rules file (see the module docs)."""
    id: str
    paths: list
    why: str = ""
    forbid: list = field(default_factory=list)
    require: list = field(default_factory=list)
    forbid_calls: list = field(default_factory=list)
    require_def: list = field(default_factory=list)
    question: str | None = None
    when: list = field(default_factory=list)
    on_fail: str = "deny"
    calibration: str | None = None
    redact: bool = False

    @property
    def name(self):
        """The rule's id as a catalog name: letters, digits and "_"."""
        n = re.sub(r"\W", "_", self.id)
        return n if n and not n[0].isdigit() else "rule_" + n

    @property
    def deterministic(self):
        return bool(self.forbid or self.require or self.forbid_calls or self.require_def) or self.question is None

    def applies(self, path):
        inc = [p for p in self.paths if not p.startswith("!")]
        exc = [p[1:] for p in self.paths if p.startswith("!")]
        return any(glob_regex(p).match(path) for p in inc) and not any(glob_regex(p).match(path) for p in exc)

    def triggered(self, added):
        """Added lines matching `when` (all non-blank added lines without it): where the question is asked."""
        if not self.when:
            return [[n, t] for n, t in added if t.strip()]
        return [[n, t] for n, t in added if any(re.search(p, t) for p in self.when)]

    def spec(self):
        """The rule as JSON: the checks close over it, so a changed rule changes the catalog's fingerprint."""
        return json.dumps({k: getattr(self, k) for k in ("id", "why", "forbid", "require", "forbid_calls", "require_def",
                                                         "question", "when", "on_fail", "redact")}, sort_keys=True)


def _toml(text):
    import tomllib
    return tomllib.loads(text)


RESERVED_NAMES = frozenset({"path", "added_lines", "result_text", "change_text", "instructions", "no_instructions", "edit",
                            "edit_verdict"})
"""What the hook's catalog names itself: the facts it gives about a change and its own parts."""
RULE_PARTS = ("_lines", "_trigger", "_judged", "_answer")       # the parts a rule adds, by suffix of its name


def load_rules(path):
    """A rules file (TOML, or JSON by its extension) → [Rule]; RulesError says what is wrong and where."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        raise RulesError(f"cannot read the rules file {path}: {e.strerror}") from None
    try:
        data = json.loads(text) if path.endswith(".json") else _toml(text)
    except ValueError as e:
        raise RulesError(f"{path}: {e}") from None
    raw = data.get("rule", data.get("rules", []))
    if not isinstance(raw, list):
        raise RulesError(f"{path}: expected [[rule]] tables")
    out, seen, owned = [], set(), set()
    for i, r in enumerate(raw):
        where = f"{path}: rule {r.get('id', i + 1) if isinstance(r, dict) else i + 1}"
        if not isinstance(r, dict):
            raise RulesError(f"{where}: not a table")
        bad = set(r) - RULE_KEYS
        if bad:
            raise RulesError(f"{where}: unknown key(s) {', '.join(sorted(bad))} (known: {', '.join(sorted(RULE_KEYS))})")
        if not r.get("id") or not isinstance(r["id"], str):
            raise RulesError(f"{where}: needs an id")
        paths = r.get("paths")
        if isinstance(paths, str):
            paths = [paths]
        if not paths or not all(isinstance(p, str) and p.strip("!") for p in paths):
            raise RulesError(f"{where}: needs paths (a list of globs)")
        kw = {}
        for k in ("forbid", "require", "forbid_calls", "require_def", "when"):
            v = r.get(k, [])
            v = [v] if isinstance(v, str) else v
            if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
                raise RulesError(f"{where}: {k} must be a list of strings")
            kw[k] = v
        for k in ("forbid", "require", "when"):
            for p in kw[k]:
                try:
                    re.compile(p)
                except re.error as e:
                    raise RulesError(f"{where}: {k} pattern {p!r}: {e}") from None
        on_fail = r.get("on_fail", "deny")
        if on_fail not in ("deny", "ask"):
            raise RulesError(f"{where}: on_fail must be \"deny\" or \"ask\", not {on_fail!r}")
        q = r.get("question")
        if q is not None and (not isinstance(q, str) or not q.strip()):
            raise RulesError(f"{where}: question must be a text")
        cal = r.get("calibration")
        if cal is not None:
            if q is None:
                raise RulesError(f"{where}: a calibration is for a rule with a question")
            cal = os.path.join(os.path.dirname(os.path.abspath(path)), cal)
        rule = Rule(r["id"], list(paths), str(r.get("why", "")), question=q, on_fail=on_fail, calibration=cal,
                    redact=bool(r.get("redact", False)), **kw)
        if rule.name in seen:
            raise RulesError(f"{where}: two rules named {rule.name}")
        if rule.name in RESERVED_NAMES:               # the hook's own facts and parts: the rule would replace one, or be
            raise RulesError(f"{where}: the id {rule.id!r} is a name the hook uses itself "    # replaced by one, and never run
                             f"({', '.join(sorted(RESERVED_NAMES))}); choose another id")
        mine = {rule.name} | {rule.name + s for s in RULE_PARTS}
        clash = sorted(mine & owned)
        if clash:
            raise RulesError(f"{where}: its checks would be named like another rule's ({', '.join(clash)}); choose "
                             "another id")
        owned |= mine
        seen.add(rule.name)
        out.append(rule)
    return out


# --------------------------------------------------------------------------------------------------- the change
@dataclass
class Change:
    """One file of a proposed edit: the tool, the project-relative path, the added lines [[line, text]] and the file after
    the edit (None when it cannot be known: the file is not readable or the edit's old text is not in it). `lines` says
    what the line numbers count: "file" (lines of the file after the edit) or "new text" (lines of the edit's new text)."""
    tool: str
    path: str
    added: list
    result: str | None
    lines: str = "file"
    deleted: bool = False

    def text(self):
        """The change as a decider reads it: the file and the added lines."""
        body = "\n".join(f"{n}: {t}" for n, t in self.added)
        return f"File: {self.path}\nAdded lines ({'line numbers of the file after the edit' if self.lines == 'file' else 'lines of the new text'}):\n{body}"


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            return fh.read()
    except OSError:
        return None


def added_lines(before, after):
    """Lines of `after` that are not in `before` (a line diff) → [[line number in after, text]]."""
    import difflib
    a, b = before.splitlines(), after.splitlines()
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            out += [[j + 1, b[j]] for j in range(j1, j2)]
    return out


def _apply(cur, edits):
    """Apply [(old, new, replace_all)] to the text in order → the new text, or None when an old text is not there."""
    for old, new, every in edits:
        if old == "":
            if cur != "":
                return None
            cur = new
            continue
        if old not in cur:
            return None
        cur = cur.replace(old, new) if every else cur.replace(old, new, 1)
    return cur


def _loose(edits):
    """Without the file: the lines each edit's new text adds against its old text, numbered in the new text."""
    out = []
    for old, new, _ in edits:
        out += added_lines(old, new)
    return out


def _patch_sections(patch):
    """Codex's apply_patch envelope → [(action, path, lines, move_to)]."""
    out, cur = [], None
    for line in patch.splitlines():
        m = re.match(r"\*\*\* (Add|Update|Delete) File: (.+)", line)
        if m:
            cur = [m.group(1), m.group(2).strip(), [], None]
            out.append(cur)
        elif line.startswith("*** Move to: ") and cur is not None:
            cur[3] = line[len("*** Move to: "):].strip()
        elif line.startswith(("*** Begin Patch", "*** End Patch", "*** End of File")):
            continue
        elif cur is not None:
            cur[2].append(line)
    return out


def _apply_hunks(cur, lines):
    """The Update section's hunks applied to the current text → the new text, or None when a hunk's context is not found."""
    src = cur.splitlines()
    hunks, h = [], None
    for ln in lines:
        if ln.startswith("@@"):
            h = [[], []]
            hunks.append(h)
            continue
        if h is None:
            h = [[], []]
            hunks.append(h)
        tag, body = (ln[:1], ln[1:]) if ln[:1] in (" ", "-", "+") else (" ", ln)
        if tag in (" ", "-"):
            h[0].append(body)
        if tag in (" ", "+"):
            h[1].append(body)
    pos, out = 0, []
    for old, new in hunks:
        if not old:                                    # nothing to anchor on: the lines go at the end
            out += src[pos:] + new
            pos = len(src)
            continue
        for i in range(pos, len(src) - len(old) + 1):
            if src[i:i + len(old)] == old:
                out += src[pos:i] + new
                pos = i + len(old)
                break
        else:
            return None
    out += src[pos:]
    return "\n".join(out) + ("\n" if cur.endswith("\n") or not cur else "")


def changes_of(payload, root):
    """The hook's payload → [Change] (one per file; Codex's apply_patch may touch several). ValueError when it is not an
    edit this hook reads."""
    tool = payload.get("tool_name") or ""
    ti = payload.get("tool_input") or {}
    if not isinstance(ti, dict):
        raise ValueError("tool_input is not an object")
    patch = next((v for v in ti.values() if isinstance(v, str) and "*** Begin Patch" in v), None)
    if patch is not None:
        return _patch_changes(tool or "apply_patch", patch, root)
    fp = ti.get("file_path") or ti.get("path")
    if not isinstance(fp, str) or not fp:
        raise ValueError(f"{tool or 'the tool'}: no file_path in tool_input")
    abs_path = os.path.abspath(os.path.join(root, fp))
    rel = relpath(fp, root)
    cur = _read(abs_path)
    if tool == "Write" or ("content" in ti and "old_string" not in ti and "edits" not in ti):
        content = ti.get("content") if isinstance(ti.get("content"), str) else ""
        return [Change(tool or "Write", rel, added_lines(cur or "", content), content)]
    if isinstance(ti.get("edits"), list):
        edits = [(e.get("old_string", ""), e.get("new_string", ""), bool(e.get("replace_all")))
                 for e in ti["edits"] if isinstance(e, dict)]
    elif "new_string" in ti:
        edits = [(ti.get("old_string", ""), ti.get("new_string", ""), bool(ti.get("replace_all")))]
    else:
        raise ValueError(f"{tool}: no content, new_string or edits in tool_input")
    edits = [(o if isinstance(o, str) else "", n if isinstance(n, str) else "", r) for o, n, r in edits]
    result = _apply(cur, edits) if cur is not None else (_apply("", edits) if all(o == "" for o, _, _ in edits) else None)
    if result is None:
        return [Change(tool, rel, _loose(edits), None, lines="new text")]
    return [Change(tool, rel, added_lines(cur or "", result), result)]


def _patch_changes(tool, patch, root):
    out = []
    for action, path, lines, move_to in _patch_sections(patch):
        target = move_to or path
        rel = relpath(target, root)
        if action == "Delete":
            out.append(Change(tool, relpath(path, root), [], None, deleted=True))
        elif action == "Add":
            content = "\n".join(ln[1:] if ln.startswith("+") else ln for ln in lines) + "\n"
            out.append(Change(tool, rel, added_lines("", content), content))
        else:
            cur = _read(os.path.abspath(os.path.join(root, path)))
            result = _apply_hunks(cur, lines) if cur is not None else None
            if result is None:
                plus = [ln[1:] for ln in lines if ln.startswith("+")]
                out.append(Change(tool, rel, [[i + 1, t] for i, t in enumerate(plus)], None, lines="new text"))
            else:
                out.append(Change(tool, rel, added_lines(cur, result), result))
    if not out:
        raise ValueError("apply_patch: no file sections in the patch")
    return out


# --------------------------------------------------------------------------------------------------- the checks
_ADDRESSED = re.compile(
    r"(?i)\b(?:note|message|attention|instructions?)\s+(?:to|for)\s+(?:the\s+|any\s+)?(?:ai|llm|assistant|agent|model|"
    r"reviewer|claude|codex|copilot|linter|hook|guard|solvi)s?\b"
    r"|\b(?:ai|llm|assistant|agent|claude|codex|copilot|reviewers?|hooks?|guard|solvi)\b[^\n]{0,60}\b(?:must|should|"
    r"may|can|please)\s+(?:allow|approve|accept|skip|ignore|bypass|let)\b"
    r"|\b(?:ignore|disregard|skip|bypass|override)\s+(?:the\s+|all\s+|any\s+|previous\s+|prior\s+|above\s+)*"
    r"(?:rules?|instructions?|policy|policies|checks?|guard|hooks?|review)\b"
    r"|\b(?:pre-?approved|already\s+approved|approved\s+by\s+(?:security|legal|the\s+\w+\s+team))\b"
    r"|\b(?:this|the)\s+(?:rule|check|policy)\s+(?:does\s+not|doesn't)\s+apply\b")


_CUES = re.compile(r"(?i)ignor|disregard|forget|overrid|system|instruct|assistant|answer|prompt|\[/?inst|<\|?/?\w*\|?>|###|"
                   r"игнор|забудь|инструкц|систем")


def instruction_lines(added):
    """Added lines with instruction-like text addressed to a reviewer or an agent: overrides and role tags
    (solvi.core.deciders.perturb, without its action rules: code says "send" and "delete" all the time) and claims of an approval
    ("pre-approved", "ignore the rules", "note for the AI reviewer"). A heuristic: it tells a person, it never allows."""
    out = []
    for n, t in added:
        if _ADDRESSED.search(t):
            out.append([n, t])
        elif _CUES.search(t):                          # solvi.core.deciders.perturb only where one of its cue words is
            from ..core.deciders.perturb import injection_spans
            if injection_spans(t, actions=False):
                out.append([n, t])
    return out


def _mask(text, pattern):
    return re.sub(pattern, lambda m: m.group(0)[:4] + "…" if len(m.group(0)) > 4 else "…", text)


def _call_name(node):
    import ast
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_hits(patterns, added, result, path):
    """Added lines that call a forbidden Python function (dotted names with globs; "name(shell=True)" only when that
    keyword is passed as a true constant: True, 1) → [[line, text, call]]; SyntaxError when the code does not parse.
    A name is also read through the module's own imports (`import subprocess as sp` → sp.run is subprocess.run, `from
    os import system` → system is os.system), and a call is reported when any of its lines is an added line (at the
    first of them)."""
    import ast
    from fnmatch import fnmatchcase
    specs = []
    for p in patterns:
        m = re.fullmatch(r"\s*([\w.*?\[\]]+)\s*(?:\(\s*(\w+)\s*=\s*True\s*\))?\s*", p)
        specs.append((m.group(1), m.group(2)) if m else (p, None))
    lines = {n: t for n, t in added}
    text = result if result is not None else "\n".join(t for _, t in added)
    tree = ast.parse(text, filename=path)
    alias = {}                                         # a local name → the dotted name it was imported as
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            alias.update({a.asname: a.name for a in node.names if a.asname})
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            alias.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names if a.name != "*"})
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name is None:
            continue
        head, dot, rest = name.partition(".")
        names = [name] + ([alias[head] + dot + rest] if head in alias else [])
        for pat, kw in specs:
            hit = next((x for x in names if fnmatchcase(x, pat)), None)
            if hit is not None and (kw is None or any(k.arg == kw and isinstance(k.value, ast.Constant)
                                                      and bool(k.value.value) for k in node.keywords)):
                span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
                at = [n if result is not None else added[n - 1][0] for n in span]
                n = next((x for x in at if x in lines), None)
                if n is not None:
                    out.append([n, lines[n], hit + (f"({kw}=True)" if kw else "")])
                    break
    return sorted(out)


def _empty_body(fn):
    import ast
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return all(isinstance(s, ast.Pass) for s in body)


def _rule_lines(spec):
    """The deterministic findings of one rule: [[line, text, what]] (a line 0 names the file as a whole)."""
    rule = json.loads(spec)

    catch_all = not (rule["forbid"] or rule["require"] or rule["forbid_calls"] or rule["require_def"])

    def lines(path, added_lines, result_text):
        if catch_all:                                  # a rule without checks: any change to its paths is a finding
            return [[0, "", "*"]]
        out = []
        for n, t in added_lines:
            for p in rule["forbid"]:
                if re.search(p, t):
                    out.append([n, _mask(t, p) if rule["redact"] else t, "forbidden pattern"])
                    break
        if rule["forbid_calls"] and path.endswith(".py"):
            try:
                out += [[n, t, f"calls {c}"] for n, t, c in _call_hits(rule["forbid_calls"], added_lines, result_text,
                                                                       path)]
            except SyntaxError as e:                   # not parsed: plain names by their text; nothing found → a
                found = []                             # person (a glob or a keyword cannot be checked without the AST)
                for n, t in added_lines:
                    for p in rule["forbid_calls"]:
                        name = re.sub(r"\(.*", "", p).strip()
                        if "*" not in name and re.search(r"(?<![\w.])" + re.escape(name) + r"\s*\(", t):
                            found.append([n, t, f"calls {name}"])
                            break
                if not found:
                    raise ValueError(f"the Python code does not parse ({e.msg}, line {e.lineno}), so the calls it "
                                     "makes cannot be checked") from None
                out += found
        if rule["require"] or rule["require_def"]:
            if result_text is None:
                raise ValueError("the file after the edit is not known (its old text is not in the file, or the file "
                                 "cannot be read), so what it must contain cannot be checked")
            for p in rule["require"]:
                if not re.search(p, result_text, re.M):
                    out.append([0, "", f"the file does not match {p!r}"])
            if rule["require_def"] and path.endswith(".py"):
                import ast
                try:
                    tree = ast.parse(result_text, filename=path)
                except SyntaxError as e:
                    raise ValueError(f"the file after the edit does not parse: {e.msg} (line {e.lineno})") from None
                defs = {f.name: f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for d in rule["require_def"]:
                    if d not in defs:
                        out.append([0, "", f"no function {d}()"])
                    elif _empty_body(defs[d]):
                        out.append([defs[d].lineno, "", f"{d}() does nothing"])
        return out
    return lines


def _part(name, args, impl):
    """A catalog part named `name` that reads the facts `args` (its signature) and returns impl(*their values): a rule's
    parts are named after the rule, and the rule's spec is in impl's closure, so it is in the part's fingerprint."""
    import inspect

    def part(**facts):
        return impl(*[facts[x] for x in args])
    part.__name__ = part.__qualname__ = name
    part.__signature__ = inspect.Signature([inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD) for x in args])
    return part


def _rule_trigger(spec):
    rule = json.loads(spec)

    def trigger(added_lines):
        return [[n, t] for n, t in added_lines if (not rule["when"] and t.strip())
                or any(re.search(p, t) for p in rule["when"])]
    return trigger


def _none_found(found):
    return not found


def _not_yes(answer):
    return answer is not True and answer != "yes"


def no_instructions(instructions):
    """No added line carries instruction-like text addressed to a reviewer or an agent."""
    return not instructions


def edit_verdict(path):
    return "allow"


def fuzzy_part(model, rule):
    """The decision part of a rule's question (a yes/no over the change; "yes" is a violation), with its calibration file
    when the rule names one. The same construction in the hook and in rules_system, so a calibration fits both."""
    part = model.decision(f"{rule.name}_answer", rule.question, "change_text", type=bool, perturb=2)
    if rule.calibration:
        part.load_calibration(rule.calibration)
    return part


def edit_system(rules, change, model=None, instructions=True):
    """The System for one change: the rules that apply to its path (and, for a question, whose `when` an added line
    matches), each rule's checks as hard checks of the question `edit`, deny checks first."""
    from ..core.catalog import Answer, Catalog, Question
    from ..core.system import System
    cat = Catalog()
    deny, ask, used = [], [], []
    for r in rules:
        if not r.applies(change.path):
            continue
        det = r.deterministic
        fuzzy = r.question is not None and bool(r.triggered(change.added))
        if change.deleted and not (det and not (r.forbid or r.require or r.forbid_calls or r.require_def)):
            continue                                   # a deleted file: only the rules for any change apply
        if not det and not fuzzy:
            continue
        used.append(r.id)
        if det:
            cat.fn(_part(f"{r.name}_lines", ["path", "added_lines", "result_text"], _rule_lines(r.spec())))
            (deny if r.on_fail == "deny" else ask).append((_part(r.name, [f"{r.name}_lines"], _none_found), r.on_fail))
        if fuzzy:
            cat.fn(_part(f"{r.name}_trigger", ["added_lines"], _rule_trigger(r.spec())))
            if model is None:                          # nobody answers the question: a triggered one asks a person
                ask.append((_part(f"{r.name}_judged", [f"{r.name}_trigger"], _none_found), "ask"))
            else:
                cat.fn(fuzzy_part(model, r))
                on = "deny" if r.calibration else "ask"
                (deny if on == "deny" else ask).append((_part(f"{r.name}_judged", [f"{r.name}_answer"], _not_yes), on))
    if instructions:
        cat.fn(_part("instructions", ["added_lines"], instruction_lines))
        ask.insert(0, (no_instructions, "ask"))
    checks = []
    for f, outcome in deny + ask:                      # the first failed check declared decides: a deny wins
        cat.check(hard=True, then={"edit": outcome})(f)
        checks.append(f.__name__)
    cat.rule("edit")(edit_verdict)
    q = Question("edit", "May the agent make this change?", Answer.choice(list(OUTCOMES)), requires=checks)
    return mark(System(cat, [q]), "hooks"), used


def rules_system():
    """Every question of the rules file in $SOLVI_HOOK_RULES (default .claude/solvi-rules.toml) as decision parts of one
    System, with the decider in $SOLVI_HOOK_DECIDER ($SOLVI_HOOK_MODEL in 0.7, removed in 0.9): for `solvi calibrate
    solvi.experimental.hooks:rules_system RULE_answer ...`."""
    from ..core.catalog import Catalog
    from ..models import load as load_model
    from ..core.system import System
    from .._loader import LoadError                      # the command prints it; a library caller gets an exception
    spec = os.environ.get("SOLVI_HOOK_DECIDER")
    if not spec and os.environ.get("SOLVI_HOOK_MODEL"):
        raise LoadError("solvi.experimental.hooks:rules_system: $SOLVI_HOOK_MODEL was renamed in 0.8 and removed in 0.9: set "
                        "$SOLVI_HOOK_DECIDER")
    if not spec:
        raise LoadError("solvi.experimental.hooks:rules_system: set SOLVI_HOOK_DECIDER to the decider the hook uses (--decider)")
    model = load_model(spec)
    cat, qs = Catalog(), []
    for r in load_rules(os.environ.get("SOLVI_HOOK_RULES", DEFAULT_RULES)):
        if r.question is not None:
            qs.append(fuzzy_part(model, r).question(cat))
    if not qs:
        raise LoadError("solvi.experimental.hooks:rules_system: no rule with a question")
    return mark(System(cat, qs), "hooks")


# --------------------------------------------------------------------------------------------------- the store
def open_store(where, root):
    """The hook's TraceStorage: a JSON-lines file opens from its head (the count and last hash, checked against the last
    line) rather than by reading every record, so a long store stays fast; other kinds as open_storage opens them. Hooks
    may run in parallel: each append locks the file and continues the chain from what the others wrote (JSONLStorage)."""
    from ..core.store import JSONLStorage, open_storage
    path = where if os.path.isabs(where) or "://" in where else os.path.join(root, where)
    if "://" in path or path.endswith((".db", ".sqlite", ".sqlite3", ".duckdb")):
        return open_storage(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return JSONLStorage(path, index=False)


def _save(store, res, meta):
    return None if store is None else store.save(res, meta)


# --------------------------------------------------------------------------------------------------- pre-edit
def _where(n, change):
    if not n:
        return "the file"
    return f"line {n}" if change.lines == "file" else f"line {n} of the new text"


def reasons(system, res, change, rules):
    """The failed checks as (outcome, rule id, line): the rule, the lines, the rule's why; outcome is what the check
    forces."""
    by_name = {r.name: r for r in rules}
    cat, vals = system.catalog, res.values
    order = {n: i for i, n in enumerate(cat.parts)}
    failed = sorted((x for x in res.trace.records if x.value is False and x.name in cat.parts
                     and cat.parts[x.name].kind == "check"), key=lambda x: order[x.name])
    out, told = [], set()

    def why(r):                                        # a rule's why once, with its first finding
        if not r.why or r.id in told:
            return ""
        told.add(r.id)
        return f". {r.why}"
    for rec in failed:
        n = rec.name
        lvl = cat.parts[n].then.get("edit", "ask")
        if n == "no_instructions":
            found = vals.get("instructions") or []
            out.append((lvl, None, "instruction-like text in the change, addressed to a reviewer or an agent (the rules do not "
                       "take instructions from the code): " + "; ".join(f"{_where(k, change)}: {_short(t)}"
                                                                         for k, t in found[:3])))
            continue
        base = n[:-len("_judged")] if n.endswith("_judged") else n
        r = by_name.get(base)
        if r is None:
            out.append((lvl, None, f"{n} failed"))
            continue
        if n.endswith("_judged"):
            trig = vals.get(f"{r.name}_trigger") or []
            at = ", ".join(_where(k, change) for k, _ in trig[:3])
            if f"{r.name}_answer" in vals:
                rec_a = next((x for x in res.trace.records if x.name == f"{r.name}_answer"), None)
                probs = (rec_a.probs or {}) if rec_a is not None else {}
                p_yes = probs.get(True, probs.get("yes"))
                g = ((rec_a.extra or {}).get("guarantee") or {}) if rec_a is not None else {}
                how = "the model says yes" + (f" (P(yes) = {p_yes:.2f})" if isinstance(p_yes, (int, float)) else "")
                how += (f"; calibrated: P(answered alone and wrong) ≤ {g['risk']:g}" if isinstance(g, dict) and
                        isinstance(g.get("risk"), (int, float)) else "") if r.calibration else \
                    "; without a calibration a person confirms"
            else:
                how = "no model is configured: a person decides"
            out.append((lvl, r.id, f"{r.id}: {r.question} — {how}" + (f"; see {at}" if at else "") + why(r)))
            continue
        hits = vals.get(f"{r.name}_lines") or []
        if hits and hits != [[0, "", "*"]]:
            where = "; ".join(f"{_where(k, change)}: {_short(t)}" if t else f"{_where(k, change)}: {what}"
                              for k, t, what in hits[:5]) + (f" (and {len(hits) - 5} more)" if len(hits) > 5 else "")
            extra = sorted({what for _, t, what in hits if t and what != "forbidden pattern"})
            out.append((lvl, r.id, f"{r.id} — {where}" + (f" [{', '.join(extra)}]" if extra else "") + why(r)))
        else:
            out.append((lvl, r.id, f"{r.id}: any change to {change.path} goes to a person" + why(r)))
    if not out:
        unresolved = res.flow.unresolved.get("edit") or []
        for rec in res.trace.records:
            if rec.error and rec.name in cat.parts and cat.parts[rec.name].kind != "check":
                base = re.sub(r"_(lines|answer|trigger)$", "", rec.name)
                r = by_name.get(base)
                who = r.id if r is not None else rec.name
                out.append(("ask", who, f"{who}: cannot be checked — {rec.error}"))
        if not out:
            out += [("ask", None, f"cannot evaluate {n}") for n in unresolved] or [("ask", None, res["edit"].why)]
    return out


def _short(t, n=120):
    t = t.strip()
    return t if len(t) <= n else t[:n - 1] + "…"


def decide_change(rules, change, model=None, instructions=True, store=None, meta=None):
    """One change → (outcome, reasons, rules used, stored id)."""
    system, used = edit_system(rules, change, model, instructions)
    state = {"path": change.path, "added_lines": change.added, "result_text": change.result}
    if model is not None:
        state["change_text"] = change.text()
    res = system.ask(state)
    r = res["edit"]
    outcome = r.answer if r.status in ("ok", "forced") and r.answer in OUTCOMES else "ask"
    found = [] if outcome == "allow" else reasons(system, res, change, rules)
    decided = {rid for lvl, rid, _ in found if lvl == outcome and rid}
    why = [t for lvl, _, t in found if lvl == outcome] + [f"also for a person: {t}" for lvl, rid, t in found
                                                          if lvl != outcome and (rid is None or rid not in decided)]
    extra = {"deleted": True} if change.deleted else {}
    if not instructions:
        extra["instruction_check"] = False
    sid = _save(store, res, dict(meta or {}, hook="pre-edit", tool=change.tool, path=change.path, outcome=outcome,
                                 reasons=[t for _, _, t in found], rules=used, **extra))
    masked = sorted({rid for _, rid, _ in found if rid and any(r.id == rid and r.redact for r in rules)})
    if sid is not None and masked:
        # a rule with redact = true matched: the change holds what must not be kept (a secret), and the stored input
        # state is the change — erase the record's content (its place, hash and answer stay; the chain verifies)
        store.redact(sid, by="solvi hook", note=f"redact = true: {', '.join(masked)}")
    return outcome, why, used, sid


def _decision_json(outcome, text, agent, approve):
    if outcome == "allow":
        if not approve or agent == "codex":             # Codex: "allow" without updatedInput is not supported
            return None
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                       "permissionDecisionReason": text}}
    if outcome == "ask" and agent == "codex":           # Codex hooks cannot ask: a deny that says a person confirms
        outcome, text = "deny", "A person must confirm this edit (Codex hooks cannot ask). " + text
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": outcome,
                                   "permissionDecisionReason": text}}


def pre_edit(payload, rules, root, model=None, store=None, instructions=True):
    """The PreToolUse payload → (outcome, message, [(change, outcome, reasons, stored id)]). A tool that is not an edit
    gets "allow" and no checks."""
    tool = payload.get("tool_name") or ""
    ti = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    if tool and tool not in EDIT_TOOLS and not any(isinstance(v, str) and "*** Begin Patch" in v for v in ti.values()):
        return "allow", f"{tool}: not an edit", []
    meta = {k: payload[k] for k in ("session_id", "tool_use_id", "prompt_id", "agent_id") if payload.get(k)}
    results = []
    for ch in changes_of(payload, root):
        o, why, _, sid = decide_change(rules, ch, model, instructions, store, meta)
        results.append((ch, o, why, sid))
    return _summary(results)


def _summary(results):
    rank = {"allow": 0, "ask": 1, "deny": 2}
    outcome = max((o for _, o, _, _ in results), key=rank.get, default="allow")
    lines = []
    for ch, o, why, _ in results:
        if o == "allow":
            continue
        head = f"solvi {'blocked' if o == 'deny' else 'asks about'} this edit of {ch.path}:"
        lines.append(head + "".join(f"\n- {w}" for w in why))
    return outcome, "\n".join(lines) or "no rule objects", results


def cmd_pre_edit(a):
    payload = _payload()
    try:
        root = project_root(payload, a.project)
        rules = load_rules(a.rules if os.path.isabs(a.rules) else os.path.join(root, a.rules))
        store = None if a.no_store else open_store(a.store, root)
        outcome, text, results = pre_edit(payload, rules, root, _model(a.model), store, not a.no_instruction_check)
        ids = [sid for *_, sid in results if sid]
        if ids and outcome != "allow":
            where = os.path.relpath(store.path, root) if hasattr(store, "path") else a.store
            text += f"\n(solvi decision {', '.join(ids)} in {where})"
    except (Exception, SystemExit) as e:  # noqa: BLE001 — a policy hook that crashes (or a model file that exits) must
        # not let the edit through silently
        outcome, text = "ask", f"solvi hook pre-edit could not check this edit ({type(e).__name__}: {e}); a person decides"
    out = _decision_json(outcome, text if outcome != "allow" else "solvi: no rule objects", a.agent, a.approve)
    if out is not None:
        print(json.dumps(out, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------------------------------- pick-skill
STOP = set("""a an and any are as at be been but by can could do does for from get go has have how i if in into is it its
just like make me my need needs not now of on or our please should so some than that the their them then there these
they this to up us use used uses using want wants was we what when where which while who why will with would you your
ask asks asked user users skill skills claude help also only via etc e g eg ie add adds change changes create new
update fix make makes write writes under over all each every more most other same""".split())


def _stem(w):
    for suf in ("ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def words(text):
    """Content words of a text, lower-cased and lightly stemmed (a set)."""
    return {_stem(w) for w in re.findall(r"[^\W_]+", text.lower()) if w not in STOP and len(w) > 1}


def read_skills(dirs):
    """Skill folders (`<dir>/<name>/SKILL.md`, or a `<dir>/<name>.md` file) → [[name, description]] sorted by name; the
    front matter's `name` (default: the folder) and `description` (a line, a quoted string or a `>` / `|` block)."""
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for entry in sorted(os.listdir(d)):
            p = os.path.join(d, entry, "SKILL.md") if os.path.isdir(os.path.join(d, entry)) else os.path.join(d, entry)
            if not p.endswith(".md") or not os.path.isfile(p):
                continue
            meta = front_matter(_read(p) or "")
            name = str(meta.get("name") or os.path.splitext(entry)[0]).strip()
            desc = str(meta.get("description") or "").strip()
            if name and name not in out:
                out[name] = desc
    return [[n, out[n]] for n in sorted(out)]


def front_matter(text):
    """The simple `key: value` front matter of a Markdown file (no YAML library): strings, quoted strings, `>` / `|`
    blocks."""
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()[1:]
    out, key, block = {}, None, None
    for ln in lines:
        if ln.strip() == "---":
            break
        if block is not None and (ln.startswith((" ", "\t")) or not ln.strip()):
            block.append(ln.strip())
            out[key] = " ".join(x for x in block if x)
            continue
        block = None
        m = re.match(r"([A-Za-z_][\w-]*)\s*:\s*(.*)$", ln)
        if not m:
            continue
        key, v = m.group(1), m.group(2).strip()
        if v in (">", "|", ">-", "|-", ">+", "|+"):
            block, out[key] = [], ""
        elif len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            out[key] = v[1:-1]
        else:
            out[key] = v
    return out


def _skill_scores(spec):
    cfg = json.loads(spec)

    def skill_scores(prompt, skills):
        """Each skill's score: the rarity-weighted words the prompt shares with its name (double) and description."""
        import math
        said = {}
        for w in re.findall(r"[^\W_]+", prompt.lower()):     # each stem shown as the prompt wrote it
            said.setdefault(_stem(w), w)
        p = words(prompt)
        toks = [(n, words(n.replace("-", " ").replace("_", " ")), words(d)) for n, d in skills]
        df = {}
        for _, nt, dt in toks:
            for w in nt | dt:
                df[w] = df.get(w, 0) + 1
        out = {}
        for n, nt, dt in toks:
            hit = sorted(p & (nt | dt))
            s = sum(math.log(1 + len(toks) / df[w]) * (2 if w in nt else 1) for w in hit)
            out[n] = [round(s, 4), [said.get(w, w) for w in hit]]
        return {"scores": out, "min_score": cfg["min_score"], "margin": cfg["margin"]}
    return skill_scores


def skill_rule(skill_scores):
    """The best skill when it reaches min_score and leads the next by more than margin; "none" below min_score; a near
    tie abstains (nothing is said)."""
    ranked = sorted(skill_scores["scores"].items(), key=lambda kv: -kv[1][0])
    if not ranked or ranked[0][1][0] < skill_scores["min_score"]:
        return "none"
    if len(ranked) > 1 and ranked[1][1][0] >= ranked[0][1][0] * (1 - skill_scores["margin"]):
        return None
    return ranked[0][0]


def skill_system(skills, model=None, min_score=1.5, margin=0.25, calibration=None):
    from ..core.catalog import Answer, Catalog, Question
    from ..core.system import System
    cat = Catalog()
    names = [n for n, _ in skills]
    if model is None:
        f = _skill_scores(json.dumps({"min_score": min_score, "margin": margin}, sort_keys=True))
        cat.fn(f)
        cat.rule("skill")(skill_rule)
    else:
        opts = {n: (d or n) for n, d in skills}
        opts["none"] = "none of these skills: the request needs none of them"
        part = model.decision("skill", "Which of these skills does this request need?", "prompt", options=opts,
                              min_margin=margin)
        if calibration:
            part.load_calibration(calibration)
        part.question(cat)
    return mark(System(cat, [Question("skill", "Which skill does this request need?", Answer.choice(names + ["none"]))]),
                "hooks")


def pick_skill(prompt, skills, model=None, min_score=1.5, margin=0.25, calibration=None, store=None, meta=None):
    """→ (the skill's name or None, the response, stored id)."""
    system = skill_system(skills, model, min_score, margin, calibration)
    res = system.ask({"prompt": prompt, "skills": skills})
    r = res["skill"]
    pick = r.answer if r.status in ("ok", "forced") and r.answer != "none" else None
    sid = _save(store, res, dict(meta or {}, hook="pick-skill", skill=pick, status=r.status))
    return pick, res, sid


def skill_context(name, skills, res):
    desc = dict((n, d) for n, d in skills).get(name, "")
    first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0] if desc else ""
    sc = res.values.get("skill_scores")
    hit = sc["scores"][name][1] if isinstance(sc, dict) and name in sc.get("scores", {}) else None
    return (f'The project skill "{name}" matches this request' + (f" ({first.rstrip('.')})" if first else "")
            + (f"; shared words: {', '.join(hit)}" if hit else "") + ". (solvi pick-skill)")


def cmd_pick_skill(a):
    payload = _payload()
    try:
        prompt = payload.get("prompt") or ""
        if not isinstance(prompt, str) or not prompt.strip() or prompt.lstrip().startswith("/"):
            return 0
        root = project_root(payload, a.project)
        dirs = [d if os.path.isabs(d) else os.path.join(root, d) for d in (a.skills_dir or [DEFAULT_SKILLS])]
        skills = read_skills(dirs)
        if not skills:
            return 0
        store = None if a.no_store else open_store(a.store, root)
        meta = {k: payload[k] for k in ("session_id", "prompt_id") if payload.get(k)}
        pick, res, _ = pick_skill(prompt, skills, _model(a.model), a.min_score, a.margin, a.calibration, store, meta)
        if pick is None:
            return 0
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                 "additionalContext": skill_context(pick, skills, res)}},
                         ensure_ascii=False))
    except (Exception, SystemExit) as e:  # noqa: BLE001 — a skill hint is optional: never stop the prompt (an exit
        # status 2 of this hook would block it)
        print(f"solvi hook pick-skill: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------------- audit
def stored_edit(stored, rules, model=None):
    """A stored pre-edit decision → (the System rebuilt from its recorded change and the rules, the response loaded with
    that catalog). The rules and the model must be the ones that decided, or the replay says what changed."""
    m = stored.meta or {}
    init = stored.data["response"]["trace"]["init"]
    ch = Change(m.get("tool", ""), init["path"], init["added_lines"], init["result_text"], deleted=bool(m.get("deleted")))
    system, _ = edit_system(rules, ch, model, m.get("instruction_check", True))
    return system, stored.response(system.catalog)


def cmd_audit(a):
    """`solvi hook audit [ID]`: a stored pre-edit decision (the last by default) — its audit, and whether it re-computes
    with the current rules (a replay of every step from its recorded inputs)."""
    root = project_root(None, a.project)
    path = a.store if os.path.isabs(a.store) else os.path.join(root, a.store)
    if not os.path.exists(path):
        print(f"solvi: no store at {path}", file=sys.stderr)
        return 2
    from ..core.store import open_storage
    store = open_storage(path)
    recs = [r for r in store.iter() if (r.meta or {}).get("hook") == "pre-edit"]
    pick = [r for r in recs if r.id == a.id] if a.id else recs[-1:]
    if not pick:
        print(f"solvi: no pre-edit decision {a.id or ''} in {path}".replace("  ", " "), file=sys.stderr)
        return 2
    rules = load_rules(a.rules if os.path.isabs(a.rules) else os.path.join(root, a.rules))
    st = pick[0]
    system, res = stored_edit(st, rules, _model(a.model))
    m = st.meta or {}
    print(f"decision {st.id}: {m.get('outcome')} — {m.get('tool')} {m.get('path')}")
    for w in m.get("reasons") or []:
        print(f"  - {w}")
    print(res.audit("edit"))
    rep = res.trace.replay(system, res.flow)
    if rep["ok"]:
        print(f"replay: every step re-computes with the current rules ({rep['steps']} steps; catalog {rep['catalog']})")
    else:
        print("replay: does not re-compute with the current rules:")
        for step, name, why in rep["mismatches"][:10]:
            print(f"  step {step} {name}: {why}")
    return 0 if rep["ok"] else 1


# --------------------------------------------------------------------------------------------------- install
def _payload():
    text = sys.stdin.read()
    try:
        d = json.loads(text) if text.strip() else {}
    except ValueError:
        d = {}
    return d if isinstance(d, dict) else {}


def _model(spec):
    if not spec:
        return None
    from ..models import load
    return load(spec)


def _solvi_command():
    import shutil
    exe = shutil.which("solvi")
    if exe:
        return _q(os.path.abspath(exe))
    return f"{_q(sys.executable)} -m solvi"


def _q(s):
    import shlex
    return shlex.quote(s)


# solvi's own entries: "solvi hook pre-edit", ".../python -m solvi hook pick-skill" — also when the path needed quoting
# ("'/opt/my tools/solvi' hook pre-edit": the closing quote sits between the command and "hook")
MARK = re.compile(r"(^|[\s/\\'\"])(solvi|-m\s+solvi)['\"]?\s+hook\s+(pre-edit|pick-skill)\b")


_PRINTS = {"echo", "printf", "grep", "cat", "true", "false", "test", ":"}   # commands that only mention solvi's words


def _ours(handler):
    """Is a hook handler one that solvi installed? Its command runs `solvi hook pre-edit | pick-skill` — as words of
    the command line (`solvi`, a path to it, `python -m solvi`, behind a launcher such as `uv run`), not inside a
    quoted argument, and not as the arguments of a command that only prints or searches ("echo solvi hook pre-edit")."""
    cmd = handler.get("command") if isinstance(handler, dict) else None
    if not isinstance(cmd, str) or not MARK.search(cmd):
        return False
    import shlex
    try:
        toks = shlex.split(cmd)
    except ValueError:                                 # an unbalanced quote: by the pattern alone
        return True
    for i, tok in enumerate(toks[:-2]):
        runs = os.path.basename(tok) in ("solvi", "solvi.exe")        # also the "solvi" of "python -m solvi"
        if runs and toks[i + 1] == "hook" and toks[i + 2] in ("pre-edit", "pick-skill"):
            return os.path.basename(toks[0]) not in _PRINTS
    return False


def _settings_path(root, agent):
    return os.path.join(root, ".claude", "settings.json") if agent == "claude" else os.path.join(root, ".codex", "hooks.json")


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except ValueError as e:
        raise SettingsError(f"{path} is not JSON ({e}); fix it or move it away, nothing was changed") from None
    if not isinstance(d, dict):
        raise SettingsError(f"{path} is not a JSON object; nothing was changed")
    return d


def _strip(settings):
    """Remove solvi's handlers → (settings, [(event, matcher, command)] removed); empty groups and events go too."""
    hooks = settings.get("hooks")
    removed = []
    if not isinstance(hooks, dict):
        return settings, removed
    for ev in list(hooks):
        groups = hooks[ev]
        if not isinstance(groups, list):
            continue
        keep = []
        for g in groups:
            if not isinstance(g, dict) or not isinstance(g.get("hooks"), list):
                keep.append(g)
                continue
            hs = [h for h in g["hooks"] if not _ours(h)]
            removed += [(ev, g.get("matcher", ""), h["command"]) for h in g["hooks"] if _ours(h)]
            if hs:
                keep.append(dict(g, hooks=hs))
            elif not any(_ours(h) for h in g["hooks"]):
                keep.append(g)
        if keep:
            hooks[ev] = keep
        else:
            del hooks[ev]
    if not hooks:
        settings.pop("hooks", None)
    return settings, removed


def _count(settings):
    hooks = settings.get("hooks") if isinstance(settings.get("hooks"), dict) else {}
    return sum(len(g.get("hooks", [])) for gs in hooks.values() if isinstance(gs, list) for g in gs if isinstance(g, dict))


def install(root, agent="claude", rules=DEFAULT_RULES, skills_dirs=None, edits=True, skills=True, model=None,
            approve=False, command=None, dry_run=False):
    """Write solvi's hook entries into the project's settings (.claude/settings.json; Codex: .codex/hooks.json), merged
    with what is there: other hooks and settings are kept, solvi's own entries are replaced. Writes the sample rules
    file when the rules file does not exist. → the lines it prints."""
    path = _settings_path(root, agent)
    settings = _load_json(path)
    before = _count(settings)
    settings, removed = _strip(settings)
    exe = command or _solvi_command()
    extra = (f" --decider {_q(model)}" if model else "") + (" --agent codex" if agent == "codex" else "")
    added = []
    hooks = settings.setdefault("hooks", {})
    if edits:
        cmd = f"{exe} hook pre-edit --rules {_q(rules)}{extra}" + (" --approve" if approve else "")
        matcher = "Edit|Write|MultiEdit" if agent == "claude" else "apply_patch|Edit|Write"
        hooks.setdefault("PreToolUse", []).append(
            {"matcher": matcher, "hooks": [{"type": "command", "command": cmd, "timeout": 120 if model else 30,
                                            "statusMessage": "solvi: checking the edit against the rules"}]})
        added.append(("PreToolUse", matcher, cmd))
    if skills:
        dirs = skills_dirs or [DEFAULT_SKILLS]
        cmd = f"{exe} hook pick-skill " + " ".join(f"--skills-dir {_q(d)}" for d in dirs) + extra.replace(" --agent codex", "")
        hooks.setdefault("UserPromptSubmit", []).append(
            {"hooks": [{"type": "command", "command": cmd, "timeout": 60 if model else 15}]})
        added.append(("UserPromptSubmit", "", cmd))
    rel = os.path.relpath(path, root)
    lines = []
    for ev, m, c in added:
        was = any(e == ev for e, _, _ in removed)
        lines.append(f"{rel}: {'replaced' if was else 'added'} {ev}{f' ({m})' if m else ''} → {c}")
    kept = before - len(removed)
    if kept:
        lines.append(f"{rel}: kept {kept} other hook handler(s)")
    rules_abs = rules if os.path.isabs(rules) else os.path.join(root, rules)
    if edits and not os.path.exists(rules_abs):
        lines.append(f"{os.path.relpath(rules_abs, root)}: written — the sample rules; edit them for your project")
        if not dry_run:
            os.makedirs(os.path.dirname(rules_abs), exist_ok=True)
            with open(rules_abs, "w", encoding="utf-8") as fh:
                fh.write(SAMPLE_RULES)
    if not dry_run:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    if not _gitignored(root):
        lines.append("note: decisions are stored in .solvi/traces/ with the changes and prompts they rest on — add "
                     ".solvi/ to .gitignore")
    return lines


def _gitignored(root):
    gi = _read(os.path.join(root, ".gitignore")) or ""
    return any(ln.strip().rstrip("/") in (".solvi", "/.solvi") for ln in gi.splitlines())


def uninstall(root, agent="claude", dry_run=False):
    """Remove solvi's hook entries from the project's settings; everything else stays. → the lines it prints."""
    path = _settings_path(root, agent)
    rel = os.path.relpath(path, root)
    if not os.path.exists(path):
        return [f"{rel}: not there, nothing to remove"]
    settings, removed = _strip(_load_json(path))
    if not removed:
        return [f"{rel}: no solvi hooks"]
    if not dry_run:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    return [f"{rel}: removed {ev}{f' ({m})' if m else ''} → {c}" for ev, m, c in removed]


def _agents(a):
    return ["claude", "codex"] if a == "both" else [a]


def cmd_install(a):
    root = os.path.abspath(a.project or os.getcwd())
    for ag in _agents(a.agent):
        for line in install(root, ag, a.rules, a.skills_dir, not a.no_edits, not a.no_skills, a.model, a.approve,
                            a.command, a.dry_run):
            print(line)
    if a.dry_run:
        print("(dry run: nothing written)")
    return 0


def cmd_uninstall(a):
    root = os.path.abspath(a.project or os.getcwd())
    for ag in _agents(a.agent):
        for line in uninstall(root, ag, a.dry_run):
            print(line)
    return 0


# --------------------------------------------------------------------------------------------------- the command
def parser():
    import argparse
    p = argparse.ArgumentParser(prog="solvi hook", description="solvi behind a coding agent's hooks (Claude Code; "
                                "Codex, preview): check edits against rules, pick a skill for a prompt, install the hooks",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--project", help="the project root (default: $CLAUDE_PROJECT_DIR, else found from the hook's cwd)")
        s.add_argument("--decider", dest="model",
                       help="a decider: a local checkpoint folder or cached Hugging Face id, llm:URL#model, "
                            "systemone:URL#model or module:attr (keys from $SOLVI_LLM_API_KEY / "
                            "$SOLVI_SYSTEMONE_API_KEY); default: deterministic only")
        s.add_argument("--model", dest="old_model", help=argparse.SUPPRESS)   # 0.7 name: refused with what to do
        s.add_argument("--store", default=HOOK_STORE, help=f"the TraceStorage for the decisions (default {HOOK_STORE}, "
                                                            "relative to the project)")
        s.add_argument("--no-store", action="store_true", help="do not store the decisions")
    e = sub.add_parser("pre-edit", help="PreToolUse on Edit|Write|MultiEdit: check the change against the rules")
    e.add_argument("--rules", default=DEFAULT_RULES, help=f"the rules file (TOML or JSON; default {DEFAULT_RULES})")
    common(e)
    e.add_argument("--agent", choices=["claude", "codex"], default="claude", help="the answer's dialect (Codex cannot ask)")
    e.add_argument("--approve", action="store_true", help="answer an explicit allow when no rule objects (skips the "
                                                          "permission prompt); default: say nothing")
    e.add_argument("--no-instruction-check", action="store_true",
                   help="do not ask about instruction-like text addressed to a reviewer in the change")
    k = sub.add_parser("pick-skill", help="UserPromptSubmit: name the one skill the prompt needs, or say nothing")
    k.add_argument("--skills-dir", action="append", help=f"a folder of skills (repeat; default {DEFAULT_SKILLS})")
    common(k)
    k.add_argument("--min-score", type=float, default=1.5, help="deterministic: the least score to pick a skill")
    k.add_argument("--margin", type=float, default=0.25,
                   help="a near tie says nothing: the second within this share of the best (with --decider: the decider's "
                        "min_margin between its top two probabilities)")
    k.add_argument("--calibration", help="with --decider: a calibration file for the skill decision")
    k.add_argument("--agent", choices=["claude", "codex"], default="claude", help=argparse.SUPPRESS)
    i = sub.add_parser("install", help="write the hook entries into the project's settings (merged; prints the changes)")
    i.add_argument("--project", help="the project root (default: the current folder)")
    i.add_argument("--agent", choices=["claude", "codex", "both"], default="claude",
                   help="claude: .claude/settings.json (default); codex: .codex/hooks.json (preview)")
    i.add_argument("--rules", default=DEFAULT_RULES, help=f"the rules file (default {DEFAULT_RULES}; the sample is "
                                                           "written there when it does not exist)")
    i.add_argument("--skills-dir", action="append", help=f"skill folders for pick-skill (default {DEFAULT_SKILLS})")
    i.add_argument("--no-skills", action="store_true", help="no UserPromptSubmit hook")
    i.add_argument("--no-edits", action="store_true", help="no PreToolUse hook")
    i.add_argument("--decider", dest="model", help="pass --decider to the hooks")
    i.add_argument("--model", dest="model", help=argparse.SUPPRESS)
    i.add_argument("--approve", action="store_true", help="pass --approve to pre-edit")
    i.add_argument("--command", help="the command that runs solvi (default: the solvi on PATH, else this Python -m solvi)")
    i.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    u = sub.add_parser("uninstall", help="remove solvi's hook entries from the project's settings")
    u.add_argument("--project", help="the project root (default: the current folder)")
    u.add_argument("--agent", choices=["claude", "codex", "both"], default="claude")
    u.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    au = sub.add_parser("audit", help="a stored pre-edit decision (the last by default): its audit and its replay "
                                      "against the current rules")
    au.add_argument("id", nargs="?", help="the stored decision's id")
    au.add_argument("--rules", default=DEFAULT_RULES, help=f"the rules file (default {DEFAULT_RULES})")
    au.add_argument("--store", default=HOOK_STORE, help=f"the store (default {HOOK_STORE})")
    au.add_argument("--decider", dest="model", help="the decider the decision used, if any")
    au.add_argument("--model", dest="model", help=argparse.SUPPRESS)
    au.add_argument("--project", help="the project root (default: found from the current folder)")
    sub.add_parser("sample-rules", help="print the sample rules file")
    return p


def main(argv=None):
    p = parser()
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    if a.cmd == "sample-rules":
        print(SAMPLE_RULES, end="")
        return 0
    if getattr(a, "old_model", None) is not None:      # status 1, not 2: an old installed hook must not block every call
        print("solvi hook: --model was renamed in 0.8 and removed in 0.9: use --decider (reinstall the hooks with "
              "`solvi hook install --decider ...`)", file=sys.stderr)
        return 1
    try:
        return {"pre-edit": cmd_pre_edit, "pick-skill": cmd_pick_skill, "install": cmd_install,
                "uninstall": cmd_uninstall, "audit": cmd_audit}[a.cmd](a)
    except SettingsError as e:                         # install / uninstall: said in one line, nothing was changed
        print(f"solvi: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Change", "changes_of", "front_matter", "glob_regex", "install", "load_rules", "main", "parser", "reasons",
           "Rule", "rules_system", "RulesError", "SAMPLE_RULES", "SettingsError", "uninstall", "words"]
