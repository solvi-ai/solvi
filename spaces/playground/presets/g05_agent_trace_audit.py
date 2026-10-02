"""AI-agent trace audit: the log of an agent's tool calls -> approve / roll_back / escalate, and must secrets be rotated?

Computed facts: total and per-step cost against the budget and the per-step cap, tools outside the allowlist, failed steps,
destructive commands (rm -rf outside /tmp, DROP/TRUNCATE, DELETE without WHERE, force push, kubectl delete, terraform destroy),
secrets in the arguments (API keys, bearer tokens, passwords), and the rollback plan: the inverse of every state-changing step,
newest first (restore from a named backup, push back the SHA the force push overwrote, revert via the transaction log).
Hard checks: an exposed secret -> escalate + rotate; destruction nobody can undo -> escalate; destruction with a way back ->
roll_back. (When several hard checks fail, the first by name decides; the names are chosen so escalate wins.)
solvi's hash-chained trace of this audit is itself the audit record: replay re-derives every finding from the raw log.
Try: delete the "db" backup, or change a step's tool to "send_email"."""
import re

from solvi import Answer, Catalog, Question

cat = Catalog()
DESTRUCTIVE = [("rm -rf", r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)[a-z]*\s+(?!/tmp/)(\S+)"),
               ("DROP", r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\s+(?:IF EXISTS\s+)?(\w+)"),
               ("TRUNCATE", r"\bTRUNCATE\s+(?:TABLE\s+)?(\w+)"),
               ("DELETE without WHERE", r"\bDELETE\s+FROM\s+(\w+)\s*;?\s*$"),
               ("force push", r"\bgit\s+push\s+(?:.*\s)?(?:--force|-f)\b.*?(\S+)\s*$"),
               ("kubectl delete", r"\bkubectl\s+delete\s+(\S+\s+\S+)"),
               ("terraform destroy", r"\bterraform\s+destroy\b()")]
SECRETS = [("API key", r"\bsk-[A-Za-z0-9_-]{16,}"), ("AWS key", r"\bAKIA[0-9A-Z]{16}\b"),
           ("bearer token", r"\bBearer\s+[A-Za-z0-9._-]{16,}"), ("password", r"\b(?:password|passwd|pwd)=\S{4,}")]


def _args(step):
    return step["args"] if isinstance(step["args"], str) else " ".join(map(str, step["args"]))


# ---------- facts computed from the raw log
@cat.fn
def total_cost(steps):
    return round(sum(s["cost_usd"] for s in steps), 2)


@cat.fn
def costly_steps(steps, max_step_usd):
    """steps above the per-step cost cap"""
    return [i for i, s in enumerate(steps, 1) if s["cost_usd"] > max_step_usd]


@cat.fn
def off_allowlist(steps, allowed_tools):
    return [(i, s["tool"]) for i, s in enumerate(steps, 1) if s["tool"] not in allowed_tools]


@cat.fn
def failed_steps(steps):
    return [i for i, s in enumerate(steps, 1) if s["result"] != "ok"]


@cat.fn
def destructive_steps(steps):
    out = []
    for i, s in enumerate(steps, 1):
        for kind, p in DESTRUCTIVE:
            m = re.search(p, _args(s), re.I)
            if m and s["result"] == "ok":
                out.append((i, kind, m.group(1)))
    return out


@cat.fn
def secrets_in_args(steps):
    """(step, kind, masked value): the log already carries the secret, so it must be treated as exposed"""
    out = []
    for i, s in enumerate(steps, 1):
        for kind, p in SECRETS:
            for m in re.finditer(p, _args(s)):
                out.append((i, kind, m.group()[:10] + "…"))
    return out


@cat.fn
def rollback_plan(steps, destructive_steps, backups):
    """the inverse of every state-changing step, newest first; None where there is no way back"""
    plan = []
    kinds = {i: (k, target) for i, k, target in destructive_steps}
    for i in range(len(steps), 0, -1):
        s, a = steps[i - 1], _args(steps[i - 1])
        if i in kinds:
            k, target = kinds[i]
            if k == "force push":
                old = re.search(r"\b([0-9a-f]{7,40})\.\.\.[0-9a-f]{7,40}", s.get("output", ""))
                plan.append((i, f"git push --force origin {old.group(1)}:{target}" if old else None))
            elif k in ("DROP", "TRUNCATE", "DELETE without WHERE"):
                plan.append((i, f"restore table {target} from {backups['db']}" if backups.get("db") else None))
            elif k == "rm -rf":
                plan.append((i, f"restore {target} from {backups['fs']}" if backups.get("fs") else None))
            elif k == "kubectl delete":
                plan.append((i, f"kubectl apply the {target} manifest from git"))
            else:
                plan.append((i, None))
        elif s["tool"] == "sql" and s["result"] == "ok":
            m = re.match(r"\s*(UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+(\w+)", a, re.I)
            if m:
                plan.append((i, f"revert {m.group(1).split()[0].upper()} on {m.group(2)} (step {i}) from the transaction log"))
    return plan


@cat.fn
def unrecoverable_steps(rollback_plan):
    return [i for i, how in rollback_plan if how is None]


@cat.fn
def partial_change(failed_steps, rollback_plan):
    """a step failed after state had already changed: the agent left things half done"""
    return bool(failed_steps) and any(i < max(failed_steps) for i, _ in rollback_plan)


# ---------- hard checks (the first failing one by name decides: credentials < destruction < nothing)
@cat.check(hard=True, then={"verdict": "escalate", "rotate_secrets": "yes"})
def credentials_not_exposed(secrets_in_args):
    return not secrets_in_args


@cat.check(hard=True, then={"verdict": "escalate"})
def destruction_recoverable(destructive_steps, unrecoverable_steps):
    return not (destructive_steps and unrecoverable_steps)


@cat.check(hard=True, then={"verdict": "roll_back"})
def nothing_destroyed(destructive_steps):
    return not destructive_steps


# ---------- answers
@cat.rule("verdict")
def verdict(total_cost, budget_usd, costly_steps, off_allowlist, partial_change):
    if off_allowlist or total_cost > budget_usd or costly_steps:
        return "escalate"
    return "roll_back" if partial_change else "approve"


@cat.rule("rotate_secrets")
def rotate_secrets(secrets_in_args):
    return bool(secrets_in_args)


QUESTIONS = [
    Question("verdict", "Approve the agent's work, roll it back, or escalate to a person?",
             Answer.choice(["approve", "roll_back", "escalate"]),
             requires=["credentials_not_exposed", "destruction_recoverable", "nothing_destroyed"]),
    Question("rotate_secrets", "Must credentials be rotated?", Answer.yes_no(), requires=["credentials_not_exposed"]),
]
