"""Episodes over an Environment and the paired bootstrap — the measurement under the fresh-environments stage.

    row = run_episode(env, policy, seed, max_steps=200)    # {"seed", "steps", "done", "failed", "progress", "actions",
                                                            #  "errors", "error"}
    mean, lo, hi = paired_bootstrap(base_scores, cand_scores, boot=2000, seed=0)   # mean of cand − base, 95% interval

Ported from a research harness (gated evolution of an agent's rules): the candidate and the current agent play the same
worlds (the same seeds), the per-world difference is bootstrapped, and the candidate must win on one set of fresh worlds
(selection) and then again, with the interval's lower bound above zero, on other fresh worlds (confirmation).

An episode reads the environment only through the Environment protocol (reset(seed), actions(state), step(action)). What
"failed" and "progress" mean is the environment's to say, in the last Outcome's info ({"failed": bool, "progress":
number}); the solvi protocol does not read info, so these two keys are a convention of this module. A policy that
raises ends its episode and is counted in "errors" (a candidate's rules must never crash)."""
from __future__ import annotations

import random


def run_episode(env, policy, seed, *, max_steps=200):
    """Play one episode: policy(state, actions) → an action, until done or max_steps → the row (module docstring)."""
    state = env.reset(seed)
    actions, info, done, errors, error = [], {}, False, 0, None
    for _ in range(max_steps):
        try:
            a = policy(state, list(env.actions(state)))
        except Exception as e:  # noqa: BLE001 — a candidate's rule that crashes is measured, not raised
            errors, error = errors + 1, f"{type(e).__name__}: {e}"
            break
        out = env.step(a)
        actions.append(a)
        state, info, done = out.state, dict(out.info or {}), bool(out.done)
        if done:
            break
    return {"seed": seed, "steps": len(actions), "done": done, "failed": bool(info.get("failed", False)),
            "progress": float(info.get("progress", 0.0)), "actions": actions, "errors": errors, "error": error}


def paired_bootstrap(base, cand, *, boot=2000, seed=0, level=0.95):
    """The mean of cand[i] − base[i] and its bootstrap interval at `level` → (mean, low, high). Same worlds, same order."""
    if len(base) != len(cand) or not base:
        raise ValueError("a paired comparison needs two equally long, non-empty lists of per-world scores")
    rng = random.Random(seed)
    d = [float(c) - float(b) for b, c in zip(base, cand)]
    m = sum(d) / len(d)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(boot))
    tail = (1 - level) / 2
    return m, bs[int(tail * boot)], bs[min(boot - 1, int((1 - tail) * boot))]


__all__ = ["paired_bootstrap", "run_episode"]
