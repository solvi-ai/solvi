"""What counts as a refusal (is_refusal, REFUSAL_WORDS): read by the refusal-regression stage and by the rule-text
prover, which have no other dependency on each other."""
from __future__ import annotations

REFUSAL_WORDS = frozenset({"abstain", "abstained", "abstains", "escalate", "escalated", "hand over", "handover",
                           "hand_over", "handed over", "defer", "deferred", "unknown", "i don't know", "don't know"})


def is_refusal(answer, question=None):
    """Did an answer refuse or hand over? None; a solvi Response (the question's result, or any question when
    `question` is None, has status "abstain"); a result or a dict whose status is "abstain" / "escalate"; a
    Decision whose `escalate` is set; a string such as "abstain", "escalate", "hand over", "unknown" (REFUSAL_WORDS,
    any case). A domain answer like "refuse" (a loan refused) or "no" is an answer, not a refusal: pass your own
    `refused=` to RefusalRegression when your system hands over with another word (e.g. "review")."""
    if answer is None:
        return True
    results = getattr(answer, "results", None)
    if isinstance(results, dict):                                     # a Response
        rs = [results[question]] if question is not None and question in results else list(results.values())
        return any(is_refusal(r) for r in rs)
    if isinstance(answer, dict):
        st = answer.get("status")
        return isinstance(st, str) and st.lower() in ("abstain", "escalate", "escalated", "handover")
    st = getattr(answer, "status", None)
    if isinstance(st, str) and st.lower() in ("abstain", "escalate", "escalated", "handover"):
        return True
    esc = getattr(answer, "escalate", None)
    if isinstance(esc, str) and esc:
        return True
    if isinstance(answer, str):
        return answer.strip().lower() in REFUSAL_WORDS
    return False


__all__ = ["is_refusal", "REFUSAL_WORDS"]
