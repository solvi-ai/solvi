"""Banking77 stream: answer alone under a promised error rate, and notice the shift.

    python banking77/score.py runs/<arm>.jsonl [risk=0.05]

A prediction per stream row: {"n", "intent": str or None, "escalate": bool}; one extra row {"drift_flag_at": n} if the
solution raised a drift flag (the first n at which it did). The shift starts at n = 1000 (unseen intents appear)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402


def part(rows, pred):
    ans = [r for r in rows if r["n"] in pred and not pred[r["n"]].get("escalate") and pred[r["n"]].get("intent")]
    wrong = sum(pred[r["n"]]["intent"] != r["intent"] for r in ans)
    return {"n": len(rows), "answered_alone": round(len(ans) / max(len(rows), 1), 3),
            "error_among_answered": round(wrong / max(len(ans), 1), 4), "wrong_answers": wrong}


def score(pred_file, risk=0.05):
    stream = read_jsonl(DATA / "banking77/prepared/stream.jsonl")
    rows = read_jsonl(pred_file)
    pred = {r["n"]: r for r in rows if "n" in r}
    flag = next((r["drift_flag_at"] for r in rows if r.get("drift_flag_at") is not None), None)
    before = [r for r in stream if r["phase"] == "before"]
    after = [r for r in stream if r["phase"] == "after"]
    out = {"risk_promised": float(risk), "before": part(before, pred), "after": part(after, pred),
           "after_known": part([r for r in after if r["known"]], pred), "after_unseen": part([r for r in after if not r["known"]], pred)}
    out["promise_kept_before"] = out["before"]["error_among_answered"] <= float(risk)
    out["promise_kept_after"] = out["after"]["error_among_answered"] <= float(risk)
    out["drift_flag_at"] = flag
    out["drift"] = "not flagged" if flag is None else "false alarm" if flag < 1000 else f"{flag - 1000} requests after the shift"
    return out


if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:]), indent=1))
