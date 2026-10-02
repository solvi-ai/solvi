"""Banking77 baseline, no solvi: TF-IDF + logistic regression over the known intents, answering alone when the top
probability is above a threshold picked on calib.jsonl for the promised error rate. No drift monitoring.

    uv run --with scikit-learn python banking77/baseline.py [--risk 0.05] [--out runs/baseline.jsonl]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DATA, read_jsonl, write_jsonl  # noqa: E402
from score import score  # noqa: E402


def main(risk, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    d = DATA / "banking77/prepared"
    fit, calib, stream = (read_jsonl(d / f"{f}.jsonl") for f in ("fit", "calib", "stream"))
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1), LogisticRegression(C=20, max_iter=2000))
    clf.fit([r["text"] for r in fit], [r["intent"] for r in fit])

    def run(rows):
        p = clf.predict_proba([r["text"] for r in rows])
        return [(clf.classes_[x.argmax()], float(x.max())) for x in p]
    cal = sorted(((c, lab == r["intent"]) for (lab, c), r in zip(run(calib), calib)), reverse=True)
    thr, wrong = 1.01, 0
    for i, (c, ok) in enumerate(cal, 1):                       # the lowest threshold whose answered part errs <= risk
        wrong += not ok
        if wrong / i <= risk:
            thr = c
    return write_jsonl(out, [{"n": r["n"], "intent": lab, "confidence": round(c, 4), "escalate": c < thr}
                             for r, (lab, c) in zip(stream, run(stream))])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--risk", type=float, default=0.05)
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "baseline.jsonl"))
    a = p.parse_args()
    print(json.dumps(score(main(a.risk, a.out), a.risk), indent=1))
