"""Reference spans for the browser parity test, computed with the PyTorch LongSpanExtractor.

  node tests/export_usecases.mjs > /tmp/usecases.json
  PYTHONPATH=<solvi>/src:. python tests/make_ref.py /tmp/usecases.json <extract-base dir or HF id> tests/parity_ref.json

Writes {"model", "settings", "texts": {id: text}, "items": [{uc, doc, text, field, desc, start, end, score, present, value}]}
and prints the solvi decisions for every sample document (a check that the rules make sense)."""
import json
import sys
import time

from solvi.core.extract import LongSpanExtractor

import solvi_docs

LONG_PARTS = [("contract", 0), ("contract", 1), ("nda", 0), ("dpa", 0), ("lease", 0), ("offer", 0), ("nda", 1), ("insurance", 0)]
LONG_FIELDS = [("governing_law", "the clause that says which state's or country's law governs the contract"),
               ("breach_notice", "the time within which the processor must notify the controller of a personal data breach")]


def main(uc_path, model, out_path):
    ucs = json.load(open(uc_path))
    ex = LongSpanExtractor.load(model, device="cpu")
    thr = lambda name: ex.thr.get(name, ex.thr_default)             # noqa: E731
    texts, items = {}, []
    t0 = time.time()
    for u in ucs:
        for di, d in enumerate(u["docs"]):
            tid = f"{u['id']}/{di}"
            texts[tid] = d["text"]
            hits = []
            for name, desc in u["fields"]:
                s, e, sc, _ = ex.predict(d["text"], desc)
                present = sc >= thr(name)
                items.append(dict(uc=u["id"], doc=d["name"], text=tid, field=name, desc=desc, start=int(s), end=int(e),
                                  score=float(sc), present=bool(present), value=d["text"][s:e]))
                hits.append(dict(name=name, desc=desc, hit=dict(present=bool(present), start=int(s), end=int(e), score=float(sc))))
            res = json.loads(solvi_docs.run(json.dumps(dict(doc=d["text"], today=d.get("today"), fields=hits, code=u["code"]))))
            print(f"\n== {u['id']} / {d['name']}  ({len(d['text'])} chars, {time.time() - t0:.0f} s)")
            for h in hits:
                v = d["text"][h["hit"]["start"]:h["hit"]["end"]] if h["hit"]["present"] else "—"
                print(f"   {h['name']:20s} {h['hit']['score']:.3f}  {v[:90]!r}")
            if "error" in res:
                print("   ERROR", res["error"])
                continue
            for a in res["answers"]:
                print(f"   -> {a['name']:22s} {str(a['answer']):14s} {a['confidence']:.2f} {a['status']:8s} {a['why'][:110]}")
            print(f"   replay ok={res['replay']['ok']}")
    by = {(u["id"], i): d["text"] for u in ucs for i, d in enumerate(u["docs"])}
    long_text = "\n\n".join(by[k] for k in LONG_PARTS)
    texts["long/0"] = long_text
    for name, desc in LONG_FIELDS:
        s, e, sc, _ = ex.predict(long_text, desc)
        items.append(dict(uc="long", doc="10-page bundle", text="long/0", field=name, desc=desc, start=int(s), end=int(e),
                          score=float(sc), present=bool(sc >= thr(name)), value=long_text[s:e]))
        print(f"\n== long ({len(long_text)} chars): {name} {sc:.3f} {long_text[s:e][:100]!r}")
    json.dump(dict(model=str(model), settings=dict(max_len=ex.max_len, stride=ex.stride, max_span=ex.max_span,
                                                   thr_default=ex.thr_default, thr=ex.thr),
                   texts=texts, items=items), open(out_path, "w"), ensure_ascii=False, indent=0)
    print(f"\n{len(items)} reference spans -> {out_path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
