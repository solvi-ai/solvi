"""What solvi renders for people — the audit, solvi.show, the compact audit, the safeguard report — over every gallery
case and the examples that print audits (12 grounded audit, 18 several models): the text whose English sha256 the golden file
holds (tests/i18n/en_golden.json) and the Russian rendering is checked on.

    uv run python tests/i18n/render.py            # rewrite the golden file — only when an English change is intended
    uv run python tests/i18n/render.py out/       # also write each text to out/ (to diff two versions)"""
from __future__ import annotations

import contextlib
import hashlib
import re
import io
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ["12_grounded_audit.py", "18_several_models.py"]


def _show(res, **kw):
    from solvi.show import show
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        show(res, **kw)
    return "\n".join(line for line in buf.getvalue().splitlines() if not line.startswith(("── time", "── время")))


def gallery(entry, lang=None):
    """Every case of one gallery entry: audit, show, compact audit; then the system's safeguard report. → (text, trace hashes)."""
    from solvi import testing
    suite = testing.load(ROOT / "gallery" / entry / "cases.json")
    system = suite.system()
    if lang is not None:
        system.lang = lang
    out, hashes = [], []
    for case in suite.cases:
        res = system.ask(suite.state(case))
        hashes.append([r.hash for r in res.trace.records])
        out += [f"### {case.get('name')}", str(res.audit()), res.audit().compact(), _show(res)]
    out += ["### safeguard_report", system.safeguard_report()]
    return "\n".join(out) + "\n", hashes


def example(name):
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = [name]
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(ROOT / "examples" / name), run_name="__main__")
    finally:
        sys.argv = argv
    return buf.getvalue()


def entries():
    return sorted(p.parent.name for p in (ROOT / "gallery").glob("*/cases.json"))


def texts(lang=None):
    """{name: rendered text} for every gallery entry and example (examples in English only)."""
    out = {e: gallery(e, lang)[0] for e in entries()}
    if lang is None:
        out.update({Path(x).stem: example(x) for x in EXAMPLES})
    return out


def sha(text):
    # a fast head's fingerprint hashes its float weights, and a decision part's its calibrated threshold (a probability);
    # their last bits differ between numpy builds (the text around them is what this checks): masked before hashing
    text = re.sub(r"(FastHead #|DecisionPart \S+ #)[0-9a-f]+", r"\1…", text)
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    got = texts()
    (here / "en_golden.json").write_text(json.dumps({k: sha(v) for k, v in got.items()}, indent=1) + "\n")
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        d.mkdir(parents=True, exist_ok=True)
        for k, v in got.items():
            (d / f"{k}.txt").write_text(v)
