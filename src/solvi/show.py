"""Print a response: answers, flow, computed_state, replay — so you can see how the system reached each answer."""
from __future__ import annotations

from . import i18n


def _rule(title, en_title, n):
    """A heading line: its English width is what it always was; other languages keep the same total width."""
    return title + "─" * max(3, len(en_title) + n - len(title))


def _flow(flow, lang):
    """Flow.__str__ in a language: the strategist's reasons for each step (i18n scope "flow") and the skipped parts."""
    if lang == i18n.DEFAULT:
        return str(flow)
    lines = []
    for i, s in enumerate(flow.steps, 1):
        p = s.part
        reasons = "; ".join(i18n.msg(r, lang, scope="flow") for r in s.reasons)
        lines.append(f"{i:2d}. {p.kind:7s} {p.name:24s} ← {', '.join(p.inputs) or '—'}   [{reasons}]")
    if flow.skipped:
        lines.append(i18n.t("fl.not_taken", lang) + ", ".join(f"{k} ({i18n.msg(v, lang)})" for k, v in sorted(flow.skipped.items())))
    return "\n".join(lines)


def show(res, catalog=None, flow=True, state=True, audit=True, lang=None):
    """Print a response. lang: "en" or "ru" (solvi.i18n; default: the language of the System that answered)."""
    lang = i18n.check(getattr(res, "lang", None) if lang is None else lang)
    t, m = i18n.t, (lambda x: i18n.msg(x, lang))

    def head(key, n):
        return _rule(t(key, lang), t(key), n)
    print(head("sh.answers", 60))
    for q, r in res.results.items():
        a = "—" if r.answer is None else r.answer
        extra = "" if r.status == "ok" else f"  [{i18n.status(r.status, lang)}]"
        print(f"  {q:16s} {a!s:12s} " + t("sh.line", lang, c=f"{r.confidence:.2f}") + extra)
        print(f"  {'':16s} " + t("sh.why", lang) + f"{m(r.why)}")
    if flow:
        print(head("sh.flow", 35))
        for line in _flow(res.flow, lang).splitlines():
            print("  " + line)
    if flow and res.trace.skipped:
        print(t("sh.skipped", lang) + ", ".join(f"{n} ({m(why)})" for n, why in res.trace.skipped))
    if flow and res.trace.schedule:
        print(t("sh.learned_order", lang))
        for line in res.trace.explain_order().splitlines():
            print("    " + i18n.msg(line, lang, scope="flow"))
    if state:
        print(head("sh.state", 52))
        text = res.state_text(lang)
        for line in text.splitlines():
            print("  " + line)
        for r in res.trace.records:
            if r.tried:
                print(t("sh.tried", lang, name=r.name, used=r.producer or "—",
                        tried=", ".join(f"{n} ({m(w)})" for n, w in r.tried)))
    if audit:
        print(head("sh.audit", 24))
        for line in res.audit(lang=lang).compact().splitlines():
            print("  " + line)
    if catalog is not None:
        rep = res.trace.replay(catalog)
        print(head("sh.replay", 54))
        print(t("sh.steps", lang, n=rep["steps"], m=len(rep["mismatches"]))
              + ("" if rep["ok"] else f": {rep['mismatches'][:3]}"))
        if rep.get("models"):
            print(t("sh.model_steps", lang) + ", ".join(f"{n} {v}" for _, n, v in rep["models"]))
    print(t("sh.time", lang, ms=f"{res.ms:.2f}"))


__all__ = ["show"]
