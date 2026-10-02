"""Human-readable reports: one decision (res.report) and a period of stored decisions (store.report), as Markdown,
self-contained HTML with every value escaped, or data; `solvi report` on the command line."""
import json
import re

import pytest

from solvi import Answer, Catalog, Decision, JSONLStorage, Question, Quote, SQLiteStorage, System
from solvi.cli import main
from solvi.report import highlight, md

DOC = 'Invoice 7781\nVendor: Acme <Tools> & "Sons"\nTotal: 1,250.00 EUR\n<script>alert(1)</script>\n'


class Classifier:
    """A stand-in model: a risk decision; `guarantee` puts an act_guard-like promise in the decision's record."""
    model_id = "test/risk"

    def __init__(self, version="1", guarantee=False):
        self.version, self.guarantee = version, guarantee


def build(version="1", guarantee=False, hallucinate=False, margin=0):
    cat = Catalog()
    clf = Classifier(version, guarantee)

    class Extractor:
        model_id = "test/vendor"
        version = "1"

    def vendor_model(doc):
        m = re.search(r"Vendor:\s*(.+)", doc)
        return Quote("<img src=x onerror=alert(1)>" if hallucinate else m.group(1), m.start(1), m.end(1))
    vendor_model.__solvi_model__ = Extractor()
    cat.extract(vendor_model, provides="vendor")

    @cat.extract(provides="vendor")
    def vendor_regex(doc):
        m = re.search(r"Vendor:\s*(.+)", doc)
        return Quote(m.group(1), m.start(1), m.end(1))

    @cat.extract
    def total(doc):
        m = re.search(r"Total:\s*([0-9,.]+)", doc)
        return Quote(float(m.group(1).replace(",", "")), m.start(1), m.end(1))

    @cat.fn(model=clf, options=["low", "high"])
    def risk(total):
        extra = {"guarantee": {"promise": "P(answered alone and wrong) <= 10%", "method": "crc", "n": 300}} \
            if clf.guarantee else {}
        return Decision("high" if total > 1000 else "low", {"high": 0.8, "low": 0.2} if total > 1000 else
                        {"high": 0.1, "low": 0.9}, extra=extra)

    @cat.check(hard=True, then={"pay": "no"})
    def known_vendor(vendor, blocked):
        return vendor not in blocked

    @cat.rule("pay")
    def pay(risk, total, limit):
        return "yes" if risk == "low" and total <= limit - margin else "review"

    return System(cat, [Question("pay", "Pay the invoice?", Answer.choice(["yes", "review", "no"]),
                                 checkpoints=["known_vendor"])])


STATE = {"doc": DOC, "blocked": [], "limit": 2000}


# ---------------------------------------------------------------- one decision
def test_decision_report_data():
    s = build(hallucinate=True)
    res = s.ask(STATE)
    d = res.report(format="data")
    a = d["answers"][0]
    assert a["question"] == "pay" and a["answer"] == "'review'" and a["text"] == "Pay the invoice?"
    kinds = {r["kind"] for r in a["rests_on"]}
    assert {"given", "quoted", "decided", "check", "rule"} <= kinds
    assert {e["kind"] for e in a["safeguards"]} == {"grounding", "fallback"}
    assert a["guarantee"].startswith("none") and a["guaranteed"] is False
    doc = d["documents"][0]
    assert doc["source"] == "doc" and {(s["start"], s["end"]) for s in doc["spans"]} == \
        {(DOC.index("Acme"), DOC.index("\nTotal")), (DOC.index("1,250"), DOC.index(" EUR"))}
    names = [m["name"] for m in d["models"]]
    assert "risk" in names and any(n.startswith("vendor (vendor_model, not used") for n in names)
    assert d["replay"]["status"] == "ok" and "not re-run" in d["replay"]["detail"]
    assert d["trace"]["steps"] == len(res.trace.records) and d["trace"]["catalog"]
    json.dumps(d)                                     # plain data


def test_guarantee_line():
    a = build(guarantee=True).ask(STATE).report(format="data")["answers"][0]
    assert a["guaranteed"] is True and "10%" in a["guarantee"]

    cat = Catalog()

    @cat.rule("big")
    def big(amount) -> bool:
        return amount > 10
    a = System(cat, [Question("big", "Big?", Answer.yes_no())]).ask({"amount": 3}).report(format="data")["answers"][0]
    assert a["guaranteed"] is None and a["guarantee"].startswith("no model decided")


def test_decision_markdown():
    res = build(hallucinate=True).ask(STATE)
    out = res.report()
    assert out.startswith("# Decision report")
    assert "## pay = 'review'" in out
    assert "**1,250.00**" in out                      # the quote, bold in its context
    assert "grounding rejected" in out and "Replay: **ok**" in out
    assert "<script>" not in out and "\\<script\\>" in out      # Markdown-escaped
    assert not re.search(r"(?<!\\)<img", out) and "\\<img src=x" in out


def test_decision_html_is_self_contained_and_escaped():
    res = build(hallucinate=True).ask(STATE)
    page = res.report(format="html")
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    body = page.split("</style>", 1)[1]
    assert "<script" not in page and "<img" not in page and "<link" not in page
    assert "src=" not in body.replace("src=x", "") and "http" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page      # the rejected (hallucinated) quote, escaped
    assert re.search(r'<mark title="pay: vendor \[\d+:\d+\]">Acme &lt;Tools&gt; &amp; &quot;Sons&quot;</mark>', page)
    assert '<mark title="pay: total' in page


def test_report_format_is_checked():
    with pytest.raises(ValueError):
        build().ask(STATE).report(format="pdf")


def test_highlight_overlaps_and_long_text():
    text = "abcdefghij"
    out = highlight(text, [{"start": 1, "end": 5, "label": "x", "ok": True}, {"start": 3, "end": 7, "label": "y", "ok": False}])
    assert out == ('a<mark title="x [1:5]">bc</mark><mark class="bad" title="x [1:5]; y [3:7]">de</mark>'
                   '<mark class="bad" title="y [3:7]">fg</mark>hij')
    long = "x" * 50_000 + "<NEEDLE>" + "y" * 50_000
    s = 50_000
    out = highlight(long, [{"start": s, "end": s + 8, "label": "n", "ok": True}], max_text=1000, context=10)
    assert "&lt;NEEDLE&gt;</mark>" in out and len(out) < 500
    assert f"… [0:{s - 10}] not shown …" in out and f"[{s + 18}:{len(long)}] not shown" in out


def test_md_escape():
    assert md("a|b*c_<d>\nnew") == "a\\|b\\*c\\_\\<d\\> new"


def test_forced_answer_and_stored_response(tmp_path):
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = build()
    s.storage = store
    store.catalog = s
    res = s.ask({**STATE, "blocked": ["Acme <Tools> & \"Sons\""]})
    assert res["pay"].status == "forced"
    back = store.get(res.stored_id)
    d = back.report(format="data")
    a = d["answers"][0]
    assert d["stored_id"] == res.stored_id and a["answer"] == "'no'"
    assert any(r["kind"] == "check" and "decided the answer" in r["note"] for r in a["rests_on"])
    assert d["replay"]["status"] == "ok"
    assert back.report(format="data", replay=False)["replay"]["status"] == "not run"


# ---------------------------------------------------------------- a period
class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        self.t += 10.0
        return self.t


@pytest.fixture(params=["jsonl", "sqlite"])
def store(request, tmp_path):
    clock = Clock()
    st = (JSONLStorage(tmp_path / "d.jsonl", clock=clock) if request.param == "jsonl"
          else SQLiteStorage(tmp_path / "d.db", clock=clock))
    s1 = System(build().catalog, build().questions.values(), storage=st)
    s1.ask(STATE)                                          # yes... no: risk high (total 1250 > 1000) → review
    s1.ask({**STATE, "doc": DOC.replace("1,250.00", "250.00")})               # yes
    s1.ask({**STATE, "blocked": ['Acme <Tools> & "Sons"']})                   # no (hard check)
    s2 = System(build(version="2", guarantee=True, margin=1).catalog, build().questions.values(), storage=st)
    s2.ask({**STATE, "doc": DOC.replace("1,250.00", "50.00")})                # yes, a new model fingerprint
    return st


def test_period_data(store):
    d = store.report(format="data")
    assert d["decisions"] == 4
    p = d["questions"]["pay"]
    assert p["asked"] == 4 and p["answers"] == {"'review'": 1, "'yes'": 2, "'no'": 1}
    assert p["status"] == {"ok": 3, "forced": 1} and p["escalation_rate"] == 0.0
    assert p["safeguards"] == {"hard_check": 1}
    assert p["model_backed"] == 3 and p["guaranteed"] == 1 and abs(p["guarantee_coverage"] - 1 / 3) < 1e-9
    assert p["deterministic"] == 1                        # the forced answer: a hard check over code
    assert [c["what"] for c in d["changes"]] == ["catalog", "model Classifier test/risk"]
    assert d["changes"][1]["seq"] == 3
    assert [x["id"] for x in p["examples"]["answers"]["'yes'"]] == [s.id for s in store.query(answer="yes")]


def test_period_filters_and_escalations(store):
    t = [s.time for s in store.iter()]
    d = store.report(since=t[1], until=t[3], format="data")
    assert d["decisions"] == 2 and d["changes"] == []
    assert store.report(question="nope", format="data")["decisions"] == 0

    cat = Catalog()

    @cat.rule("q")
    def q(x):
        return None if x < 0 else x > 1
    s = System(cat, [Question("q", "Q?", Answer.yes_no())], storage=store)
    s.ask({"x": -1})
    s.ask({"x": 5})
    p = store.report(question="q", format="data")["questions"]["q"]
    assert p["asked"] == 2 and p["escalation_rate"] == 0.5 and p["guards"] == {"rule_abstained": 1}
    assert p["examples"]["escalations"]["rule_abstained"][0]["why"].startswith("the rule abstained")


def test_period_markdown_and_html(store):
    out = store.report()
    assert out.startswith("# Decisions report") and "4 stored decision(s)" in out
    assert "## pay" in out and "hard check decided" in out and "Changes:" in out
    page = store.report(format="html")
    assert page.startswith("<!doctype html>") and "<script" not in page and "http" not in page
    assert "Guarantee coverage" in page and "1/3" in page


def test_cli_report(store, tmp_path, capsys):
    path = store.path if hasattr(store, "path") else str(tmp_path / "d.db")
    assert main(["report", path]) == 0
    assert capsys.readouterr().out.startswith("# Decisions report")
    out = tmp_path / "r.html"
    assert main(["report", path, "--question", "pay", "--html", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")
    rid = next(iter(store.iter())).id
    assert main(["report", path, "--id", rid, "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["kind"] == "decision" and d["stored_id"] == rid and d["replay"]["status"] == "not run"
    with pytest.raises(SystemExit) as e:
        main(["report", path, "--id", "nope"])
    assert e.value.code == 2


def test_derived_quote_is_not_flagged_as_a_mismatch():
    """A value derived from a quote (the text "lawyer" → "legal threat") is grounded text, not "not the text at these
    offsets"."""
    from solvi import Catalog, Question, Quote, System
    cat = Catalog()

    @cat.extract
    def threat(message: str) -> str:
        i = message.find("lawyer")
        return Quote("legal threat", i, i + 6, source="message")

    @cat.rule("priority")
    def priority(threat: str) -> bool:
        return threat == "legal threat"

    res = System(cat, [Question("priority", "High priority?")]).ask({"message": "Refund me or my lawyer calls."})
    md = res.report()
    assert "**lawyer**" in md and "NOT the text" not in md
    assert "not the text at these offsets" not in res.report(format="html")


def test_period_report_shows_non_finite_answers_untagged(tmp_path):
    import math
    cat = Catalog()

    @cat.rule("limit")
    def limit(x):
        return math.inf

    st = JSONLStorage(tmp_path / "p.jsonl")
    s = System(cat, [Question("limit", "Limit?", Answer.estimate())], storage=st)
    s.ask({"x": 1.0})
    text = st.report()
    assert "$float" not in text and "inf" in text
    from solvi.report import period
    assert "$float" not in json.dumps(period(st), default=str)


def test_report_stays_english_for_a_system_in_another_language():
    s = build(guarantee=True)
    s.lang = "ru"
    res = s.ask(STATE)
    assert re.search("[а-яА-Я]", res.audit("pay").support_line())             # the audit itself renders in Russian
    d = res.report(format="data")
    assert not re.search("[а-яА-Я]", json.dumps(d, ensure_ascii=False))
    assert not re.search("[а-яА-Я]", res.report()) and not re.search("[а-яА-Я]", res.report(format="html"))


def test_period_report_escapes_a_stored_seq_that_is_not_a_number(store):
    from solvi.report import period, render
    data = period(store)
    evil = "<img src=x onerror=alert(2)>"
    data["changes"] = [{"what": "catalog", "from": "a" * 16, "to": "b" * 16, "id": "r1", "seq": evil, "time": "t"}]
    assert evil not in render(data, "html") and "&lt;img src=x" in render(data, "html")
    assert evil not in render(data, "md")
