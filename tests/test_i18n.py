"""Languages of rendering (solvi.i18n): English stays byte for byte what it was, Russian renders, and nothing recorded —
traces, hashes, `why`, to_dict() — depends on the language."""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

from solvi import Answer, Catalog, Question, System, i18n, testing
from solvi.show import show

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "i18n"))
import render  # noqa: E402

GOLDEN = json.loads((HERE / "i18n" / "en_golden.json").read_text())


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_english_output_is_unchanged(name):
    """audit, compact audit, show and safeguard_report over every gallery case, and examples 12 and 18: the same text
    as before languages existed (sha256 in tests/i18n/en_golden.json; `uv run python tests/i18n/render.py out/` writes
    the texts to diff)."""
    examples = {Path(x).stem: x for x in render.EXAMPLES}
    text = render.example(examples[name]) if name in examples else render.gallery(name)[0]
    assert render.sha(text) == GOLDEN[name]


def _suite(entry="10_procurement_3way_match"):
    return testing.load(render.ROOT / "gallery" / entry / "cases.json")


def test_hashes_and_data_are_the_same_in_every_language():
    for entry in ("05_agent_trace_audit", "10_procurement_3way_match", "07_kyc_aml"):
        suite = _suite(entry)
        en, ru = suite.system(), suite.system()
        ru.lang = "ru"
        for case in suite.cases:
            a, b = en.ask(suite.state(case)), ru.ask(suite.state(case))
            assert [r.hash for r in a.trace.records] == [r.hash for r in b.trace.records]
            assert a.trace.init_hash == b.trace.init_hash
            assert {q: r.why for q, r in a.results.items()} == {q: r.why for q, r in b.results.items()}
            da, db = a.to_dict(), b.to_dict()
            for d in (da, db):                        # run times differ from run to run, whatever the language
                d.pop("ms"), d["trace"].pop("timings")
            assert da == db
            assert a.audit().to_dict() == b.audit().to_dict()
            assert a.safeguards == b.safeguards
            assert a.trace.replay(ru.catalog)["ok"] and b.trace.replay(en.catalog)["ok"]


def test_russian_audit_renders():
    suite = _suite()
    system = suite.system()
    system.lang = "ru"
    res = next(r for r in (system.ask(suite.state(c)) for c in suite.cases)
               if any(e["kind"] == "hard_check" for e in r.safeguards))
    text = str(res.audit())
    assert text.startswith("аудит: ответов ")
    assert "итого: уверенность" in text and "  защиты        решила жёсткая проверка ×1" in text
    assert "[решено проверкой]" in text and "  → ответ       " in text
    assert "жёсткая проверка not_duplicate не пройдена" in text          # the why, translated; the check's name is not
    assert "audit:" not in text and "safeguards" not in text and "hard check" not in text
    assert res.audit(lang="en").render() == str(res.audit(lang="en"))
    assert str(res.audit(lang="en")).startswith("audit: ")              # per call, the other way round
    assert res.audit().compact().splitlines()[0].startswith("payment: элементов опоры ")
    report = system.safeguard_report()
    assert report.startswith("запросов ") and "решила жёсткая проверка" in report
    assert system.safeguard_report(lang="en").startswith("asks ")


def test_show_in_russian():
    suite = _suite()
    system = suite.system()
    res = system.ask(suite.state(suite.cases[0]))

    def shown(lang):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            show(res, catalog=system.catalog, lang=lang)
        return buf.getvalue()
    out, en = shown("ru"), shown(None)
    for s in ("── ответы ", "почему: ", "── поток (выбран стратегом) ", "── computed_state ", "── аудит ", "── повтор трассы ",
              "  шагов ", "── время "):
        assert s in out
    assert "── answers" not in out and "why:" not in out
    assert "── answers " in en and "why: " in en                          # the response's own language: English

    def widths(text):                                                  # every heading keeps its English width
        return [len(line) for line in text.splitlines() if line.startswith("── ") and "──" in line[3:]]
    assert widths(out) == widths(en)


def test_messages_translate_only_solvi_words():
    ru = lambda s: i18n.msg(s, "ru")      # noqa: E731
    assert ru("hard check amount_ok is false") == "жёсткая проверка amount_ok не пройдена"
    assert ru("low confidence 0.40 < 0.5; would have answered 'travel' (category = 'travel')") == \
        "низкая уверенность 0.40 < 0.5; ответ был бы 'travel' (category = 'travel')"
    assert ru("rule not computed: missing inputs: category; missing category") == \
        "правило не вычислено: не хватает входов: category; не хватает category"
    assert ru("model escalated: every model of the cascade escalated (small: confidence 0.40 < 0.75 (shared threshold); "
              "would have answered 'x'; large: act 0.20 < 0.50 (shared threshold); would have answered 'y')") == \
        ("модель передала человеку: каждая модель каскада передала решение (small: уверенность 0.40 < 0.75 (общий порог); "
         "ответ был бы 'x'; large: act 0.20 < 0.50 (общий порог); ответ был бы 'y')")
    # your text is never translated: a rule's why (facts and values), an unknown message, an exception's text
    for s in ("amount = 48.6; category = 'travel'", "the customer asked twice", "KeyError: 'latency_buckets'"):
        assert ru(s) == s
    assert ru("error: KeyError: 'hard check x is false'") == "ошибка: KeyError: 'hard check x is false'"
    # English is the identity
    for s in ("hard check a is false", "anything at all; not stated"):
        assert i18n.msg(s) == s and i18n.msg(s, "en") == s


def test_every_key_has_a_translation():
    assert set(i18n.RU) == set(i18n.EN)
    for k, v in i18n.EN.items():                        # the same placeholders
        assert sorted(re.findall(r"\{(\w+)\}", v)) == sorted(re.findall(r"\{(\w+)\}", i18n.RU[k])), k
    for en, ru in i18n.MESSAGES_RU + i18n.FLOW_RU:
        assert {m for m in re.findall(r"\{(\w+)(?::\w+)?\}", en)} == set(re.findall(r"\{(\w+)\}", ru)), en


def test_lang_is_checked():
    cat = Catalog()

    @cat.rule("ok")
    def ok(x):
        return "yes" if x else "no"
    qs = [Question("ok", "Is it ok?", Answer.yes_no())]
    with pytest.raises(ValueError, match="lang must be one of"):
        System(cat, qs, lang="de")
    s = System(cat, qs, lang="ru")
    res = s.ask({"x": 1})
    assert res.lang == "ru" and "lang" not in res.to_dict()
    assert str(res.audit()).splitlines()[2].startswith("ok = 'yes'  [ок]  уверенность 1.00  ← вычислено: ok")
    with pytest.raises(ValueError):
        res.audit(lang="xx")


def test_an_exceptions_text_is_never_translated_and_text_in_messages_are():
    ru = lambda s: i18n.msg(s, "ru")      # noqa: E731
    # the text of your exception stays yours, even when it reads like one of solvi's messages
    for s in ("ValueError: missing invoice number", "LookupError: not stated", "RuntimeError: timed out after 3 s",
              "app.errors.BillingException: no value"):
        assert ru(s) == s
    assert ru("rule not computed: ValueError: missing invoice number") == \
        "правило не вычислено: ValueError: missing invoice number"
    assert ru("small: confidence 0.40 < 0.75") == "small: уверенность 0.40 < 0.75"        # a nested reason still is
    # the part that failed further up (System._caused_by): its exception stays, solvi's own reasons are translated
    assert ru("hard check day_allowed could not be evaluated: missing inputs: violations; "
              "caused by spec: ValueError: no slot in the plan") == \
        ("жёсткую проверку day_allowed не удалось вычислить: не хватает входов: violations; "
         "причина — spec: ValueError: no slot in the plan")
    assert ru("rule not computed: missing inputs: total; caused by rate: timed out after 2 s and fee: rejected by "
              "validate") == ("правило не вычислено: не хватает входов: total; причина — rate: время истекло: 2 с и fee: "
                              "отклонено validate")
    # text in (ask_text): solvi's own messages
    for en, want in (
            ("not stated in the text: amount, currency; rule not computed: missing inputs: amount",
             "не указано в тексте: amount, currency; правило не вычислено: не хватает входов: amount"),
            ("the entry point is unsure, nothing was asked: model escalated: entry point unsure — refund 0.41 < 0.60",
             "вопрос по тексту не выбран, ничего не спрошено: модель передала человеку: вопрос не выбран уверенно — refund 0.41 < 0.60"),
            ("model escalated: entry point unsure — refund 0.48 vs cancel 0.45 (margin < 0.1)",
             "модель передала человеку: вопрос не выбран уверенно — refund 0.48 против cancel 0.45 (разрыв < 0.1)"),
            ("found with confidence 0.30 < 0.50", "найдено с уверенностью 0.30 < 0.50"),
            ("no parser reads list[str]", "нет разбора для типа list[str]"),
            ("cannot parse as date: '12 September': the year is not stated (pass today= to read it in today's year)",
             "не читается как date: '12 September': the year is not stated (pass today= to read it in today's year)")):
        assert ru(en) == want, en
