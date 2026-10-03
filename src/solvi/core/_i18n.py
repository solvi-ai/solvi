"""Languages for what solvi renders for people: the audit, solvi.show, the safeguard report and the messages solvi writes
itself (safeguard labels, the `why` of an answer, rejection and escalation reasons).

    system = System(cat, QUESTIONS, lang="ru")      # res.audit(), show(res), system.safeguard_summary() in Russian
    print(res.audit(lang="ru"))                     # or per call; show(res, lang="ru"); system.safeguard_summary(lang="ru")

English is the default and is what solvi records: every message that enters a trace, a stored response or a hash — a
rejection reason, `Result.why`, an escalation message, the plan — is written in English, and a language other than English
changes only the rendering. So traces, hashes, `to_dict()` and replay are the same whatever the language.

What is translated: solvi's own words — headings and labels, safeguard names, statuses and provenance kinds, and the
messages solvi builds from templates (`msg`). What is never translated: anything that came from you — fact, part and
question names, values, options, quoted text, the text of your exceptions and validators. A message solvi does not know
(a template added later, a message from your own code) is shown as it is, in English.

Languages: "en", "ru". A catalog for another language is a dict like RU below, added to LANGS."""
from __future__ import annotations

import re

DEFAULT = "en"

# --- words and templates: key → text. English here is exactly what solvi printed before languages existed.
EN = {
    # safeguards (solvi.audit.LABEL)
    "sg.grounding": "grounding rejected", "sg.type_rejected": "type rejected", "sg.outside_options": "outside the options",
    "sg.rule_abstained": "rule abstained", "sg.low_confidence": "low confidence", "sg.validator": "rejected by validate",
    "sg.hard_check": "hard check decided", "sg.constraint_repair": "constraint repair", "sg.fallback": "fallback producer",
    "sg.escalated": "model escalated", "sg.evidence_missing": "evidence missing", "sg.timeout": "timed out", "sg.instruction": "answer depends on an instruction-like sentence",
    "sg.memory": "memory of corrections disagrees",
    # statuses and provenance kinds
    "status.ok": "ok", "status.abstain": "abstain", "status.forced": "forced",
    "prov.given": "given", "prov.computed": "computed", "prov.quoted": "quoted", "prov.decided": "decided",
    "prov.learned": "learned", "prov.proposed": "proposed",
    # the audit
    "au.head": "audit: {n} answer(s), {m} model output(s), {k} safeguard event(s)",
    "au.overall": "overall: confidence {c}{weakest}; {a}/{t} answered{abstained}{not_stated}",
    "au.weakest": ", weakest {q} {c}", "au.abstained": ", abstained: {qs}", "au.not_stated": ", not stated: {qs}",
    "au.infeasible": "; constraints violated",
    "au.answer": "{q} = {shown}  [{status}]  confidence {c}", "au.prov": "  ← {prov}", "au.by": " by {src}",
    "col.given": "given", "col.computed": "computed", "col.quoted": "quoted", "col.decided": "decided",
    "col.learned": "learned", "col.check": "check", "col.rule": "rule", "col.evidence": "evidence", "col.span": "span",
    "col.scores": "scores", "col.constraint": "constraint", "col.not_run": "not run", "col.answer": "→ answer",
    "col.support": "support", "col.guarantee": "guarantee", "col.safeguards": "safeguards",
    "au.failed": "— rejected/failed: {err}", "au.rejected": "REJECTED — {err}",
    "au.literal": "literal", "au.derived": "derived from", "au.converted": "converted from",
    "au.confidence": "  confidence {c}", "au.reads": "reads {x}", "au.ignored": "ignored {x}",
    "au.hard": "hard", "au.soft": "soft", "au.unknown_check": "hard or soft: not recorded", "au.decides": ", decides the answer", "au.not_computed": " — not computed: {err}",
    "au.unchecked": "  in the text; support not checked", "au.verified": "  verified", "au.not_in_text": "  NOT IN THE TEXT",
    "au.model": "  [model]", "au.satisfied": "satisfied", "au.broken": "BROKEN",
    "au.support": "{n} items ({parts}): {share} deterministic", "au.from_models": ", {n} from models", "au.none": "none",
    "au.kind": "{n} {kind}", "kind.given": "given", "kind.computed": "computed", "kind.quoted": "quoted",
    "kind.quoted_by_model": "quoted by model", "kind.decided": "decided", "kind.learned": "learned",
    "kind.proposed": "proposed",
    "au.none_fired": "none fired", "au.compact": "{q}: {support}; safeguards: {safeguards}",
    "x.act": "act {x}", "x.expected": "expected {x}", "x.margin": "margin {x}", "x.candidates": "candidates {x}",
    "x.pass": "one pass with {x}", "x.own_pass": "own pass (shared pass fell back)",
    "x.truncated": "read {read} of {of} input tokens (the rest was cut)",
    # several models (solvi.multi) in the audit
    "m.cascade": "cascade     ", "m.stage_answered": "stage {i} answered", "m.every_stage": "every stage escalated",
    "m.calls": "; {n} model(s) called", "m.stage": "  stage {i}   ", "m.answered": "answered",
    "m.escalated": "escalated — {e}", "m.alone": "answers alone", "m.vote": "vote        rule {rule}: ",
    "m.all_agree": "all agree", "m.disagree": "they disagree", "m.one_vote": "  vote      ", "m.route": "route       by {by} → ",
    # a memory of corrections (solvi.memory) in the audit
    "mem.head": "memory      {n} case(s) stored, {k} near: ", "mem.proposes": "proposes {v} (strength {s}, agreement {a})",
    "mem.abstains": "abstains: {why}", "mem.agrees": " → agrees with the model", "mem.escalated": " → escalated the decision",
    "mem.answered": " → answered in place of the model ({x})", "mem.disagrees": " → disagrees (already escalated)",
    "mem.agrees_late": " → agrees (already escalated)",
    "mem.case": "  case {id} {v}  distance {d}  {src}",
    # solvi.show
    "sh.answers": "── answers ", "sh.why": "why: ", "sh.line": "confidence {c}",
    "sh.flow": "── flow (chosen by the strategist) ", "sh.skipped": "  skipped at run time: ",
    "sh.learned_order": "  hard checks in learned order:", "sh.state": "── computed_state ",
    "sh.tried": "  {name}: used {used}; tried {tried}", "sh.audit": "── audit (support of each answer, safeguards) ",
    "sh.replay": "── trace replay ", "sh.steps": "  steps {n}, mismatches {m}", "sh.model_steps": "  model steps: ",
    "sh.time": "── time {ms} ms", "fl.not_taken": "not taken: ",
    # Response.computed_state
    "cs.quote": "   quote [{s}:{e}]", "cs.confidence": " confidence {c}", "cs.model": "   model {m}", "cs.error": "   ERROR: {err}",
    # System.safeguard_summary
    "rp.head": "asks {asks}, answers {answers}, abstained {abstained}, model outputs {model_outputs}",
}

RU = {
    "sg.grounding": "не подтверждено текстом", "sg.type_rejected": "не прошло проверку типа",
    "sg.outside_options": "вне вариантов ответа", "sg.rule_abstained": "правило воздержалось",
    "sg.low_confidence": "низкая уверенность", "sg.validator": "отклонено validate",
    "sg.hard_check": "решила жёсткая проверка", "sg.constraint_repair": "исправлено ограничением",
    "sg.fallback": "запасной источник", "sg.escalated": "модель передала человеку",
    "sg.evidence_missing": "нет подтверждающей цитаты", "sg.timeout": "время истекло", "sg.instruction": "ответ зависит от фразы-инструкции во входе",
    "sg.memory": "память исправлений не согласна",
    "status.ok": "ок", "status.abstain": "воздержание", "status.forced": "решено проверкой",
    "prov.given": "дано", "prov.computed": "вычислено", "prov.quoted": "цитата", "prov.decided": "решение модели",
    "prov.learned": "обучено", "prov.proposed": "предложено",
    "au.head": "аудит: ответов {n}, выходов моделей {m}, срабатываний защит {k}",
    "au.overall": "итого: уверенность {c}{weakest}; отвечено {a} из {t}{abstained}{not_stated}",
    "au.weakest": ", слабейший ответ {q} {c}", "au.abstained": ", воздержались: {qs}", "au.not_stated": ", не указано: {qs}",
    "au.infeasible": "; ограничения нарушены",
    "au.answer": "{q} = {shown}  [{status}]  уверенность {c}", "au.prov": "  ← {prov}", "au.by": ": {src}",
    "col.given": "дано", "col.computed": "вычислено", "col.quoted": "цитата", "col.decided": "решено",
    "col.learned": "обучено", "col.check": "проверка", "col.rule": "правило", "col.evidence": "подтверждение",
    "col.span": "фрагмент", "col.scores": "оценки", "col.constraint": "ограничение", "col.not_run": "не запущено",
    "col.answer": "→ ответ", "col.support": "опора", "col.guarantee": "гарантия", "col.safeguards": "защиты",
    "au.failed": "— отклонено или ошибка: {err}", "au.rejected": "ОТКЛОНЕНО — {err}",
    "au.literal": "дословно", "au.derived": "выведено из", "au.converted": "преобразовано из",
    "au.confidence": "  уверенность {c}", "au.reads": "читает {x}", "au.ignored": "не использует {x}",
    "au.hard": "жёсткая", "au.soft": "мягкая", "au.unknown_check": "жёсткая или мягкая: не записано", "au.decides": ", решает ответ", "au.not_computed": " — не вычислено: {err}",
    "au.unchecked": "  есть в тексте; подтверждение не проверено", "au.verified": "  проверено",
    "au.not_in_text": "  НЕТ В ТЕКСТЕ", "au.model": "  [модель]", "au.satisfied": "выполнено", "au.broken": "НАРУШЕНО",
    "au.support": "элементов опоры {n} ({parts}): детерминировано {share}", "au.from_models": ", от моделей {n}",
    "au.none": "нет", "au.kind": "{kind} {n}", "kind.given": "дано", "kind.computed": "вычислено", "kind.quoted": "цитат",
    "kind.quoted_by_model": "цитат моделью", "kind.decided": "решений модели", "kind.learned": "обучено",
    "kind.proposed": "предложено",
    "au.none_fired": "не сработали", "au.compact": "{q}: {support}; защиты: {safeguards}",
    "x.act": "act {x}", "x.expected": "ожидаемое {x}", "x.margin": "разрыв {x}", "x.candidates": "кандидаты {x}",
    "x.pass": "один проход вместе с {x}", "x.own_pass": "свой проход (общий не удался)",
    "x.truncated": "прочитано {read} из {of} токенов входа (остальное отрезано)",
    "m.cascade": "каскад        ", "m.stage_answered": "ответила ступень {i}", "m.every_stage": "все ступени передали человеку",
    "m.calls": "; вызвано моделей: {n}", "m.stage": "  ступень {i}   ", "m.answered": "ответила",
    "m.escalated": "передала человеку — {e}", "m.alone": "отвечает сама", "m.vote": "голосование   правило {rule}: ",
    "m.all_agree": "все согласны", "m.disagree": "мнения расходятся", "m.one_vote": "  голос       ",
    "m.route": "маршрут       по {by} → ",
    "mem.head": "память        случаев {n}, близких {k}: ", "mem.proposes": "предлагает {v} (сила {s}, согласие {a})",
    "mem.abstains": "воздерживается: {why}", "mem.agrees": " → согласна с моделью", "mem.escalated": " → передала решение человеку",
    "mem.answered": " → ответила вместо модели ({x})", "mem.disagrees": " → не согласна (решение уже передано)",
    "mem.agrees_late": " → согласна (решение уже передано)",
    "mem.case": "  случай {id} {v}  расстояние {d}  {src}",
    "sh.answers": "── ответы ", "sh.why": "почему: ", "sh.line": "уверенность {c}",
    "sh.flow": "── поток (выбран стратегом) ", "sh.skipped": "  пропущено при выполнении: ",
    "sh.learned_order": "  жёсткие проверки в выученном порядке:", "sh.state": "── computed_state ",
    "sh.tried": "  {name}: использован {used}; испробованы {tried}",
    "sh.audit": "── аудит (опора каждого ответа, защиты) ", "sh.replay": "── повтор трассы ",
    "sh.steps": "  шагов {n}, расхождений {m}", "sh.model_steps": "  шаги моделей: ", "sh.time": "── время {ms} мс",
    "fl.not_taken": "не взято: ",
    "cs.quote": "   цитата [{s}:{e}]", "cs.confidence": " уверенность {c}", "cs.model": "   модель {m}",
    "cs.error": "   ОШИБКА: {err}",
    "rp.head": "запросов {asks}, ответов {answers}, воздержаний {abstained}, выходов моделей {model_outputs}",
}

# --- messages solvi builds (the `why` of an answer, rejection and escalation reasons, skipped parts): English template →
# translation. {name} is copied as it is (a name, a value, your text) and holds no "; "; {name:any} is copied too and may
# hold anything; {name:msg} is itself a message and is translated. The first template that matches the whole message
# wins; a message no template matches is split at "; " and each part is tried on its own; what still matches nothing
# stays as it is.
MESSAGES_RU = [
    # answers (System._answer)
    ("hard check {f} is false and no answer is set for it", "жёсткая проверка {f} не пройдена, а ответ для этого случая не задан"),
    ("hard check {f} is false: {r}", "жёсткая проверка {f} не пройдена: {r}"),
    ("hard check {f} is false", "жёсткая проверка {f} не пройдена"),
    ("hard check {f} could not be evaluated: {e:msg}", "жёсткую проверку {f} не удалось вычислить: {e}"),
    ("no value", "нет значения"),
    ("cannot compute: {fs}", "нельзя вычислить: {fs}"),
    ("rule not computed: {e:msg}", "правило не вычислено: {e}"),
    ("no step", "шага нет"),
    ("missing inputs: {fs}", "не хватает входов: {fs}"),
    # the part further up whose own failure left the inputs missing (System._caused_by): one, or several joined by "and"
    ("caused by {who}: {x:msg} and {more:msg}", "причина — {who}: {x} и {more}"),
    ("caused by {who}: {x:msg}", "причина — {who}: {x}"),
    ("missing {fs}", "не хватает {fs}"),
    ("the rule abstained (returned None)", "правило воздержалось (вернуло None)"),
    ("rule returned {v}, not one of the answer options", "правило вернуло {v} — это не вариант ответа"),
    ("no rule and the answer head is not fitted (fit)", "нет правила, и голова ответа не обучена (fit)"),
    ("head features not computed: {fs}", "признаки головы ответа не вычислены: {fs}"),
    ("the answer head gave no finite probabilities (non-finite {fs})",
     "голова ответа не дала конечных вероятностей (не конечны {fs})"),
    ("the answer head gave no finite probabilities", "голова ответа не дала конечных вероятностей"),
    ("failed checks: {fs}", "не пройдены проверки: {fs}"),
    ("low confidence {c} < {m}; would have answered {a:any} ({w:msg})",
     "низкая уверенность {c} < {m}; ответ был бы {a} ({w})"),
    ("not stated", "не указано"),
    # text in (System.ask_text, solvi.core.textin)
    ("not stated in the text: {fs}", "не указано в тексте: {fs}"),
    ("not stated in the text", "не указано в тексте"),
    ("the entry point is unsure, nothing was asked: {x:msg}", "вопрос по тексту не выбран, ничего не спрошено: {x}"),
    ("entry point unsure — {a} {p} vs {b} {q} (margin < {m})", "вопрос не выбран уверенно — {a} {p} против {b} {q} (разрыв < {m})"),
    ("entry point unsure — {a} {p} < {m}", "вопрос не выбран уверенно — {a} {p} < {m}"),
    ("found with confidence {c} < {m}", "найдено с уверенностью {c} < {m}"),
    ("no parser reads {t}", "нет разбора для типа {t}"),
    ("cannot parse as {k}: {why:any}", "не читается как {k}: {why}"),
    ("does not re-derive from its quote: {why:any}", "не восстанавливается из своей цитаты: {why}"),
    ("no supporting quote (require_evidence)", "нет подтверждающей цитаты (require_evidence)"),
    ("would have answered {a:any}", "ответ был бы {a}"),
    # rejections (accepts, primitives, typed facts, the executor)
    ("confidence {c} < {m} (escalate_below)", "уверенность {c} < {m} (escalate_below)"),
    ("confidence {c} < {m} (shared threshold)", "уверенность {c} < {m} (общий порог)"),
    ("confidence {c} < {m}", "уверенность {c} < {m}"),
    ("act {c} < {m} (shared threshold)", "act {c} < {m} (общий порог)"),
    ("margin {d} < {m} between {a} ({p}) and {b} ({q})", "разрыв {d} < {m} между {a} ({p}) и {b} ({q})"),
    ("candidates at {c}: {xs}", "кандидаты при покрытии {c}: {xs}"),
    ("model escalated: every model of the cascade escalated ({x:msg})",
     "модель передала человеку: каждая модель каскада передала решение ({x})"),
    ("model escalated: the models disagree ({rule}): {xs}", "модель передала человеку: модели расходятся ({rule}): {xs}"),
    ("model escalated: the models agree on {v:any} ({rule}), but {x:msg}",
     "модель передала человеку: модели согласны на {v} ({rule}), но {x}"),
    ("model escalated: {x:msg}", "модель передала человеку: {x}"),
    ("memory of corrections disagrees: {n} similar corrected case(s) say {v} (strength {s})",
     "память исправлений не согласна: похожих исправленных случаев {n}, в них {v} (сила {s})"),
    ("the memory holds no corrected cases", "в памяти нет исправленных случаев"),
    ("no corrected case within distance {r}", "нет исправленного случая ближе {r}"),
    ("similar cases disagree: {xs} (agreement {a} < {m})", "похожие случаи расходятся: {xs} (согласие {a} < {m})"),
    ("too little support: strength {s} < {m}", "мало опоры: сила {s} < {m}"),
    ("act {p} < {t}", "act {p} < {t}"),
    ("decision {v} is outside the options {opts} (return type {t})", "решение {v} вне вариантов {opts} (тип результата {t})"),
    ("decision {v} is outside the options {opts}", "решение {v} вне вариантов {opts}"),
    ("recorded decision {v} is outside the options {opts}", "записанное решение {v} вне вариантов {opts}"),
    ("ranking {v} is outside the options: {err}", "ранжирование {v} вне вариантов: {err}"),
    ("estimate {v} is outside the options: {err}", "оценка {v} вне вариантов: {err}"),
    ("answered Unknown (not stated), which the question does not allow: declare Maybe[...]",
     "ответ Unknown (не указано), а вопрос этого не допускает: объявите Maybe[...]"),
    ("not grounded: evidence {v} is not in {src}", "не подтверждено текстом: цитаты {v} нет в {src}"),
    ("not grounded: evidence {v} is not the text at {at} ({t})", "не подтверждено текстом: цитата {v} — не текст в {at} ({t})"),
    ("not grounded: span {v} is not the text at {at} ({t})", "не подтверждено текстом: фрагмент {v} — не текст в {at} ({t})"),
    ("not grounded: span {v} is not in {src}", "не подтверждено текстом: фрагмента {v} нет в {src}"),
    ("not grounded: a span answer is a Quote or a text in {src}, not {v}",
     "не подтверждено текстом: ответ-фрагмент — это Quote или текст из {src}, а не {v}"),
    ("not grounded: recorded {v} is not the text at {at}", "не подтверждено текстом: записанное {v} — не текст в {at}"),
    ("not grounded: {v} is not the text at {at} ({t})", "не подтверждено текстом: {v} — не текст в {at} ({t})"),
    ("not grounded: {x}", "не подтверждено текстом: {x}"),
    ("quote outside the text: evidence {at} of {src}", "цитата за пределами текста: цитата {at} из {src}"),
    ("quote outside the text: span {at} of {src}", "цитата за пределами текста: фрагмент {at} из {src}"),
    ("quote outside the text", "цитата за пределами текста"),
    ("rejected by validate", "отклонено validate"),
    ("validate raised {x}", "validate выбросил {x}"),
    ("type rejected: given {k} = {v} is not {t} ({err})", "не прошло проверку типа: дано {k} = {v} — не {t} ({err})"),
    ("type rejected: given {k} = {v} is not {t}", "не прошло проверку типа: дано {k} = {v} — не {t}"),
    ("type rejected: returned {v}, not {t} ({err})", "не прошло проверку типа: возвращено {v}, а не {t} ({err})"),
    ("type rejected: span {v} is not {t} ({err})", "не прошло проверку типа: фрагмент {v} — не {t} ({err})"),
    ("type rejected: {x} = {v} is not {t} ({err})", "не прошло проверку типа: {x} = {v} — не {t} ({err})"),
    ("type rejected: {x}", "не прошло проверку типа: {x}"),
    ("timed out after {t} s", "время истекло: {t} с"),
    ("error: {t}: {x}", "ошибка: {t}: {x}"),
    ("no producer accepted: {x}", "ни один источник не принят: {x}"),
    ("accepted", "принят"),
    ("shadow: agrees", "тень: совпадает"),
    ("shadow: differs", "тень: расходится"),
    ("shadow: {x:msg}", "тень: {x}"),
    # safeguard details, parts not run
    ("{p} used after {ps} rejected", "использован {p}, после того как отклонены {ps}"),
    ("changed from {a} to satisfy {cs}", "изменено с {a}, чтобы выполнить {cs}"),
    ("not needed: hard check {cs} failed", "не понадобилось: не пройдена жёсткая проверка {cs}"),
    ("not needed", "не понадобилось"),
    ("no inputs", "нет входов"),
    ("not needed for questions", "не нужно для вопросов"),
    # guarantees (act_guard, calibrate_for)
    ("P(answered alone and wrong) ≤ {r} for inputs like the calibration examples",
     "P(ответ без человека и неверный) ≤ {r} для входов, похожих на калибровочные примеры"),
    ("error among the answers given alone ≤ {e} with probability ≥ {p}, for inputs like the calibration examples",
     "ошибка среди ответов без человека ≤ {e} с вероятностью ≥ {p} для входов, похожих на калибровочные примеры"),
    ("none: the error was measured on the calibration examples only",
     "нет: ошибка измерена только на калибровочных примерах"),
    ("none for some decisions: their thresholds were not calibrated on your data (see act_guard)",
     "нет для части решений: их пороги не откалиброваны на ваших данных (см. act_guard)"),
    ("{p:msg} ({m}, n = {n})", "{p} ({m}, n = {n})"),
    # a nested reason: "<who>: <reason>" (a stage of a cascade, a vote, a producer). A template that starts with a
    # placeholder is tried last, on a message of one part only: it matches almost anything with a colon
    ("{who}: {x:msg}", "{who}: {x}"),
]

# the strategist's reasons for a step of the flow (Flow.__str__), and the learned order: a scope of their own, since
# "rule {q}" or "answer {q}" would also match your own text in a `why`
FLOW_RU = [
    ("input for {f}", "вход для {f}"),
    ("rule {q}", "правило {q}"),
    ("answer feature {q}", "признак ответа {q}"),
    ("uses hint {q}", "подсказка uses {q}"),
    ("{q}: flow not narrowed (no rule, fit or uses)", "{q}: поток не сужен (нет rule, fit или uses)"),
    ("checkpoint {q}", "контрольная точка {q}"),
    ("check on computed: {fs}", "проверка вычисленного: {fs}"),
    ("answer {q}", "ответ {q}"),
    ("default order: hard checks and their inputs first, in flow order",
     "порядок по умолчанию: сначала жёсткие проверки и их входы, в порядке потока"),
]

LANGS = {"en": (EN, [], []), "ru": ({**EN, **RU}, MESSAGES_RU, FLOW_RU)}


def check(lang):
    """A language code → itself; None → the default ("en"); an unknown code → ValueError."""
    if lang is None:
        return DEFAULT
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {sorted(LANGS)}, not {lang!r}")
    return lang


def t(key, lang=None, **kw):
    """A word or template of the catalog, in a language, formatted with kw."""
    s = LANGS[check(lang)][0][key]
    return s.format(**kw) if kw else s


def label(kind, lang=None):
    """A safeguard's label (grounding → "grounding rejected")."""
    return t("sg." + kind, lang)


def status(s, lang=None):
    return LANGS[check(lang)][0].get("status." + str(s), s)


def provenance(p, lang=None):
    return LANGS[check(lang)][0].get("prov." + str(p), p) if p else p


def width(keys, lang=None, en=12):
    """The column width for a set of labels: the English width as it always was; in another language, the longest label
    plus a space (at least the English width)."""
    lang = check(lang)
    if lang == DEFAULT:
        return en
    return max(en, max(len(t(k, lang)) for k in keys) + 1)


_compiled = {}
_EXCEPTION = re.compile(r"[\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Timeout)\b: ")


def _compile(table):
    out = []
    for en, tr in table:
        rx, pos, msgs = "", 0, set()
        for m in re.finditer(r"\{(\w+)(?::(msg|any))?\}", en):
            rx += re.escape(en[pos:m.start()]) + (f"(?P<{m.group(1)}>.+?)" if m.group(2) else f"(?P<{m.group(1)}>[^;]+?)")
            if m.group(2) == "msg":
                msgs.add(m.group(1))
            pos = m.end()
        rx += re.escape(en[pos:])
        out.append((re.compile(rx, re.S), tr, msgs, en.startswith("{")))   # last: a template that starts with a value
    return out


def _table(lang, scope):
    key = (lang, scope)
    if key not in _compiled:
        _compiled[key] = _compile(LANGS[lang][1] if scope == "msg" else LANGS[lang][2])
    return _compiled[key]


def msg(text, lang=None, scope="msg", _depth=0):
    """A message solvi wrote (in English, as recorded) → the same message in a language. Unknown messages, and the parts
    of a message that came from you, stay as they are; English returns the text unchanged."""
    lang = check(lang)
    if lang == DEFAULT or not isinstance(text, str) or not text or _depth > 8:
        return text
    table = _table(lang, scope)
    got = _match([x for x in table if not x[3]], text, lang, scope, _depth)
    if got is not None:
        return got
    if "; " in text:
        pieces = text.split("; ")
        return "; ".join(msg(p, lang, scope, _depth + 1) for p in pieces)
    if _EXCEPTION.match(text):                        # "<ExceptionType>: <its text>" came from you: never translated,
        return text                                   # even when the text reads like one of solvi's messages
    got = _match([x for x in table if x[3]], text, lang, scope, _depth)     # "<who>: <reason>", one part only
    return text if got is None else got


def _match(table, text, lang, scope, depth):
    for rx, tr, msgs, _ in table:
        m = rx.fullmatch(text)
        if m:
            parts = {k: (msg(v, lang, scope, depth + 1) if k in msgs else v) for k, v in m.groupdict().items()}
            return re.sub(r"\{(\w+)\}", lambda x: parts[x.group(1)], tr)
    return None


__all__ = ["check", "DEFAULT", "EN", "FLOW_RU", "label", "MESSAGES_RU", "msg", "provenance", "RU", "status", "t",
           "width"]
