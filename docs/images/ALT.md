# Alt text for the solvi 0.5.0 launch images

Numbers come from the model cards (solvi-ai/solvi-large, solvi-ai/solvi-base) and exps_v2/RESULTS/experiments/L19.md.

## cover_1600x900.png, cover_linkedin_1200x627.png

- EN: Title "Let the model propose. Let the code decide." with the subtitle "solvi 0.5 — typed, verifiable decisions in
  Python" and a four-step flow: propose (model), check (types, quotes, rules), decide (code), trace (hash-chained).
- RU: Заголовок «Let the model propose. Let the code decide.» и подзаголовок «solvi 0.5 — типизированные, проверяемые
  решения на Python»; схема из четырёх шагов: предложить (модель), проверить (типы, цитаты, правила), решить (код),
  записать след (цепочка хешей).

## architecture.png / architecture_ru.png

- EN: solvi flow diagram. Input (text, JSON or a pydantic state) goes to models that propose (a decider for choice, score,
  yes/no, not stated, span and evidence quote; an extractor for fields), then to deterministic checks (answer types, quote
  grounding by offsets, hard rules, constraints), then code computes the answer, and everything goes into a hash-chained
  trace with an audit. When a check fails, the exits are reject, abstain or escalate.
- RU: Схема solvi. Вход (текст, JSON или состояние pydantic) попадает к моделям, которые предлагают (решатель: выбор,
  оценка, да/нет, «не сказано», фрагмент, цитата-довод; извлекатель: поля), затем к проверкам кодом (типы ответов,
  привязка цитат по смещениям, жёсткие правила, ограничения), затем код считает ответ, и всё пишется в след с цепочкой
  хешей и аудит. Если проверка не прошла — отклонить, воздержаться или эскалировать.

## results_typed.png / results_typed_ru.png

- EN: Bar chart, typed-decisions test (2000 questions), zero-shot accuracy: Laya base 36.0%, GLiNER2.5-Decide 50.3%,
  solvi-base 54.5%, solvi-large 54.5%; solvi-large fitted per process on 300 examples 64.9%. Gold labels come from a
  ~4B teacher, so accuracy means agreement with that teacher.
- RU: Столбчатая диаграмма, тест typed-decisions (2000 вопросов), точность без примеров: Laya base 36.0%,
  GLiNER2.5-Decide 50.3%, solvi-base 54.5%, solvi-large 54.5%; solvi-large после подгонки под процесс на 300 примерах
  64.9%. Эталон — метки учителя (~4B), точность означает согласие с ним.

## results_fd.png / results_fd_ru.png

- EN: Bar chart, Fast Decisions public dev split, our harness. GLiNER2.5-Decide 62.9% zero-shot (with 64 examples not
  measured); solvi-large 59.4% zero-shot, 63.0% with 64 labelled examples; solvi-base 56.3% zero-shot, 60.4% with 64;
  Laya 48.9% zero-shot. GLiNER is ahead zero-shot; solvi-large is level after 64 examples. Official test-split numbers are
  not comparable.
- RU: Столбчатая диаграмма, Fast Decisions, публичная dev-часть, наша обвязка. GLiNER2.5-Decide 62.9% без примеров (с 64
  примерами не измеряли); solvi-large 59.4% без примеров и 63.0% с 64 размеченными; solvi-base 56.3% и 60.4%; Laya
  48.9%. Без примеров впереди GLiNER, solvi-large вровень после 64 примеров. Официальные числа на test-части несравнимы.

## evidence.png / evidence_ru.png

- EN: Bar chart, ContractNLI test: share of evidence quotes that support the answer. Previous base (L14g) 23%, solvi-base
  46%, solvi-large 62%. In-distribution: the ContractNLI train split was used for training.
- RU: Столбчатая диаграмма, ContractNLI test: доля цитат-доводов, поддерживающих ответ. Прежний base (L14g) 23%,
  solvi-base 46%, solvi-large 62%. Тест в распределении: train-часть ContractNLI была в обучении.

## playground_audit.png

- EN: Screenshot of the solvi playground in the browser, preset "New in 0.5 · Answer primitives": five questions answered
  in 1.30 ms, "Trace replay: OK", a table of answers (signed = yes; amount = 149.9 quoted as "149.90" at [67:73]; damaged =
  yes with the quoted sentence) and the audit of "signed": given input, rule, a satisfied constraint, the answer, 100%
  deterministic, no safeguards fired.
- RU: Снимок песочницы solvi в браузере, пресет «New in 0.5 · Answer primitives»: пять вопросов за 1.30 мс, «Trace
  replay: OK», таблица ответов (signed = yes; amount = 149.9 с цитатой «149.90» на [67:73]; damaged = yes с цитатой
  предложения) и аудит ответа signed: вход, правило, выполненное ограничение, ответ, 100% детерминированно, защиты не
  срабатывали.

## playground_tamper.png

- EN: Screenshot of the playground's "Tamper with the trace" panel: the hash chain of five steps, step 1 answer:signed
  changed from True to 999 with the attacker recomputing every hash, and the result "Caught. Replay reports 1 mismatch;
  the first one is at step 1 answer:signed, the exact step you changed" — value 999 ≠ recomputed True.
- RU: Снимок панели «Tamper with the trace» в песочнице: цепочка хешей из пяти шагов, шаг 1 answer:signed изменён с True
  на 999, атакующий пересчитал все хеши, и итог «Caught»: повтор находит расхождение ровно на изменённом шаге 1 —
  значение 999 ≠ пересчитанное True.

## realms_l17.png

- EN: Screenshot of solvi realms after 100 turns with the L17 policy-net faction: the strategy map with four factions'
  territories, cities and units, and the decision card of archer #156: order_military = attack, confidence 0.97; why: "L17
  policy net: attack +0.06, move +0.03, fortify -0.00, defend -0.08; laws removed: none; check: ok → attack"; the answer
  check (laws re-verified after the net) ok; hard check keeps_capital_defender passed.
- RU: Снимок solvi realms после 100 ходов с фракцией L17: карта с территориями четырёх фракций, городами и отрядами, и
  карточка решения лучника #156: order_military = attack, уверенность 0.97; почему: «L17 policy net: attack +0.06,
  move +0.03, fortify -0.00, defend -0.08; laws removed: none; check: ok → attack»; повторная проверка законов после
  сети — ok; жёсткая проверка keeps_capital_defender пройдена.

## code_audit.png

- EN: Code screenshot from how-to 01 (support triage): a hard check no_legal_threat that forces priority high and route
  legal, a priority rule returning Scale["low", "normal", "high"], a ticket threatening a lawyer, and print(res3.audit(
  "priority")). Below, the real output: priority = 'high' [forced], the quoted words "lawyer" and "Third time" with
  offsets, the hard check deciding the answer, 100% deterministic support; replay ok; a forged trace is caught
  ("record modified after execution", "value True ≠ recomputed False").
- RU: Снимок кода из how-to 01 (разбор обращений): жёсткая проверка no_legal_threat, которая ставит приоритет high и
  очередь legal, правило приоритета с типом Scale["low", "normal", "high"], обращение с угрозой юристом и
  print(res3.audit("priority")). Ниже — настоящий вывод: priority = 'high' [forced], цитаты «lawyer» и «Third time» со
  смещениями, ответ решила жёсткая проверка, опора на 100% детерминирована; повтор проходит, подделанный след пойман
  («record modified after execution», «value True ≠ recomputed False»).
