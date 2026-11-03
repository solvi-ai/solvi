# S298: безопасный режим композиции

База: 11131e4524c33d0bc5dfa457b1eec3ced3a31b2b. Ветка codex/growth-drift-guard.

Исходный отчёт: stuff/handoffs/solvi_handoff_2026-11-02_src/solvi_handoff_2026-11-02_defects_ratchet/bag_reports/solvi_1.4.0.md:808. SHA256 fcb85f2c2789c69cb87e7280a9bc20c56628bbb41a6e9df177ca890829161bf2. В исходном кейсе сертификация и loosening меняют alone-share, вызывая ложные тревоги на неизменном CLINC. Текущая база вызывает monitor.adapted() после certified/loosened/recertify, собирая новую reference; одновременный shift может стать этой reference. RiskDetector питается отдельно в sampling.py и не сбрасывается самим threshold move, но только собственной тревогой.

Реализован узкий guarded mode: Dispatcher с promise и monitor.adapted без detector отвергается ValueError при создании, до первого решения. Явный allow_reference_retake=True допускает прежнюю композицию с документированной слепой зоной. Это не новая статистическая гарантия. При detector поведение мониторинга и confidence bars неизменно; раздельный reset confidence/alone каналов не реализован, поскольку текущая совместная калибровка CUSUM требует отдельного проекта.

Проверка реальным runtime: tests/test_stream_reference_guard.py использует настоящий Dispatcher, StreamPromise, RiskDetector, DriftMonitor и rule-based системы с искусственными labels. Одинаковый тест на базовом source даёт 1 failed / 1 passed: отсутствовал требуемый отказ. После изменения 2 passed. Label-регрессия сертифицирует clean stream, проверяет отсутствие тревоги, вызывает ровно действие monitor.adapted(), которое dispatcher делает при threshold move, и проверяет сохранение detector.n и тревогу после смены gold. Важно: одновременный естественный loosening+shift не воспроизведён, вызов adapted здесь явный.

Фокусные проверки после изменения: 4 passed, 2 warnings in 3.30s, включая существующий unchanged-stream certification+loosening контроль и существующий реальный score shift. Все Python-запуски через stuff/audit/run.sh, 4 CPU/BLAS threads. Логи вне clone: research_growth/growth_2026_11/s298_drift_{baseline,focused,controls}_20261103.log. Никаких fullsuite, quality eval, Colab и push.

Остатки: исторический CLINC 20-seed false-alarm rate не повторялся; label-only detector не гарантирует обнаружение без роста sampled deployed risk, sampling увеличивает задержку; ARL не является horizon false-alarm probability. Потребители legacy promise+monitor без detector теперь должны явно включить opt-out либо добавить detector. Внешние API guides и все call sites не мигрированы. S298 целиком не объявляется закрытым.

## Integration follow-up

`allow_reference_retake` теперь принимает только настоящий bool; проверка стоит первой в конструкторе, до чтения System.questions и любой работы конструктора. Значение хранится в `self.allow_reference_retake`. `Dispatcher.config()` добавляет tuple `('allow_reference_retake', True)` только для включённого waiver: сохранённые dispatch records связывают waiver через configuration fingerprint, а False не меняет прежний digest format.

Добавлены семь invalid-type controls (включая строку 'false', числа, None и контейнеры) с sentinel System, который запрещает любые обращения до type check. Проверено наличие True в digest inputs, отсутствие default key и равенство default digest прежнему набору arguments. Настоящий explicit-waiver Dispatcher сохраняет решения в JSONLStorage; stored config совпадает с текущим, replay_all и replay_stream проходят, fresh dispatcher с тем же waiver воспроизводит stream.

Миграция описана в docs/migration-1.5-candidate.md: нужен independent label detector либо явное согласие на blind spot; detector не доказывает обнаружение confidence-only shift без роста deployed labelled risk; confidence bars и calibration не меняются.

Ближайший пакет из пяти файлов дал 70 passed / 1 failed in 18.53s. Единственный failure был в новом тесте replay: тест не создавал необходимое storage. Исправлено только создание тестового JSONLStorage, production source после этого не менялся. Финальный focused rerun: 11 passed in 0.67s (последний точный elapsed в s298_drift_final_focused_20261103.log); остальные 60 nearest tests уже прошли. Итог: 71 distinct targeted tests подтверждены между этими запусками; полного повторного пакета и full suite не было. Лог ближайшего пакета: s298_drift_integration_20261103.log. Все Python-запуски через wrapper; source/diff checks выполнены. Широкая compatibility миграция сторонних callers и исторические statistical controls по-прежнему не заявлены.
