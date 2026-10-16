# `solvi.experimental.injection`

A calibrated detector of instruction-like text in tool outputs (experimental): `InjectionDetector().fit(texts, spans)`,
`.calibrate(ordinary_texts, max_flag_rate)`, `.spans(text)` (the shape of `injection_spans`).
Opt-in, no shipped weights; the fixed rules stay the default. Fit and calibrate it on the outputs your own tools return:
its measured gain held only on the injection family it was fitted on ([status](../experimental.md#injection)).

::: solvi.experimental.injection
