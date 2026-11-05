# `solvi.core.deciders.llm`

Any OpenAI-compatible chat-completions server (OpenAI, OpenRouter, vLLM, llama.cpp, Ollama) as a solvi decider.

::: solvi.core.deciders.llm


## Transport settings

`llm(...)`, `LLMScorer(...)` and the chart `LLMProposer(...)` forward explicit
`timeout`, `retries`, and `backoff` to the shared transport. Existing defaults are
unchanged: 60 seconds per attempt, two additional retries, and one second initial
backoff. The System One client shares the same validation but keeps its own
30-second default timeout.

`timeout` must be finite and strictly positive. `backoff` must be finite and
nonnegative; zero disables the delay. Numeric strings for these seconds values
remain accepted. `retries` must be a nonnegative integer count, not a boolean,
floating-point number or string; negative counts are rejected rather than clamped.
Malformed settings raise a field-specific `ValueError` at construction, before
any request or sleep. A retry waits `backoff * 2**k` seconds and sends the same
request again; timeout is per attempt, not an overall deadline. These options do
not add retries for authentication/configuration errors or change format fallback.
