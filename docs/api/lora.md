# `solvi.experimental.lora`

A LoRA adapter on the decider for one question (experimental): `solvi.experimental.lora.adapt_lora(part, examples)`, `part.load_lora`, `solvi.experimental.lora.remove_lora(part)`.

Training on the CPU (`device="cpu"`) creates no CUDA context, so it works beside a GPU that is busy or full.

::: solvi.experimental.lora
