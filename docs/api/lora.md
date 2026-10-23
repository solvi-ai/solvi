# `solvi.experimental.lora`

A LoRA adapter on the decider for one question (experimental): `solvi.experimental.lora.adapt_lora(part, examples)`, `part.load_lora`, `solvi.experimental.lora.remove_lora(part)`.

Training on the CPU (`device="cpu"`) creates no CUDA context, so it works beside a GPU that is busy or full.

An adapter belongs to a decision part (its name, on its model), not to the question's text: two parts asking the same
task and options in one process — the folds of a cross-validation — each have their own, and a part never scores with
another part's adapter (a `SharedQuestionWarning` names both parts once). The `lora` extra needs `peft>=0.19`; an older
peft is refused before any training.

Versions of one part (the adapter loop, `solvi.experimental.stream.lora_learner`): `swap(part, adapter, calibration)`
changes the adapter and its calibration (`calibration(part)`: the thresholds, the guarantee, the conformal set, the
adaptation) together under the part's version lock (`version_lock(part)`, taken by each of the part's decisions) and
the scorer's lock, so a decision of the part never sees an adapter without its own thresholds, and a decision on
another question of the same model is not disturbed (the model's adapter slots are replaced, never changed in place) —
except through a combination (Cascade, Vote, Route) over the part, which reads the part without its version lock and
can pair one version's scores with the other's thresholds during a swap. adapt_lora and load_lora leave that window
open (the adapter is on, the thresholds are cleared until act_guard runs). Swapping back is bit-exact (the
part's and the model's fingerprints, the probabilities). `stage(part, adapter)` installs an adapter inactive, outside
any slot; `scoring(part, adapter)` scores the part's question with it on the calling thread only.
`adapter_by_hash(folder, hash)` finds an adapter file by the hash a decision recorded. Unmeasured beyond the tests.

## Install on x86_64 macOS (Intel)

The last torch for x86_64 macOS is 2.2.2. transformers 5 and NumPy 2 do not run with it, and on Python 3.12 the
transformers 4.x ModernBERT module calls `torch.compile` when it is imported, which Dynamo refuses there ("Dynamo is
not supported on Python 3.12+"). A working set, reported by a user on an Intel Mac with Python 3.12:

```bash
pip install "torch==2.2.2" "numpy<2" "transformers==4.48.3" "peft>=0.19" "solvi[lora]"
export SOLVI_NO_COMPILE=1          # torch.compile becomes a no-op while the decider's encoder is imported and built
```

`pip install "solvi[lora]"` on that platform asks for `transformers<5` and `numpy<2` by itself (environment markers
for x86_64 macOS only; other platforms are unchanged). `SOLVI_NO_COMPILE=1` must be set before the decider is loaded
(`solvi.models.decider(..., backend="torch")` / `DecideModel.load`); it also turns ModernBERT's compiled MLP off at run
time. The ONNX backend (`backend="onnx"`, `solvi[onnx]`) needs neither torch nor this setting.

::: solvi.experimental.lora
