"""Train a LoRA adapter for one decision offline, on a GPU (experimental) — for solvi-large, or thousands of examples.

`solvi.experimental.lora.adapt_lora(part, examples)` trains in-process and only on solvi-base-sized checkpoints (on a CPU it takes minutes). This
script trains the same adapter (the same recipe, solvi.experimental.lora.train) on any checkpoint and device, and writes a file that
`part.load_lora(path)` loads wherever the decision runs — a CPU is fine for answering:

    uv run --with torch --with transformers --with peft python tools/adapt_lora_gpu.py \\
        solvi-ai/solvi-large labels.jsonl --task "Which team should handle this ticket?" \\
        --options billing technical shipping [--fact email] [--kind choice] [--holdout 300] [--out team.lora.safetensors]

LABELS: JSON lines (or a .csv), one labelled example per row: `label` plus the input — the fact the part reads as a column
(`--fact`, default "doc"), or a `text` / `input` column, or else the other columns as a state (as `solvi calibrate`
reads them). The question must be asked exactly as the catalog asks it (task, options, descriptions via --descriptions
a JSON object, kind, other): the adapter is refused by a part asking another question.

--holdout N keeps N rows out of training and prints the held-out accuracy before / after and act_guard at --risk on
them; calibrate the part again where it runs (part.act_guard or `solvi calibrate`) after load_lora — or save both with
part.save_calibration there. Measured on a GPU (Colab G4): about 20 s for 300 examples on solvi-base, 26 s on solvi-large."""
from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="a solvi-decide checkpoint folder or Hugging Face id (needs model.safetensors)")
    ap.add_argument("labels", help="labelled examples: JSON lines or .csv with a 'label' column and the input")
    ap.add_argument("--task", required=True, help="the question, as the catalog asks it")
    ap.add_argument("--options", nargs="+", help="the options (not needed for --kind noul)")
    ap.add_argument("--descriptions", help='the options\' descriptions, a JSON object {"option": "description"}')
    ap.add_argument("--kind", default="choice", choices=["choice", "multi", "score", "noul"])
    ap.add_argument("--other", default=None, help='the "other" option (default: found by name; "false": none)')
    ap.add_argument("--fact", default="doc", help="the fact the decision reads (default doc)")
    ap.add_argument("--holdout", type=int, default=0, help="rows kept out of training for the before / after check")
    ap.add_argument("--risk", type=float, default=0.10, help="act_guard's risk on the held-out rows (default 0.10)")
    ap.add_argument("--r", type=int, default=8, help="the adapter's rank (default 8)")
    ap.add_argument("--epochs", type=float, default=6, help="passes over the examples (default 6)")
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (default 3e-4)")
    ap.add_argument("--max-updates", type=int, default=400, help="at most this many updates of 8 examples (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda (default when available), cpu, cuda:1, ...")
    ap.add_argument("--out", default=None, help="the adapter file (default <fact>.lora.safetensors)")
    a = ap.parse_args(argv)

    import torch

    from solvi.cli._calibrate import examples_of, read_rows
    from solvi.core.deciders import DecideModel
    from solvi.experimental.lora import adapt
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("adapt_lora_gpu: no GPU found — training on the CPU (slow for solvi-large)", file=sys.stderr)
    model = DecideModel.load(a.model, backend="torch", device=device)
    opts = a.options or ()
    desc = json.loads(a.descriptions) if a.descriptions else None
    other = False if a.other == "false" else a.other
    part = model.decision(a.fact, a.task, a.fact, opts, descriptions=desc, kind=a.kind, other=other)
    examples = examples_of(part, read_rows(a.labels))
    t0 = time.time()
    rep = adapt(part, examples, r=a.r, epochs=a.epochs, holdout=a.holdout or None, seed=a.seed, device=device,
                lr=a.lr, max_risk=a.risk, max_updates=a.max_updates, allow_large=True)   # warnings: the estimate, few examples
    out = a.out or f"{a.fact}.lora.safetensors"
    part.save_lora(out)
    print(f"adapter #{rep['adapter']}: {rep['k']} examples, {rep['updates']} updates on {rep['device']} in "
          f"{time.time() - t0:.0f} s, {rep['size_mb']} MB → {out}")
    h = rep["holdout"]
    if h:
        g = h["act_guard"]
        print(f"held out {h['n']}: accuracy {h['accuracy_before']:.3f} → {h['accuracy_after']:.3f}; act_guard at risk "
              f"{a.risk:g}: answered alone {g['answered']:.1%}, risk {g['risk']:.3f}")
    print(f"load it where the decision runs: part.load_lora({out!r}), then calibrate (part.act_guard / solvi calibrate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
