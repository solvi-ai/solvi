"""Decisions with a model: a cross-encoder that picks among options described in words ("decider").

The decider (solvi-decide, ModernBERT) reads one sequence per decision

    [mode] task[opt] option 1[opt] option 2 ... [SEP] text

and gives one logit per option marker in a single pass: `[unused1]` = choose one (softmax over the options), `[unused2]` =
every option that applies (a sigmoid per option). An option may carry a description ("label: description").

In solvi a decider is a catalog part like any other: `model.decision(...)` returns a function that returns
`solvi.Decision(value, probs)`. The value is one of the declared options by construction, the part's provenance is `decided`
and the model's identity (weights, calibration and adaptation of this part) is in the trace, so the closed set,
`min_confidence`, constraints with joint decoding, hard checks, the audit and the stats apply unchanged.

On top of the raw logits, per (task, options):
  - label-bias correction without labels (`adapt`): the mean logit of each option over unlabelled texts of the domain is
    subtracted before the softmax (the decider likes some labels regardless of the text; +7 points in research L14b);
  - few-shot adaptation "S" (`fit`, `teach`): a per-option shift and a shared scale fitted on k labelled examples (L-BFGS),
    with a temperature fitted on out-of-fold predictions, so confidences are calibrated; `teach` updates the shift at once;
  - "other" / "none" as an abstain threshold: such an option is not scored by the model; it is chosen when the best real
    option's calibrated probability is below a threshold (fitted on labelled examples that include it, else the default).

Backends: "torch" (`solvi[model]`) or "onnx" (`solvi[onnx]`: onnxruntime + tokenizers, no torch), or any object with
`logits(items)` (tests, other models): see DecideModel."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from .core import Decision, Quote

OPT, ONE, MANY = "[unused0]", "[unused1]", "[unused2]"
OTHER_NAMES = ("other", "none", "none of the above", "none of these", "other / none", "nothing")
DEFAULT_T = {"l14b_decider v1": 1.45}          # temperature fitted on the training pool's validation split (L14b)


@dataclass(frozen=True)
class Item:
    """One decision to score: the scorer returns one logit (or a [single, multi] pair) per option."""
    task: str
    options: tuple
    descriptions: tuple | None
    text: str
    multi: bool = False


def prompt(task, options, descriptions=None, multi=False):
    """The decider's first segment: "[mode] task[opt] option ..." (an option with a description is "label: description")."""
    opts = [o if not descriptions or not descriptions[i] else f"{o}: {descriptions[i]}" for i, o in enumerate(options)]
    return f"{MANY if multi else ONE} {task}" + "".join(f"{OPT} {o}" for o in opts)


# ------------------------------------------------------------------------------------------------ backends
class _Encoder:
    """Tokenization with the checkpoint's tokenizer.json (the `tokenizers` library): the prompt is kept whole, only the text is
    truncated; returns token ids and the positions of the option markers."""

    def __init__(self, path, max_len):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len, strategy="only_second")
        self.max_len = max_len
        self.opt_id = self.tok.token_to_id(OPT)
        pad = next((self.tok.token_to_id(t) for t in ("[PAD]", "<pad>") if self.tok.token_to_id(t) is not None), 0)
        self.pad_id = pad
        if self.opt_id is None:
            raise ValueError(f"the tokenizer has no {OPT} token: not a solvi-decide checkpoint")

    def encode(self, it):
        p = prompt(it.task, it.options, it.descriptions, it.multi)
        try:
            enc = self.tok.encode(p, it.text or " ")
        except Exception as e:  # noqa: BLE001  (the prompt alone is longer than max_len)
            raise ValueError(f"task and options do not fit in {self.max_len} tokens: {e}") from None
        ids = enc.ids
        pos = [i for i, t in enumerate(ids) if t == self.opt_id]
        if len(pos) != len(it.options):
            raise ValueError(f"found {len(pos)} option markers for {len(it.options)} options (an option contains {OPT}, "
                             f"or the options do not fit in {self.max_len} tokens)")
        return ids, pos


def _batches(encs, bs):
    order = sorted(range(len(encs)), key=lambda i: len(encs[i][0]))
    for b in range(0, len(order), bs):
        yield order[b:b + bs]


class OnnxScorer:
    """onnxruntime session over onnx/model_*.onnx (inputs input_ids, attention_mask; output logits [B, L, 2])."""

    def __init__(self, path, max_len=512, device=None, onnx_file=None, bs=16):
        import onnxruntime as ort
        self.enc = _Encoder(path, max_len)
        self.file = onnx_file or _onnx_file(path)
        if self.file is None:
            raise FileNotFoundError(f"no ONNX model in {path}/onnx")
        providers = ["CPUExecutionProvider"]
        if device and str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        self.sess = ort.InferenceSession(self.file, providers=providers)
        self.names = [i.name for i in self.sess.get_inputs()]
        self.bs = bs
        self.tag = "onnx:" + os.path.basename(self.file)

    def logits(self, items):
        encs = [self.enc.encode(it) for it in items]
        out = [None] * len(items)
        for ch in _batches(encs, self.bs):
            L = max(len(encs[i][0]) for i in ch)
            ids = np.full((len(ch), L), self.enc.pad_id, dtype=np.int64)
            att = np.zeros((len(ch), L), dtype=np.int64)
            for r, i in enumerate(ch):
                n = len(encs[i][0])
                ids[r, :n], att[r, :n] = encs[i][0], 1
            feed = {"input_ids": ids, "attention_mask": att}
            lg = self.sess.run(None, {k: v for k, v in feed.items() if k in self.names})[0].astype(np.float32)
            for r, i in enumerate(ch):
                out[i] = lg[r, encs[i][1]]
        return out


class TorchScorer:
    """The checkpoint's encoder (transformers AutoModel from config.json) + the option head, weights from model.safetensors."""

    def __init__(self, path, max_len=512, device=None, bs=16):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = _Encoder(path, max_len)
        sd = load_file(os.path.join(path, "model.safetensors"))
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path))
        h = encoder.config.hidden_size
        n_out = sd["head.3.weight"].shape[0] if "head.3.weight" in sd else 2
        head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.LayerNorm(h), torch.nn.Linear(h, n_out))

        class Decider(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc, self.head = encoder, head

            def forward(self, input_ids, attention_mask):
                return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

        self.model = Decider()
        self.model.load_state_dict({k: v.float() for k, v in sd.items()})
        self.model.eval().to(self.device)
        self.bs = bs
        self.tag = "torch"

    def logits(self, items):
        torch = self.torch
        encs = [self.enc.encode(it) for it in items]
        out = [None] * len(items)
        with torch.no_grad():
            for ch in _batches(encs, self.bs):
                L = max(len(encs[i][0]) for i in ch)
                ids = torch.full((len(ch), L), self.enc.pad_id, dtype=torch.long)
                att = torch.zeros((len(ch), L), dtype=torch.long)
                for r, i in enumerate(ch):
                    n = len(encs[i][0])
                    ids[r, :n] = torch.tensor(encs[i][0])
                    att[r, :n] = 1
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                    lg = self.model(ids.to(self.device), att.to(self.device)).float().cpu().numpy()
                for r, i in enumerate(ch):
                    out[i] = lg[r, encs[i][1]]
        return out


def _onnx_file(path):
    d = os.path.join(path, "onnx")
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".onnx"))
    for pref in ("model_fp16.onnx", "model.onnx", "model_fp32.onnx"):
        if pref in files:
            return os.path.join(d, pref)
    return os.path.join(d, files[0]) if files else None


def _file_fingerprint(paths, chunks=32, size=4096):
    """Names, sizes and evenly spaced 4 KB chunks of the checkpoint files: ~ms, and any retrained checkpoint differs."""
    h = hashlib.sha256()
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        n = os.path.getsize(p)
        h.update(f"{os.path.basename(p)}:{n}".encode())
        with open(p, "rb") as fh:
            if n <= chunks * size:
                h.update(fh.read())
            else:
                for k in range(chunks):
                    fh.seek(k * (n - size) // (chunks - 1))
                    h.update(fh.read(size))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ calibration of one decision
@dataclass
class Adaptation:
    """What solvi learned for one (task, options): the label-bias correction, the few-shot shift / scale, the temperature and
    the "other" threshold. Part of the fingerprint of every decision that uses it."""
    bias: list | None = None             # centered mean logit per option over unlabelled texts (subtracted)
    n_unlabelled: int = 0
    scale: float | None = None           # a in (a·z + b) / temperature (None: 1 / the model's temperature)
    shift: list | None = None            # b, per option
    temperature: float = 1.0             # fitted on out-of-fold predictions
    other_threshold: float | None = None  # fitted on labelled examples that include "other"
    n_labelled: int = 0
    examples: list = field(default_factory=list)   # [(raw logits, label)] kept for teach and refits

    def params(self):
        """Everything that changes the output (the examples only through the fitted parameters)."""
        d = asdict(self)
        d.pop("examples")
        return d


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _fit_shift(Z, Y, multi, a0, lam=1.0, iters=300, init=None):
    """S: minimize the mean loss of a·z + b + λ/n·(‖b‖² + (a − a0)²) with L-BFGS. Z [n, K]; Y label indices (single choice)
    or a 0/1 matrix [n, K] (multi-label)."""
    from scipy.optimize import minimize
    n, K = Z.shape
    oh = Y if multi else np.eye(K)[Y]

    def f(w):
        a, b = w[0], w[1:]
        s = a * Z + b
        if multi:
            p = _sig(s)
            loss = -np.mean(np.sum(oh * np.log(p + 1e-12) + (1 - oh) * np.log(1 - p + 1e-12), 1))
            g = (p - oh) / n
        else:
            m = s.max(1, keepdims=True)
            ls = s - m - np.log(np.exp(s - m).sum(1, keepdims=True))
            loss = -np.mean((oh * ls).sum(1))
            g = (np.exp(ls) - oh) / n
        loss += lam / n * (np.sum(b ** 2) + (a - a0) ** 2)
        return loss, np.concatenate([[float((g * Z).sum()) + 2 * lam / n * (a - a0)], g.sum(0) + 2 * lam / n * b])
    w0 = np.concatenate([[a0], np.zeros(K)]) if init is None else np.asarray(init, float)
    w = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": iters}).x
    return float(w[0]), w[1:]


def _fit_temperature(oof, multi, prior=1.0):
    """Temperature on out-of-fold scores: minimum mean NLL + prior·(log t)²/n (a pull towards 1, so a few separable
    examples do not sharpen the model without limit). A single-choice row labelled None ("other": no option fits) has a
    uniform target — sharpening on texts that fit no option is penalized, which keeps the "other" threshold meaningful."""
    n = max(1, len(oof))
    best = None
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 60)):
        if multi:
            nll = -np.mean([np.mean(y * np.log(_sig(s / t) + 1e-12) + (1 - y) * np.log(1 - _sig(s / t) + 1e-12)) for s, y in oof])
        else:
            nll = -np.mean([np.log(_softmax(s / t)[y] + 1e-12) if y is not None else np.mean(np.log(_softmax(s / t) + 1e-12))
                            for s, y in oof])
        nll += prior * np.log(t) ** 2 / n
        if best is None or nll < best[0]:
            best = (nll, float(t))
    return best[1]


class _Spec:
    """A decision's options: all of them, the ones the model scores (without "other"), their descriptions, the mode."""

    def __init__(self, task, options, descriptions=None, multi=False, other=None):
        if isinstance(options, dict):
            descriptions = {**options, **(descriptions or {})}
            options = list(options)
        options = list(options)
        if not options:
            raise ValueError("a decision needs options")
        if len(set(options)) != len(options):
            raise ValueError(f"duplicate options: {options}")
        if other is None:
            found = [o for o in options if isinstance(o, str) and o.strip().lower() in OTHER_NAMES]
            other = found[-1] if found else None
        elif other is False:
            other = None
        elif other not in options:
            raise ValueError(f"other={other!r} is not one of the options")
        self.task, self.options, self.multi, self.other = task, options, bool(multi), other
        self.real = [o for o in options if o != other]
        if not self.real:
            raise ValueError("a decision needs at least one option besides 'other'")
        d = descriptions or {}
        self.descriptions = {o: d[o] for o in options if d.get(o)}
        self.key = (task, tuple(self.real), tuple(d.get(o) or "" for o in self.real), self.multi)

    def item(self, text):
        desc = tuple(self.descriptions.get(o, "") for o in self.real)
        return Item(self.task, tuple(str(o) for o in self.real), desc if any(desc) else None, text, self.multi)

    def describe(self):
        return {"task": self.task, "options": self.options, "descriptions": self.descriptions, "multi": self.multi,
                "other": self.other}


def _text(v):
    return v.value if isinstance(v, Quote) else ("" if v is None else str(v))


# ------------------------------------------------------------------------------------------------ the model
class DecideModel:
    """A decider: text + task + options → a probability per option.

    DecideModel.load(path_or_hf_id, device=None, backend="auto") loads a solvi-decide checkpoint (a folder with config.json,
    tokenizer.json, solvi_decide.json, model.safetensors and/or onnx/model_fp16.onnx; or a Hugging Face id). `backend`:
    "torch", "onnx" or "auto" (ONNX when the file and onnxruntime are there, else torch).

    DecideModel(scorer, meta=None, model_id=...) wraps any object with `logits(items)` → one array per Item, [K] or [K, 2]
    ([:, 0] for choose-one, [:, 1] for multi-label) and optionally `fingerprint()`."""

    deterministic = True

    def __init__(self, scorer, meta=None, model_id=None, path=None, backend=None, cache_size=4096):
        self.scorer = scorer
        self.meta = dict(meta or {})
        self.model_id = model_id or getattr(scorer, "model_id", None) or type(scorer).__name__
        self.path = path
        self.backend = backend or getattr(scorer, "tag", type(scorer).__name__)
        fmt = self.meta.get("format", "")
        cal = self.meta.get("calibration", {}) if isinstance(self.meta.get("calibration"), dict) else {}
        self.temperature = float(self.meta.get("temperature", cal.get("temperature", DEFAULT_T.get(fmt, 1.0))))
        self.temperature_multi = float(self.meta.get("temperature_multi", cal.get("temperature_multi", 1.0)))
        self.other_threshold = float(self.meta.get("other_threshold", cal.get("other_threshold", 0.5)))
        self.multi_threshold = float(self.meta.get("multi_threshold", cal.get("multi_threshold", 0.5)))
        self.adaptations: dict[tuple, Adaptation] = {}
        self._cache, self._cache_size, self._lock = OrderedDict(), cache_size, threading.Lock()
        self._wfp = None

    # --- loading and identity
    @classmethod
    def load(cls, path_or_id, device=None, backend="auto", max_len=None, bs=16):
        path = str(path_or_id)
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            allow = None
            if backend == "onnx":
                allow = ["*.json", "onnx/*"]
            elif backend == "torch":
                allow = ["*.json", "*.safetensors"]
            path = snapshot_download(path, allow_patterns=allow)
        meta_file = os.path.join(path, "solvi_decide.json")
        if not os.path.isfile(meta_file):
            raise FileNotFoundError(f"{path} has no solvi_decide.json: not a solvi-decide checkpoint")
        meta = json.load(open(meta_file))
        fmt = str(meta.get("format", ""))
        if fmt and not fmt.startswith("l14b_decider"):
            raise ValueError(f"unknown decider format {fmt!r} (this solvi reads 'l14b_decider v1')")
        n = int(max_len or meta.get("max_len", 512))
        if backend == "auto":
            try:
                import onnxruntime  # noqa: F401
                backend = "onnx" if _onnx_file(path) else "torch"
            except ImportError:
                backend = "torch"
        if backend == "onnx":
            scorer = OnnxScorer(path, n, device, bs=bs)
            weights = scorer.file
        elif backend == "torch":
            scorer = TorchScorer(path, n, device, bs=bs)
            weights = os.path.join(path, "model.safetensors")
        else:
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        m = cls(scorer, meta, model_id=str(path_or_id), path=path, backend=scorer.tag)
        m._wfp = _file_fingerprint([os.path.join(path, f) for f in ("config.json", "solvi_decide.json", "tokenizer.json")]
                                   + [weights])
        return m

    def weights_fingerprint(self):
        """A hash of the checkpoint (files, or the scorer's own fingerprint), the backend and the default calibration."""
        if self._wfp is None:
            fp = getattr(self.scorer, "fingerprint", None)
            self._wfp = str(fp()) if callable(fp) else f"unversioned:{type(self.scorer).__name__}"
        from .provenance import digest
        return digest("DecideModel", self._wfp, self.backend, self.temperature, self.temperature_multi, self.other_threshold,
                      self.multi_threshold)

    def fingerprint(self):
        """The checkpoint plus every adaptation (so a replay knows the calibration a decision used)."""
        from .provenance import digest
        return digest(self.weights_fingerprint(), {repr(k): a.params() for k, a in sorted(self.adaptations.items(), key=repr)})

    def metadata(self):
        """The checkpoint's metadata (without the training history), the default calibration and every adaptation."""
        base = {k: v for k, v in self.meta.items() if k not in ("hist", "init_meta")}
        return {"model_id": self.model_id, "backend": self.backend, "weights": self.weights_fingerprint(),
                "temperature": self.temperature, "temperature_multi": self.temperature_multi,
                "other_threshold": self.other_threshold, "meta": base,
                "adaptations": [{"task": k[0], "options": list(k[1]), "descriptions": list(k[2]), "multi": k[3], **a.params()}
                                for k, a in self.adaptations.items()]}

    # --- raw logits
    def _raw(self, specs_texts):
        """[(spec, text)] → raw logits of the mode's column, [K] each (cached by text and options)."""
        out, todo = [None] * len(specs_texts), []
        with self._lock:
            for i, (sp, t) in enumerate(specs_texts):
                k = (sp.key, t)
                if k in self._cache:
                    self._cache.move_to_end(k)
                    out[i] = self._cache[k]
                else:
                    todo.append(i)
        if todo:
            items = [specs_texts[i][0].item(specs_texts[i][1]) for i in todo]
            got = self.scorer.logits(items)
            with self._lock:
                for i, lg in zip(todo, got):
                    lg = np.asarray(lg, dtype=np.float64)
                    sp = specs_texts[i][0]
                    z = lg if lg.ndim == 1 else lg[:, 1 if (sp.multi and lg.shape[1] > 1) else 0]
                    if z.shape != (len(sp.real),):
                        raise ValueError(f"the scorer returned {z.shape} logits for {len(sp.real)} options")
                    out[i] = z
                    self._cache[(sp.key, specs_texts[i][1])] = z
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return out

    def logits(self, text, task, options, descriptions=None, multi=False, other=None):
        """Raw logits of the scored options ("other" excluded): {option: logit}, or a list of them for a list of texts."""
        sp = _Spec(task, options, descriptions, multi, other)
        texts = [text] if isinstance(text, str) else list(text)
        zs = self._raw([(sp, t) for t in texts])
        res = [{o: float(v) for o, v in zip(sp.real, z)} for z in zs]
        return res[0] if isinstance(text, str) else res

    # --- calibrated scores
    def _scores(self, sp, z):
        """Raw logits → calibrated scores s (softmax / sigmoid input) with this (task, options)'s adaptation."""
        a = self.adaptations.get(sp.key)
        z = np.asarray(z, float)
        if a is not None and a.bias is not None:
            z = z - np.asarray(a.bias)
        T = self.temperature_multi if sp.multi else self.temperature
        scale = a.scale if a is not None and a.scale is not None else 1.0 / T
        shift = np.asarray(a.shift) if a is not None and a.shift is not None else 0.0
        tau = a.temperature if a is not None else 1.0
        return (scale * z + shift) / tau

    def _decision(self, sp, z):
        s = self._scores(sp, z)
        a = self.adaptations.get(sp.key)
        if sp.multi:
            p = _sig(s)
            probs = {o: float(v) for o, v in zip(sp.real, p)}
            chosen = tuple(o for o in sp.real if probs[o] >= self.multi_threshold)
            conf = float(np.min(np.maximum(p, 1 - p)))
            if sp.other is not None:
                probs[sp.other] = float(1 - p.max())
                if not chosen:
                    chosen = (sp.other,)
            order = {o: i for i, o in enumerate(sp.options)}
            return Decision(tuple(sorted(chosen, key=order.get)), {o: probs[o] for o in sp.options}, confidence=conf)
        q = _softmax(s)
        best = int(q.argmax())
        m = float(q[best])
        if sp.other is None:
            return Decision(sp.real[best], {o: float(v) for o, v in zip(sp.real, q)})
        thr = a.other_threshold if a is not None and a.other_threshold is not None else self.other_threshold
        thr = min(max(thr, 1e-6), 1 - 1e-6)
        g = thr * (1 - m) / (1 - thr)            # other's odds: other is the most probable option exactly when m < thr
        pi = g / (1 + g)
        probs = {o: float((1 - pi) * v) for o, v in zip(sp.real, q)}
        probs[sp.other] = float(pi)
        probs = {o: probs[o] for o in sp.options}
        if m < thr:
            return Decision(sp.other, probs, confidence=1 - m)
        return Decision(sp.real[best], probs, confidence=m)

    def decide(self, text, task, options, descriptions=None, multi=False, other=None):
        """→ Decision(value, probs) (a list of them for a list of texts). The value is always one of the options."""
        sp = _Spec(task, options, descriptions, multi, other)
        one = isinstance(text, (str, Quote))
        texts = [text] if one else list(text)
        out = [self._decision(sp, z) for z in self._raw([(sp, _text(t)) for t in texts])]
        return out[0] if one else out

    def score(self, text, task, options, descriptions=None, multi=False, other=None):
        """→ {option: probability} (a list of them for a list of texts): softmax over the options for a single choice, a
        sigmoid per option with multi=True; the adaptation of this (task, options) applied, "other" by its threshold."""
        d = self.decide(text, task, options, descriptions, multi, other)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    # --- adaptation
    def adaptation(self, task, options, descriptions=None, multi=False, other=None, create=False):
        sp = _Spec(task, options, descriptions, multi, other)
        if create and sp.key not in self.adaptations:
            self.adaptations[sp.key] = Adaptation()
        return self.adaptations.get(sp.key)

    def adapt(self, texts, task, options, descriptions=None, multi=False, other=None):
        """Label-bias correction without labels: the mean logit of each option over unlabelled texts of the domain (centered
        over the options) is subtracted before the softmax / sigmoid. → the Adaptation. A later fit is kept consistent."""
        sp = _Spec(task, options, descriptions, multi, other)
        texts = [_text(t) for t in texts]
        if not texts:
            raise ValueError("adapt needs unlabelled texts")
        Z = np.array(self._raw([(sp, t) for t in texts]))
        mean = Z.mean(0)
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.bias = [float(v) for v in mean - mean.mean()]
        a.n_unlabelled = len(texts)
        if a.examples:
            self._refit(sp, a)
        return a

    def fit(self, examples, task, options, descriptions=None, multi=False, other=None, lam=1.0, folds=4):
        """Few-shot adaptation "S" from labelled examples [(text, correct)]: a per-option shift and a shared scale on the
        (bias-corrected) logits by L-BFGS, a temperature on out-of-fold predictions, and — when some examples are labelled
        "other" — the "other" threshold that maximizes out-of-fold accuracy. Replaces earlier examples. → the Adaptation."""
        sp = _Spec(task, options, descriptions, multi, other)
        ex = [(_text(t), y) for t, y in examples]
        Z = self._raw([(sp, t) for t, _ in ex])
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples = [(list(map(float, z)), _label(sp, y)) for z, (_, y) in zip(Z, ex)]
        self._refit(sp, a, lam, folds)
        return a

    def teach(self, text, correct, task, options, descriptions=None, multi=False, other=None, lam=1.0):
        """One labelled example, absorbed at once: the shift / scale is refitted from the kept examples (warm start, K + 1
        parameters — about a millisecond or two); the temperature and the "other" threshold stay until the next fit.
        → the update time in ms (the model's forward pass, if the text was not scored before, is not included)."""
        sp = _Spec(task, options, descriptions, multi, other)
        z = self._raw([(sp, _text(text))])[0]
        t0 = time.perf_counter()
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples.append((list(map(float, z)), _label(sp, correct)))
        self._refit(sp, a, lam, folds=0, warm=True)
        return (time.perf_counter() - t0) * 1000

    def _refit(self, sp, a, lam=1.0, folds=4, warm=False):
        T = self.temperature_multi if sp.multi else self.temperature
        a0 = 1.0 / T
        bias = np.asarray(a.bias) if a.bias is not None else 0.0
        if sp.multi:
            rows = [(np.asarray(z) - bias, y) for z, y in a.examples]
            Z = np.array([z for z, _ in rows])
            Y = np.array([[float(o in y) for o in sp.real] for _, y in rows])
            real_idx = list(range(len(rows)))
        else:
            rows = [(np.asarray(z) - bias, y) for z, y in a.examples]
            real_idx = [i for i, (_, y) in enumerate(rows) if y != sp.other]
            Z = np.array([rows[i][0] for i in real_idx]) if real_idx else np.zeros((0, len(sp.real)))
            Y = np.array([sp.real.index(rows[i][1]) for i in real_idx], dtype=int)
        a.n_labelled = len(a.examples)
        if len(real_idx) == 0:
            return
        init = None
        if warm and a.scale is not None and a.shift is not None:
            init = np.concatenate([[a.scale], np.asarray(a.shift)])
        a.scale, b = _fit_shift(Z, Y, sp.multi, a0, lam, iters=60 if warm else 300, init=init)
        a.shift = [float(v) for v in b]
        if warm:
            return
        # temperature and "other" threshold from out-of-fold predictions
        n = len(rows)
        fold = np.arange(n) % folds if folds and n >= 2 * folds else None
        oof = []                                  # (index into rows, scores before temperature)
        if fold is not None:
            for f in range(folds):
                tr = [j for j, i in enumerate(real_idx) if fold[i] != f]
                if len(tr) < 2:
                    continue
                af, bf = _fit_shift(Z[tr], Y[tr], sp.multi, a0, lam)
                for i in range(n):
                    if fold[i] == f:
                        oof.append((i, af * rows[i][0] + bf))
        a.temperature = 1.0
        if sp.multi:
            pairs = [(s, np.array([float(o in rows[i][1]) for o in sp.real])) for i, s in oof]
            if len(pairs) >= 4:
                a.temperature = _fit_temperature(pairs, True)
            return
        pairs = [(s, None if rows[i][1] == sp.other else sp.real.index(rows[i][1])) for i, s in oof]
        if sum(y is not None for _, y in pairs) >= 4:
            a.temperature = _fit_temperature(pairs, False)
        labels = [rows[i][1] for i, _ in oof]
        n_other = sum(y == sp.other for y in labels)
        if sp.other is not None and n_other >= 3 and len(labels) - n_other >= 3:
            # the threshold with the best out-of-fold accuracy; among equals, the one closest to the model's default
            ms = [(float(_softmax(s / a.temperature).max()), sp.real[int(np.argmax(s))], rows[i][1]) for i, s in oof]
            grid = sorted({0.0, *np.round(np.linspace(0.05, 0.95, 19), 2)})
            accs = [(float(np.mean([(sp.other if m < thr else p) == y for m, p, y in ms])), float(thr)) for thr in grid]
            top = max(acc for acc, _ in accs)
            a.other_threshold = min((t for acc, t in accs if acc >= top - 1e-12), key=lambda t: abs(t - self.other_threshold))

    def reset(self, task=None, options=None, descriptions=None, multi=False, other=None):
        """Forget the adaptation of one (task, options), or every adaptation."""
        if task is None:
            self.adaptations.clear()
        else:
            self.adaptations.pop(_Spec(task, options, descriptions, multi, other).key, None)

    def save_adaptations(self, path):
        """Write every adaptation (with its kept examples' logits) to a JSON file, with the checkpoint's fingerprint."""
        data = {"weights": self.weights_fingerprint(), "model_id": self.model_id,
                "adaptations": [{"key": [k[0], list(k[1]), list(k[2]), k[3]], **asdict(a)} for k, a in self.adaptations.items()]}
        with open(path, "w") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=_json_default)

    def load_adaptations(self, path, strict=True):
        """Read adaptations written by save_adaptations. strict: refuse them if the checkpoint differs (the bias and shift
        were fitted on another model's logits)."""
        data = json.load(open(path))
        if strict and data.get("weights") != self.weights_fingerprint():
            raise ValueError(f"adaptations were fitted on checkpoint #{data.get('weights')}, this is #{self.weights_fingerprint()}")
        for d in data["adaptations"]:
            k = d.pop("key")
            ex = [(list(z), tuple(y) if isinstance(y, list) else y) for z, y in d.pop("examples", [])]
            self.adaptations[(k[0], tuple(k[1]), tuple(k[2]), bool(k[3]))] = Adaptation(**d, examples=ex)
        return self

    # --- catalog part
    def decision(self, name, task, text_fact="doc", options=(), descriptions=None, multi=False, other=None):
        """A catalog part: text_fact (a fact name, or a list of them joined by new lines) → Decision(value, probs).
        Register with `cat.fn(part)` (a fact other parts read; `cat.fn(min_confidence=...)(part)` rejects unsure ones) or
        make it a question's answer with `part.question(cat)` / `cat.rule("q")(part)`. The value is one of the options by
        construction; the options are the part's closed set; provenance `decided`; the trace records the model, whose
        fingerprint covers the checkpoint and this part's adaptation."""
        return DecisionPart(self, name, task, text_fact, options, descriptions, multi, other)


def _label(sp, y):
    if sp.multi:
        vals = [y] if isinstance(y, str) else list(y or [])
        bad = [v for v in vals if v not in sp.options]
        if bad:
            raise ValueError(f"{bad} not among {sp.options}")
        return tuple(v for v in sp.real if v in vals)
    if y not in sp.options:
        raise ValueError(f"{y!r} is not one of {sp.options}")
    return y


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


class DecisionPart:
    """A decider bound to one decision (name, task, options, the fact it reads). Callable as a catalog function; it is also
    the model recorded in the trace: `fingerprint()` covers the checkpoint and this decision's adaptation only, so teaching
    one decision does not mark the others as changed."""

    deterministic = True

    def __init__(self, model, name, task, text_fact, options, descriptions=None, multi=False, other=None):
        self.model = model
        self.spec = _Spec(task, options, descriptions, multi, other)
        self.facts = [text_fact] if isinstance(text_fact, str) else list(text_fact)
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = task
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts])
        self.__solvi_model__ = self
        self.__solvi_provenance__ = "decided"
        self.__solvi_options__ = list(self.spec.options)
        self.__solvi_decision__ = self

    # identity recorded in the trace
    @property
    def model_id(self):
        return self.model.model_id

    @property
    def available(self):
        return getattr(self.model, "available", True)

    @property
    def options(self):
        return list(self.spec.options)

    @property
    def multi(self):
        return self.spec.multi

    @property
    def task(self):
        return self.spec.task

    @property
    def adaptation(self):
        return self.model.adaptations.get(self.spec.key)

    def fingerprint(self):
        from .provenance import digest
        a = self.adaptation
        return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None)

    def text_of(self, facts):
        """The text this part reads, from a dict of facts."""
        return "\n".join(_text(facts[f]) for f in self.facts)

    def __call__(self, *args, **kw):
        vals = dict(zip(self.facts, args))
        vals.update(kw)
        return self.decide(self.text_of(vals))

    def __repr__(self):
        return f"DecisionPart({self.__name__!r}, options={self.spec.options}, model={self.model.model_id!r})"

    # the model's methods for this decision
    def decide(self, text):
        texts = [text] if isinstance(text, (str, Quote)) else list(text)
        out = [self.model._decision(self.spec, z) for z in self.model._raw([(self.spec, _text(t)) for t in texts])]
        return out[0] if isinstance(text, (str, Quote)) else out

    def score(self, text):
        d = self.decide(text)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def _kw(self):
        return dict(task=self.spec.task, options=self.spec.options, descriptions=self.spec.descriptions,
                    multi=self.spec.multi, other=self.spec.other or False)

    def adapt(self, texts):
        """Label-bias correction from unlabelled texts of the domain (see DecideModel.adapt)."""
        return self.model.adapt(texts, **self._kw())

    def fit(self, examples, lam=1.0, folds=4):
        """Few-shot "S" from [(text, correct)] (see DecideModel.fit)."""
        return self.model.fit(examples, lam=lam, folds=folds, **self._kw())

    def teach(self, text, correct):
        """One correction, absorbed at once (see DecideModel.teach). → ms."""
        return self.model.teach(text, correct, **self._kw())

    def reset(self):
        self.model.reset(**self._kw())

    def question(self, cat, name=None, text=None, min_confidence=None, checkpoints=None):
        """Make this decision the answer of a question: registers it as the question's rule (`cat.rule(name)(self)`) and
        returns the Question (choice, or multi for a multi-label decision, with the option descriptions). System.teach on
        that question teaches this decision."""
        from .core import Answer, Question
        name = name or self.__name__
        cat.rule(name)(self)
        opts = {o: self.spec.descriptions.get(o, "") for o in self.spec.options} if self.spec.descriptions else self.spec.options
        at = Answer.multi(opts) if self.spec.multi else Answer.choice(opts)
        return Question(name, text or self.spec.task, at, checkpoints=list(checkpoints or []), min_confidence=min_confidence)


def decision_of(catalog, question):
    """The DecisionPart behind a question's answer: its rule is a decision, or a rule that only passes a decided fact on
    (e.g. `def team(route): return route`). → the part or None."""
    rule = catalog.rules.get(question)
    if rule is None or rule.func is None:
        return None
    d = getattr(rule.func, "__solvi_decision__", None)
    if d is None and len(rule.inputs) == 1:
        part = catalog.parts.get(rule.inputs[0])
        if part is not None and part.alternatives is None:
            d = getattr(part.func, "__solvi_decision__", None)
    return d
