"""DecideModel: a decider checkpoint (or any scorer) — load, score, decide, adapt, and make decision parts. (Part of
solvi.decide, which re-exports every name.)"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict

import numpy as np

from .. import _deprecate
from ..core import Decision, Quote, Unknown
from ..provenance import ESCALATED
from ..provenance import digest
from .kinds import DEFAULT_T, FORMAT, LEGACY_FORMAT, NULL_SOURCE, TYPED2_FORMAT, TYPED_FORMAT, WIRE, _kind, _spec_names, _unused_options
from .state import _single, _text
from .wire import Logits, Pass, _typed_span, decode_pointer, pointer_evidence, prompt
from .capabilities import _max_len_long, capabilities
from .backends import BlockUnsupported, LongInputWarning, OnnxScorer, TorchScorer, _Encoder, _file_fingerprint, _onnx_file, need
from .adapt import Adaptation, _Spec, _basis, _fit_shift, _fit_temperature, _from_type, _given, _is_type, _sig, _softmax, _usable, lora_key
from .gate import Facts, act_features
from .part import DecisionPart


class DecideModel:
    """A decider: an input (a text or a state) + a question (task, options, kind) → a probability per option.

    DecideModel.load(path_or_hf_id, device=None, backend="auto") loads a solvi-decide checkpoint (a folder with config.json,
    tokenizer.json, solvi_decide.json, model.safetensors and/or onnx/model_fp16.onnx; or a Hugging Face id). `backend`:
    "torch", "onnx" or "auto" (ONNX when the file and onnxruntime are there, else torch). solvi_decide.json declares what the
    checkpoint can do (modes, act head, several questions per pass, temperatures, thresholds: docs/decide_format.md).

    DecideModel(scorer, meta=None, model_id=...) wraps any object with `logits(items)` → per Item an array [K] or [K, C] (a
    column per mode: [:, 0] choose-one, [:, 1] multi-label, more as `meta["columns"]` says) or {"logits": array, "act":
    logit}; optionally `logits_pass(passes)` → per Pass a list of those (one per question) and `fingerprint()`."""

    long_len = property(lambda self: (_deprecate.renamed("DecideModel.long_len", "max_len_long"), self.max_len_long)[1],
                        doc="Deprecated (removed in 0.9): max_len_long.")

    deterministic = True

    def __init__(self, scorer, meta=None, model_id=None, path=None, backend=None, cache_size=4096, multi_question=None,
                 act=None, max_len_long=None):
        self.scorer = scorer
        self.meta = dict(meta or {})
        self.model_id = model_id or getattr(scorer, "model_id", None) or type(scorer).__name__
        self.path = path
        self.backend = backend or getattr(scorer, "tag", type(scorer).__name__)
        self.caps = capabilities(self.meta, multi_question, act)
        if max_len_long is not None:
            max_len_long = _max_len_long(max_len_long, getattr(getattr(scorer, "enc", None), "max_len", 0)
                                         or self.meta.get("max_len", 512))
        self._overrides = {k: v for k, v in (("multi_question", multi_question), ("act", act),
                                             ("max_len_long", max_len_long)) if v is not None}
        self._warned = set()                # the long-input warnings already given (once per model)
        fmt = self.meta.get("format", "")
        cal = self.meta.get("calibration", {}) if isinstance(self.meta.get("calibration"), dict) else {}
        T = self.meta.get("temperature", cal.get("temperature", DEFAULT_T.get(fmt, 1.0)))
        td = T if isinstance(T, dict) else {}
        self.temperature = float(td.get("single", td.get("choice", 1.0)) if td else T)
        self.temperature_multi = float(td.get("multi", self.meta.get("temperature_multi", cal.get("temperature_multi", 1.0))))
        a = self.caps["act"] or {}
        self.temperatures = {"single": self.temperature, "multi": self.temperature_multi,
                             "score": float(td.get("score", self.temperature)), "noul": float(td.get("noul", self.temperature)),
                             "act": float(a.get("temperature", 1.0))}
        if self.caps["version"] >= 3:               # rank, number, span (older checkpoints: the dict above, as before)
            self.temperatures.update(rank=float(td.get("rank", self.temperature)),
                                     number=float(td.get("number", td.get("score", self.temperature))),
                                     span=float(td.get("span", 1.0)))
        th = self.meta.get("thresholds") if isinstance(self.meta.get("thresholds"), dict) else {}
        self.other_threshold = float(th.get("other", self.meta.get("other_threshold", cal.get("other_threshold", 0.5))))
        self.multi_threshold = float(th.get("multi", self.meta.get("multi_threshold", cal.get("multi_threshold", 0.5))))
        self.act_threshold = float(a.get("threshold", th.get("act", 0.5)))
        self.act_thresholds = {float(k): float(v) for k, v in (a.get("threshold_for_error") or {}).items()}
        self.escalate_below = th.get("escalate_below")          # a default for parts that set none (None: no default)
        self.adaptations: dict[tuple, Adaptation] = {}
        self.loras: dict = {}                # lora_key → solvi.lora.LoraAdapter (experimental: part.adapt_lora)
        self._cache, self._cache_size, self._lock = OrderedDict(), cache_size, threading.Lock()
        self._pass_lock = threading.Lock()
        self._block_failed = False
        self._wfp = None
        self.passes = 0                     # sequences the network encoded (a shared pass counts once)

    # --- loading and identity
    @classmethod
    def load(cls, path_or_id, device=None, backend="auto", max_len=None, bs=16, multi_question=None, act=None,
             max_len_long=None):
        """path_or_id: a checkpoint folder (`~` is expanded) or a Hugging Face id (downloaded once, then read from the
        cache). multi_question / act: override what solvi_decide.json declares (for experiments, e.g. testing a checkpoint
        in multi-question passes); both are part of the fingerprint. max_len: the tokens of one ordinary pass (default:
        the checkpoint's `max_len`); it also sets long="retrieve"'s budget. max_len_long: the length long="full" reads
        whole (default: the checkpoint's `max_len_long`; a checkpoint that declares none refuses long="full" unless it is
        given here — with a warning: it was not trained on long inputs). Part of the fingerprint."""
        if backend not in ("auto", "onnx", "torch"):     # before anything is downloaded
            raise ValueError(f'backend must be "torch", "onnx" or "auto", not {backend!r}')
        path = os.path.expanduser(str(path_or_id))
        if not os.path.isdir(path):
            if os.path.isabs(path) or path.startswith((".", "~")):    # a path, not a Hugging Face id: do not ask the hub
                raise FileNotFoundError(f"{path}: no such folder (a checkpoint is a folder with solvi_decide.json, or a "
                                        "Hugging Face id like solvi-ai/solvi-base)")
            snapshot_download = need("huggingface_hub", "onnx", "loading a decider by its Hugging Face id").snapshot_download
            allow = None
            if backend == "onnx":
                allow = ["*.json", "onnx/*"]
            elif backend == "torch":
                allow = ["*.json", "*.safetensors"]
            path = snapshot_download(path, allow_patterns=allow)
        meta_file = os.path.join(path, "solvi_decide.json")
        if not os.path.isfile(meta_file):
            raise FileNotFoundError(f"{path} has no solvi_decide.json: not a solvi-decide checkpoint")
        with open(meta_file) as fh:
            meta = json.load(fh)
        fmt = str(meta.get("format", ""))
        if fmt and not (fmt.startswith("l14b_decider") or
                        re.match(r"^(l14f typed v1|l14g typed v2|solvi_decide v[23])(\.\d+)*$", fmt)):
            raise ValueError(f"unknown decider format {fmt!r} (this solvi reads {LEGACY_FORMAT!r}, {TYPED_FORMAT!r}, "
                             f"{FORMAT!r} (with subformat {TYPED2_FORMAT!r}) and 'solvi_decide v3')")
        caps = capabilities(meta, multi_question, act)
        n = int(max_len or meta.get("max_len", 512))
        if backend == "auto":
            try:
                import onnxruntime  # noqa: F401
                backend = "onnx" if _onnx_file(path) else "torch"
            except ImportError:
                backend = "torch"
            if backend == "torch":
                try:
                    import torch  # noqa: F401
                except ImportError:
                    raise ImportError('DecideModel.load needs a runtime: pip install "solvi[onnx]" (onnxruntime, for a '
                                      'checkpoint with an onnx/ folder) or "solvi[model]" (torch)') from None
        if backend == "onnx":
            scorer = OnnxScorer(path, n, device, bs=bs, caps=caps)
            weights = scorer.file
        elif backend == "torch":
            scorer = TorchScorer(path, n, device, bs=bs, caps=caps)
            weights = os.path.join(path, "model.safetensors")
        else:
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        m = cls(scorer, meta, model_id=str(path_or_id), path=path, backend=scorer.tag, multi_question=multi_question, act=act,
                max_len_long=max_len_long)
        cfg_file = os.path.join(path, "config.json")
        pos = None
        if os.path.isfile(cfg_file):
            with open(cfg_file) as fh:
                pos = json.load(fh).get("max_position_embeddings")
        if m.max_len_long is not None and pos and m.max_len_long > int(pos):
            raise ValueError(f"max_len_long {m.max_len_long} is beyond the encoder's max_position_embeddings ({pos}) in "
                             f"{cfg_file}")
        m._wfp = _file_fingerprint([os.path.join(path, f) for f in ("config.json", "solvi_decide.json", "tokenizer.json")]
                                   + [weights])
        return m

    @property
    def batchable(self):
        """Can several questions about one input share a forward pass (declared by the checkpoint, and the scorer has
        logits_pass)?"""
        return self.caps["max_questions"] > 1 and callable(getattr(self.scorer, "logits_pass", None))

    @property
    def has_act(self):
        """Does the checkpoint give an act / escalate signal per question?"""
        return self.caps["act"] is not None

    def act_threshold_for(self, error):
        """The act threshold the checkpoint ships for a target error rate (its `act.threshold_for_error` table: the entry
        with the largest error ≤ the target). ValueError when it ships none — use calibrate_for on your examples."""
        ok = [e for e in self.act_thresholds if e <= error + 1e-12]
        if not ok:
            raise ValueError(f"the checkpoint has no act threshold for error ≤ {error} (it has {sorted(self.act_thresholds)}); "
                             "use part.calibrate_for(examples, max_error=...)")
        return self.act_thresholds[max(ok)]

    def wire(self, kind):
        """How a question kind is asked: natively when the checkpoint was trained on it, else as a single choice (score:
        the levels; noul: the options "yes" / "no"; rank: the options, ordered by probability; number: the bins, as a score
        when the checkpoint has scores). A span is native only."""
        w = WIRE[kind]
        if w in ("single", "multi") or w in self.caps["modes"]:
            return w
        return "score" if w == "number" and "score" in self.caps["modes"] else "single"

    @property
    def has_not_stated(self):
        """Does the checkpoint give a "not stated" output (an l14g checkpoint's `unknown`)?"""
        return self.caps.get("unknown") is not None

    @property
    def has_unknown(self):
        """Deprecated (removed in 0.9): has_not_stated."""
        _deprecate.renamed("DecideModel.has_unknown", "has_not_stated")
        return self.has_not_stated

    @property
    def has_pointer(self):
        """Can the checkpoint point at a piece of its input (span answers, evidence quotes)?"""
        return self.caps.get("pointer") is not None and "span" in self.caps["modes"]

    def _T(self, sp):
        w = self.wire(sp.kind)
        return self.temperature_multi if w == "multi" else (self.temperature if w == "single" else self.temperatures[w])

    def weights_fingerprint(self):
        """A hash of the checkpoint (files, or the scorer's own fingerprint), the backend, the default calibration and (for
        checkpoints in the v2 format, or with overrides) the declared capabilities."""
        if self._wfp is None:
            fp = getattr(self.scorer, "fingerprint", None)
            self._wfp = str(fp()) if callable(fp) else f"unversioned:{type(self.scorer).__name__}"
        parts = [self._wfp, self.backend, self.temperature, self.temperature_multi, self.other_threshold, self.multi_threshold]
        if self.caps["version"] >= 2 or self._overrides:        # the 'l14b_decider v1' format hashes exactly as before
            parts.append({"caps": {k: v for k, v in self.caps.items()}, "temperatures": self.temperatures,
                          "act_threshold": self.act_threshold, "escalate_below": self.escalate_below,
                          "overrides": self._overrides})
        return digest("DecideModel", *parts)

    def fingerprint(self):
        """The checkpoint plus every adaptation (so a replay knows the calibration a decision used)."""
        from ..provenance import digest
        ads = {repr(k): a.params() for k, a in sorted(self.adaptations.items(), key=repr)}
        if self.loras:                       # a model without adapters hashes as before
            return digest(self.weights_fingerprint(), ads, {repr(k): ad.hash for k, ad in sorted(self.loras.items(), key=repr)})
        return digest(self.weights_fingerprint(), ads)

    def metadata(self):
        """The checkpoint's metadata (without the training history), capabilities, the default calibration and every
        adaptation."""
        base = {k: v for k, v in self.meta.items() if k not in ("hist", "init_meta")}
        return {"model_id": self.model_id, "backend": self.backend, "weights": self.weights_fingerprint(),
                "capabilities": dict(self.caps), "temperature": self.temperature, "temperature_multi": self.temperature_multi,
                "temperatures": dict(self.temperatures), "other_threshold": self.other_threshold,
                "act_threshold": self.act_threshold, "meta": base,
                "adaptations": [{"task": k[0], "options": list(k[1]), "descriptions": list(k[2]), "multi": k[3], **a.params()}
                                for k, a in self.adaptations.items()],
                **({"loras": [{"task": k[0], "options": [o for o, _ in k[1]], **ad.describe()} for k, ad in self.loras.items()]}
                   if self.loras else {})}

    @property
    def state_format(self):
        """The serialization of states for this checkpoint: the first of its declared ones that solvi writes ("paths" by
        default; a text-only checkpoint reads the "paths" lines as text)."""
        return self.caps["state_format"]

    @property
    def max_len(self):
        """The tokens this checkpoint reads in one sequence (question and input): the encoder's, else the checkpoint's
        `max_len`, else 512."""
        enc = getattr(self.scorer, "enc", None)
        return int(getattr(enc, "max_len", 0) or self.meta.get("max_len") or getattr(self.scorer, "max_len", 0) or 512)

    @property
    def max_len_long(self):
        """The tokens long="full" reads whole (question and input): the load(max_len_long=...) override, else the
        checkpoint's `max_len_long`, else None (the checkpoint was not trained on long inputs)."""
        n = self._overrides.get("max_len_long", self.caps.get("max_len_long"))
        return None if n is None else int(n)

    @property
    def long_declared(self):
        """Does the checkpoint itself declare a long-input length (`max_len_long` in solvi_decide.json)?"""
        return self.caps.get("max_len_long") is not None

    def _warn_once(self, key, message, category=LongInputWarning):
        if key not in self._warned:
            self._warned.add(key)
            import warnings
            warnings.warn(message, category, stacklevel=4)

    def on_cpu(self):
        """Does the network run on a CPU (a torch scorer on "cpu", an ONNX session without CUDA)? None when unknown (a
        stand-in or remote scorer)."""
        sc = self.scorer
        dev = getattr(sc, "device", None)
        if isinstance(sc, TorchScorer) and dev is not None:
            return str(dev).startswith("cpu")
        sess = getattr(sc, "sess", None)
        if isinstance(sc, OnnxScorer) and sess is not None:
            return "CUDAExecutionProvider" not in sess.get_providers()
        return None

    def count_tokens(self, text):
        """Tokens of a text for this checkpoint: its tokenizer when it has one, else solvi.longdoc.approx_tokens."""
        enc = getattr(self.scorer, "enc", None)
        tok = getattr(enc, "raw", None) or getattr(enc, "tok", None)      # raw: no truncation at max_len
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False).ids)
        from ..longdoc import approx_tokens
        return approx_tokens(text)

    def text(self, v):
        """An input (a text, a Quote, a scalar or a state) → the text this checkpoint reads."""
        return _text(v, self.caps["state_format"])

    def truncation(self, specs, text, read_len=0):
        """What one ordinary pass leaves unread of an input → None when it reads all of it, else {"input_tokens",
        "read_tokens", "question_tokens", "max_len"}. `specs`: the question, or the questions of a shared pass. None for
        a scorer that reads the text as it is (an LLM, a hosted decision model) and in the block layout when the input
        fits its budget."""
        enc = getattr(self.scorer, "enc", None)
        if not isinstance(enc, _Encoder) or not isinstance(text, str):
            return None
        specs = [specs] if isinstance(specs, _Spec) else list(specs)
        items = tuple(self._item(sp, text, read_len) for sp in specs)
        mq = getattr(self.scorer, "mq", None)
        if (not read_len and self.block and not self._block_failed and not any(sp.pointer for sp in specs)
                and mq is not None and mq.get("layout") == "block"):     # the block layout budgets the input itself
            q = sum(len(enc.raw.encode(prompt(it.task, it.options, it.descriptions, mode=it.mode, markers=enc.markers),
                                       add_special_tokens=False).ids) + 1 for it in items) + 2
            left, total = int(mq["max_len"]) - q, self.count_tokens(text or " ")
            return None if total <= left else {"input_tokens": total, "read_tokens": max(0, left), "question_tokens": q,
                                               "max_len": int(mq["max_len"])}
        return enc.read(items, text)

    # --- raw logits
    def _pick(self, sp, o, text=None):
        """One scorer output → (the question's logits [K], the act logit or None). From an l14g checkpoint the logits also
        carry the "not stated" logit (`.unknown`) and the decoded pointer (`.pointer`)."""
        act = unk = ptr = why = info = None
        transient = False
        if isinstance(o, dict):
            why, info, transient = o.get("escalate"), o.get("info"), bool(o.get("transient"))
            o, act, unk, ptr = o.get("logits"), o.get("act"), o.get("unknown"), o.get("pointer")
            if o is None and sp.kind == "span":
                o = np.zeros(0)
        lg = np.asarray(o, dtype=np.float64)
        if lg.ndim == 2:
            col = self.caps["columns"].get(self.wire(sp.kind), 0)
            lg = lg[:, col if col < lg.shape[1] else 0]
        if sp.kind == "noul" and lg.shape == (1,):     # a single yes log-odds
            lg = np.array([lg[0], 0.0])
        if lg.shape != (len(sp.real),):
            raise ValueError(f"the scorer returned {lg.shape} logits for {len(sp.real)} options")
        if unk is not None or (ptr is not None and sp.pointer) or why or info:
            lg = lg.view(Logits)
            lg.unknown = None if unk is None else float(unk)
            if ptr is not None and sp.pointer:
                if "spans" in ptr:                    # already decoded by the scorer (quotes it located itself)
                    lg.pointer = {"null": float(ptr.get("null", 0.0)), "spans": list(ptr["spans"])}
                else:
                    pc = self.caps.get("pointer") or {}
                    lg.pointer = decode_pointer(ptr, text or "", pc.get("max_span_tokens", 40),
                                                temperature=self.temperatures.get("span", 1.0))
            lg.escalate, lg.info, lg.transient = (str(why) if why else None), info, transient
        return lg, (None if act is None else float(act))

    def _item(self, sp, text, read_len=0):
        it = sp.item(text, self.wire(sp.kind), self.caps["noul_labels"])
        return dataclasses.replace(it, max_len=int(read_len)) if read_len else it

    @property
    def block(self):
        """Does this model score in the block layout (declared by the checkpoint, and the scorer has logits_pass)? Then
        every question — alone or with others — is scored in that layout: its answer does not depend on the other
        questions of its pass, and adapt / fit / teach see the same logits as the runtime."""
        mq = self.caps["multi_question"]
        return mq is not None and mq["layout"] == "block" and callable(getattr(self.scorer, "logits_pass", None))

    def _lora_name(self, sp):
        ad = self.loras.get(lora_key(sp)) if self.loras else None
        return None if ad is None else ad.name

    def _using(self, name):
        """The scorer with this LoRA adapter active (None: none) while scoring — a no-op for a model without adapters."""
        if not self.loras:
            import contextlib
            return contextlib.nullcontext()
        from ..lora import using
        return using(self.scorer, name)

    def _forget_cached(self, key):
        """Drop the cached logits of one question (every option order): its adapter changed."""
        with self._lock:
            for k in [k for k in self._cache if lora_key(k[0]) == key]:
                del self._cache[k]

    def _score(self, pairs, read_len=0):
        """[(spec, text)] → scorer outputs, in order; the questions with a LoRA adapter (solvi.lora) are scored with it
        active, apart from the others. read_len: see _score_plain."""
        if not self.loras:
            return self._score_plain(pairs, read_len)
        groups = OrderedDict()
        for i, (sp, _) in enumerate(pairs):
            groups.setdefault(self._lora_name(sp), []).append(i)
        out, used = [None] * len(pairs), False
        for name, ix in groups.items():
            with self._using(name):
                got, u = self._score_plain([pairs[i] for i in ix], read_len)
            used = used or u
            for i, o in zip(ix, got):
                out[i] = o
        return out, used

    def _score_plain(self, pairs, read_len=0):
        """[(spec, text)] → scorer outputs, in order: block passes (one per text, at most max_questions each) for a block
        model — questions that do not fit together go one per pass, and an export without the block layout falls back to
        one question per sequence — else one sequence per question. Questions that need the pointer (span answers,
        evidence) always go one per sequence (the full layout: in the block layout the input does not see the question).
        read_len: a whole long text (long="full") — one sequence per question of up to read_len tokens."""
        if read_len:
            items = [self._item(sp, t, read_len) for sp, t in pairs]
            out = self.scorer.logits(items)
            self.passes += len(items)
            return out, False
        if self.block and not self._block_failed and any(sp.pointer for sp, _ in pairs):
            ptr = [i for i, (sp, _) in enumerate(pairs) if sp.pointer]
            rest = [i for i, (sp, _) in enumerate(pairs) if not sp.pointer]
            out, used = [None] * len(pairs), False
            if rest:
                got, used = self._score_plain([pairs[i] for i in rest])
                for i, o in zip(rest, got):
                    out[i] = o
            got = self.scorer.logits([self._item(*pairs[i]) for i in ptr])
            self.passes += len(ptr)
            for i, o in zip(ptr, got):
                out[i] = o
            return out, used
        if self.block and not self._block_failed:
            by_text = OrderedDict()
            for i, (sp, t) in enumerate(pairs):
                by_text.setdefault(t, []).append(i)
            n = self.caps["max_questions"]
            for size in (n, 1):
                chunks = [ix[k:k + size] for ix in by_text.values() for k in range(0, len(ix), size)]
                try:
                    got = self.scorer.logits_pass([Pass(pairs[c[0]][1], tuple(self._item(*pairs[i]) for i in c))
                                                   for c in chunks])
                except BlockUnsupported as e:        # said once: several questions were asked for and are not shared
                    self._block_failed = True
                    self._warn_once("block", f"several questions per pass are not available: {e} "
                                             f"({getattr(self.scorer, 'tag', type(self.scorer).__name__)}); each question "
                                             "is scored in a pass of its own", UserWarning)
                    break
                except ValueError:                    # too long together: smaller passes, then one per sequence
                    continue
                self.passes += len(chunks)
                out = [None] * len(pairs)
                for c, res in zip(chunks, got):
                    for i, o in zip(c, res):
                        out[i] = o
                return out, True
        items = [self._item(*pr) for pr in pairs]
        out = self.scorer.logits(items)
        self.passes += len(items)
        return out, False

    def _raw_full(self, specs_texts, info=None, read_len=0):
        """[(spec, text)] → [(raw logits of the mode's column [K], act logit or None)] (cached by text and question).
        info: a dict that receives "block": whether what was scored now went through block passes. read_len: read each
        text whole up to that many tokens (long="full"; cached apart)."""
        out, todo = [None] * len(specs_texts), []

        def key(sp, t):                               # "not stated" allowed is another question to the scorer: a
            k = (sp.key, t, "unknown") if sp.unknown else (sp.key, t)    # Maybe[...] part and a plain one do not share a reply
            return k + (("read", int(read_len)),) if read_len else k
        with self._lock:
            for i, (sp, t) in enumerate(specs_texts):
                k = key(sp, t)
                if k in self._cache:
                    self._cache.move_to_end(k)
                    out[i] = self._cache[k]
                else:
                    todo.append(i)
        if todo:
            got, used_block = self._score([specs_texts[i] for i in todo], read_len)
            if info is not None:
                info["block"] = used_block
            with self._lock:
                for i, o in zip(todo, got):
                    sp = specs_texts[i][0]
                    out[i] = self._pick(sp, o, specs_texts[i][1])
                    if not getattr(out[i][0], "transient", False):     # a server that did not answer: ask again
                        self._cache[key(sp, specs_texts[i][1])] = out[i]
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return out

    def _raw(self, specs_texts):
        return [z for z, _ in self._raw_full(specs_texts)]

    def _raw_pass(self, specs, text):
        """Several questions about one text in one forward pass → [(logits, act, shared)]. Block layout: the questions not
        cached yet go in one pass (an answer does not depend on its neighbours, so the cache is per question). Other
        layouts: the pass is cached as a whole; one question per pass (shared False) when the scorer has no logits_pass or
        the questions do not fit together."""
        if self.block:
            info = {"block": not self._block_failed}
            with self._pass_lock:                     # parallel steps of one pass: the first runs it, the others read it
                res = self._raw_full([(sp, text) for sp in specs], info)
            return [(z, a, info["block"]) for z, a in res]
        sig = tuple(sp.key for sp in specs)
        keys = [(sp.key, text, sig) for sp in specs]

        def cached():
            with self._lock:
                if all(k in self._cache for k in keys):
                    for k in keys:
                        self._cache.move_to_end(k)
                    return [self._cache[k] for k in keys]
            return None
        got = cached()
        if got is not None:
            return got
        with self._pass_lock:
            got = cached()
            if got is not None:
                return got
            outs = None
            if self.batchable and not any(self._lora_name(sp) for sp in specs):   # an adapted question: alone
                items = tuple(self._item(sp, text) for sp in specs)
                try:
                    with self._using(None):
                        outs = self.scorer.logits_pass([Pass(text, items)])[0]
                    self.passes += 1
                except ValueError:                    # too long together: one question per pass
                    outs = None
            if outs is None:
                res = [(z, a, False) for z, a in self._raw_full([(sp, text) for sp in specs])]
            else:
                res = [(*self._pick(sp, o, text), True) for sp, o in zip(specs, outs)]
            with self._lock:
                for k, v in zip(keys, res):
                    self._cache[k] = v
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return res

    def logits(self, text, task, options, descriptions=None, multi=False, other=None, kind=None):
        """Raw logits of the scored options ("other" excluded): {option: logit}, or a list of them for a list of inputs."""
        sp = _Spec(task, options, descriptions, multi, other, kind)
        one = _single(text)
        zs = self._raw([(sp, self.text(t)) for t in ([text] if one else list(text))])
        res = [{o: float(v) for o, v in zip(sp.real, z)} for z in zs]
        return res[0] if one else res

    # --- calibrated scores
    def _scores(self, sp, z):
        """Raw logits → calibrated scores s (softmax / sigmoid input) with this question's adaptation."""
        a = self.adaptations.get(sp.key)
        z = np.asarray(z, float)
        if a is not None and a.bias is not None:
            z = z - np.asarray(a.bias)
        scale = a.scale if a is not None and a.scale is not None else 1.0 / self._T(sp)
        shift = np.asarray(a.shift) if a is not None and a.shift is not None else 0.0
        tau = a.temperature if a is not None else 1.0
        return (scale * z + shift) / tau

    def _decision(self, sp, z):
        """Calibrated logits → Decision. Questions of solvi 0.5 without "not stated" decide exactly as before; span, rank and
        number questions and those allowing "not stated" (from an l14g checkpoint's outputs) go through _decision_v3; a
        question asking for evidence gets the pointer's quotes."""
        u, ptr = getattr(z, "unknown", None), getattr(z, "pointer", None)
        if sp.kind == "span":
            d = self._span_decision(sp, ptr, getattr(z, "escalate", None))
        else:
            if (sp.unknown and u is not None) or sp.kind in ("rank", "number"):
                d = self._decision_v3(sp, z, u if sp.unknown else None)
            else:
                d = self._decision_v1(sp, z)
            if sp.evidence and ptr is not None and d.value is not Unknown:
                ev = (self.caps.get("pointer") or {}).get("evidence") or {}
                d.evidence = pointer_evidence(ptr, ev.get("threshold", 0.15), min(sp.evidence, ev.get("max_spans", 3)))
        info, why = getattr(z, "info", None), getattr(z, "escalate", None)
        if info:
            d.extra.update(info)
        if why and d.escalate is None:              # the scorer could not give a usable output: never a guess
            d.escalate = f"{ESCALATED}: {why}"
            if sp.kind != "span":                   # nor a value: uniform logits would read as the first option
                d.value, d.probs, d.confidence = None, {}, 0.0
        return d

    def _decision_v3(self, sp, z, u):
        """rank / number, and any kind with "not stated": for single, score, noul, rank, number the "not stated" logit u
        competes with the options in one softmax (the l14g contract), for multi it is a sigmoid. choice / noul: the most
        probable of the options and "not stated"; score / rank / number: "not stated" when p(not stated) ≥ the threshold
        (0.5), else the median level / the order by probability / the median bin, from the probabilities given it is
        stated. Confidence: p(answer) — for a ranking its Plackett–Luce probability, for a number the mass of its interval
        — times p(stated)."""
        thr = float((self.caps.get("unknown") or {}).get("threshold", 0.5))
        if sp.multi:
            d = self._decision_v1(sp, z)
            pu = float(_sig(u))
            if pu >= thr:
                return Decision(Unknown, {**d.probs, Unknown: pu}, confidence=pu)
            d.probs[Unknown] = pu
            d.confidence = min(d.conf, 1 - pu)
            return d
        s = self._scores(sp, z)
        K = len(sp.real)
        q_all = _softmax(np.append(s, u / self._T(sp))) if u is not None else _softmax(s)
        pn = float(q_all[K]) if u is not None else 0.0
        q = q_all[:K]
        cond = q / max(float(q.sum()), 1e-300)
        probs = {o: float(v) for o, v in zip(sp.real, q)}
        if u is not None:
            probs[Unknown] = pn
        if sp.kind in ("choice", "noul"):
            best = int(np.argmax(q_all))
            if best == K:
                return Decision(Unknown, probs, confidence=pn)
            return Decision(sp.out(sp.real[best]), probs, confidence=float(q[best]))
        if u is not None and pn >= thr:
            return Decision(Unknown, probs, confidence=pn)
        if sp.kind == "rank":
            from ..primitives import plackett_luce
            order = tuple(sp.real[i] for i in sorted(range(K), key=lambda i: (-cond[i], i))[: sp.k or K])
            conf = plackett_luce(order, dict(zip(sp.real, cond))) * (1 - pn)
            return Decision(order, probs, confidence=conf, extra={"k": len(order)})
        if sp.kind == "number":
            from types import SimpleNamespace

            from ..primitives import estimate_of
            value, interval, mass = estimate_of(SimpleNamespace(bins=sp.edges, coverage=sp.coverage), list(cond))
            return Decision(value, probs, confidence=mass * (1 - pn), extra={"interval": interval, "coverage": sp.coverage})
        levels = sp.real                               # score
        acc, med = 0.0, levels[-1]
        for o, p in zip(levels, cond):
            acc += p
            if acc >= 0.5 - 1e-12:
                med = o
                break
        rank = float(np.dot(np.arange(K), cond))
        numeric = all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in levels)
        expected = float(np.dot(np.asarray(levels, float), cond)) if numeric else rank
        value = {"median": med, "mode": levels[int(cond.argmax())], "expected": levels[int(round(rank))]}[sp.score_value]
        return Decision(value, probs, confidence=float(q[levels.index(value)]), extra={"expected": expected, "median": med})

    def _span_decision(self, sp, ptr, why=None):
        """The pointer's best span (its text literally from the input), or "not stated" when the null span is at least as
        probable and the question allows it; when it does not (a Span without Maybe), such a span escalates. Confidence:
        p(span) among the null span and every span (renormalized without the null span when "not stated" is not
        allowed). why: the scorer's own reason when it gave no usable output (an LLM's passage not in the text, a server
        that did not answer) — the escalation says that, not that the checkpoint has no pointer."""
        if ptr is None:
            return Decision(Quote("", 0, 0, NULL_SOURCE), {}, confidence=0.0,
                            escalate=f"{ESCALATED}: {why}" if why else
                            "the checkpoint gave no pointer output for this span question")
        spans, pn = ptr["spans"], float(ptr["null"])
        if sp.unknown and (not spans or pn >= spans[0][0]):
            return Decision(Unknown, {Unknown: pn}, confidence=pn, extra={"p_null": pn})
        if not spans:
            return Decision(Quote("", 0, 0, NULL_SOURCE), {}, confidence=0.0, escalate="the pointer found no span")
        k, mass = _typed_span(spans, sp.vtype)
        p, a, b, t = spans[k]
        p = p if mass is None else mass
        conf = float(p) if sp.unknown else min(1.0, float(p) / max(1e-12, 1 - pn))
        extra = {"p_null": pn}
        if k:
            extra["trimmed"] = spans[0][3]            # the best span did not parse as the type; its part that did
        d = Decision(Quote(t, a, b, NULL_SOURCE, conf), {Unknown: pn} if sp.unknown else {}, confidence=conf, extra=extra)
        if not sp.unknown and pn >= spans[0][0]:      # the model itself says "no span": not an answer to give alone
            d.escalate = (f"{ESCALATED}: the text may not state it — no span (p {pn:.2f}) is at least as probable as the "
                          f"best span (p {spans[0][0]:.2f}); would have answered {t!r}")
        return d

    def _decision_v1(self, sp, z):
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
        if sp.kind == "noul":
            lab = "yes" if q[0] >= 0.5 else "no"
            return Decision(sp.out(lab), {"yes": float(q[0]), "no": float(q[1])}, confidence=float(q.max()))
        if sp.kind == "score":
            levels = sp.real
            acc, med = 0.0, levels[-1]
            for o, p in zip(levels, q):
                acc += p
                if acc >= 0.5 - 1e-12:
                    med = o
                    break
            rank = float(np.dot(np.arange(len(levels)), q))
            numeric = all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in levels)
            expected = float(np.dot(np.asarray(levels, float), q)) if numeric else rank
            value = {"median": med, "mode": levels[int(q.argmax())], "expected": levels[int(round(rank))]}[sp.score_value]
            return Decision(value, {o: float(v) for o, v in zip(levels, q)}, confidence=float(q[levels.index(value)]),
                            extra={"expected": expected, "median": med})
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

    def act_probability(self, sp, d, act):
        """The probability that the model's answer is right, from its act logit: sigmoid(logit / T_act), or the checkpoint's
        act calibrator — a logistic regression over ACT_FEATURES of the calibrated decision (see docs/decide_format.md)."""
        cal = (self.caps["act"] or {}).get("calibrator")
        if not cal:
            return float(_sig(act / self.temperatures["act"]))
        x = act_features(sp, d, act)
        return float(_sig(float(cal.get("bias", 0.0)) + sum(float(w) * x[f] for f, w in zip(cal["features"], cal["weights"]))))

    def _finish(self, sp, d, act, escalate_below=None, act_threshold=None, use_act=None):
        """Act or escalate: the model's act signal (its probability is recorded; below the threshold → "model escalated"),
        else the calibrated confidence below escalate_below (→ "confidence … < …", the low-confidence safeguard)."""
        if self.loras:
            name = self._lora_name(sp)
            if name is not None:
                d.extra["lora"] = {"adapter": self.loras[lora_key(sp)].hash, "experimental": True}
        if act is not None:
            p = self.act_probability(sp, d, act)
            d.extra["act"] = p
            thr = self.act_threshold if act_threshold is None else act_threshold
            if use_act is not False and p < thr:
                d.escalate = f"{ESCALATED}: act {p:.2f} < {thr:.2f}; would have answered {d.value!r}"
        eb = self.escalate_below if escalate_below is None else escalate_below
        if d.escalate is None and eb is not None and d.conf < eb:
            d.escalate = f"confidence {d.conf:.2f} < {eb:.2f} (escalate_below); would have answered {d.value!r}"
        return d

    @_deprecate.kwargs(escalate_below="min_confidence")
    @_spec_names
    def decide(self, text, task, options, descriptions=None, multi=False, other=None, kind=None, min_confidence=None,
               **spec):
        """→ Decision(value, probs) (a list of them for a list of inputs). The value is always one of the options; a text
        or a state (dict, list, pydantic model: see state_text). `spec`: not_stated=, k=, bins=, unit=, coverage=,
        evidence= (see decision). min_confidence: escalate below this confidence (`escalate_below=` in 0.7)."""
        escalate_below = min_confidence
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        one = _single(text)
        texts = [text] if one else list(text)
        texts = [self.text(t) for t in texts]
        raws = self._raw_full([(sp, t) for t in texts])
        out = [self._finish(sp, self._decision(sp, z), a, escalate_below) for z, a in raws]
        for d, t in zip(out, texts):                  # an input read cut: marked, and a warning once per question
            cut = self.truncation(sp, t)
            if cut:
                d.extra["truncated"] = cut
                self._warn_once(("truncated", sp.key),
                                f"the input has {cut['input_tokens']} tokens and one pass reads {cut['read_tokens']} of "
                                f"them (max_len {cut['max_len']}, the question takes {cut['question_tokens']}): the rest "
                                "was cut, and the decision's extra[\"truncated\"] says so. A decision part with "
                                "long=\"retrieve\" reads a long input by its relevant sections.")
        return out[0] if one else out

    def score(self, text, task, options, descriptions=None, multi=False, other=None, kind=None):
        """→ {option: probability} (a list of them for a list of inputs): softmax over the options for a single choice, a
        score or yes/no, a sigmoid per option with multi=True; the adaptation of this question applied, "other" by its
        threshold."""
        d = self.decide(text, task, options, descriptions, multi, other, kind)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def decide_pass(self, text, parts, names=None):
        """Several decision parts about one input, scored together → [Decision] (one forward pass when the checkpoint
        supports it, else one per question). What the runtime does for parts grouped in `flow.batches`."""
        parts = list(parts)
        t = self.text(text)
        long = any(p.long is not None and p._too_long(t) for p in parts)    # a long text: each part retrieves its own
        if len(parts) > 1 and self.batchable and all(p.option_order != "average" for p in parts) and not long:
            firsts = [(self._decision(p.spec, z), a, shared)
                      for p, (z, a, shared) in zip(parts, self._raw_pass([p.spec for p in parts], t))]
        else:
            firsts = [(*p._initial(t), False) for p in parts]
        names = list(names) if names else [p.__name__ for p in parts]
        out = []
        for p, (d0, a, shared) in zip(parts, firsts):
            ctx = p._ctx(t, vals=text if isinstance(text, Facts) else None, raw=text)
            if shared:
                ctx["specs"] = [q.spec for q in parts]            # the pass read the input after all their questions
            d = p._finish(d0, a, ctx=ctx)
            d.extra["pass"] = {"with": names, "shared": shared}
            out.append(d)
        return out

    # --- adaptation
    def adaptation(self, task, options, descriptions=None, multi=False, other=None, create=False, kind=None):
        sp = _Spec(task, options, descriptions, multi, other, kind)
        if create and sp.key not in self.adaptations:
            self.adaptations[sp.key] = Adaptation()
        return self.adaptations.get(sp.key)

    @_spec_names
    def adapt(self, texts, task, options, descriptions=None, multi=False, other=None, kind=None, logits=None, **spec):
        """Label-bias correction without labels: the mean logit of each option over unlabelled inputs of the domain
        (centered over the options) is subtracted before the softmax / sigmoid. → the Adaptation. A later fit is kept
        consistent. logits: the inputs' logits already computed (one per text, in the question's option order) — what a
        DecisionPart passes, so the correction is fitted on the signal it decides on (option_order="average", long)."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        if sp.kind == "span":
            raise ValueError("a span question has no options to adapt")
        texts = [self.text(t) for t in texts]
        if not texts:
            raise ValueError("adapt needs unlabelled texts")
        Z = np.array(_usable(self._raw([(sp, t) for t in texts])) if logits is None else _given(logits, len(texts)))
        mean = Z.mean(0)
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.bias = [float(v) for v in mean - mean.mean()]
        a.n_unlabelled = len(texts)
        if a.examples:
            self._refit(sp, a)
        return a

    @_spec_names
    def fit(self, examples, task, options, descriptions=None, multi=False, other=None, lam=1.0, folds=4, kind=None,
            logits=None, **spec):
        """Few-shot adaptation "S" from labelled examples [(input, correct)]: a shift and a shared scale on the
        (bias-corrected) logits by L-BFGS (per option; a score: a tilt and a spread over the levels; yes/no: one bias), a
        temperature on out-of-fold predictions, and — when some examples are labelled "other" — the "other" threshold that
        maximizes out-of-fold accuracy. Replaces earlier examples. Examples labelled "not stated" are left out (the "not
        stated" logit is not adapted). logits: the examples' logits already computed (one per example; see adapt).
        → the Adaptation."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        examples = list(examples)
        given = None if logits is None else _given(logits, len(examples))
        keep = [i for i, (_, y) in enumerate(examples) if y is not Unknown]
        ex = [(self.text(examples[i][0]), examples[i][1]) for i in keep]
        Z = _usable(self._raw([(sp, t) for t, _ in ex])) if given is None else [given[i] for i in keep]
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples = [(list(map(float, z)), sp.label(y)) for z, (_, y) in zip(Z, ex)]
        self._refit(sp, a, lam, folds)
        return a

    @_spec_names
    def teach(self, text, correct, task, options, descriptions=None, multi=False, other=None, lam=1.0, kind=None,
              logits=None, **spec):
        """One labelled example, absorbed at once: the shift / scale is refitted from the kept examples (warm start, K + 1
        parameters or fewer — about a millisecond or two); the temperature and the "other" threshold stay until the next
        fit. logits: the input's logits already computed (see adapt). → the update time in ms (the model's forward pass,
        if the input was not scored before, is not included)."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        z = _usable(self._raw([(sp, self.text(text))]))[0] if logits is None else _given([logits], 1)[0]
        t0 = time.perf_counter()
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples.append((list(map(float, z)), sp.label(correct)))
        self._refit(sp, a, lam, folds=0, warm=True)
        return (time.perf_counter() - t0) * 1000

    def _refit(self, sp, a, lam=1.0, folds=4, warm=False):
        a0 = 1.0 / self._T(sp)
        basis = _basis(sp)
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
        a.scale, b = _fit_shift(Z, Y, sp.multi, a0, lam, iters=60 if warm else 300, init=init, basis=basis)
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
                af, bf = _fit_shift(Z[tr], Y[tr], sp.multi, a0, lam, basis=basis)
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

    @_spec_names
    def reset(self, task=None, options=None, descriptions=None, multi=False, other=None, kind=None, **spec):
        """Forget the adaptation of one question, or every adaptation."""
        if task is None:
            self.adaptations.clear()
        else:
            self.adaptations.pop(_Spec(task, options, descriptions, multi, other, kind, **spec).key, None)

    def save_adaptations(self, path):
        """Write every adaptation (with its kept examples' logits) to a JSON file, with the checkpoint's fingerprint."""
        data = {"weights": self.weights_fingerprint(), "model_id": self.model_id,
                "adaptations": [{"key": [k[0], list(k[1]), list(k[2]), *k[3:]], **asdict(a)} for k, a in self.adaptations.items()]}
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
            self.adaptations[(k[0], tuple(k[1]), tuple(k[2]), k[3] if isinstance(k[3], str) else bool(k[3]), *k[4:])] = \
                Adaptation(**d, examples=ex)             # k[4:]: "pointer" / "evidence" questions (files before 0.8 lost it)
        return self

    # --- catalog parts
    @_deprecate.kwargs(escalate_below="min_confidence", act_threshold="min_act", target_error="max_error",
                       unknown="not_stated")
    def decision(self, name, task, text_fact="doc", options=(), descriptions=None, multi=False, other=None, *, kind=None,
                 type=None, min_confidence=None, min_act=None, use_act=None, max_error=None,
                 score_value=None, not_stated=False, k=None, bins=None, unit=None, coverage=None, evidence=False,
                 option_order="canonical", permutations=4, min_margin=None, long=None, top_k=None, rerank=False,
                 perturb=0, retrieve_query=None, _shared=frozenset()):
        """A catalog part: text_fact (a fact name, or a list of them) → Decision(value, probs).

        The question: `options` (a list, or {option: description}) and `kind` ("choice", "multi", "score", "noul"; default
        choice, or multi with multi=True) — or a Python type, as `type=` or in place of the options: Literal[...] / an Enum
        (choice), list[Literal[...]] (multi), Scale[...] (score), bool (noul: the value is True / False). The input: a
        text fact is read as it is (several joined by new lines); a state (dict, list, pydantic model) by state_text
        (several facts: {fact: value}).

        Escalation: the model's act signal when it has one (use_act=False ignores it; min_act overrides the
        checkpoint's threshold; max_error=0.1 takes the checkpoint's threshold for that error rate), and a calibrated
        confidence below min_confidence (see calibrate_for); an escalated decision is rejected — the fact is missing, the
        answer abstains ("model escalated" / "low confidence" in the audit and stats). min_margin=0.1: also escalate when
        the two most probable answers are closer than that (a near tie is where a misleading text flips the choice).

        Option order (choice and multi questions): "canonical" (the default) asks in sorted order, so how a caller lists
        the options cannot change the answer (the part's options are then in that order); "given" asks as listed (0.5.0);
        "average" averages the model's logits over `permutations` rotations of the list (each costs a forward pass) —
        against a model's preference for positions.

        perturb=k: ask again on up to k variants of the input without its instruction-like sentences ("ignore the rules
        and answer X", "SYSTEM: ...", "the correct answer is X" — deterministic rules, solvi.perturb) and escalate when
        the answer changes ("answer depends on an instruction-like sentence: ..."). An input without such sentences costs
        nothing extra; one with them costs up to k forward passes.

        Register with `cat.fn(part)` (a fact other parts read) or make it a question's answer with `part.question(cat)`.
        The value is one of the options by construction; the options are the part's closed set; provenance `decided`; the
        trace records the model, whose fingerprint covers the checkpoint and this part's adaptation and thresholds.

        Answer primitives (an l14g checkpoint, docs/decide_format.md §9): `Maybe[T]` or not_stated=True — "not stated" is an
        answer (solvi.Unknown); `Rank[Literal[...], k]` or kind="rank", k= — the options best first; `Estimate[edges]` or
        kind="number", bins=, unit=, coverage= — a number over bins; `Span[T]` or kind="span" — a piece of the (one, given)
        text fact, coerced to T when it is a question's answer; evidence=True (or a number) — supporting quotes from the
        pointer. A checkpoint that cannot give what is asked raises here.

        Long texts: long=None cuts a text beyond max_len (the tokenizer truncates it, as before); long="retrieve" splits
        it into sections, selects the top_k that bear on the question by BM25 (rerank=True: re-ordered by the decider's own
        relevance, one yes / no pass per candidate section) and decides on them; spans and evidence point into the whole
        text, and the sections read are in the decision's extra["long"] (solvi.longdoc). top_k=None (the default): sections
        of about 170 tokens — budget / 170, at least 3 (3 at max_len 512, 12 at 2048). retrieve_query: the words the
        sections are searched by, in place of the question's own (its task, options and descriptions) — the labels the
        document writes next to the value ("Invoice No Contract No Ref"), or the document's language when the question
        is asked in another one; the decider still reads the question as it is. long="full" (a checkpoint trained
        on long inputs: `max_len_long` in its solvi_decide.json) reads a text that does not fit max_len whole, up to
        max_len_long tokens, and retrieves within max_len_long beyond that (recorded in extra["long"]); a GPU mode — on a
        CPU a whole 8k-token text takes seconds per question.

        An option the question's kind does not use is a ValueError, not ignored: score_value= (score questions; default
        "median"), k= (rank), bins= / unit= / coverage= (number; coverage default 0.8), other= (choice and multi),
        min_margin= (not multi), top_k= / rerank= (with long=), min_act= / max_error= (a checkpoint with an act
        head), and kind= that contradicts multi=True.

        Names (0.8): min_confidence= (0.7: escalate_below=), min_act= (act_threshold=), max_error= (target_error=),
        not_stated= (unknown=) — the old ones work with a SolviDeprecationWarning until 0.9; the part keeps the thresholds as
        `part.min_confidence` / `part.min_act`."""
        escalate_below, act_threshold, target_error, unknown = min_confidence, min_act, max_error, not_stated
        given = {"score_value": score_value, "k": k, "bins": bins, "unit": unit, "coverage": coverage, "other": other,
                 "min_margin": min_margin, "top_k": top_k, "rerank": rerank or None, "min_act": act_threshold,
                 "max_error": target_error}
        as_bool, extra = False, {}
        if type is None and _is_type(options):
            type, options = options, ()
        if type is not None:
            kind, options, descriptions, as_bool, extra = _from_type(type, kind, options, descriptions)
        if target_error is not None and act_threshold is None:
            act_threshold = self.act_threshold_for(target_error)
        prim = {"unknown": bool(unknown or extra.get("unknown")), "k": extra.get("k", k), "bins": extra.get("bins", bins),
                "unit": extra.get("unit", unit), "coverage": extra.get("coverage", coverage),
                "vtype": extra.get("vtype")}
        if extra.get("labels") is not None:
            options = extra["labels"]
        pc = self.caps.get("pointer") or {}
        prim["evidence"] = (int((pc.get("evidence") or {}).get("max_spans", 3)) if evidence is True else int(evidence or 0))
        k_ = _kind(kind, multi)
        _unused_options(name, k_, kind, multi, long, self.has_act, {x: v for x, v in given.items() if x not in _shared})
        if prim["coverage"] is None:
            prim["coverage"] = 0.8
        if score_value is None:
            score_value = "median"
        if (k_ == "span" or prim["evidence"]) and not self.has_pointer:
            raise ValueError(f"{name}: this checkpoint cannot point at its input (span answers, evidence): it declares no "
                             "'pointer' / 'span' mode (docs/decide_format.md §9)")
        if prim["unknown"] and not self.has_not_stated:
            raise ValueError(f"{name}: this checkpoint has no 'not stated' output (declare 'unknown', docs/decide_format.md "
                             "§9); drop Maybe[...] / not_stated=True")
        if option_order not in ("given", "canonical", "average"):
            raise ValueError('option_order must be "given", "canonical" or "average"')
        if long not in (None, "retrieve", "full"):
            raise ValueError('long must be None (truncate), "retrieve" or "full"')
        if long == "full":
            self._check_full(name)
        return DecisionPart(self, name, task, text_fact, options, descriptions, multi, other, kind=kind, as_bool=as_bool,
                            escalate_below=escalate_below, act_threshold=act_threshold, use_act=use_act,
                            option_order=option_order, permutations=permutations, min_margin=min_margin,
                            long=long, top_k=top_k, rerank=rerank, perturb=perturb, retrieve_query=retrieve_query,
                            score_value=score_value, **{x: v for x, v in prim.items() if v not in (None, False, 0)
                                                         or x == "coverage"})

    def _check_full(self, name):
        """long="full" needs a long-input length: declared by the checkpoint, or forced at load (with a warning)."""
        L = self.max_len_long
        if L is None:
            raise ValueError(
                f'{name}: long="full" needs a checkpoint trained on long inputs — this one declares no "max_len_long" in its '
                f'solvi_decide.json (it reads {self.max_len} tokens). Use long="retrieve": it finds the sections that bear on '
                'the question and reads those. To read whole anyway, load with DecideModel.load(path, max_len_long=N) '
                '(a model trained on short inputs was not trained to read long texts: long="retrieve" is the safer '
                'choice)')
        if L <= self.max_len:
            raise ValueError(f'{name}: long="full" reads up to max_len_long = {L} tokens, not more than max_len = '
                             f'{self.max_len}: nothing to read whole')
        declared = self.caps.get("max_len_long")
        if declared is None or L > int(declared):
            self._warn_once("untrained", (
                f"{self.model_id}: long=\"full\" reads up to {L} tokens, but the checkpoint "
                + ("declares no long-input length" if declared is None else f"was trained on up to {declared}")
                + ": it was not trained on inputs that long, so its answers and quotes on them are not what it learned; "
                "prefer long=\"retrieve\", which reads the sections that bear on the question."))

    def decisions(self, schema, text_fact="doc", fields=None, **kw):
        """One decision part per field of a pydantic model class: the field's type is the question (bool, Literal[...],
        an Enum, Scale[...], list[Literal[...]]), its description the task (else its title, else its name), and
        `json_schema_extra` may carry "options" ({option: description}), "min_confidence", "min_act", "use_act",
        "max_error", "other", "score_value" (the 0.7 keys "escalate_below", "act_threshold", "target_error" still read,
        with a SolviDeprecationWarning). → {field: DecisionPart} in field order."""
        import typing

        from ..typed import Bins, Ordinal, RankOf, SpanOf
        out = {}
        for name, fi in schema.model_fields.items():
            if fields is not None and name not in fields:
                continue
            t = fi.annotation
            marks = [m for m in fi.metadata if isinstance(m, (Ordinal, SpanOf, RankOf, Bins))]
            if marks:                                  # pydantic moves Annotated metadata to the field: put solvi's back
                t = typing.Annotated[(t, *marks)]
            extra = fi.json_schema_extra if isinstance(fi.json_schema_extra, dict) else {}
            args = dict(kw)
            old = {"escalate_below": "min_confidence", "act_threshold": "min_act", "target_error": "max_error"}
            for k in ("min_confidence", "min_act", "use_act", "max_error", "other", "score_value", "kind",
                      "evidence", "coverage", "unit", *old):
                if k in extra:
                    if k in old:
                        _deprecate.renamed(f"json_schema_extra {k!r}", repr(old[k]), stacklevel=3)
                    args[old.get(k, k)] = extra[k]
            task = fi.description or fi.title or name.replace("_", " ").capitalize() + "?"
            out[name] = self.decision(name, task, text_fact, options=extra.get("options") or (), type=t, **args,
                                      _shared=frozenset(kw))      # an option given to every field applies where it can
        if fields is not None and set(fields) - set(out):
            raise ValueError(f"fields {sorted(set(fields) - set(out))} are not fields of {schema.__name__} "
                             f"({', '.join(schema.model_fields)})")
        return out

    def questions(self, cat, schema, text_fact="doc", fields=None, min_confidence=None, **kw):
        """decisions(...) registered as the answers of questions named after the fields → [Question]."""
        return [p.question(cat, min_confidence=min_confidence) for p in self.decisions(schema, text_fact, fields, **kw).values()]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


__all__ = ["DecideModel"]
