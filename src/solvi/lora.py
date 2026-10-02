"""LoRA adapters on the decider, one per question (experimental): `part.adapt_lora(examples)`.

When a question has a hundred labelled answers or more, a shift and a scale on the logits (`part.fit`) stop improving: they
cannot change what the model reads in the input. A LoRA adapter can: low-rank updates (rank 8) of the encoder's attention
and MLP weights in every layer, plus the last layer of the output head, trained on this question's examples with the
rest of the checkpoint frozen. Measured on solvi-base (typed decisions of four processes, the adapter against `fit` on
the same examples, per process): +2.9 / +4.2 / +6.0 / +9.2 points at 32 / 100 / 300 / 1000 examples — `fit` levels off
near 63% accuracy, the adapter reaches 72.6% at 1000. The adapter is 3.2 MB (bf16); training on a CPU takes minutes
(about 4 at 100 examples and 13 at 300 on 4 server cores; a laptop is slower), on a GPU about 20 seconds.

    part = model.decision("team", "Which team?", "email", TEAMS)      # DecideModel.load(..., backend="torch")
    report = part.adapt_lora(labelled, holdout=300)                   # 300 of the examples calibrate act_guard
    part.save_calibration("team.calib.json")                          # + team.calib.lora.safetensors beside it
    ...
    part.remove_lora()                                                # roll back

What changes and what does not:

- The adapter is active only while this question is scored; every other question of the same model is scored by the
  checkpoint as it was (to the bit). Its hash is part of the part's fingerprint (and the model's), and every decision
  records it in extra["lora"], so a replay knows which weights answered.
- Confidences after LoRA are overconfident (calibration error 1.5–3× that of `fit` in the measurements above), so the
  escalation must be recalibrated on labels not used for training: `holdout=` runs act_guard on them (the risk guarantee
  held at 0.10, and the adapter answered alone 53% of the time against 45% for `fit`, at 300 examples). The question's
  earlier adaptation and thresholds are cleared: they were fitted on the model without the adapter.
- Deterministic for a fixed seed on a CPU (the same examples, seed, torch version and thread count give the same adapter,
  and the same hash).
- solvi-base-sized checkpoints only; for solvi-large or thousands of examples, tools/adapt_lora_gpu.py (in the solvi
  repository, not installed by pip) trains the same adapter on a GPU, and `part.load_lora(path)` loads it.

Needs the torch backend and peft: `pip install "solvi[lora]"`."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import random
import threading
import time
import warnings
from dataclasses import dataclass, field

FORMAT = "solvi lora v1"
TARGET = r".*layers\.\d+\.(attn\.(Wqkv|Wo)|mlp\.(Wi|Wo))"     # ModernBERT: attention and MLP of every layer
HEAD = "head.3."                                              # the output head's last layer is trained too
BS = 8                                                        # examples per update
MAX_HIDDEN = 768                                              # solvi-base; larger checkpoints: the GPU script
GPU_SCRIPT = ("tools/adapt_lora_gpu.py (in the solvi repository, not installed by pip: "
              "https://github.com/solvi-ai/solvi/blob/main/tools/adapt_lora_gpu.py)")
MIN_EXAMPLES, FEW_EXAMPLES = 8, 100
_state_lock = threading.Lock()
_warned = [False]


class LoraWarning(UserWarning):
    """adapt_lora's notices: the time it will take, few examples, no held-out calibration."""


def _experimental():
    if not _warned[0]:
        _warned[0] = True
        from .learning import ExperimentalWarning
        warnings.warn("part.adapt_lora / load_lora are experimental: the API, the recipe and the file format may change",
                      ExperimentalWarning, stacklevel=4)


def _lora_key(part):
    from .decide import lora_key
    return lora_key(part.spec)


@dataclass
class LoraAdapter:
    """A trained adapter: its tensors (float32 on the CPU, values exact in bf16 — the file stores bf16), its config (rank,
    alpha, targets, the question) and what it was trained on (info). `hash` covers the config and the tensors."""
    tensors: dict
    config: dict
    info: dict = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = _hash(self.tensors, self.config)

    @property
    def name(self):
        return "solvi_" + self.hash

    @property
    def size_bytes(self):
        return sum(int(t.numel()) * 2 for t in self.tensors.values())

    def describe(self):
        """What the trace and the metadata show."""
        return {"adapter": self.hash, "r": self.config.get("r"), "k": self.info.get("k"),
                "updates": self.info.get("updates"), "size_mb": round(self.size_bytes / 2 ** 20, 2), "experimental": True}


def _hash(tensors, config):
    import torch
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for k in sorted(tensors):
        t = tensors[k].detach().to("cpu", torch.bfloat16).contiguous()
        h.update(f"{k}{tuple(t.shape)}".encode())
        h.update(t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()[:16]


def _peft_config(config):
    from peft import LoraConfig
    return LoraConfig(r=int(config["r"]), lora_alpha=int(config["alpha"]), target_modules=config["target"],
                      lora_dropout=0.0, bias="none")


# ------------------------------------------------------------------------------------------------ scope
def check(part, allow_large=False):
    """Refuse what adapt_lora cannot adapt (ValueError / TypeError / ImportError with what to do instead)."""
    from .decide import DecisionPart, TorchScorer
    if not isinstance(part, DecisionPart):
        raise TypeError(f"adapt_lora adapts a model decision (a DecisionPart), not a {type(part).__name__}: a rule or a "
                        "learned head has no encoder to adapt (use fit)")
    scorer = getattr(part.model, "scorer", None)
    if not isinstance(scorer, TorchScorer):
        name = type(scorer).__name__
        if name == "LLMScorer":
            why = "an LLM decider (solvi.llm) has no encoder weights here to adapt; use fit and act_guard on its answers"
        elif name == "OnnxScorer":
            why = ('the decider runs on ONNX, and LoRA trains the torch weights: load it with DecideModel.load(..., '
                   'backend="torch") (pip install "solvi[lora]")')
        else:
            why = f'LoRA needs the torch decider (DecideModel.load(..., backend="torch")), not a {name}'
        raise ValueError(f"adapt_lora({part.__name__}): {why}")
    try:
        import peft  # noqa: F401
    except ImportError:
        raise ImportError('adapt_lora needs peft: pip install "solvi[lora]"') from None
    if getattr(part, "long", None) == "full":
        raise ValueError(f'adapt_lora({part.__name__}): an adapter is trained on ordinary passes, not on whole long texts '
                         '(long="full"); adapt this question with long="retrieve" (or long=None)')
    if part.spec.kind in ("rank", "number", "span"):
        raise ValueError(f"adapt_lora({part.__name__}): {part.spec.kind} questions are not adapted from labels (choice, "
                         "multi, score and yes/no questions are)")
    hidden = int(getattr(getattr(scorer.model.enc, "config", None), "hidden_size", 0) or 0)
    if hidden > MAX_HIDDEN and not allow_large:
        n = sum(p.numel() for p in scorer.model.parameters()) / 1e6
        raise ValueError(f"adapt_lora({part.__name__}): this decider has {n:.0f}M parameters; in-process training is for "
                         f"solvi-base (on a CPU a solvi-large adapter takes 40 minutes and more at 300 examples). Train it "
                         f"on a GPU with {GPU_SCRIPT} and load it with part.load_lora(path)")


# ------------------------------------------------------------------------------------------------ adapters on the scorer
def _state(scorer):
    with _state_lock:
        st = getattr(scorer, "_lora", None)
        if st is None:
            head = scorer.model.head[3]
            st = scorer._lora = {"lock": threading.RLock(), "active": None, "heads": {},
                                 "head0": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    return st


def _layers(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    return [m for m in model.modules() if isinstance(m, BaseTunerLayer)]


def _activate(scorer, st, name, force=False):
    """Switch the scorer's network to adapter `name` (None: the checkpoint as it is): the LoRA layers and the head."""
    if st["active"] == name and not force:
        return
    import torch
    for m in _layers(scorer.model):
        if name is None:
            m.enable_adapters(False)
        else:
            m.enable_adapters(True)
            m.set_adapter(name, inference_mode=True)
    with torch.no_grad():
        scorer.model.head[3].load_state_dict(st["head0"] if name is None else st["heads"][name])
    st["active"] = name


@contextlib.contextmanager
def using(scorer, name):
    """Score with adapter `name` active (None: without any); one thread at a time."""
    st = _state(scorer)
    with st["lock"]:
        _activate(scorer, st, name)
        yield


def install(scorer, ad):
    """Add an adapter to the scorer's network (inactive until a question that has it is scored)."""
    import torch
    from peft import inject_adapter_in_model
    st = _state(scorer)
    with st["lock"]:
        if ad.name in st["heads"]:
            return
        _activate(scorer, st, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inject_adapter_in_model(_peft_config(ad.config), scorer.model, adapter_name=ad.name)
        tag = f".{ad.name}."
        mine = {n.replace(tag, "."): p for n, p in scorer.model.named_parameters() if tag in n}
        body = {k for k in ad.tensors if not k.startswith(HEAD)}
        heads = {k[len(HEAD):]: v for k, v in ad.tensors.items() if k.startswith(HEAD)}
        ok = set(mine) == body and set(heads) == set(st["head0"])
        if ok:
            with torch.no_grad():
                for k in body:
                    mine[k].copy_(ad.tensors[k].to(mine[k].device, mine[k].dtype))
            for p in mine.values():
                p.requires_grad_(False)
            dev = next(iter(st["head0"].values())).device
            st["heads"][ad.name] = {k: v.to(dev, st["head0"][k].dtype) for k, v in heads.items()}
        else:
            _delete(scorer, ad.name)
        _activate(scorer, st, None, force=True)
        if not ok:
            raise ValueError("the adapter does not fit this checkpoint (other layers or sizes): it was trained on another "
                             "decider")


def _delete(scorer, name):
    from peft.tuners.tuners_utils import delete_adapter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        delete_adapter(scorer.model, name, prefix="lora_")


def uninstall(scorer, name):
    st = _state(scorer)
    with st["lock"]:
        if st["active"] == name:
            _activate(scorer, st, None)
        _delete(scorer, name)
        st["heads"].pop(name, None)
        _activate(scorer, st, None, force=True)


# ------------------------------------------------------------------------------------------------ the part's state
def _snapshot(part):
    return {"adaptation": copy.deepcopy(part.model.adaptations.get(part.spec.key)),
            "escalate_below": part.escalate_below, "act_threshold": part.act_threshold,
            "guarantee": copy.deepcopy(part.guarantee), "conformal_set": copy.deepcopy(part.conformal_set),
            "groups": part.groups, "signature": part.__signature__}


def _restore(part, snap):
    if snap["adaptation"] is None:
        part.model.adaptations.pop(part.spec.key, None)
    else:
        part.model.adaptations[part.spec.key] = snap["adaptation"]
    part.escalate_below, part.act_threshold = snap["escalate_below"], snap["act_threshold"]
    part.guarantee, part.conformal_set, part.groups = snap["guarantee"], snap["conformal_set"], snap["groups"]
    part.__signature__ = snap["signature"]


def _clear(part):
    """Forget what was fitted on the model without this adapter → the names of what was cleared."""
    import inspect
    cleared = []
    if part.model.adaptations.pop(part.spec.key, None) is not None:
        cleared.append("adaptation")
    for a in ("escalate_below", "act_threshold", "guarantee", "conformal_set", "groups"):
        if getattr(part, a) is not None:
            cleared.append(a)
            setattr(part, a, None)
    part.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                            for f in part.facts])
    return cleared


def _set(part, ad):
    """Make `ad` this question's adapter (replacing an earlier one) → what was cleared."""
    if getattr(part, "_lora_undo", None) is None:
        part._lora_undo = _snapshot(part)
    key, m = _lora_key(part), part.model
    old = m.loras.get(key)
    install(m.scorer, ad)
    m.loras[key] = ad
    if old is not None and old.name != ad.name:
        uninstall(m.scorer, old.name)
    m._forget_cached(key)
    return _clear(part)


def remove(part):
    """part.remove_lora (see there)."""
    ad = part.lora
    if ad is None:
        return None
    key, m = _lora_key(part), part.model
    m.loras.pop(key, None)
    uninstall(m.scorer, ad.name)
    m._forget_cached(key)
    undo = getattr(part, "_lora_undo", None)
    if undo is not None:
        _restore(part, undo)
        part._lora_undo = None
    else:
        _clear(part)
    return ad.hash


# ------------------------------------------------------------------------------------------------ training
def n_updates(k, epochs, max_updates=400):
    """Updates of BS examples: `epochs` passes over k examples, at least 40 and at most max_updates."""
    return int(min(max_updates, max(40, math.ceil(epochs * k / BS))))


def _texts(part, xs):
    """Inputs → the texts a decision reads (a long input: its retrieved window)."""
    ts = [part._input_text(x) for x in xs]
    if part.long is not None:
        ts = [part.long_input(t) if part._too_long(t) else t for t in ts]
    return ts


def _duration(s):
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


def train(part, rows, *, r=8, alpha=None, epochs=6, lr=3e-4, seed=0, device=None, max_updates=400, notify=None):
    """Train an adapter on [(text, label)] (labels as part.spec.label gives them) → LoraAdapter. notify(estimate_seconds,
    updates, seconds_per_update, device) is called after the first update is timed, before training."""
    import torch
    from peft import inject_adapter_in_model
    model, sp = part.model, part.spec
    scorer = model.scorer
    dev = torch.device(device or scorer.device)
    k = len(rows)
    U = n_updates(k, epochs, max_updates)
    config = {"format": FORMAT, "r": int(r), "alpha": int(alpha if alpha is not None else 2 * r), "target": TARGET,
              "head": HEAD.rstrip("."), "question": repr(_lora_key(part))}
    st = _state(scorer)
    with st["lock"]:                               # a copy of the checkpoint as it is: scoring goes on meanwhile
        _activate(scorer, st, None)
        net = copy.deepcopy(scorer.model)
    net.to("cpu")
    with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(seed)                    # the adapter's initialization (on the CPU: the same on any device)
        inject_adapter_in_model(_peft_config(config), net, adapter_name="solvi_training")
    net.to(dev)
    net.eval()                                     # no dropout: the decider has none, and eval keeps it deterministic
    for m in _layers(net):
        m.enable_adapters(True)
        m.set_adapter("solvi_training")
    for n, p in net.named_parameters():
        p.requires_grad_(".solvi_training." in n or n.startswith(HEAD))
    params = [p for p in net.parameters() if p.requires_grad]

    block = model.block and not model._block_failed and not sp.pointer
    col = model.caps["columns"].get(model.wire(sp.kind), 0)
    encs = []
    for t, _ in rows:
        it = model._item(sp, t)
        encs.append(scorer.enc.encode_block((it,), t, scorer.mq["max_len"]) if block else scorer.enc.encode((it,), t))
    if sp.multi:
        targets = [torch.tensor([float(o in y) for o in sp.real]) for _, y in rows]
    else:
        targets = [torch.tensor(sp.real.index(y)) for _, y in rows]
    cuda = dev.type == "cuda"

    def loss_of(idx):
        e = [encs[i] for i in idx]
        ids, att, pids, masks = scorer.pack(e, block)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            out = net(*scorer.inputs(ids, att, pids, masks, device=dev))
        ls = []
        for rr, i in enumerate(idx):
            z = out[rr, torch.as_tensor(e[rr][1][0][1], device=dev), col].float()
            y = targets[i].to(dev)
            ls.append(torch.nn.functional.binary_cross_entropy_with_logits(z, y) if sp.multi else
                      torch.nn.functional.cross_entropy(z[None], y[None]))
        return torch.stack(ls).mean()

    t0 = time.perf_counter()                       # time one update (not applied) for the estimate
    loss_of(list(range(min(BS, k)))).backward()
    net.zero_grad(set_to_none=True)
    per = time.perf_counter() - t0
    if notify is not None:
        notify(per * U, U, per, str(dev))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-6)
    warm = max(1, U // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: (u + 1) / warm if u < warm else max(0.0, (U - u) / max(1, U - warm)))
    rng = random.Random(seed)
    order, pos, losses = [], 0, []
    t1 = time.perf_counter()
    for _ in range(U):
        b = []
        while len(b) < min(BS, k):
            if pos >= len(order):
                order, pos = list(range(k)), 0
                rng.shuffle(order)
            b.append(order[pos])
            pos += 1
        loss = loss_of(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    seconds = time.perf_counter() - t1
    tensors = {}
    for n, p in net.named_parameters():
        if ".solvi_training." in n:
            tensors[n.replace(".solvi_training.", ".")] = p.detach().to("cpu", torch.bfloat16).float()
        elif n.startswith(HEAD):
            tensors[n] = p.detach().to("cpu", torch.bfloat16).float()
    from . import __version__
    info = {"k": k, "updates": U, "epochs": epochs, "lr": lr, "seed": seed, "batch": BS, "device": str(dev),
            "seconds": round(seconds, 1), "loss_first": round(sum(losses[:5]) / len(losses[:5]), 4),
            "loss_last": round(sum(losses[-5:]) / len(losses[-5:]), 4), "weights": model.weights_fingerprint(),
            "model_id": str(model.model_id), "question": list(map(str, (sp.task, *sp.real))), "kind": sp.kind,
            "solvi": __version__, "torch": torch.__version__, "threads": torch.get_num_threads()}
    return LoraAdapter(tensors, config, info)


def _split(examples, holdout, seed):
    ex = list(examples)
    if holdout is None or holdout is False or holdout == 0:
        return ex, []
    if isinstance(holdout, (list, tuple)):
        return ex, list(holdout)
    if isinstance(holdout, bool) or not isinstance(holdout, (int, float)):
        raise ValueError("holdout: a list of [(input, correct)], a share of the examples (0 < holdout < 1) or a number of them")
    n = int(round(holdout * len(ex))) if isinstance(holdout, float) and holdout < 1 else int(holdout)
    if not 0 < n < len(ex):
        raise ValueError(f"holdout={holdout!r} leaves {len(ex) - n} of {len(ex)} examples for training")
    idx = list(range(len(ex)))
    random.Random(seed).shuffle(idx)
    held = set(idx[:n])
    return [e for i, e in enumerate(ex) if i not in held], [e for i, e in enumerate(ex) if i in held]


def _accuracy(part, hold):
    _, ok, _, _ = part._labelled(hold, "confidence")
    return float(sum(ok) / len(ok))


def adapt(part, examples, *, r=8, epochs=6, holdout=None, seed=0, device=None, lr=3e-4, max_risk=0.10,
          signal="confidence", max_updates=400, allow_large=False):
    """part.adapt_lora (see there). allow_large: in-process training of a checkpoint larger than solvi-base (what
    tools/adapt_lora_gpu.py passes, on a GPU)."""
    from .core import Unknown
    _experimental()
    check(part, allow_large)
    sp = part.spec
    train_ex, hold = _split(examples, holdout, seed)
    keep = [(x, y) for x, y in train_ex if y is not Unknown]
    labels = [sp.label(y) for _, y in keep]
    pairs = [(x, y) for (x, _), y in zip(keep, labels) if sp.multi or sp.other is None or y != sp.other]
    skipped = len(train_ex) - len(pairs)          # "not stated" and "other": nothing to train the options on
    k = len(pairs)
    name = part.__name__
    if k < MIN_EXAMPLES:
        raise ValueError(f"adapt_lora({name}): {k} usable examples; it needs at least {MIN_EXAMPLES} (and about "
                         f"{FEW_EXAMPLES} or more to beat fit)")
    if k < FEW_EXAMPLES:
        warnings.warn(f"adapt_lora({name}): {k} examples — below about {FEW_EXAMPLES}, part.fit (and System.fit for questions "
                      "without a model) is about as accurate and takes milliseconds (the adapter gained ~3 points at 32 "
                      "examples in our measurements, within noise on some tasks)", LoraWarning, stacklevel=3)
    before = _accuracy(part, hold) if hold else None
    rows = list(zip(_texts(part, [x for x, _ in pairs]), [y for _, y in pairs]))

    est_s = []

    def notify(est, U, per, dev):
        est_s.append(round(est, 1))
        extra = (f"; for more examples or a larger model, {GPU_SCRIPT} trains it on a GPU in seconds"
                 if est > 20 * 60 else "")
        warnings.warn(f"adapt_lora({name}): {k} examples, {U} updates on {dev} — about {_duration(est)} "
                      f"(one update took {per:.1f} s here){extra}", LoraWarning, stacklevel=4)

    ad = train(part, rows, r=r, epochs=epochs, lr=lr, seed=seed, device=device, max_updates=max_updates, notify=notify)
    cleared = _set(part, ad)
    out = {"adapter": ad.hash, "k": k, "skipped": skipped, "updates": ad.info["updates"], "seconds": ad.info["seconds"],
           "estimate_seconds": est_s[0] if est_s else None, "size_mb": round(ad.size_bytes / 2 ** 20, 2), "device": ad.info["device"],
           "loss": [ad.info["loss_first"], ad.info["loss_last"]], "cleared": cleared, "holdout": None,
           "experimental": True}
    if hold:
        after = _accuracy(part, hold)
        guard = part.act_guard(hold, max_risk=max_risk, signal=signal)
        out["holdout"] = {"n": len(hold), "accuracy_before": before, "accuracy_after": after, "act_guard": guard}
    else:
        warnings.warn(f"adapt_lora({name}): no held-out labels, so escalation is not calibrated — confidences after LoRA "
                      "are overconfident. Call part.act_guard(held_out, max_risk=0.10) on labels not used for training (~300 "
                      "is typical), or pass holdout=", LoraWarning, stacklevel=3)
    return out


# ------------------------------------------------------------------------------------------------ files
def save(part, path):
    """part.save_lora (see there) → path."""
    import torch
    from safetensors.torch import save_file
    ad = part.lora
    if ad is None:
        raise ValueError(f"{part.__name__} has no LoRA adapter (adapt_lora or load_lora first)")
    meta = {"format": FORMAT, "hash": ad.hash, "config": json.dumps(ad.config, sort_keys=True),
            "info": json.dumps(ad.info, default=str)}
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in ad.tensors.items()}, str(path), metadata=meta)
    return path


def read(path):
    """An adapter file → LoraAdapter (ValueError if it is not one, or its content does not match its hash)."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k).float() for k in f.keys()}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a solvi LoRA adapter (format {meta.get('format')!r}, expected {FORMAT!r})")
    ad = LoraAdapter(tensors, json.loads(meta["config"]), json.loads(meta.get("info") or "{}"))
    if ad.hash != meta.get("hash"):
        raise ValueError(f"{path}: the adapter's content does not match its hash #{meta.get('hash')} (#{ad.hash})")
    return ad


def load(part, path, strict=True, expect=None):
    """part.load_lora (see there); expect: the hash a calibration file names."""
    _experimental()
    check(part, allow_large=True)                  # loading (inference) is fine for any size
    ad = read(path)
    if expect is not None and ad.hash != expect:
        raise ValueError(f"{path}: adapter #{ad.hash}, the calibration was made with #{expect}")
    if strict:
        if ad.config.get("question") != repr(_lora_key(part)):
            raise ValueError(f"{path}: an adapter for another question ({ad.info.get('question')}); this part asks "
                             f"{[part.spec.task, *map(str, part.spec.real)]}")
        if ad.info.get("weights") != part.model.weights_fingerprint():
            raise ValueError(f"{path}: trained on checkpoint #{ad.info.get('weights')} ({ad.info.get('model_id')}), this "
                             f"is #{part.model.weights_fingerprint()} — train it again for this checkpoint")
    _set(part, ad)
    return ad


__all__ = ["LoraAdapter", "LoraWarning", "adapt", "check", "load", "n_updates", "read", "remove", "save", "train"]
