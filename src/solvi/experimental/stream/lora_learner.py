"""The LoRA learner of the adapter loop (experimental; stage 3): `LoraLearner`, a batch learner that trains a LoRA
adapter on the decider for one question, and `LoraVersion`, a frozen version (an adapter and its calibration) that the
loop scores, freezes, serves and rolls back. Needs the torch decider and peft (`pip install "solvi[lora]"`).

    from solvi.experimental.stream import AdapterLoop
    from solvi.experimental.stream.lora_learner import LoraLearner

    part = model.decision("intent", "Which intent?", "text", INTENTS)        # DecideModel.load(path, backend="torch")
    learner = LoraLearner(part, dir="loop/")                                 # runner="subprocess" by default
    loop = AdapterLoop(learner, question="intent", initial=learner.version0(), system=part, versions_dir="loop/",
                       max_risk=0.05, secret=key)

What a training is: `train(job)` trains an adapter **from the base checkpoint every time** (no warm start: a
`warm_start` setting is refused) on `job.rows` only, with the policy's `r`, `lr`, `updates` and `seed`, in the runner
(`stream.runner`: a subprocess by default — it loads the checkpoint again, `nice` 10, threads capped at
`budget.cpu_threads`, the device picked by `budget.device`, "auto": a GPU with `free_gpu_mb` free, else the CPU; a GPU
out of memory falls back to the CPU). Before training it estimates the time from the first update and refuses when the
estimate is above `budget.max_train_s` (the loop records the refusal). Then it runs `act_guard` at the loop's α on the
calib labels (`job.calib`: labels whose keys are never trained on) on a copy of the part with the new adapter, and
returns a `LoraVersion` whose `act_threshold` is that threshold. The runner writes `train_<job id>.log` (a heartbeat
every 30 s) and the adapter file `train_<job id>.lora.safetensors`; the version is staged on the scorer (installed,
inactive) when it is read back.

`LoraVersion` is both the loop's frozen model and its `Served`: `score(x)` → (the answer, the signal its threshold is
on), computed with its adapter on this thread only (`solvi.experimental.lora.scoring`: the served part, other
questions and the fingerprints are untouched); `install(system)` is `solvi.experimental.lora.swap` — the adapter and its
thresholds change together under the part's version lock, so no decision of the part sees one without the other —
and `uninstall` swaps back what it replaced. `fingerprint()` = digest(adapter hash, calibration fingerprint).
`LoraVersion.of_part(part)` is version 0: the part as attached, its adapter (or none) and its thresholds. `save(dir,
v)` writes `<v>.lora.safetensors` and `<v>.calib.json`; `load(dir, v, expect_fp=, part=)` reads them back (the
adapter's content against its hash, the version against `expect_fp`).

Limits: determinism holds for a fixed device and thread count (the adapter file is bf16 either way, so a GPU-trained
adapter loads bit-exact on a CPU); a combination over the part (Cascade, Vote, Route) reads the part's internals
without its version lock — the swap's atomicity covers the part's own decisions (call, decide, in_pass); a part's
correction memory is not consulted while calibrating; trained versions stay staged on the scorer (about 3 MB each for
solvi-base) for the loop's life. Unmeasured: no stand has run on the LoRA loop."""
from __future__ import annotations

import copy
import inspect
import json
import math
import pickle
from pathlib import Path

from ...core.provenance import digest
from .runner import TrainingFailed, make_runner

TARGET = "solvi.experimental.stream.lora_learner:train_target"
CALIB_FORMAT = "solvi lora version calibration v1"


def _lora():
    from .. import lora
    return lora


def _signature(names):
    return inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD) for f in names])


def calibration_from(rec) -> dict:
    """A calibration record (solvi.experimental.lora.calibration_record) → the calibration a swap restores. A version
    the learner trained has no adaptation and no thresholds per group; a record with either is refused."""
    if rec.get("adaptation") is not None or rec.get("groups") is not None:
        raise ValueError("a LoRA version's calibration has no adaptation and no thresholds per group")
    return {"adaptation": None, "escalate_below": rec.get("escalate_below"), "act_threshold": rec.get("act_threshold"),
            "guarantee": copy.deepcopy(rec.get("guarantee")), "conformal_set": copy.deepcopy(rec.get("conformal_set")),
            "groups": None, "signature": None if rec.get("signature") is None else _signature(rec["signature"])}


class LoraVersion:
    """See the module docstring. part: the DecisionPart it serves (None: read from files, not bound yet — `bind`);
    adapter: a LoraAdapter or None (the checkpoint itself); calibration: solvi.experimental.lora.calibration's dict;
    info: what it was trained on; signal: the signal score returns ("auto": the threshold's, as recorded by act_guard)."""

    def __init__(self, part, adapter, calibration, *, info=None, signal="auto"):
        self.part, self.adapter, self.calibration = None, adapter, calibration
        self.info, self.signal = dict(info or {}), signal
        L = _lora()
        self._fp = digest("LoraVersion", None if adapter is None else adapter.hash, L.calibration_fingerprint(calibration))
        self._replaced = None
        if part is not None:
            self.bind(part)

    @classmethod
    def of_part(cls, part, *, signal="auto"):
        """Version 0: the part as it is now — its adapter (or none) and its calibration."""
        L = _lora()
        return cls(part, part.lora, L.calibration(part), info={"version0": True}, signal=signal)

    def bind(self, part):
        """Serve and score on `part` (its version lock is created; the adapter is staged on its scorer) → self."""
        L = _lora()
        L.version_lock(part)
        if self.adapter is not None:
            L.stage(part, self.adapter)
        self.part = part
        return self

    def _bound(self):
        if self.part is None:
            raise ValueError("this LoRA version is not bound to a part: LoraVersion.load(..., part=) or bind(part)")
        return self.part

    # --- Frozen
    def fingerprint(self) -> str:
        return self._fp

    @property
    def threshold(self):
        """The act_guard threshold on its signal (escalate_below for "confidence", act_threshold for "act"); None when
        it has none; inf when act_guard found none at the promised risk."""
        c = self.calibration
        sig = (c.get("guarantee") or {}).get("signal")
        if sig == "act":
            return c.get("act_threshold")
        if sig == "confidence":
            return c.get("escalate_below")
        return c.get("escalate_below") if c.get("escalate_below") is not None else c.get("act_threshold")

    @property
    def act_threshold(self):
        """The threshold the adapter loop's pre-freeze report applies to score's signal (= threshold)."""
        return self.threshold

    def _sig_name(self):
        if self.signal != "auto":
            return self.signal
        return (self.calibration.get("guarantee") or {}).get("signal")

    def score(self, x):
        """→ (the answer, the signal): the part's question scored with this version's adapter on this thread only
        (no threshold applied); the signal is the one its threshold is on (act or confidence)."""
        part = self._bound()
        L = _lora()
        with L.scoring(part, self.adapter):
            text = part._input_text(x)
            if part.long is not None and part._too_long(text):
                text = part.long_input(text)
            z, a = part._raw([text])[0]
            d = part.model._decision(part.spec, z)
            d = part.model._finish(part.spec, d, a, -math.inf, -math.inf, part.use_act)
        name = self._sig_name()
        if name == "act" and d.extra.get("act") is not None:
            s = float(d.extra["act"])
        elif name == "confidence":
            s = float(d.conf)
        else:
            s = part._signal(d)[1]
        return d.value, s

    # --- Served
    def install(self, system=None) -> dict:
        """Serve this version on its part (system: unused — the part is the version's own): the atomic swap. → {"from",
        "to"}: the part's fingerprints."""
        part = self._bound()
        L = _lora()
        prev = (part.lora, L.calibration(part))
        out = L.swap(part, self.adapter, self.calibration)
        self._replaced = prev
        return {"from": out["from"], "to": out["to"]}

    def uninstall(self, system=None) -> None:
        """Back to what install replaced (atomically)."""
        if self._replaced is None:
            return
        ad, cal = self._replaced
        _lora().swap(self._bound(), ad, cal)
        self._replaced = None

    def save(self, dir, version: int) -> list:
        """→ [<v>.lora.safetensors (when it has an adapter), <v>.calib.json]."""
        L = _lora()
        d = Path(dir)
        out = []
        if self.adapter is not None:
            out.append(self.adapter.save(d / f"{int(version)}.lora.safetensors"))
        rec = {"format": CALIB_FORMAT, "adapter": None if self.adapter is None else self.adapter.hash,
               "calibration": L.calibration_record(self.calibration), "fingerprint": self._fp, "signal": self.signal,
               "info": self.info}
        p = d / f"{int(version)}.calib.json"
        p.write_text(json.dumps(rec, default=str, sort_keys=True))
        return out + [p]

    @classmethod
    def load(cls, dir, version: int, *, expect_fp: str, part=None) -> "LoraVersion":
        """The version saved by save; ValueError when a file is not the recorded one (the adapter's content against its
        hash, the calibration's adapter, the version's fingerprint against expect_fp)."""
        L = _lora()
        d = Path(dir)
        rec = json.loads((d / f"{int(version)}.calib.json").read_text())
        if rec.get("format") != CALIB_FORMAT:
            raise ValueError(f"version {version}: not a LoRA version's calibration ({rec.get('format')!r})")
        ad = None
        if rec.get("adapter") is not None:
            ad = L.read(d / f"{int(version)}.lora.safetensors")
            if ad.hash != rec["adapter"]:
                raise ValueError(f"version {version}: adapter #{ad.hash}, the calibration names #{rec['adapter']}")
        v = cls(None, ad, calibration_from(rec["calibration"]), info=rec.get("info"), signal=rec.get("signal", "auto"))
        if v.fingerprint() != expect_fp:
            raise ValueError(f"version {version}: fingerprint {v.fingerprint()[:12]}, recorded {str(expect_fp)[:12]}")
        return v if part is None else v.bind(part)

    def __repr__(self):
        a = None if self.adapter is None else self.adapter.hash
        return f"LoraVersion(adapter {a}, {self._fp[:12]}, threshold {self.threshold})"


def _part_bytes(part):
    """The part without its model (and its correction memory), pickled: what the training process rebuilds it from."""
    p = copy.copy(part)
    p.model = None
    p.correction_memory = None
    p.__dict__.pop("_lora_undo", None)
    p.__solvi_model__ = p.__solvi_decision__ = p
    try:
        return pickle.dumps(p)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"LoraLearner: the part {part.__name__} does not pickle without its model ({e}); the training "
                        "process rebuilds it from that") from e


class LoraLearner:
    """See the module docstring. part: the question's DecisionPart (torch backend, loaded from a checkpoint folder);
    dir: where the logs and adapter files go; runner: "subprocess" (default), "thread" or a runner object
    (stream.runner); signal: act_guard's signal ("auto"); epochs: when the policy's settings give no `updates`;
    max_risk: act_guard's risk when the job does not carry the loop's α."""

    def __init__(self, part, dir, *, runner="subprocess", signal="auto", epochs=6, max_risk=0.05):
        L = _lora()
        L.check(part)
        if not getattr(part.model, "path", None):
            raise ValueError("LoraLearner: the training process loads the checkpoint again from its folder; this model "
                             "has no path (DecideModel.load(folder, backend=\"torch\"))")
        self.part, self.dir = part, Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runner = make_runner(runner)
        self.signal, self.epochs, self.max_risk = signal, int(epochs), float(max_risk)
        self._seen = 0
        L.version_lock(part)

    def update(self, x, y, *, key=None, source=None) -> None:
        """A train label: counted; the rows of a training come with its job (the loop keeps them)."""
        self._seen += 1

    def seen(self) -> int:
        return self._seen

    def version0(self) -> LoraVersion:
        """The part as it is now, as version 0 (the loop's initial=)."""
        return LoraVersion.of_part(self.part, signal=self.signal)

    def _payload(self, job):
        L = _lora()
        part, sp = self.part, self.part.spec
        from ...core.catalog import Unknown
        hyper = dict(job.hyper or {})
        if hyper.get("warm_start"):
            raise ValueError("warm_start is not built: every version is trained from the base checkpoint")
        keep = [(r.x, r.y) for r in job.rows if r.y is not Unknown]
        labels = [sp.label(y) for _, y in keep]
        pairs = [(x, y) for (x, _), y in zip(keep, labels) if sp.multi or sp.other is None or y != sp.other]
        if len(pairs) < L.MIN_EXAMPLES:
            raise ValueError(f"{len(pairs)} usable train rows; a LoRA training needs at least {L.MIN_EXAMPLES}")
        calib = [(r.x, r.y) for r in getattr(job, "calib", ())]
        if not calib:
            raise ValueError("no calib labels: act_guard needs labels the adapter never trained on")
        rows = list(zip(L._texts(part, [x for x, _ in pairs]), [y for _, y in pairs]))
        m = part.model
        mq = getattr(m.scorer, "mq", None) or {}
        alpha = getattr(job, "alpha", None)
        return {"model": {"path": m.path, "max_len": mq.get("max_len"), "overrides": dict(getattr(m, "_overrides", {})),
                          "weights": m.weights_fingerprint()},
                "part": _part_bytes(part), "rows": rows, "calib": calib,
                "hyper": {"r": int(hyper.get("r", 8)), "lr": float(hyper.get("lr", 3e-4)),
                          "seed": int(hyper.get("seed", 0)), "updates": hyper.get("updates"), "epochs": self.epochs},
                "max_risk": float(alpha if alpha is not None else self.max_risk), "signal": self.signal,
                "out": str(self.dir / f"train_{job.id}.lora.safetensors")}

    def train(self, job) -> LoraVersion:
        """One training (see the module docstring) → LoraVersion. Raises stream.runner's TrainingRefused (the estimate
        is above max_train_s), TrainingFailed (the training raised) or RunnerError (the process died)."""
        from .policy import Budget
        L = _lora()
        payload = self._payload(job)
        budget = getattr(job, "budget", None) or Budget()
        status = getattr(job, "status", None)
        out = self.runner.run(TARGET, payload, name=f"train_{job.id}", workdir=self.dir, budget=budget, status=status)
        ad = L.read(out["adapter_file"])
        if ad.hash != out.get("adapter"):
            raise TrainingFailed(f"the adapter file holds #{ad.hash}, the training reported #{out.get('adapter')}")
        cal = calibration_from(out["calibration"])
        if L.calibration_fingerprint(cal) != out.get("calibration_fp"):
            raise TrainingFailed("the calibration read back is not the one the training made")
        info = {"train": ad.info, "device": out.get("device"), "estimate_s": out.get("estimate_s"),
                "act_guard": out.get("act_guard"), "job": job.id, "log": str(self.dir / f"train_{job.id}.log")}
        return LoraVersion(self.part, ad, cal, info=info, signal=self.signal)

    def __repr__(self):
        return f"LoraLearner({self.part.__name__!r}, {self.dir}, runner {getattr(self.runner, 'kind', self.runner)})"


def train_target(payload, ctx) -> dict:
    """The training itself, in the runner (a subprocess, or a thread): load the checkpoint, rebuild the part, train
    from the base, act_guard on the calib labels, write the adapter file → what LoraLearner.train reads back."""
    import torch
    L = _lora()
    from ...core.deciders import DecideModel
    m, hy = payload["model"], payload["hyper"]

    def build(dev):
        kw = {"max_len": m["max_len"]} if m.get("max_len") else {}
        model = DecideModel.load(m["path"], device=dev, backend="torch", **kw, **m.get("overrides", {}))
        if model.weights_fingerprint() != m["weights"]:
            raise ValueError(f"the checkpoint at {m['path']} is #{model.weights_fingerprint()}, the part's is "
                             f"#{m['weights']}")
        part = pickle.loads(payload["part"])  # noqa: S301 — the payload the parent process made
        part.model = model
        return part

    if not getattr(ctx, "in_process", False):            # a process of its own: its torch threads capped (a thread
        torch.set_num_threads(ctx.threads)                 # runner leaves the serving process's as they are)

    def fit(dev):
        ctx.set_phase(f"load {dev}")
        part = build(dev)
        ctx.set_phase(f"train {dev}")
        ad = L.train(part, payload["rows"], r=hy["r"], lr=hy["lr"], seed=hy["seed"], device=dev,
                     epochs=hy.get("epochs", 6), updates=hy.get("updates"),
                     notify=lambda est, U, per, d: ctx.estimate(est),
                     log=lambda u, U, loss: ctx.progress(u, U, loss=round(loss, 4)))
        return part, ad

    dev = ctx.device
    try:
        part, ad = fit(dev)
    except torch.cuda.OutOfMemoryError:
        if not str(dev).startswith("cuda"):
            raise
        ctx.log(f"out of GPU memory on {dev}: training on the CPU")
        torch.cuda.empty_cache()
        dev = ctx.device = "cpu"
        part, ad = fit(dev)
    ctx.set_phase("calibrate")
    L._set(part, ad)
    rep = part.act_guard(payload["calib"], max_risk=payload["max_risk"], signal=payload["signal"])
    cal = L.calibration(part)
    ad.save(payload["out"])
    return {"adapter": ad.hash, "adapter_file": payload["out"], "calibration": L.calibration_record(cal),
            "calibration_fp": L.calibration_fingerprint(cal), "device": dev,
            "act_guard": {k: rep.get(k) for k in ("signal", "threshold", "answered", "error", "risk", "n")}}


__all__ = ["LoraLearner", "LoraVersion", "calibration_from", "train_target"]
