"""What learns in the closed learning loop (experimental; stage 3b): the `Learner` protocol and two learners.

A `Learner` is the live model: it takes one oracle label at a time (`update(x, y)`), is never served, and gives a
`Frozen` copy (`snapshot()`) that never changes again. A `Frozen` model scores an input → (its would-be answer, the
guard signal) and has a fingerprint; `LearningLoop` (stream.versions) freezes, certifies and serves those copies.

- `HashedTextLearner` — for text streams: hashed word 1–2-grams and character 2–5-grams (CRC-32 into 2^18 buckets,
  L2-normalised counts) and an online softmax (one SGD step per label, plain numpy, sparse weights); the score is the
  largest probability. Classes are the given options, or (options=None) every label seen so far, in order of first
  sight. Re-implemented from a published description; not tuned, unmeasured.
- `FastHeadLearner` — for typed-state questions: solvi's own `FastHead` (the head `System.fit` builds) over facts; the
  first `min_fit` labels are kept and fitted once, every later one is `FastHead.teach` (a rank-one update; its refit
  schedule as `System.fit`'s). The score is the head's largest probability. `facts=` turns an input into the facts
  row (`system.facts_for`), so the input can be a System's init state.

- `HashedTextTrainer` — a batch learner (the stand-in for the LoRA learner of stage 3, no torch): `train(job)` fits a
  fresh `HashedTextLearner` on the job's rows (shuffled by the job's seed, `epochs` passes) and freezes it; every
  version is trained from scratch, never warm-started.

A batch learner has `train(job: TrainJob) → Frozen` (one training on `job.rows`, the rows the policy selected); the
adapter loop (`stream.loop.AdapterLoop`) calls it when the policy asks for a training and calls `snapshot()` on an
online learner (one without `train`). Its `update` may take `key=` and `source=` keywords (read once from its
signature). The contract: `train(job)` trains on `job.rows` only, or the learner exposes `keys()` — every content key
it has trained on (a learner that keeps the labels `update` gave it, or warm-starts from an earlier version). The loop
records as the version's training keys the keys of the rows it handed over (or, online, of every label it fed) united
with `keys()` when the learner has it; a sample on any of them is "seen", never "novel".

Threads (stage 3b): unless the loop runs with `wait=True`, a batch learner's `train(job)` runs on a thread of its own
while the decision thread keeps calling `update`, `seen` and `keys`. So `train(job)` must read `job` only (or its own
state under its own lock), and `update` / `seen` / `keys` must not wait for a training. `wait=True` restores the serial
form (the training inside the step or the control).

`Frozen.as_head(feature)` wraps a frozen model as an answer head (`solvi.core.Head`: options, features, predict,
contributions, fingerprint; fit and teach refused), which `solvi.core.versions.serve_head` installs on a System."""
from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class Frozen(Protocol):
    def score(self, x) -> tuple[Any, float]: ...      # (the would-be answer, the guard signal)

    def fingerprint(self) -> str: ...


class Learner(Protocol):
    def update(self, x, y) -> None: ...               # one oracle label (the live learner only)

    def snapshot(self) -> Frozen: ...                 # a copy that never changes again (online learners)

    def seen(self) -> int: ...                        # labels taken so far


class BatchLearner(Protocol):
    def update(self, x, y) -> None: ...               # one train label (may also take key=, source=)

    def train(self, job: "TrainJob") -> Frozen: ...   # one training on job.rows only (else expose keys()); on
    #                                                   another thread than update / seen / keys: read job only

    def seen(self) -> int: ...


@dataclass
class TrainJob:
    """One training the adapter loop runs. id: "<version>.<attempt>"; version: the version it would become; rows: the
    selected train rows (stream.policy.LabelRow; empty for an online learner, which freezes a snapshot); hyper: the
    policy's settings (r, lr, updates, seed, ...); trigger: "schedule" (the policy's want_train) or "train_now";
    started: the wall clock; state: "running", "done", "failed", "refused" or "stale"; why: the refusal or the error;
    fingerprint and seconds: the frozen model's and the wall time, once done; frozen: whether the pre-freeze gates
    passed. What the loop also hands a batch learner (stage 3): calib — the calib labels (stream.policy.LabelRow, the
    latest per key; never trained on: the LoRA learner runs act_guard on them); budget — the policy's Budget (the
    runner's threads, device and max_train_s); alpha — the loop's promised rate; status — the dict the loop's
    status()["job"] shows, which a runner updates in place (pid, log, device, eta, last line); cancel — a
    threading.Event the loop sets when it drops the training (close, an error stop): a runner that takes cancel= stops
    it."""
    id: str
    version: int
    rows: tuple = ()
    hyper: dict = field(default_factory=dict)
    trigger: str = "schedule"
    started: float = 0.0
    state: str = "running"
    why: str | None = None
    fingerprint: str | None = None
    seconds: float = 0.0
    frozen: bool | None = None
    calib: tuple = ()
    budget: Any = None
    alpha: float | None = None
    status: dict | None = None
    cancel: Any = field(default=None, repr=False, compare=False)


class FrozenError(TypeError):
    """A frozen model was asked to learn."""


class _AsHead:
    """A frozen model as an answer head (solvi.core.Head): it reads the facts `features`, never learns."""

    def __init__(self, frozen, options, features, read):
        self.frozen, self.options, self.features, self._read = frozen, list(options), list(features), read

    def fit(self, rows, answers, features):
        raise FrozenError("a served version is frozen: it is never fitted again (the loop's live learner learns)")

    def teach(self, row, answer):
        raise FrozenError("a served version is frozen: labels go to the loop's live learner")

    def predict(self, row):
        p = self.frozen.probs(self._read(row))
        return {o: float(p.get(o, 0.0)) for o in self.options}

    def contributions(self, row):
        a, s = self.frozen.score(self._read(row))
        return {f: float(s) if math.isfinite(s) else 0.0 for f in self.features}

    def fingerprint(self):
        return self.frozen.fingerprint()

    def __repr__(self):
        return f"served head {self.frozen.fingerprint()[:12]} over {self.features}"


# ───────────────────────────────────────────────────────────── hashed n-grams + online softmax
def _features(text, n_features, words, chars) -> dict:
    t = str(text).lower()
    counts: dict = {}

    def add(key):
        i = zlib.crc32(key.encode()) % n_features
        counts[i] = counts.get(i, 0.0) + 1.0
    toks = re.findall(r"\w+", t)
    for n in range(words[0], words[1] + 1):
        for j in range(len(toks) - n + 1):
            add("w|" + " ".join(toks[j:j + n]))
    padded = f" {' '.join(toks)} "
    for n in range(chars[0], chars[1] + 1):
        for j in range(len(padded) - n + 1):
            add("c|" + padded[j:j + n])
    norm = math.sqrt(math.fsum(v * v for v in counts.values())) or 1.0
    return {i: v / norm for i, v in sorted(counts.items())}


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class FrozenText:
    """A frozen copy of a HashedTextLearner (see the module docstring)."""

    def __init__(self, classes, weights, bias, config):
        self.classes, self._w, self._b, self.config = list(classes), weights, bias, dict(config)
        self._fp = None

    def _z(self, x):
        c = len(self.classes)
        z = self._b[:c].copy()
        for i, v in _features(x, self.config["n_features"], self.config["words"], self.config["chars"]).items():
            w = self._w.get(i)
            if w is not None:
                z[:len(w)] += v * w[:c]
        return z

    def probs(self, x) -> dict:
        if not self.classes:
            return {}
        return dict(zip(self.classes, (float(p) for p in _softmax(self._z(x)))))

    def score(self, x):
        if not self.classes:
            return None, -math.inf
        p = _softmax(self._z(x))
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])

    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256()
            h.update(repr((self.classes, sorted(self.config.items()))).encode())
            h.update(np.round(self._b, 12).tobytes())
            for i in sorted(self._w):
                h.update(i.to_bytes(8, "little"))
                h.update(np.round(self._w[i], 12).tobytes())
            self._fp = h.hexdigest()
        return self._fp

    def as_head(self, feature: str = "text", options=None):
        """This version as an answer head reading the text fact `feature`."""
        return _AsHead(self, options if options is not None else self.classes, [feature], lambda row: row[feature])

    def __repr__(self):
        return f"FrozenText({len(self.classes)} classes, {self.fingerprint()[:12]})"


class HashedTextLearner:
    """See the module docstring. options: the closed set of answers (None: open, a class per label seen); lr: the SGD
    step; n_features: hash buckets; words / chars: n-gram ranges."""

    def __init__(self, options=None, *, lr: float = 0.5, n_features: int = 2 ** 18, words=(1, 2), chars=(2, 5)):
        if lr <= 0:
            raise ValueError("lr is a positive step")
        self.closed = options is not None
        self.classes = list(options or [])
        self.lr, self.n_features = float(lr), int(n_features)
        self.words, self.chars = tuple(words), tuple(chars)
        self._w: dict = {}
        self._b = np.zeros(max(1, len(self.classes)))
        self._seen = 0

    def _config(self):
        return {"n_features": self.n_features, "words": self.words, "chars": self.chars, "lr": self.lr}

    def seen(self) -> int:
        return self._seen

    def update(self, x, y) -> None:
        if y not in self.classes:
            if self.closed:
                raise ValueError(f"{y!r} is not one of the options")
            self.classes.append(y)
        c = len(self.classes)
        if len(self._b) < c:
            self._b = np.concatenate([self._b, np.zeros(c - len(self._b))])
        feats = _features(x, self.n_features, self.words, self.chars)
        z = self._b[:c].copy()
        for i, v in feats.items():
            w = self._w.get(i)
            if w is not None:
                z[:len(w)] += v * w[:c]
        g = _softmax(z)
        g[self.classes.index(y)] -= 1.0
        for i, v in feats.items():
            w = self._w.get(i)
            if w is None or len(w) < c:
                w = np.concatenate([w if w is not None else np.zeros(0), np.zeros(c - (0 if w is None else len(w)))])
                self._w[i] = w
            w -= self.lr * v * g
        self._b[:c] -= self.lr * g
        self._seen += 1

    def snapshot(self) -> FrozenText:
        return FrozenText(self.classes, {i: w.copy() for i, w in self._w.items()}, self._b.copy(), self._config())

    def __repr__(self):
        return f"HashedTextLearner({len(self.classes)} classes, seen {self._seen})"


class HashedTextTrainer:
    """See the module docstring: a batch learner over HashedTextLearner. options, lr, n_features, words, chars: as
    HashedTextLearner; epochs: passes over the rows (job.hyper["epochs"] overrides)."""

    def __init__(self, options=None, *, lr: float = 0.5, n_features: int = 2 ** 18, words=(1, 2), chars=(2, 5),
                 epochs: int = 1):
        HashedTextLearner(options, lr=lr, n_features=n_features, words=words, chars=chars)    # the arguments, checked
        if int(epochs) < 1:
            raise ValueError("epochs is at least 1")
        self._kw = {"options": options, "lr": lr, "n_features": n_features, "words": words, "chars": chars}
        self.epochs, self._seen = int(epochs), 0

    def update(self, x, y, **_) -> None:
        self._seen += 1

    def seen(self) -> int:
        return self._seen

    def train(self, job) -> FrozenText:
        live = HashedTextLearner(self._kw["options"], **{k: v for k, v in self._kw.items() if k != "options"})
        rows = list(job.rows)
        rng = random.Random(f"train|{job.hyper.get('seed', job.version)}")
        for _ in range(int(job.hyper.get("epochs", self.epochs))):
            rng.shuffle(rows)
            for r in rows:
                live.update(r.x, r.y)
        return live.snapshot()

    def __repr__(self):
        return f"HashedTextTrainer(seen {self._seen})"


# ───────────────────────────────────────────────────────────── solvi's FastHead over facts
class FrozenFast:
    """A frozen copy of a FastHeadLearner (None head: not fitted yet — it answers nothing, score -inf)."""

    def __init__(self, head, options, facts):
        self.head, self.options, self._facts = head, list(options), facts

    @classmethod
    def of_system(cls, system, question):
        """The fitted answer head of a System's question, frozen (a copy) over the facts of its init state: version 0
        of a LearningLoop that starts from System 1 (`LearningLoop(initial=FrozenFast.of_system(system, q))`)."""
        head = system.heads.get(question)
        if head is None:
            raise ValueError(f"{question!r} has no fitted answer head (System.fit first)")
        return cls(copy.deepcopy(head), system.questions[question].answer.options, system.facts_for)

    def _row(self, x):
        return self._facts(x) if self._facts is not None else x

    def probs(self, x) -> dict:
        if self.head is None:
            return {}
        return self.head.predict(self._row(x))

    def score(self, x):
        if self.head is None:
            return None, -math.inf
        p = self.head.predict(self._row(x))
        vals = [p[o] for o in self.options]
        if not all(math.isfinite(v) for v in vals):
            return None, -math.inf
        k = int(np.argmax(vals))
        return self.options[k], float(vals[k])

    def fingerprint(self) -> str:
        from ...core.provenance import digest
        return digest("FrozenFast", self.options, None if self.head is None else self.head.fingerprint())

    def as_head(self, feature=None, options=None):
        """This version as an answer head over the head's own facts (feature: unused, kept for one signature)."""
        feats = [] if self.head is None else list(self.head.features)
        return _AsHead(self, options if options is not None else self.options, feats, lambda row: row)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_facts"] = None                                # a function of a live System is not pickled
        return d

    def __repr__(self):
        return f"FrozenFast({self.fingerprint()[:12]})"


class FastHeadLearner:
    """See the module docstring. options: the answers; features: the facts the head may read (None: every key of the
    first rows); min_fit: labels kept before the first fit; facts: input → facts row (None: the input is the row);
    lam, refit, refit_until: as FastHead."""

    def __init__(self, options, features=None, *, min_fit: int = 20, facts=None, lam=None, refit=2.0,
                 refit_until=2000):
        if int(min_fit) < 2:
            raise ValueError("min_fit is at least 2 labels")
        self.options, self.features, self.min_fit, self.facts = list(options), features, int(min_fit), facts
        self._kw = {"lam": lam, "refit": refit, "refit_until": refit_until}
        self.head = None
        self._rows, self._ys = [], []
        self._seen = 0

    @classmethod
    def for_system(cls, system, question, features=None, **kw):
        """A learner for a System's question: its options, and its facts from the init state (`system.facts_for`)."""
        q = system.questions[question]
        return cls(q.answer.options, features, facts=system.facts_for, **kw)

    def seen(self) -> int:
        return self._seen

    def update(self, x, y) -> None:
        if y not in self.options:
            raise ValueError(f"{y!r} is not one of the options")
        row = self.facts(x) if self.facts is not None else dict(x)
        if self.head is None:
            self._rows.append(row)
            self._ys.append(y)
            if len(self._rows) >= self.min_fit:
                from ...core.deciders.heads import FastHead
                feats = self.features if self.features is not None else sorted(self._rows[0])
                self.head = FastHead(self.options, **self._kw).fit(self._rows, self._ys, list(feats))
                self._rows, self._ys = [], []
        else:
            self.head.teach(row, y)
        self._seen += 1

    def snapshot(self) -> FrozenFast:
        return FrozenFast(copy.deepcopy(self.head), self.options, self.facts)

    def __repr__(self):
        return f"FastHeadLearner({len(self.options)} options, seen {self._seen})"


__all__ = ["BatchLearner", "FastHeadLearner", "Frozen", "FrozenError", "FrozenFast", "FrozenText", "HashedTextLearner",
           "HashedTextTrainer", "Learner", "TrainJob"]
