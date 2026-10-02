"""Answer head for questions without a rule: features from computed_state → answer probabilities.
A number is encoded as its value plus thresholds at training quantiles ("greater than threshold" steps).
Multinomial logistic regression with L2 (Adam, 300 steps — milliseconds; one-vs-rest ridge could not express the middle one
of ordered classes). Features are chosen by greedy forward selection on 5-fold cross-validation accuracy (gain ≥ 1 pt):
the selected facts define the question's flow.

System.fit built this head before 0.8; it now builds a FastHead (below), whose selection is scored by the leave-one-out
squared error (accuracy kept nothing on imbalanced questions). Head stays for code that builds one itself; its
Featurizer is shared with FastHead.

FastHead, VecFeaturizer and CandidateHead were solvi.fast up to 0.7 (the module still reads, with a warning):

Fast answer head: closed-form ridge regression, learned in milliseconds and updated instantly from every correction.

Features come from computed facts (numbers with quantile steps, booleans, categories — as in Head) and from vector facts such
as a document embedding (see LongSpanExtractor.embedder). Answers are one-vs-rest ridge scores turned into probabilities.
The head keeps the inverse of (XᵀX + λI), so `update` is a rank-one Sherman–Morrison step: a new labeled example is absorbed in
about a millisecond without retraining and without touching the rest of the system. Leave-one-out accuracy is exact and free
(the ridge hat matrix).

A rank-one step keeps what the first fit chose: the ridge strength, the featurizer (number scales, the known values of each
category) and whether pairwise products are used. Chosen on a handful of rows they are often wrong for hundreds (on open
tabular sets a head started on 10 rows and taught up to 300 was 5 points less accurate than one fitted on all 300). So the
head keeps its examples and, each time their number doubles (`refit=2.0`), fits again on all of them — the same as a fresh
fit on those rows — and goes on with rank-one steps. The amortised cost stays a constant per update; the update that
triggers a refit is as slow as a fit. Past `refit_until` examples (2000) it stops refitting and drops the kept rows."""
from __future__ import annotations

import numpy as np



class Featurizer:
    def __init__(self):
        self.spec = {}              # fact → ("num", mean, std) | ("bool",) | ("cat", [values])

    def fit(self, rows, facts):
        for f in facts:
            vs = [r.get(f) for r in rows if r.get(f) is not None]
            if not vs:
                continue
            if all(isinstance(v, bool) for v in vs):
                self.spec[f] = ("bool",)
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs):
                a = np.array(vs, float)
                qs = sorted(set(np.quantile(a, [0.125 * i for i in range(1, 8)]).round(6).tolist()))
                self.spec[f] = ("num", float(a.mean()), float(a.std() or 1.0), qs)
            elif all(isinstance(v, str) for v in vs) and len(set(vs)) <= 20:
                self.spec[f] = ("cat", sorted(set(vs)))
        return self

    def cols(self, facts):
        out = []
        for f in facts:
            s = self.spec.get(f)
            if s is None:
                continue
            if s[0] == "cat":
                out += [(f, v) for v in s[1]]
            elif s[0] == "num":
                out += [(f, None)] + [(f, q) for q in s[3]]
            else:
                out += [(f, None)]
        return out

    def row(self, r, facts):
        x = []
        for f in facts:
            s = self.spec.get(f)
            if s is None:
                continue
            v = r.get(f)
            if s[0] == "bool":
                x.append(0.0 if v is None else float(bool(v)) * 2 - 1)
            elif s[0] == "num":
                ok = isinstance(v, (int, float))
                x.append(0.0 if not ok else (float(v) - s[1]) / s[2])
                x += [(1.0 if float(v) > q else -1.0) if ok else 0.0 for q in s[3]]   # quantile thresholds: step features
            else:
                x += [1.0 if v == c else 0.0 for c in s[1]]
        return x


def _softmax_fit(X, y, k, l2=1e-2, steps=300, lr=0.1):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    W = np.zeros((d + 1, k))
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    Y = np.eye(k)[y]
    for t in range(1, steps + 1):
        Z = Xb @ W
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = Xb.T @ (P - Y) / n + l2 * np.vstack([W[:-1], np.zeros((1, k))])
        m = 0.9 * m + 0.1 * G
        v = 0.999 * v + 0.001 * G * G
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W


def _cv_acc(X, y, k, folds=5):
    idx = np.arange(len(y))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    hit = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        W = _softmax_fit(X[tr], y[tr], k)
        Z = np.hstack([X[te], np.ones((len(te), 1))]) @ W
        hit += int((Z.argmax(1) == y[te]).sum())
    return hit / len(y)


class Head:
    def __init__(self, options):
        self.options = list(options)
        self.features = []
        self.W = None
        self.T = 1.0
        self.fz = None
        self.loo_acc = None
        self._fp = None

    def fingerprint(self):
        """A stable hash of the head's parameters (recorded with every answer it gives)."""
        if getattr(self, "_fp", None) is None:
            from .provenance import digest
            self._fp = digest("Head", self.options, self.features, self.W, self.T, getattr(self, "prior", None),
                              None if self.fz is None else self.fz.spec)
        return self._fp

    def fit(self, rows, answers, candidates, min_gain=0.01):
        self._fp = None
        y = np.array([self.options.index(a) for a in answers])
        k = len(self.options)
        self.fz = Featurizer().fit(rows, candidates)
        cands = [f for f in candidates if f in self.fz.spec]
        chosen, cur = [], float(np.bincount(y, minlength=k).max() / len(y))
        while True:
            trial = []
            for f in cands:
                if f in chosen:
                    continue
                X = np.array([self.fz.row(r, chosen + [f]) for r in rows])
                trial.append((_cv_acc(X, y, k), f))
            if not trial:
                break
            acc, f = max(trial)
            if acc < cur + min_gain:
                break
            chosen.append(f)
            cur = acc
        self.features, self.cv_acc = chosen, cur
        self.loo_acc = cur
        if not chosen:
            self.W = None
            self.prior = np.bincount(y, minlength=k) / len(y)
            return self
        X = np.array([self.fz.row(r, chosen) for r in rows])
        self.W = _softmax_fit(X, y, k)
        self.T = 1.0
        return self

    def predict(self, row):
        if self.W is None:
            p = self.prior
        else:
            x = np.array(self.fz.row(row, self.features) + [1.0])
            z = x @ self.W / self.T
            p = np.exp(z - z.max())
            p /= p.sum()
        return {o: float(pi) for o, pi in zip(self.options, p)}

    def contributions(self, row):
        """Contribution of each feature to the chosen answer (for explanations)."""
        if self.W is None:
            return {}
        p = self.predict(row)
        j = self.options.index(max(p, key=p.get))
        out, i = {}, 0
        for f in self.features:
            n = len(self.fz.cols([f]))
            xs = self.fz.row(row, [f])
            out[f] = float(np.dot(xs, self.W[i:i + n, j]))
            i += n
        return out


# --- the fast head (solvi.fast up to 0.7)

def _numeric_vector(v):
    if isinstance(v, np.ndarray):
        return v.ndim == 1 and np.issubdtype(v.dtype, np.number)
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _why_dropped(rows, f):
    vs = [r.get(f) for r in rows]
    if all(v is None for v in vs):
        return "never computed in the examples"
    kinds = sorted({type(v).__name__ for v in vs if v is not None})
    missing = sum(v is None for v in vs)
    return f"mixed or unsupported values ({', '.join(kinds)}{f', {missing} missing' if missing else ''})"


class VecFeaturizer(Featurizer):
    """Featurizer plus fixed-length numeric vectors (lists / arrays), standardized per dimension."""

    def fit(self, rows, facts):
        super().fit(rows, facts)
        for f in facts:
            vs = [r.get(f) for r in rows if r.get(f) is not None]
            if vs and all(_numeric_vector(v) for v in vs) and len({len(v) for v in vs}) == 1:
                a = np.array([np.asarray(v, float) for v in vs])
                self.spec[f] = ("vec", a.mean(0), a.std(0) + 1e-6)
        return self

    def cols(self, facts):
        out = []
        for f in facts:
            s = self.spec.get(f)
            if s is not None and s[0] == "vec":
                out += [(f, i) for i in range(len(s[1]))]
            else:
                out += super().cols([f])
        return out

    def compact(self, r, facts):
        """One value per number (standardized), ±1 per boolean, one-hot categories; vectors left out — the base for pairs."""
        x = []
        for f in facts:
            s = self.spec.get(f)
            if s is None or s[0] == "vec":
                continue
            v = r.get(f)
            if s[0] == "num":
                x.append((float(v) - s[1]) / s[2] if isinstance(v, (int, float)) else 0.0)
            elif s[0] == "bool":
                x.append(0.0 if v is None else float(bool(v)) * 2 - 1)
            else:
                x += [1.0 if v == c else 0.0 for c in s[1]]
        return x

    def row(self, r, facts):
        x = []
        for f in facts:
            s = self.spec.get(f)
            if s is not None and s[0] == "vec":
                v = r.get(f)
                x += [0.0] * len(s[1]) if v is None else list((np.asarray(v, float) - s[1]) / s[2])
            else:
                x += super().row(r, [f])
        return x


_LAMS = (0.1, 0.3, 1, 3, 10, 30, 100)


def _loo_error(X, Y):
    """The ridge's exact leave-one-out squared error (the best over FastHead's ridge strengths): mean over the examples
    of the squared distance between the one-hot answer and the prediction made without that example."""
    e, V = np.linalg.eigh(X.T @ X)
    XV, VB = X @ V, V.T @ (X.T @ Y)
    best = None
    for lam in _LAMS:
        inv = 1.0 / (e + lam)
        h = np.einsum("ij,j,ij->i", XV, inv, XV)
        loo = Y - (Y - XV @ (inv[:, None] * VB)) / (1 - h)[:, None]
        err = float(((loo - Y) ** 2).sum(1).mean())
        best = err if best is None or err < best else best
    return best


def select_features(rows, answers, options, features, min_gain=0.0):
    """Greedy forward selection for a FastHead: start from no feature (the answers' shares) and add, one at a time, the
    fact that lowers the exact leave-one-out squared error most; stop when none lowers it by more than `min_gain` × the
    error of the answers' shares. → (chosen, {fact: error after adding it}, {fact that cannot be encoded: why}). A
    proper score, so a rare answer counts:
    accuracy, the old criterion, kept nothing on questions where the most frequent answer is 80-90% of the examples.
    Each fact is chosen with its own encoding only (no pairwise products while choosing)."""
    fz = VecFeaturizer().fit(rows, features)
    cands = [f for f in features if f in fz.spec]
    Y = np.eye(len(options))[[options.index(a) for a in answers]]
    ones = np.ones((len(rows), 1))
    cur = float(((Y - Y.mean(0)) ** 2).sum(1).mean())
    gain = float(min_gain) * cur
    cols = {f: np.array([fz.row(r, [f]) for r in rows]) for f in cands}
    chosen, path = [], {}
    while len(chosen) < len(cands):
        trial = [(_loo_error(np.hstack([cols[c] for c in chosen] + [cols[f], ones]), Y), i, f)
                 for i, f in enumerate(cands) if f not in chosen]
        err, _, f = min(trial)
        if cur - err <= gain:
            break
        chosen.append(f)
        path[f] = round(err, 6)
        cur = err
    return chosen, path, {f: _why_dropped(rows, f) for f in features if f not in fz.spec}


class FastHead:
    def __init__(self, options, lam=None, pairs=None, refit=2.0, refit_until=2000):
        """lam: ridge strength (None — chosen by exact leave-one-out accuracy); pairs: add pairwise products of the base features
        (None — only when there are at most 40 of them, so interactions and middle classes can be expressed). refit: when
        `update` brings the number of examples to `refit` times the number of the last fit, fit again on all of them (the
        ridge strength, the featurizer and the pairs decision are chosen again, as given here); None — never, and no rows are
        kept. refit_until: no refit beyond this many examples; the kept rows are dropped once none is due."""
        if refit is not None and refit and refit <= 1:
            raise ValueError("refit must be a growth factor above 1 (2.0: refit when the examples double) or None")
        self.options = list(options)
        self.lam = lam
        self.pairs = pairs
        self._lam0, self._pairs0 = lam, pairs         # as given: what a refit chooses again
        self.refit = float(refit) if refit else None
        self.refit_until = int(refit_until)
        self._rows = None                             # the examples kept for the next refit (None: none due)
        self._answers = None
        self._requested = None
        self.fitted_on = 0                            # number of examples at the last fit (or refit)
        self.features = []
        self.fz = None
        self.Ainv = None                  # (XᵀX + λI)⁻¹ over [features, bias]
        self.B = None                     # XᵀY
        self.W = None
        self.n = 0
        self.loo_acc = None
        self._fp = None

    def fingerprint(self):
        """A stable hash of the head's parameters (recorded with every answer it gives; changes with fit and each update)."""
        if getattr(self, "_fp", None) is None:
            from .provenance import digest
            self._fp = digest("FastHead", self.options, self.features, self.W, self.lam, self.pairs,
                              None if self.fz is None else self.fz.spec)
        return self._fp

    @property
    def cv_acc(self):                     # same name as Head, for code that prints it
        return self.loo_acc

    def _x(self, row):
        b = np.array(self.fz.row(row, self.features))
        if self.pairs:
            c = np.array(self.fz.compact(row, self.features))
            k = np.triu_indices(len(c), 1)
            pair = np.outer(c, c)[k] if np.isfinite(c).all() else np.full(len(k[0]), np.nan)   # inf · 0: no warning
            b = np.concatenate([b, pair])
        return np.append(b, 1.0)

    def fit(self, rows, answers, features):
        self.lam, self.pairs = getattr(self, "_lam0", self.lam), getattr(self, "_pairs0", self.pairs)
        self.fz = VecFeaturizer().fit(rows, features)
        self.features = [f for f in features if f in self.fz.spec]
        self.dropped = {f: _why_dropped(rows, f) for f in features if f not in self.fz.spec}   # asked for, but unusable
        if self.pairs is None:
            self.pairs = len(self.fz.compact(rows[0], self.features)) <= 40
        X = np.array([self._x(r) for r in rows])
        Y = np.eye(len(self.options))[[self.options.index(a) for a in answers]]
        B = X.T @ Y
        e, V = np.linalg.eigh(X.T @ X)                               # one decomposition serves every λ
        XV, VB = X @ V, V.T @ B
        best = None
        for lam in ([self.lam] if self.lam is not None else [0.1, 0.3, 1, 3, 10, 30, 100]):
            inv = 1.0 / (e + lam)
            h = np.einsum("ij,j,ij->i", XV, inv, XV)                  # leverages → exact leave-one-out predictions
            fit = XV @ (inv[:, None] * VB)
            loo = Y - (Y - fit) / (1 - h)[:, None]
            acc = float((loo.argmax(1) == Y.argmax(1)).mean())
            if best is None or acc > best[0]:
                best = (acc, lam, inv)
        self.loo_acc, self.lam, inv = best
        self.Ainv = (V * inv) @ V.T
        self.W = self.Ainv @ B
        self.B = B
        self.n = self.fitted_on = len(rows)
        self._fp = None
        refit = getattr(self, "refit", None)
        if refit and self._next_refit() <= self.refit_until:
            self._rows, self._answers, self._requested = [dict(r) for r in rows], list(answers), list(features)
        else:
            self._rows = self._answers = self._requested = None
        return self

    def _next_refit(self):
        import math
        return max(self.fitted_on + 1, math.ceil(self.fitted_on * self.refit))

    def update(self, row, answer):
        """Absorb one labeled example (rank-one update of the inverse; a refit on all kept examples when their number reaches
        the refit schedule); returns the time it took in ms."""
        import time
        t0 = time.perf_counter()
        k = self.options.index(answer)                # an unknown answer fails here, before anything changes
        rows = getattr(self, "_rows", None)
        if rows is not None and len(rows) + 1 >= self._next_refit():
            self.fit(rows + [row], self._answers + [answer], self._requested)
            return (time.perf_counter() - t0) * 1000
        x = self._x(row)
        y = np.eye(len(self.options))[k]
        Ax = self.Ainv @ x
        self.Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.B += np.outer(x, y)
        self.W = self.Ainv @ self.B
        self.n += 1
        self._fp = None
        if rows is not None:
            rows.append(dict(row))
            self._answers.append(answer)
        return (time.perf_counter() - t0) * 1000

    def scores(self, row):
        x = self._x(row)
        if not np.isfinite(x).all():                  # a NaN / inf feature: no score (the System abstains on it)
            return np.full(self.W.shape[1], np.nan)
        return x @ self.W

    def predict(self, row):
        s = np.clip(self.scores(row), 1e-3, None)
        p = s / s.sum()
        return {o: float(v) for o, v in zip(self.options, p)}

    def contributions(self, row):
        """Per fact: its own weight times value (pairwise terms are split equally between the two facts)."""
        x = self._x(row)
        a = int(np.argmax(self.scores(row)))
        owner = [f for f in self.features for _ in range(len(self.fz.cols([f])))]
        out = {f: 0.0 for f in self.features}
        for j, f in enumerate(owner):
            out[f] += float(x[j] * self.W[j, a])
        if self.pairs:
            k = len(owner)
            for (i, j), c in zip(zip(*np.triu_indices(k, 1)), x[k:-1] * self.W[k:-1, a]):
                out[owner[i]] += c / 2
                out[owner[j]] += c / 2
        return out


class CandidateHead:
    """A choice among candidates that change with every decision, learned from the candidates' own features.

    An answer head has fixed options. An agent's step has other candidates each time — the exits of a room, the rows a
    search returned, the tools on offer — so there is nothing to attach a per-option weight to, and fit / teach have no
    key to learn under. What does carry over from step to step is what a candidate *is*: its distance, its kind, whether
    it was a dead end. This head learns "is this candidate the one to take?" over those features (a FastHead: ridge,
    fitted in milliseconds, taught by one correction) and chooses the candidate it scores highest. relative=True also
    gives each numeric feature relative to the other candidates of the step (its gap to the smallest and the largest).

        head = CandidateHead(["kind", "distance", "reward", "dead_end"]).fit(steps)   # steps: [(candidates, chosen index)]
        i, probs = head.choose(candidates)             # candidates: [{feature: value}]; probs sum to 1 over them
        head.teach(candidates, 2)                      # one correction: the candidate that should have been taken

    Labels come from a rule you are replacing, from people, or from outcomes — and an outcome label must be the
    criterion of the sub-goal the step served (progress towards the goal for "go on", a level gained for "train"): one
    global measure teaches the head to ignore every step that does not move it.

    Measured on two tasks. Candidates scored by a hidden formula over four features (4–8 per step, 200 test steps):
    0.81 after 30 steps and 0.93 after 300, against 0.51–0.55 for "nearest that is open" / "largest reward"; taught
    one step at a time from 30 to 300: 0.89, 1 ms per step; with 20% of the labels wrong: 0.85. Where to train in a
    game, on its real data (16 features of a place, 180 test situations): 0.81 after 420 situations (0.67 after 100)
    against 0.23 for the nearest place. relative=True changed these by −5 and +2 points: off by default. The head
    learns the rule it is shown — in the game it reproduced the navigation rule and replaced the model there; it did
    not beat the rule."""

    def __init__(self, features, relative=False, pairs=None, refit=2.0):
        self.features = list(features)
        self.relative = bool(relative)
        self.head = FastHead(["yes", "no"], pairs=pairs, refit=refit)
        self.n = 0

    def rows(self, candidates):
        """The rows the head reads: each candidate's features and, for numbers, the gaps to the step's min and max."""
        cands = [dict(c) for c in candidates]
        if not self.relative:
            return [{f: c.get(f) for f in self.features} for c in cands]
        out = [{f: c.get(f) for f in self.features} for c in cands]
        for f in self.features:
            vs = [c.get(f) for c in cands]
            nums = [v for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(nums) != len(vs) or not nums:
                continue
            lo, hi = min(nums), max(nums)
            for r, v in zip(out, vs):
                r[f + ":above_min"], r[f + ":below_max"] = v - lo, hi - v
        return out

    def _names(self, rows):
        return list(dict.fromkeys(k for r in rows for k in r))

    def fit(self, steps):
        """steps: [(candidates, chosen)] — chosen: the index of the right candidate, or a set of indexes when several are
        as good. The others of the step are its "no" examples."""
        rows, answers = [], []
        for candidates, chosen in steps:
            good = {chosen} if isinstance(chosen, int) else set(chosen)
            rs = self.rows(candidates)
            if not rs or any(not 0 <= i < len(rs) for i in good):
                raise ValueError(f"chosen {chosen!r} is not among {len(rs)} candidate(s)")
            rows += rs
            answers += ["yes" if i in good else "no" for i in range(len(rs))]
        if len(set(answers)) < 2:
            raise ValueError("the steps need a chosen candidate and at least one other")
        self.head.fit(rows, answers, self._names(rows))
        self.n = len(steps)
        return self

    def scores(self, candidates):
        """The head's score of each candidate being the one to take (higher: better), in the candidates' order."""
        return [float(self.head.scores(r)[0]) for r in self.rows(candidates)]

    def choose(self, candidates):
        """→ (the index of the best candidate, [probability per candidate]) — ties go to the earlier candidate."""
        s = np.array(self.scores(candidates))
        if not len(s):
            raise ValueError("nothing to choose from")
        p = np.exp(4.0 * (s - s.max()))
        p = p / p.sum()
        return int(np.argmax(s)), [float(x) for x in p]

    def teach(self, candidates, chosen):
        """One correction: this candidate (index, or a set of them) was the one to take. → ms."""
        good = {chosen} if isinstance(chosen, int) else set(chosen)
        return sum(self.head.update(r, "yes" if i in good else "no") for i, r in enumerate(self.rows(candidates)))

    def fingerprint(self):
        from .provenance import digest
        return digest("CandidateHead", self.features, self.relative, self.head.fingerprint())
