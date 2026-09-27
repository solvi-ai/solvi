"""Answer head for questions without a rule: features from computed_state → answer probabilities.
A number is encoded as its value plus thresholds at training quantiles ("greater than threshold" steps).
Multinomial logistic regression with L2 (Adam, 300 steps — milliseconds; one-vs-rest ridge could not express the middle one
of ordered classes). Features are chosen by greedy forward selection on 5-fold cross-validation accuracy (gain ≥ 1 pt):
the selected facts define the question's flow."""
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
