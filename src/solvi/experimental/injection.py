"""A calibrated detector of instruction-like text in tool outputs (experimental, opt-in, no shipped weights).

`solvi.core.deciders.perturb.injection_spans` is a fixed rule set: it flags ordinary text (an address, a payment line,
"please reply") and misses polite attacks without its words. This detector scores each sentence of a text with a
logistic model over two kinds of features — the rule families that sentence trips in the fixed detector (role,
override, address, direct, action; with and without the action rules) and hashed word uni- and bigrams — fitted on
labelled texts, and flags at a threshold calibrated on ordinary texts to a false-flag rate you choose.

    det = InjectionDetector().fit(texts, positive_spans)        # spans: per text, the injected [(start, end)] (or [])
    det.calibrate(ordinary_texts, max_flag_rate=0.05)
    det.spans(tool_output)                                       # → [(start, end)], like injection_spans

The fixed rules stay the default; fit and calibrate this on the outputs your own tools return. On one public
benchmark's injected tool outputs it found 76% of held-out injections against the rules' 23%, but only on the
injection family it was fitted on (a plain TF-IDF classifier did the same, and on AgentDojo-style injections the rules
found 78%, this 53%); see `solvi.experimental.STATUS["injection"]`.

Deterministic: stable hashing (crc32), full-batch L-BFGS from zeros. `to_dict()` / `from_dict()` keep a fitted
detector as JSON data (its non-zero weights and threshold) with a fingerprint."""
from __future__ import annotations

import hashlib
import json
import re
import zlib

import numpy as np

from . import warn_on_import

warn_on_import(__name__)

FAMILIES = ("role", "override", "address", "direct", "action")
_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_N_RULE = 2 * len(FAMILIES) + 2          # family × (actions on / off), "quoted", bias-like "any rule"


def _rule_features(sentence):
    from ..core.deciders.perturb import instruction_rule, quoted_instructions
    out = []
    for k, actions in enumerate((True, False)):
        r = instruction_rule(sentence, actions=actions)
        if r is not None:
            out.append(k * len(FAMILIES) + (FAMILIES.index(r) if r in FAMILIES else len(FAMILIES) - 1))
    if quoted_instructions(sentence, actions=True):
        out.append(2 * len(FAMILIES))
    if out:
        out.append(2 * len(FAMILIES) + 1)
    return out


def _sentences(text):
    """The units scored: perturb.sentences, and a long unit cut into windows of at most 400 characters at spaces
    (a JSON tool output is one line)."""
    from ..core.deciders.perturb import sentences
    out = []
    for a, b in sentences(text):
        while b - a > 400:
            cut = text.rfind(" ", a + 200, a + 400)
            cut = cut if cut > a else a + 400
            out.append((a, cut))
            a = cut
            while a < b and text[a] == " ":
                a += 1
        if b > a:
            out.append((a, b))
    return out


class InjectionDetector:
    """A per-sentence logistic score of "this sentence addresses the agent", with a calibrated threshold.

    dim: hashed n-gram buckets (2**18); l2: the penalty on the weights (not on the intercept); balanced: weigh the two
    classes equally in the fit (injected sentences are rare)."""

    def __init__(self, dim=2 ** 18, l2=1.0, balanced=True, max_iter=300):
        self.dim, self.l2, self.balanced, self.max_iter = int(dim), float(l2), bool(balanced), int(max_iter)
        self.w = None
        self.b = 0.0
        self.threshold = 0.5

    # --- features
    def _row(self, sentence):
        idx = [self.dim + j for j in _rule_features(sentence)]
        toks = [t.lower() for t in _WORD.findall(sentence)]
        grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
        counts = {}
        for g in grams:
            h = zlib.crc32(g.encode("utf-8")) % self.dim
            counts[h] = counts.get(h, 0) + 1
        vals = {h: 1.0 + np.log(c) for h, c in counts.items()}
        nrm = float(np.sqrt(sum(v * v for v in vals.values()))) or 1.0
        cols = list(vals) + idx
        data = [v / nrm for v in vals.values()] + [1.0] * len(idx)
        return cols, data

    def _matrix(self, sentences):
        from scipy.sparse import csr_matrix
        indptr, cols, data = [0], [], []
        for s in sentences:
            c, d = self._row(s)
            cols += c
            data += d
            indptr.append(len(cols))
        return csr_matrix((np.array(data, float), np.array(cols, np.int64), np.array(indptr, np.int64)),
                          shape=(len(sentences), self.dim + _N_RULE))

    # --- fitting
    @staticmethod
    def label_sentences(text, spans):
        """The scored units of a text with a label each: 1 when it overlaps an injected span by at least half of the
        unit or half of the span, else 0 → [(start, end, label)]."""
        out = []
        for a, b in _sentences(text):
            lab = 0
            for x, y in spans or ():
                ov = min(b, y) - max(a, x)
                if ov > 0 and (ov >= (b - a) / 2 or ov >= (y - x) / 2):
                    lab = 1
                    break
            out.append((a, b, lab))
        return out

    def fit(self, texts, spans):
        """texts and, per text, the injected spans [(start, end)] ([] for an ordinary text) → self."""
        from scipy.optimize import minimize
        sents, y = [], []
        for t, sp in zip(texts, spans):
            for a, b, lab in self.label_sentences(t, sp):
                sents.append(t[a:b])
                y.append(lab)
        if not sents or len(set(y)) < 2:
            raise ValueError("fit needs sentences of both kinds (injected and ordinary)")
        X, y = self._matrix(sents), np.asarray(y, float)
        n, pos = len(y), y.sum()
        sw = np.where(y == 1, n / (2 * pos), n / (2 * (n - pos))) if self.balanced else np.ones(n)
        d = X.shape[1]

        def f(theta):
            w, b = theta[:d], theta[d]
            z = X @ w + b
            p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
            loss = -(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / n + \
                0.5 * self.l2 * (w @ w) / n
            g = sw * (p - y) / n
            return loss, np.concatenate([X.T @ g + self.l2 * w / n, [g.sum()]])
        r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.w, self.b = r.x[:d], float(r.x[d])
        return self

    # --- scoring
    def sentence_scores(self, text):
        """[(start, end, p)] for each scored unit of the text."""
        if self.w is None:
            raise ValueError("the detector is not fitted")
        if not isinstance(text, str) or not text.strip():
            return []
        units = _sentences(text)
        if not units:
            return []
        z = self._matrix([text[a:b] for a, b in units]) @ self.w + self.b
        p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        return [(a, b, float(q)) for (a, b), q in zip(units, p)]

    def score(self, text):
        """The text's score: its highest sentence score (0 for an empty text)."""
        return max((p for _, _, p in self.sentence_scores(text)), default=0.0)

    def calibrate(self, ordinary_texts, max_flag_rate=0.05):
        """Set the threshold so that at most `max_flag_rate` of these ordinary (not injected) texts are flagged:
        the smallest threshold above the scores of all but that share of them. → self."""
        s = np.sort(np.array([self.score(t) for t in ordinary_texts], float))[::-1]
        if not len(s):
            raise ValueError("calibrate needs ordinary texts")
        k = int(np.floor(max_flag_rate * len(s)))          # how many may be flagged
        self.threshold = float(np.nextafter(s[k], 1.0)) if k < len(s) else 0.0
        return self

    def flags(self, text):
        return self.score(text) >= self.threshold

    def spans(self, text):
        """The flagged sentences, merged → [(start, end)] (the shape of injection_spans)."""
        out = []
        for a, b, p in self.sentence_scores(text):
            if p >= self.threshold:
                if out and a <= out[-1][1] + 1:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
        return out

    # --- persistence
    def to_dict(self):
        if self.w is None:
            raise ValueError("the detector is not fitted")
        nz = np.flatnonzero(np.abs(self.w) > 1e-9)
        d = {"dim": self.dim, "l2": self.l2, "balanced": self.balanced, "b": self.b, "threshold": self.threshold,
             "w": [[int(i), float(self.w[i])] for i in nz]}
        d["fingerprint"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
        return d

    @classmethod
    def from_dict(cls, d):
        det = cls(dim=d["dim"], l2=d["l2"], balanced=d["balanced"])
        det.w = np.zeros(det.dim + _N_RULE)
        for i, v in d["w"]:
            det.w[i] = v
        det.b, det.threshold = float(d["b"]), float(d["threshold"])
        return det


__all__ = ["FAMILIES", "InjectionDetector"]
