"""The Banking77 decider, which is not solvi: the baseline's classifier (TF-IDF + logistic regression) with
nearest-neighbour signals, and an act head — the probability that the classifier's answer is right — trained to say
"wrong" on intents the classifier has never seen. `decider(rows, intents)` wraps both behind solvi's scorer protocol
(docs/decide_format.md), so the classifier is a solvi decision part with an act signal."""
import hashlib
import random

import numpy as np

TASK = "What is the customer's request about?"


def label(intent):
    """The option as the decider names it: the intent's name in words."""
    return intent.replace("_", " ")


class Clf:
    """The baseline's classifier (the same TF-IDF and LogisticRegression settings) plus nearest-neighbour signals."""

    def __init__(self, rows, C=20):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts, y = [r["text"] for r in rows], [r["intent"] for r in rows]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.X, self.CX = self.vec.fit_transform(texts), self.cvec.fit_transform(texts)
        self.lr = LogisticRegression(C=C, max_iter=2000).fit(self.X, y)
        self.classes = list(self.lr.classes_)
        self.y = np.array([self.classes.index(v) for v in y])
        self.unigrams = TfidfVectorizer(ngram_range=(1, 1)).fit(texts)

    def signals(self, texts, k=10):
        """→ (log-probabilities [n, K], features [n, 13]): the classifier's margins and entropy, how close the nearest
        training texts are (word and character n-grams) and whether they share the answer, length, known words."""
        X, CX = self.vec.transform(texts), self.cvec.transform(texts)
        raw = self.lr.decision_function(X)
        z = raw - raw.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        p = np.exp(logp)
        top = np.argsort(-p, 1)[:, :2]
        p1, p2 = p[np.arange(len(p)), top[:, 0]], p[np.arange(len(p)), top[:, 1]]
        rs = np.sort(raw, 1)
        ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
        feats = [np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)), p1 - p2, ent, rs[:, -1], rs[:, -1] - rs[:, -2]]
        for M, Q in ((self.X, X), (self.CX, CX)):
            S = (Q @ M.T).toarray()
            idx = np.argsort(-S, 1)[:, :k]
            pred = top[:, 0]
            feats += [S[np.arange(len(S)), idx[:, 0]], np.where(self.y[None, :] == pred[:, None], S, -1).max(1),
                      (self.y[idx] == pred[:, None]).mean(1)]
        an, voc = self.unigrams.build_analyzer(), self.unigrams.vocabulary_
        toks = [an(t) for t in texts]
        feats += [np.log1p([len(t) for t in toks]), np.array([np.mean([w in voc for w in t]) if t else 0.0 for t in toks])]
        return logp, np.stack(feats, 1)


def act_head(rows, intents, folds=3, seed=0):
    """A logistic regression P(the answer is right) from the signals. Its training rows come from `rows` alone: `folds`
    times a classifier is fitted without one group of intents and one part of the examples; the held-out examples of
    its own intents give right / wrong, the left-out intents give "wrong" (the classifier cannot name them)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    its = sorted(intents)
    random.Random(seed).shuffle(its)
    groups = [set(its[i::folds]) for i in range(folds)]
    rng = random.Random(seed + 1)
    part = {id(r): rng.randrange(folds) for r in rows}
    F, ok = [], []
    for j, g in enumerate(groups):
        c = Clf([r for r in rows if r["intent"] not in g and part[id(r)] != j])
        for rs in ([r for r in rows if r["intent"] not in g and part[id(r)] == j], [r for r in rows if r["intent"] in g]):
            logp, f = c.signals([r["text"] for r in rs])
            F.append(f)
            ok += [int(c.classes[i] == r["intent"]) for i, r in zip(logp.argmax(1), rs)]
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.concatenate(F), np.array(ok))


class Scorer:
    """The classifier and its act head behind solvi's scorer protocol: per item the log-probabilities of its options
    and the act logit."""

    def __init__(self, clf, head):
        self.clf, self.head = clf, head
        self.index = {label(c): i for i, c in enumerate(clf.classes)}
        self.model_id = "banking77/tfidf-lr+act"
        self._fp = hashlib.sha256(clf.lr.coef_.tobytes()).hexdigest()[:16]

    def fingerprint(self):
        return self._fp

    def logits(self, items):
        logp, F = self.clf.signals([it.text for it in items])
        p = np.clip(self.head.predict_proba(F)[:, 1], 1e-6, 1 - 1e-6)
        act = np.log(p / (1 - p))
        return [{"logits": logp[i, [self.index[o] for o in it.options]], "act": float(act[i])} for i, it in enumerate(items)]


def decider(rows, intents):
    """A solvi decision part over `intents`, fitted on `rows`: it never escalates by itself (min_act=0), so whatever
    gates it — a guarantee, an open-set gate — decides alone."""
    from solvi.core.deciders import DecideModel
    rows = [r for r in rows if r["intent"] in set(intents)]
    model = DecideModel(Scorer(Clf(rows), act_head(rows, intents)), meta={"format": "stand-in", "temperature": 1.0, "act": {}},
                        cache_size=10 ** 6)
    return model.decision("intent", TASK, "text", [label(i) for i in intents], min_act=0.0)
