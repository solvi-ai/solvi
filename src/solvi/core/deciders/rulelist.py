"""Rules learned from examples instead of hand-written ones: an ordered "if feature then answer" list (a decision list) that
can be read and audited. Features are built from computed facts: words of a string, leading digits of numbers in a string
(codes), rounded numbers, category values. Fitting is greedy: at each step take the rule with the best precision on the
not-yet-covered examples (with support ≥ min_support) while precision ≥ min_precision; last comes the default answer (the most
frequent among the rest)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict


def words(text):
    """The upper-cased words and numbers of a text, in any script: runs of letters, digits, combining marks and
    apostrophes ("ул. Северная, 12" → УЛ, СЕВЕРНАЯ, 12; "Zürich" → ZÜRICH)."""
    text = unicodedata.normalize("NFC", text).upper()
    if text.isascii():
        return re.findall(r"[A-Z0-9']+", text)
    out, cur = [], []
    for c in text:
        if c == "'" or unicodedata.category(c)[0] in "LNM":
            cur.append(c)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def literals(row, facts, ngrams=1):
    out = set()
    for f in facts:
        v = row.get(f)
        if v is None:
            continue
        if isinstance(v, bool):
            out.add(f"{f} is {v}")
        elif isinstance(v, (int, float)):
            out.add(f"{f} ≈ {round(float(v), 0):g}")
        elif isinstance(v, str):
            tokens = words(v)
            for width in range(2, ngrams + 1):
                for start in range(len(tokens) - width + 1):
                    out.add(f"{f} has '{' '.join(tokens[start:start + width])}'")
            for t in tokens:
                out.add(f"{f} has '{t}'")
                if t.isdigit() and len(t) >= 5:
                    out.add(f"{f} has number starting '{t[:2]}'")
        else:
            out.add(f"{f} = {v!r}")
    return out


class RuleList:
    def __init__(self, facts, min_support=3, min_precision=0.8, max_rules=40, *, max_literals=1, ngrams=1,
                 negations=False, confirm=None):
        for name, value in (("max_literals", max_literals), ("ngrams", ngrams)):
            if type(value) is not int or not 1 <= value <= 3:
                raise ValueError(f"{name} must be an integer from 1 to 3")
        if type(negations) is not bool:
            raise TypeError("negations must be bool")
        if confirm is not None and (not isinstance(confirm, str) or not confirm or confirm in facts):
            raise ValueError("confirm must name a separate proposal fact, outside the learned features")
        self.confirm = confirm
        self.max_literals, self.ngrams, self.negations = max_literals, ngrams, negations
        self.facts, self.min_support, self.min_precision, self.max_rules = facts, min_support, min_precision, max_rules
        self.rules, self.default, self.truncated = [], None, 0
        self.quality_mode = False
        self.quality = None

    def _proposal(self, row):
        import math
        proposal = row.get(self.confirm)
        if not isinstance(proposal, dict) or "value" not in proposal:
            return None
        confidence = proposal.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1 or proposal["value"] is None):
            return None
        return proposal["value"], float(confidence)

    def calibrate(self, rows, answers):
        """Per-firing-rule smoothed precision on independent, explicitly held-out rows."""
        if not rows or len(rows) != len(answers):
            raise ValueError("rule calibration needs held-out rows and one answer per row")
        counts = [[0, 0, 0, 0] for _ in range(len(self.rules) + 1)]
        for row, answer in zip(rows, answers):
            value, rule = self.predict(row)
            bucket = self.rules.index(rule) if rule is not None else len(self.rules)
            if self.confirm is not None:
                proposal = self._proposal(row)
                if proposal is None:
                    counts[bucket][3] += 1
                    continue
                if proposal[0] != value:
                    counts[bucket][2] += 1
                    continue
            counts[bucket][0] += value == answer
            counts[bucket][1] += 1
        quality = [{"correct": k, "n": n, "act": (k + 1) / (n + 2) if n else None,
                    "disagreed": disagreed, "unscored": unscored} for k, n, disagreed, unscored in counts]
        self.quality = quality
        return quality

    def decision(self, row):
        from ..catalog import Decision
        value, rule = self.predict(row)
        bucket = self.rules.index(rule) if rule is not None else len(self.rules)
        quality = None if self.quality is None else self.quality[bucket]
        act = None if quality is None else quality["act"]
        reason = "no independent held-out support for this rule" if act is None else None
        if self.confirm is not None:
            proposal = self._proposal(row)
            if proposal is None or proposal[0] != value:
                reason = "no scored model proposal" if proposal is None else "rule and model disagree"
                act = None
            elif act is not None:
                act = min(act, proposal[1])
        return Decision(value, confidence=0.0 if act is None else act,
                        escalate=reason,
                        extra={"act": 0.0 if act is None else act, "rule_quality": quality, "rule_index": bucket})

    def fingerprint(self):
        from ..provenance import digest
        return digest("RuleList", self.facts, self.rules, self.default, self.max_literals, self.ngrams,
                      self.negations, self.confirm, self.quality_mode, self.quality)

    def fit(self, rows, answers):
        """Learn the list from rows of facts and their answers; a second fit starts over (it replaces the list). When
        the list stops at max_rules while another rule would still pass min_precision, a warning says how many
        examples were left to the default answer (and `truncated` is that number; 0 otherwise)."""
        if not rows or len(rows) != len(answers):
            raise ValueError(f"a rule list is fitted on examples, one answer for each: got {len(rows)} rows and "
                             f"{len(answers)} answers")
        self.rules, self.default, self.truncated = [], None, 0
        self.quality = None
        L = [literals(r, self.facts, self.ngrams) for r in rows]
        if self.max_literals > 1 or self.negations or self.ngrams > 1:
            return self._fit_conditions(L, answers)
        remaining = list(range(len(rows)))
        while remaining:
            best = self._best(L, answers, remaining)
            if best is None or best[0][0] < self.min_precision:
                break
            if len(self.rules) >= self.max_rules:     # another rule would pass: say the list was cut, not finished
                self.truncated = len(remaining)
                import warnings
                warnings.warn(f"the rule list stopped at max_rules={self.max_rules} with {len(remaining)} of {len(rows)} "
                              f"examples left to the default answer while more rules pass min_precision "
                              f"{self.min_precision:g} (the next: if {best[1]} → {best[2]!r}, {best[3]}/{best[4]}); "
                              "raise max_rules to learn them", UserWarning, stacklevel=2)
                break
            _, lit, a, k, n = best
            self.rules.append({"if": lit, "then": a, "support": k, "covered": n})
            remaining = [i for i in remaining if lit not in L[i]]
        self.default = Counter(answers[i] for i in remaining).most_common(1)[0][0] if remaining else Counter(answers).most_common(1)[0][0]
        return self

    def _best(self, L, answers, remaining):
        """The rule with the best smoothed precision on the remaining examples → ((precision, support), literal,
        answer, support, covered), or None."""
        cnt = defaultdict(Counter)
        for i in remaining:
            for lit in L[i]:
                cnt[lit][answers[i]] += 1
        best = None
        for lit, c in sorted(cnt.items()):          # sorted: ties break the same way on every run
            a, k = c.most_common(1)[0]
            n = sum(c.values())
            if k < self.min_support:
                continue
            prec = (k + 1) / (n + 2)                     # smoothed precision
            key = (prec, k)
            if best is None or key > best[0]:
                best = (key, lit, a, k, n)
        return best

    def _fit_conditions(self, rows, answers):
        """Greedy ordered conjunctions: bounded beam search, not exhaustive CN2/RIPPER."""
        import warnings

        remaining = set(range(len(rows)))
        vocabulary = sorted(set().union(*rows))
        postings = {lit: {i for i, row in enumerate(rows) if lit in row} for lit in vocabulary}
        terms = [(lit, False) for lit in vocabulary]
        if self.negations:
            terms += [(lit, True) for lit in vocabulary]
        terms.sort()
        while remaining:
            beam, best = [((), remaining)], None
            for depth in range(self.max_literals):
                candidates = []
                for condition, covered in beam:
                    for term in terms:
                        if condition and term <= condition[-1]:
                            continue
                        if any(lit == term[0] for lit, _ in condition):
                            continue
                        hit = covered - postings[term[0]] if term[1] else covered & postings[term[0]]
                        if len(hit) < self.min_support:
                            continue
                        counts = Counter(answers[i] for i in sorted(hit))
                        answer, support = counts.most_common(1)[0]
                        if support < self.min_support:
                            continue
                        cond = condition + (term,)
                        rank = (-(support + 1) / (len(hit) + 2), -support, len(cond), cond)
                        candidates.append((rank, cond, hit, answer, support))
                candidates.sort(key=lambda c: c[0])
                if not candidates:
                    break
                if best is None or candidates[0][0] < best[0]:
                    best = candidates[0]
                beam = [(c[1], c[2]) for c in candidates[:32]]
            if best is None or -best[0][0] < self.min_precision:
                break
            if len(self.rules) >= self.max_rules:
                self.truncated = len(remaining)
                warnings.warn(f"the rule list stopped at max_rules={self.max_rules} with {self.truncated} examples "
                              "left while more rules pass min_precision", UserWarning, stacklevel=2)
                break
            _, condition, hit, answer, support = best
            positive = tuple(lit for lit, negative in condition if not negative)
            negative = tuple(lit for lit, negated in condition if negated)
            display = " AND ".join(("NOT " if negated else "") + lit for lit, negated in condition)
            self.rules.append({"if": display, "all": positive, "unless": negative, "then": answer,
                               "support": support, "covered": len(hit)})
            remaining -= hit
        self.default = Counter(answers[i] for i in sorted(remaining)).most_common(1)[0][0] if remaining else Counter(answers).most_common(1)[0][0]
        return self

    def predict(self, row):
        ls = literals(row, self.facts, self.ngrams)
        for r in self.rules:
            matches = (all(lit in ls for lit in r["all"]) and not any(lit in ls for lit in r["unless"])) \
                if "all" in r else r["if"] in ls
            if matches:
                return r["then"], r
        return self.default, None

    def __str__(self):
        lines = [f"{i + 1:2d}. if {r['if']} → {r['then']}   ({r['support']}/{r['covered']})" for i, r in enumerate(self.rules)]
        return "\n".join(lines + [f"else → {self.default}"])


__all__ = ["literals", "RuleList", "words"]
