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


def literals(row, facts):
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
            for t in words(v):
                out.add(f"{f} has '{t}'")
                if t.isdigit() and len(t) >= 5:
                    out.add(f"{f} has number starting '{t[:2]}'")
        else:
            out.add(f"{f} = {v!r}")
    return out


class RuleList:
    def __init__(self, facts, min_support=3, min_precision=0.8, max_rules=40):
        self.facts, self.min_support, self.min_precision, self.max_rules = facts, min_support, min_precision, max_rules
        self.rules, self.default, self.truncated = [], None, 0

    def fit(self, rows, answers):
        """Learn the list from rows of facts and their answers; a second fit starts over (it replaces the list). When
        the list stops at max_rules while another rule would still pass min_precision, a warning says how many
        examples were left to the default answer (and `truncated` is that number; 0 otherwise)."""
        if not rows or len(rows) != len(answers):
            raise ValueError(f"a rule list is fitted on examples, one answer for each: got {len(rows)} rows and "
                             f"{len(answers)} answers")
        self.rules, self.default, self.truncated = [], None, 0
        L = [literals(r, self.facts) for r in rows]
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

    def predict(self, row):
        ls = literals(row, self.facts)
        for r in self.rules:
            if r["if"] in ls:
                return r["then"], r
        return self.default, None

    def __str__(self):
        lines = [f"{i + 1:2d}. if {r['if']} → {r['then']}   ({r['support']}/{r['covered']})" for i, r in enumerate(self.rules)]
        return "\n".join(lines + [f"else → {self.default}"])


__all__ = ["literals", "RuleList", "words"]
