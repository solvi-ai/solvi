"""Long documents: find first, then decide. A text longer than the decider reads (its max_len) is cut into sections by code
(headings, then paragraphs, then sentences), a cheap lexical scorer (BM25, standard library only) — and optionally the
decider's own relevance — picks the few sections that bear on the question, the decider reads only those, and every
offset it gives (a span answer, an evidence quote) is mapped back into the full document.

    doc = LongDocument(contract, max_tokens=200)
    doc.sections                          # [Section(start, end, heading, index)], in document order, covering the text
    top = doc.select("How many days' notice does termination require?", k=3)
    win = doc.window([s for s, _ in top])  # the selected sections joined, in document order
    win.to_doc(12, 30)                     # an offset range in the window → the same text's range in the document

    part = decider.decision("notice", "Notice period for termination?", "contract", Span[str], long="retrieve", top_k=3)

With `long="retrieve"` a decision part does this whenever its text does not fit; the sections it read (offsets, heading,
score) are recorded in the decision's `extra["long"]`, so they are in the trace, hashed and re-checked by replay (the
selection is deterministic). Texts that fit are decided as before, with nothing recorded."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+", re.U)
_TOK = re.compile(r"\w+|[^\w\s]", re.U)
_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "by", "with", "as", "at", "this",
         "that", "it", "its", "from", "what", "which", "who", "how", "does", "do", "any", "there", "if", "not", "no", "yes",
         "will", "shall", "can", "may", "was", "were", "has", "have", "any"}
HEADING = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]+\S.*|(?:(?:article|section|clause|schedule|annex|appendix|part|chapter)\s+[\dIVXLC]+[.:)]?|§\s*\d+|"
    r"\d{1,2}(?:\.\d{1,2})*\.?|[IVXLC]{1,5}\.)[ \t]+\S.{0,100}|[A-Z][A-Z0-9 ,&'/()\-]{3,80})[ \t]*$", re.I | re.M)
_CAPS = re.compile(r"^[ \t]*[A-Z][A-Z0-9 ,&'/()\-]{3,80}[ \t]*$", re.M)


def approx_tokens(text):
    """A tokenizer-free estimate of a subword model's tokens: words and punctuation marks × 1.3 (rounded up)."""
    return math.ceil(len(_TOK.findall(text)) * 1.3)


def terms(text):
    """Lowercase word terms without stop words, a plural "s" stripped (termination / terminations match)."""
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


@dataclass
class Section:
    """doc[start:end]; `heading`: the heading line of the part of the document it belongs to (None before any heading)."""
    start: int
    end: int
    heading: str | None = None
    index: int = 0


@dataclass
class Window:
    """Selected sections joined in document order: `text`, and `pieces` [(window start, document start, length)]."""
    text: str
    pieces: list = field(default_factory=list)
    sections: list = field(default_factory=list)

    def to_doc(self, start, end):
        """A range of the window → the same characters' range in the document, or None when it spans two sections (or
        a separator between them)."""
        for w, d, n in self.pieces:
            if w <= start and end <= w + n:
                return d + (start - w), d + (end - w)
        return None


class BM25:
    """Okapi BM25 over a list of documents' terms (k1, b as usual); score(query_terms) → one score per document."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query):
        q = Counter(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * ln / (self.avg or 1.0))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class LongDocument:
    """A text as sections that each fit `max_tokens` (counted by `count`, default approx_tokens): split at headings (a
    Markdown "#", "1.", "1.2", "Article 5", "Section 3", "§ 4", a Roman numeral, an ALL-CAPS line), then at blank lines,
    then at sentence ends, then at spaces; the sections cover the text in order (only whitespace between them)."""

    def __init__(self, text, max_tokens=256, count=None):
        self.text = text
        self.max_tokens = max(8, int(max_tokens))
        self.count = count or approx_tokens
        self.sections = self._split()

    def __len__(self):
        return len(self.sections)

    def section_text(self, s):
        return self.text[s.start:s.end]

    # --- splitting
    def _split(self):
        t = self.text
        heads = [m for m in HEADING.finditer(t) if m.group().strip() and _is_heading(m, t)]
        bounds = [0] + [m.start() for m in heads if m.start() > 0] + [len(t)]
        titles = {m.start(): m.group().strip() for m in heads}
        out = []
        for a, b in zip(bounds, bounds[1:]):
            head = titles.get(a)
            for s, e in self._fit(a, b):
                s, e = _trim(t, s, e)
                if e > s:
                    out.append(Section(s, e, head, len(out)))
        return out

    def _fit(self, a, b):
        """[a, b) → ranges that fit max_tokens: whole, else by paragraphs, sentences, spaces (greedy, in order)."""
        a, b = _trim(self.text, a, b)
        if b <= a or self.count(self.text[a:b]) <= self.max_tokens:
            return [(a, b)] if b > a else []
        for sep in (r"\n[ \t]*\n", r"(?<=[.!?;])\s+", r"\s+"):
            cuts = [m.end() for m in re.finditer(sep, self.text[a:b])]
            parts, prev = [], a
            for c in cuts + [b - a]:
                if a + c > prev:
                    parts.append((prev, a + c))
                prev = a + c
            if len(parts) > 1:
                return self._group(parts)
        n = max(1, (b - a) // 2)                          # no space at all: halve
        return self._fit(a, a + n) + self._fit(a + n, b)

    def _group(self, parts):
        """Consecutive parts merged greedily while they fit (a part too big is split first, so a heading line joins the
        start of its body)."""
        flat = []
        for s, e in parts:
            flat += [(s, e)] if self.count(self.text[s:e]) <= self.max_tokens else self._fit(s, e)
        out, cur = [], None
        for s, e in flat:
            if cur is not None and self.count(self.text[cur[0]:e]) <= self.max_tokens:
                cur = (cur[0], e)
                continue
            if cur is not None:
                out.append(cur)
            cur = (s, e)
        if cur is not None:
            out.append(cur)
        return out

    # --- selection
    def scores(self, query):
        """BM25 of each section (its heading counts as part of it) for the query."""
        docs = [terms((s.heading or "") + "\n" + self.section_text(s)) for s in self.sections]
        return BM25(docs).score(terms(query))

    def select(self, query, k=3, budget=None, rerank=None, rerank_top=None):
        """The sections that bear on `query`, best first → [(Section, score)]: the top k by BM25 that fit `budget` tokens
        together (ties: the earlier section). rerank: a function [section text] → [relevance] (e.g. the decider's p(yes),
        see DecisionPart) applied to the best `rerank_top` (default 3k) by BM25; its score orders them instead (BM25 breaks
        ties). Sections with no query term at all are kept only when nothing matches (then the first ones)."""
        bm = self.scores(query)
        order = sorted(range(len(self.sections)), key=lambda i: (-bm[i], i))
        score = {i: bm[i] for i in order}
        if rerank is not None and order:
            top = order[: (rerank_top or 3 * k)]
            rel = rerank([self.section_text(self.sections[i]) for i in top])
            score = {i: float(r) for i, r in zip(top, rel)}
            order = sorted(top, key=lambda i: (-score[i], -bm[i], i))
        out, used = [], 0
        for i in order:
            if len(out) >= k:
                break
            n = self.count(self.section_text(self.sections[i]))
            if budget is not None and out and used + n > budget:
                continue
            out.append((self.sections[i], score[i]))
            used += n
        return out

    def window(self, sections, sep="\n\n"):
        """The sections joined in document order → Window (offsets map back with to_doc)."""
        parts, pieces, pos = [], [], 0
        chosen = sorted(sections, key=lambda s: s.start)
        for j, s in enumerate(chosen):
            if j:
                parts.append(sep)
                pos += len(sep)
            txt = self.section_text(s)
            pieces.append((pos, s.start, len(txt)))
            parts.append(txt)
            pos += len(txt)
        return Window("".join(parts), pieces, chosen)


def _is_heading(m, text):
    """A heading line is short and stands alone: an ALL-CAPS line needs a letter run, a numbered line must not be a list
    item inside a paragraph (the previous line is blank, or the line is itself short)."""
    line = m.group().strip()
    if len(line) > 110:
        return False
    if _CAPS.fullmatch(m.group()) and not re.search(r"[A-Z]{3}", line):
        return False
    before = text[:m.start()]
    prev = before.rstrip(" \t").rsplit("\n", 2)
    blank_before = m.start() == 0 or before.endswith("\n\n") or (len(prev) >= 2 and not prev[-1].strip())
    return blank_before or len(line) <= 60


def _trim(t, s, e):
    while s < e and t[s].isspace():
        s += 1
    while e > s and t[e - 1].isspace():
        e -= 1
    return s, e
