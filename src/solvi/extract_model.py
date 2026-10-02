"""Deprecated (0.8; removed in 0.9): `SpanExtractor` is gone — use `solvi.extract_long.LongSpanExtractor`.

`SpanExtractor` cut one field per forward pass and had no caller; `LongSpanExtractor` does the same (a field by its
description, two pointer heads) over windows of any length, with "no answer" thresholds and save / load. Migration:
`SpanExtractor(model_name, max_len)` → `LongSpanExtractor(model_name, max_len)`; `fit([(text, desc, (s, e) | None)])`
→ `fit([(text, desc, (s, e) | None)])`; `predict([(text, desc)])[0]` → `predict(text, desc)[:2]` plus its score; the
`@cat.extract` wrapper → `field(name, desc)`. Importing this module warns."""
from __future__ import annotations

from . import _deprecate
from .extract_long import LongSpanExtractor

_deprecate.renamed("solvi.extract_model.SpanExtractor", "solvi.extract_long.LongSpanExtractor", stacklevel=3)

SpanExtractor = LongSpanExtractor

__all__ = ["SpanExtractor"]
