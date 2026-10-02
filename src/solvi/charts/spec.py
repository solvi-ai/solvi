"""ChartSpec: the typed intermediate a proposer writes — a chart type, series of labelled values, and for every value the
quote in the source it was read from."""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCALES = {"": 1, "thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}
KINDS = ("bar", "line", "pie")


class SourceQuote(BaseModel):
    """A passage of the source, copied character for character; `start` is its offset (optional: without it the
    checker looks the passage up, and one that occurs several times — as whole words and numbers — must verify at every
    such place, else the point is dropped: give `start`)."""
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)


class Point(BaseModel):
    """One value: a category label, the number (in the chart's scale) and the quote that states it."""
    model_config = ConfigDict(extra="forbid")
    label: str
    value: Decimal
    quote: SourceQuote | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _dec(cls, v):
        if isinstance(v, float):
            return Decimal(repr(v))
        if isinstance(v, str):
            v = v.strip()
            if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", v):     # "1,200" / "1,200.5": thousands commas only
                v = v.replace(",", "")
        return v

    @field_validator("value")
    @classmethod
    def _finite(cls, v):
        if not v.is_finite():
            raise ValueError("a value must be a finite number")
        return v


class Series(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    unit: str | None = None          # None: the chart's unit
    points: list[Point]


class ChartSpec(BaseModel):
    """kind: bar | line | pie. unit: "%", a currency ("USD", "$", "EUR", "₽" …), "pp", a word ("tonnes", "users") or ""
    for plain counts. scale: the values are in thousands / millions / billions of the unit ("" = as is). total: a total
    the source states for the values (checked like any value; the parts must add up to it)."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["bar", "line", "pie"]
    title: str = ""
    unit: str = ""
    scale: Literal["", "thousand", "million", "billion"] = ""
    series: list[Series] = Field(min_length=1)
    total: Point | None = None


# --------------------------------------------------------------------------------------------- what the checker passes on
class VerifiedPoint(BaseModel):
    """A value that verified: the number (in the chart's scale), where the source states it and how it is written there."""
    label: str
    value: Decimal
    start: int
    end: int
    as_written: str


class VerifiedSeries(BaseModel):
    name: str = ""
    points: list[VerifiedPoint | None]      # aligned with VerifiedChart.categories; None: proposed, not verified


class VerifiedChart(BaseModel):
    """The only input of the renderer: every number in it is in the source at [start, end)."""
    kind: Literal["bar", "line", "pie"]
    title: str
    unit: str                                # canonical: "%", "pp", "USD", a word, or ""
    unit_label: str = ""                     # the unit as the spec writes it ("tonnes"), for the caption
    scale: Literal["", "thousand", "million", "billion"] = ""
    categories: list[str]
    series: list[VerifiedSeries]
    total: VerifiedPoint | None = None


__all__ = ["ChartSpec", "KINDS", "Point", "SCALES", "Series", "SourceQuote", "VerifiedChart", "VerifiedPoint",
           "VerifiedSeries"]
