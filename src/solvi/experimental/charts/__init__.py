"""Verified charts: a text with numbers → a chart in which every number is quoted from the text.

    from solvi.experimental.charts import chart, RuleProposer, LLMProposer
    run = chart(text, "revenue by region")                 # the rule-based proposer by default
    run = chart(text, proposer=LLMProposer("http://127.0.0.1:8080/v1", "qwen2.5-7b-instruct"))
    open("chart.svg", "w").write(run.output)               # None when nothing verified
    print(run.report())                                     # kept / dropped / changed, and why

The first specialist of `solvi.experimental.specialist`: a proposer writes a ChartSpec (a chart type, series of labelled values,
each value with its quote in the text, a unit, a scale, a title); the checker verifies every value — the quote is in
the text, it holds that number (thousands separators, decimals, "4.2 billion", "15%", "1 500 000 руб" are read; an
ambiguous "1.000" or "3 100" is not), with the chart's unit (percent is not percentage points, dollars are not euros,
"480 employees" is not 480 "stores") and scale — and the chart type against the data (a pie only for shares that add up
to 100% or to a stated total, a line needs two points, one unit per chart). A value that does not verify is not drawn:
the report says what was dropped and why, and the chart marks the gap. The renderer draws a deterministic, accessible
SVG from the verified values only, and the trace replays: the same checks, the same bytes."""
from __future__ import annotations

from ..specialist import Checked, Specialist
from .check import ChartChecker, canon_unit
from .propose import FixedProposer, LLMProposer, RuleProposer, spec_json_schema
from .render import PALETTE, Drawing, contrast, render_svg
from .spec import ChartSpec, Point, Series, SourceQuote, VerifiedChart, VerifiedPoint, VerifiedSeries
from .. import warn_on_import

warn_on_import(__name__)


class ChartSpecialist(Specialist):
    """Text → verified chart (see the module docs). decimal: "." or "," when the text's locale says which one is the
    decimal separator."""

    name = "charts"
    version = "1"
    spec_type = ChartSpec

    def __init__(self, proposer=None, *, decimal=None):
        super().__init__(proposer or RuleProposer())
        self.checker = ChartChecker(decimal=decimal)

    def check(self, spec, source) -> Checked:
        return self.checker(spec, source)

    def render(self, checked: Checked):
        return self.draw(checked).svg

    def draw(self, checked: Checked) -> Drawing:
        """The Drawing (the SVG with its layout: text boxes, fonts, notes) of a check's verified chart."""
        return render_svg(checked.verified, proposed=checked.meta.get("proposed"))


def chart(text, question=None, *, proposer=None, decimal=None):
    """A text (and a question) → Run: `.output` (the SVG, or None), `.report()`, `.issues`, `.trace`."""
    return ChartSpecialist(proposer, decimal=decimal).run(text, question)


__all__ = ["ChartChecker", "ChartSpec", "ChartSpecialist", "Drawing", "FixedProposer", "LLMProposer", "PALETTE", "Point",
           "RuleProposer", "Series", "SourceQuote", "VerifiedChart", "VerifiedPoint", "VerifiedSeries", "canon_unit",
           "chart", "contrast", "render_svg", "spec_json_schema"]
