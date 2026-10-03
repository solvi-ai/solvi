"""A verified chart: a text with numbers → an SVG in which every number is quoted from the text.

Chart makers (and LLMs) get numbers wrong: a digit swapped, a percentage drawn as a count, a share that was never in
the text, a pie of answers that add up to 108%. Here a proposer writes a typed chart spec with a quote for every value;
code checks each value against the text, draws only what verified, and says what it dropped and why.

  1. a press release → a pie of revenue by region (the rule-based proposer, no model);
  2. a quarterly table → a line of revenue, checked against the total the text states;
  3. a careless model (a stand-in that makes the usual mistakes) on the same press release: a swapped digit, a share
     that is not in the text, percentage points drawn as percent, a value with no quote, a pie that no longer adds up —
     each dropped or changed, with the reason;
  4. the trace: stored as JSON, replayed — the same checks and the same SVG bytes; an edited record is caught.

Swap the proposer for a model: `LLMProposer("http://127.0.0.1:8080/v1", "qwen2.5-7b-instruct")` (any OpenAI-compatible
server), or any function `(text, question) -> ChartSpec | dict`. The checks stay the same.

Run:  uv run python examples/21_verified_chart.py [OUT_DIR]      (the SVGs go to OUT_DIR, default a temporary folder)"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from solvi.experimental.charts import ChartSpecialist, FixedProposer, chart

PRESS = """ACME Corp. reports third-quarter 2025 results

BERLIN, 14 October 2025 — ACME Corp. today reported revenue of $4.2 billion for the third quarter, up 12% from a year
earlier. By region, Europe accounted for 42% of revenue, North America for 35% and Asia-Pacific for 23%. Operating
margin improved by 3 percentage points to 18%. Headcount grew 15% to 480 employees."""

QUARTERS = """Quarterly revenue, fiscal 2025 (unaudited)
Q1 2025: $1.2 million
Q2 2025: $1.5 million
Q3 2025: $1.9 million
Q4 2025: $2.3 million
Total revenue for the year was $6.9 million."""

CARELESS = {                      # what a careless model might propose for "revenue by region"
    "kind": "pie", "title": "ACME revenue by region, Q3 2025", "unit": "%",
    "series": [{"points": [
        {"label": "Europe", "value": 42, "quote": {"text": "Europe accounted for 42%"}},
        {"label": "North America", "value": 53, "quote": {"text": "North America for 35%"}},      # digits swapped
        {"label": "Asia-Pacific", "value": 23, "quote": {"text": "Asia-Pacific for 23%"}},
        {"label": "Latin America", "value": 7, "quote": {"text": "Latin America for 7%"}},       # not in the text
        {"label": "Margin gain", "value": 3, "quote": {"text": "3 percentage points"}},           # pp, not %
        {"label": "Other", "value": 2}]}]}                                                       # no quote


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    print("1. A press release → revenue by region")
    r1 = chart(PRESS, "revenue by region")
    print(r1.report())
    saved.append(out / "21_region_pie.svg")
    saved[-1].write_text(r1.output)

    print("\n2. A quarterly table → revenue by quarter, against the stated total")
    r2 = chart(QUARTERS, "quarterly revenue")
    print(r2.report())
    saved.append(out / "21_quarters_line.svg")
    saved[-1].write_text(r2.output)

    print("\n3. A careless model on the same press release")
    r3 = ChartSpecialist(FixedProposer(CARELESS, id="careless stand-in")).run(PRESS, "revenue by region")
    print(r3.report())
    saved.append(out / "21_careless_model.svg")
    saved[-1].write_text(r3.output)
    print(f"   drawn: {r3.checked.meta['verified']} of {r3.checked.meta['proposed']} proposed values; "
          f"kind: {r3.proposal.kind} → {r3.checked.verified.kind}")

    print("\n4. The trace: stored, replayed, tampered")
    record = json.loads(json.dumps(r3.to_dict()))
    sp = ChartSpecialist()
    rep = sp.replay(record, PRESS)
    print(f"   replay: {rep.ok} (the same checks, the same {len(r3.output.encode())} SVG bytes)")
    record["trace"][1]["data"]["spec"]["series"][0]["points"][1]["value"] = "35"      # someone "fixes" the record
    rep = sp.replay(record, PRESS)
    print(f"   replay of an edited record: {rep.ok} — {rep.problems[0]}")

    print("\nSVGs:")
    for p in saved:
        print(f"   {p}")
    return r1, r2, r3


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="solvi_charts_")))
