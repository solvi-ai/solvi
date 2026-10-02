"""Documents by description: review a contract with fields defined only in words — no labeled examples. The general extractor
reads the whole contract in windows, finds each described clause (or says it is absent) and cites it; rules turn the clauses
into typed answers. Needs `pip install "solvi[model]"`.

Run:  uv run --extra model python examples/08_contracts_by_description.py
Model: SOLVI_MODEL (a Hugging Face id or a local directory), default solvi-ai/extract-base.
For production accuracy label ~100 documents and fine-tune (LongSpanExtractor.fit; benchmarks/datasets has the loaders)."""
from __future__ import annotations

import os
import re

from solvi import Answer, Catalog, Question, System
from solvi.extract_long import LongSpanExtractor
from solvi.show import show

CONTRACT = """MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is entered into as of March 3, 2021 by and between Northwind Analytics, Inc.,
a Delaware corporation ("Provider"), and Blue Harbor Logistics LLC ("Customer").

1. SERVICES. Provider shall provide the data processing services described in each Statement of Work.

2. FEES. Customer shall pay the fees set forth in each Statement of Work within thirty (30) days of the invoice date.

3. TERM AND TERMINATION. This Agreement commences on the Effective Date and continues for three (3) years. Either party may
terminate this Agreement for convenience upon sixty (60) days prior written notice to the other party.

4. CONFIDENTIALITY. Each party shall hold the other party's Confidential Information in strict confidence.

5. LIMITATION OF LIABILITY. In no event shall either party's aggregate liability arising out of this Agreement exceed the total
fees paid by Customer in the twelve (12) months preceding the claim.

6. NON-SOLICITATION. During the term and for one (1) year thereafter, neither party shall solicit the employees of the other.

7. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State of New York,
without regard to its conflict of laws principles.

8. MISCELLANEOUS. This Agreement constitutes the entire agreement between the parties.
"""

FIELDS = {"governing_law": "the clause that says which state's or country's law governs the contract",
          "termination_notice": "the notice period required to terminate the agreement",
          "liability_cap": "a clause that caps or limits the amount of liability of a party",
          "non_compete": "a clause that restricts a party from competing in a business, market or area"}
STATES = ["Delaware", "New York", "California", "other"]
WORDS = {"thirty": 30, "sixty": 60, "ninety": 90, "fifteen": 15, "ten": 10}

extractor = LongSpanExtractor.load(os.environ.get("SOLVI_MODEL", "solvi-ai/extract-base"))
cat = Catalog()
for name, description in FIELDS.items():
    cat.extract(extractor.field(name, description))


@cat.fn
def notice_days(termination_notice):
    m = re.search(r"(\d+)\s*\)?\s*(day|month)", termination_notice) or re.search(r"(" + "|".join(WORDS) + r")", termination_notice)
    if not m:
        raise ValueError("no notice period in the clause")
    n = int(m.group(1)) if m.group(1).isdigit() else WORDS[m.group(1)]
    return n * 30 if "month" in termination_notice else n


@cat.rule("law")
def law(governing_law):
    if not governing_law:
        raise ValueError("no governing-law clause found")
    return next((s for s in STATES[:3] if s.lower() in governing_law.lower()), "other")


@cat.rule("notice_ok")
def notice_ok(notice_days):
    return notice_days >= 30                               # our policy: at least 30 days to find a replacement


@cat.rule("liability_capped")
def liability_capped(liability_cap):
    return bool(liability_cap)


@cat.rule("has_non_compete")
def has_non_compete(non_compete):
    return bool(non_compete)


QUESTIONS = [Question("law", "Which law governs?", Answer.choice(STATES)),
             Question("notice_ok", "Termination notice at least 30 days?", Answer.yes_no()),
             Question("liability_capped", "Is liability capped?", Answer.yes_no()),
             Question("has_non_compete", "Is there a non-compete?", Answer.yes_no())]

if __name__ == "__main__":
    show(System(cat, QUESTIONS).ask({"doc": CONTRACT}), cat)
