export default {
  id: "contract",
  title: "Services contract review",
  domain: "Legal",
  why: String.raw`A contract review checklist is a set of clauses plus policy. Each clause is defined in plain English (no training, no labels), the model reads the whole contract in overlapping windows and quotes the clause or says it is absent, and Python turns the clause into a typed answer: the notice period is computed from the words "sixty (60) days" or "three (3) months" and compared with your 30-day policy. A missing governing-law clause makes that question abstain rather than guess a state.`,
  fields: [
    ["governing_law", "the clause that says which state's or country's law governs the contract"],
    ["termination_notice", "the notice period required to terminate the agreement"],
    ["liability_cap", "a clause that caps or limits the amount of liability of a party"],
    ["non_compete", "a clause that restricts a party from competing or operating in a business, market or area"],
  ],
  code: String.raw`STATES = ["Delaware", "New York", "California", "Texas", "other"]

@cat.fn
def notice_days(termination_notice):
    return days(termination_notice)          # "sixty (60) days" -> 60, "three (3) months" -> 90

@cat.rule("law")
def law(governing_law):
    need(governing_law, "a governing-law clause")
    return next((s for s in STATES[:-1] if mentions(governing_law, s)), "other")

@cat.fn
def capped(liability_cap):
    return found(liability_cap)

@cat.fn
def restricted(non_compete):
    return found(non_compete)

@cat.rule("notice_ok")
def notice_ok(notice_days):
    return notice_days >= 30                 # our policy: at least 30 days to find a replacement

@cat.rule("liability_capped")
def liability_capped(capped):
    return capped

@cat.rule("has_non_compete")
def has_non_compete(restricted):
    return restricted

@cat.rule("sign")                            # a rule reads facts (not other answers)
def sign(notice_days, capped, restricted):
    if notice_days < 30 or not capped:
        return "negotiate"
    return "legal review" if restricted else "sign"

QUESTIONS = [
    Question("law", "Which law governs?", Answer.choice(STATES)),
    Question("notice_ok", "Termination notice at least 30 days?", Answer.yes_no()),
    Question("liability_capped", "Is liability capped?", Answer.yes_no()),
    Question("has_non_compete", "Is there a non-compete?", Answer.yes_no()),
    Question("sign", "Recommendation", Answer.choice(["sign", "legal review", "negotiate"])),
]
`,
  docs: [
    {
      name: "Master services agreement (2 pages)",
      text: String.raw`MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is made and entered into as of February 1, 2024 (the "Effective Date") by and
between Orbital Data Systems, Inc., a Delaware corporation with its principal place of business at 410 Market Street, Wilmington,
Delaware 19801 ("Provider"), and Greenfield Retail Group, LLC, an Ohio limited liability company with its principal place of
business at 2200 Lakeside Avenue, Cleveland, Ohio 44114 ("Customer"). Provider and Customer are each a "Party" and together the
"Parties".

1. DEFINITIONS
1.1 "Deliverables" means all reports, software configurations and other materials that Provider delivers to Customer under a
Statement of Work.
1.2 "Statement of Work" or "SOW" means a document signed by both Parties that describes specific Services, Deliverables, fees and
a schedule, and that references this Agreement.

2. SERVICES
2.1 Provider shall perform the data engineering, analytics and hosting services described in each Statement of Work (the
"Services") in a professional and workmanlike manner, using appropriately qualified personnel.
2.2 Each Statement of Work is governed by this Agreement. If a Statement of Work conflicts with this Agreement, this Agreement
controls unless the Statement of Work expressly states that it overrides a specific section of this Agreement.

3. FEES AND PAYMENT
3.1 Customer shall pay the fees set out in each Statement of Work. Provider shall invoice Customer monthly in arrears.
3.2 Customer shall pay each undisputed invoice within forty-five (45) days after the invoice date. Late payments bear interest at
the lesser of one percent (1%) per month or the maximum rate permitted by law.
3.3 Fees are exclusive of taxes. Customer is responsible for all sales, use and similar taxes, other than taxes on Provider's
net income.

4. TERM AND TERMINATION
4.1 This Agreement begins on the Effective Date and continues for an initial term of two (2) years. Thereafter it renews
automatically for successive one (1) year periods unless either Party gives written notice of non-renewal at least ninety (90)
days before the end of the then-current term.
4.2 Either Party may terminate this Agreement or any Statement of Work for convenience upon sixty (60) days' prior written notice
to the other Party.
4.3 Either Party may terminate this Agreement upon written notice if the other Party materially breaches this Agreement and fails
to cure the breach within thirty (30) days after receiving notice describing it.
4.4 Upon termination, Customer shall pay Provider for all Services performed through the effective date of termination.

5. CONFIDENTIALITY
5.1 Each Party shall hold the other Party's Confidential Information in strict confidence, use it only to perform or receive the
Services, and disclose it only to its employees and contractors who need to know it and are bound by written obligations at least
as protective as this Section.
5.2 The obligations in this Section 5 survive for three (3) years after the termination or expiration of this Agreement.

6. INTELLECTUAL PROPERTY
6.1 Upon full payment, Customer owns all Deliverables created specifically for Customer. Provider retains all rights in its
pre-existing tools, libraries and know-how and grants Customer a non-exclusive, perpetual license to use them as part of the
Deliverables.

7. WARRANTIES
7.1 Provider warrants that the Services will conform in all material respects to the applicable Statement of Work for ninety (90)
days after delivery. EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, NEITHER PARTY MAKES ANY OTHER WARRANTY, EXPRESS OR IMPLIED.

8. INDEMNIFICATION
8.1 Provider shall defend and indemnify Customer against any third-party claim alleging that the Deliverables infringe a United
States patent, copyright or trademark.

9. LIMITATION OF LIABILITY
9.1 EXCEPT FOR BREACH OF SECTION 5 OR A PARTY'S INDEMNIFICATION OBLIGATIONS, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY
ARISING OUT OF OR RELATING TO THIS AGREEMENT EXCEED THE TOTAL FEES PAID BY CUSTOMER TO PROVIDER UNDER THIS AGREEMENT IN THE TWELVE
(12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.
9.2 Neither Party shall be liable for any indirect, incidental, consequential or punitive damages, or for lost profits.

10. NON-SOLICITATION
10.1 During the term of this Agreement and for twelve (12) months thereafter, neither Party shall directly solicit for employment
any employee of the other Party who was involved in the Services, except through general advertisements.

11. GOVERNING LAW
11.1 This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its
conflict of laws principles. The state and federal courts located in New Castle County, Delaware have exclusive jurisdiction.

12. MISCELLANEOUS
12.1 Notices must be in writing and delivered by hand, by courier or by e-mail with confirmation of receipt.
12.2 Neither Party may assign this Agreement without the other Party's prior written consent, except to a successor in a merger
or sale of substantially all of its assets.
12.3 This Agreement, together with all Statements of Work, is the entire agreement between the Parties and supersedes all prior
agreements on its subject matter. It may be amended only in a writing signed by both Parties.

IN WITNESS WHEREOF, the Parties have executed this Agreement as of the Effective Date.

ORBITAL DATA SYSTEMS, INC.                    GREENFIELD RETAIL GROUP, LLC
By: Maria Chen, Chief Executive Officer        By: Daniel Okafor, Chief Operating Officer
`,
    },
    {
      name: "Supply agreement with a non-compete",
      text: String.raw`SUPPLY AGREEMENT

This Supply Agreement (the "Agreement") is dated June 15, 2023 (the "Effective Date") and is made between Redline Components Corp.,
a Texas corporation having its offices at 8800 Industrial Parkway, Houston, Texas 77041 ("Supplier"), and Apex Outdoor Equipment,
Inc., a Colorado corporation having its offices at 5100 Wynkoop Street, Denver, Colorado 80216 ("Buyer").

RECITALS
Supplier manufactures aluminum frame assemblies and related components. Buyer designs and sells camping and outdoor equipment and
wishes to purchase such components from Supplier on the terms set out below.

1. PRODUCTS AND FORECASTS
1.1 Supplier shall manufacture and sell to Buyer the products listed in Exhibit A (the "Products") in accordance with Buyer's
purchase orders.
1.2 Buyer shall provide a rolling twelve (12) month non-binding forecast of its requirements, updated monthly. The first three
(3) months of each forecast are binding.

2. ORDERS AND DELIVERY
2.1 Each purchase order shall specify Products, quantities, delivery dates and delivery location. Supplier shall accept or reject a
purchase order within five (5) business days.
2.2 Supplier shall deliver the Products FCA Supplier's facility (Incoterms 2020). Title and risk of loss pass to Buyer upon
delivery to Buyer's carrier.

3. PRICE AND PAYMENT
3.1 Prices for the Products are set out in Exhibit A and are fixed for the first twelve (12) months after the Effective Date.
Thereafter Supplier may adjust prices once per calendar year upon sixty (60) days' written notice, by no more than the change in
the Producer Price Index for aluminum products.
3.2 Buyer shall pay each invoice within thirty (30) days of the date of invoice by wire transfer.

4. QUALITY AND WARRANTY
4.1 Supplier warrants that for eighteen (18) months after delivery the Products will conform to the specifications in Exhibit B
and be free from defects in materials and workmanship.
4.2 Buyer's exclusive remedy for non-conforming Products is, at Supplier's option, repair, replacement or a refund of the price.

5. TERM AND TERMINATION
5.1 This Agreement has an initial term of three (3) years from the Effective Date and thereafter automatically renews for
successive one (1) year renewal terms unless terminated as provided below.
5.2 Either party may terminate this Agreement for convenience by giving the other party not less than three (3) months' prior
written notice.
5.3 Either party may terminate this Agreement immediately by written notice if the other party becomes insolvent, makes an
assignment for the benefit of creditors, or is the subject of bankruptcy proceedings.

6. NON-COMPETITION
6.1 During the term of this Agreement and for eighteen (18) months after its termination or expiration, Supplier shall not,
directly or through any affiliate, manufacture for or sell to any third party in North America any frame assembly that is
substantially similar to or competes with the Products designed for Buyer.

7. LIMITATION OF LIABILITY
7.1 In no event shall Supplier's total liability arising out of or in connection with this Agreement, whether in contract, tort or
otherwise, exceed the amounts paid by Buyer to Supplier under this Agreement during the six (6) months immediately preceding the
event giving rise to the claim.
7.2 Neither party shall be liable to the other for any loss of profits or for any indirect or consequential damages.

8. CONFIDENTIALITY
8.1 Each party shall keep confidential the specifications, prices and other non-public information received from the other party
and shall use such information only to perform this Agreement.

9. GOVERNING LAW AND DISPUTES
9.1 This Agreement shall be governed by the laws of the State of Texas, excluding the United Nations Convention on Contracts for
the International Sale of Goods. Disputes shall be resolved by binding arbitration in Houston, Texas.

10. GENERAL
10.1 Neither party is liable for any failure to perform caused by events beyond its reasonable control.
10.2 This Agreement, including its Exhibits, constitutes the entire agreement between the parties and supersedes all prior
negotiations and understandings.

Signed for and on behalf of REDLINE COMPONENTS CORP. by Luis Ortega, President
Signed for and on behalf of APEX OUTDOOR EQUIPMENT, INC. by Hannah Brooks, VP Procurement
`,
    },
    {
      name: "Short MSA, 15-day notice, no cap",
      text: String.raw`MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is entered into as of March 3, 2021 by and between Northwind Analytics, Inc.,
a Delaware corporation ("Provider"), and Blue Harbor Logistics LLC ("Customer").

1. SERVICES. Provider shall provide the data processing services described in each Statement of Work.

2. FEES. Customer shall pay the fees set forth in each Statement of Work within thirty (30) days of the invoice date.

3. TERM AND TERMINATION. This Agreement commences on the Effective Date and continues for three (3) years. Either party may
terminate this Agreement for convenience upon fifteen (15) days prior written notice to the other party.

4. CONFIDENTIALITY. Each party shall hold the other party's Confidential Information in strict confidence.

5. NON-SOLICITATION. During the term and for one (1) year thereafter, neither party shall solicit the employees of the other.

6. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State of New York,
without regard to its conflict of laws principles.

7. MISCELLANEOUS. This Agreement constitutes the entire agreement between the parties.
`,
    },
  ],
};
