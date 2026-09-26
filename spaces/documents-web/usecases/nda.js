export default {
  id: "nda",
  title: "NDA key terms",
  domain: "Legal",
  why: String.raw`NDAs pile up in sales and partnership pipelines, and most need the same four facts checked against a playbook: who signed, when, which law, and how long. Here every fact is a quoted span, the term is turned into years by Python, and the playbook is three lines of code you can change. The documents never leave your laptop, which matters for exactly this kind of paper.`,
  fields: [
    ["effective_date", "the date when the agreement was signed or entered into"],
    ["jurisdiction", "the state or country whose law governs the agreement"],
    ["parties", "the names of the parties that enter into the agreement"],
    ["term", "how long the agreement remains in effect"],
  ],
  code: String.raw`ACCEPTED_LAW = ["California", "New York", "Delaware", "England"]

@cat.fn
def term_years(term):
    return days(term) / 365                  # "two (2) years" -> 2.0

@cat.fn
def signed_on(effective_date):
    return parse_date(effective_date)

@cat.rule("term_ok")
def term_ok(term_years):
    return term_years <= 3                   # playbook: at most 3 years

@cat.rule("law_ok")
def law_ok(jurisdiction):
    need(jurisdiction, "the governing law")
    return any(mentions(jurisdiction, s) for s in ACCEPTED_LAW)

@cat.rule("recent")
def recent(signed_on, today):
    return (today - signed_on).days <= 365

QUESTIONS = [
    Question("term_ok", "Term at most 3 years?", Answer.yes_no()),
    Question("law_ok", "Governing law on our accepted list?", Answer.yes_no()),
    Question("recent", "Signed within the last 12 months?", Answer.yes_no()),
]
`,
  docs: [
    {
      name: "Mutual NDA (California)", today: "2026-03-01",
      text: String.raw`MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement (this "Agreement") is entered into on September 9, 2025 by and between Lumen Biotech, Inc.,
a California corporation located at 1550 Owens Street, San Francisco, California 94158 ("Lumen"), and Castellan Ventures LP, a
Massachusetts limited partnership located at 100 Federal Street, Boston, Massachusetts 02110 ("Castellan"). Each of Lumen and
Castellan may disclose or receive Confidential Information and is referred to as a "Disclosing Party" or a "Receiving Party".

1. PURPOSE. The parties wish to evaluate a possible investment by Castellan in Lumen (the "Purpose") and, in connection with the
Purpose, may disclose to each other confidential business, scientific and financial information.

2. CONFIDENTIAL INFORMATION. "Confidential Information" means any non-public information disclosed by a Disclosing Party to a
Receiving Party, in any form, that is marked as confidential or that a reasonable person would understand to be confidential,
including research data, formulations, clinical plans, financial projections and the terms of any proposed transaction.

3. EXCLUSIONS. Confidential Information does not include information that (a) is or becomes publicly available through no fault of
the Receiving Party; (b) was known to the Receiving Party before disclosure without a duty of confidentiality; (c) is received from
a third party without a duty of confidentiality; or (d) is independently developed by the Receiving Party without use of the
Disclosing Party's Confidential Information.

4. OBLIGATIONS. The Receiving Party shall (a) use Confidential Information solely for the Purpose; (b) protect it with at least
the same degree of care it uses for its own confidential information, and no less than reasonable care; and (c) disclose it only
to its directors, officers, employees and professional advisers who need to know it for the Purpose and who are bound by
confidentiality obligations no less protective than this Agreement.

5. COMPELLED DISCLOSURE. If the Receiving Party is required by law or court order to disclose Confidential Information, it shall,
where legally permitted, give the Disclosing Party prompt notice so that the Disclosing Party may seek a protective order.

6. RETURN OF MATERIALS. Upon the Disclosing Party's written request, the Receiving Party shall promptly return or destroy all
Confidential Information in its possession, except for copies retained in automatic electronic backups or as required by law.

7. NO LICENSE; NO OBLIGATION. Nothing in this Agreement grants either party any license under any patent, copyright or other
intellectual property right. Neither party is obligated to proceed with any transaction.

8. TERM AND TERMINATION. This Agreement remains in effect for two (2) years from the date first written above. Either party may
terminate this Agreement at any time upon fifteen (15) days' written notice to the other party. The Receiving Party's obligations
with respect to Confidential Information disclosed before termination survive for three (3) years after termination.

9. REMEDIES. Each party acknowledges that unauthorized disclosure of Confidential Information may cause irreparable harm for which
monetary damages would be an inadequate remedy, and that the Disclosing Party is entitled to seek injunctive relief in addition to
any other remedies available at law or in equity.

10. GOVERNING LAW. This Agreement is governed by the laws of the State of California, without giving effect to any choice of law
rules. Any dispute shall be brought exclusively in the state or federal courts located in San Francisco County, California.

11. GENERAL. This Agreement is the entire agreement of the parties regarding its subject matter. It may be amended only in a
writing signed by both parties, and may be executed in counterparts, including by electronic signature.

LUMEN BIOTECH, INC.                           CASTELLAN VENTURES LP
By: Priya Raman, Chief Financial Officer      By: Castellan GP LLC, its general partner
                                              By: Thomas Weller, Managing Partner
`,
    },
    {
      name: "One-way confidentiality agreement (England)", today: "2026-09-26",
      text: String.raw`CONFIDENTIALITY AGREEMENT

This Confidentiality Agreement (the "Agreement") is made on the 14th day of January 2026 (the "Effective Date")

BETWEEN

(1) Halvorsen Marine Robotics AS, a company incorporated in Norway with organisation number 921 554 310, whose registered
office is at Strandkaien 12, 5013 Bergen, Norway (the "Discloser"); and

(2) Kestrel Systems Integration Ltd, a company registered in England and Wales under number 08812457, whose registered office
is at 3 Wharf Road, London N1 7GR, United Kingdom (the "Recipient").

BACKGROUND
The Discloser intends to share technical information about its autonomous hull-inspection vehicles with the Recipient so that
the Recipient can prepare a proposal for integrating those vehicles into port-security systems (the "Project").

1. CONFIDENTIAL INFORMATION
1.1 "Confidential Information" means all information relating to the Project or to the Discloser's business, products,
software, designs, test results, customers and prices, disclosed by or on behalf of the Discloser before or after the date of
this Agreement, in any form.

2. OBLIGATIONS OF THE RECIPIENT
2.1 The Recipient shall keep the Confidential Information secret, shall not use it except for the Project, and shall not
disclose it to any person other than its employees and subcontractors who need to know it for the Project and who are bound
by written confidentiality obligations no less strict than these.
2.2 The Recipient shall not reverse engineer, decompile or analyse the composition of any sample, prototype or software
supplied by the Discloser.

3. DURATION
3.1 This Agreement shall continue in force for a period of five (5) years from the Effective Date, and the obligations in
clause 2 shall survive any termination for so long as the information remains confidential.

4. RETURN OF INFORMATION
4.1 Within ten (10) days of a written request, the Recipient shall return or destroy all documents and materials containing
Confidential Information and certify in writing that it has done so.

5. GOVERNING LAW AND JURISDICTION
5.1 This Agreement and any dispute or claim arising out of it shall be governed by and construed in accordance with the law of
England and Wales, and the courts of England and Wales shall have exclusive jurisdiction.

Signed by Ingrid Halvorsen, Managing Director, for and on behalf of Halvorsen Marine Robotics AS
Signed by Oliver Grant, Director, for and on behalf of Kestrel Systems Integration Ltd
`,
    },
  ],
};
