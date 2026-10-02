export default {
  id: "invoice",
  title: "Supplier invoice",
  domain: "Finance",
  why: String.raw`Accounts payable is where a confident wrong answer costs money: a mistyped IBAN sends a payment to the wrong account, an overdue invoice needs a person. The model finds the invoice number, due date, total and IBAN and cites them; a checksum on the IBAN and the due date are hard checks, so no model score can push such an invoice into the payment run.`,
  fields: [
    ["invoice_number", "the invoice number"],
    ["due_date", "the date on which payment is due"],
    ["total", "What is the total amount due?"],
    ["iban", "the IBAN bank account number for payment"],
  ],
  code: String.raw`INPUTS = {"approval_limit": 10000}         # above this a manager signs off

@cat.fn
def amount(total):
    return money(total)

@cat.fn
def due(due_date):
    return parse_date(due_date)

@cat.check(hard=True, then={"pay": "reject"})
def iban_checksum(iban):
    """hard: the IBAN must pass the mod-97 checksum"""
    return iban_ok(iban)

@cat.check(hard=True, then={"pay": "escalate"})
def not_overdue(due, today):
    """hard: an invoice past its due date goes to a person"""
    return today <= due

@cat.rule("pay")
def pay(amount, approval_limit, invoice_number, due, iban_checksum):
    # reads due and iban_checksum too: if either cannot be computed, the answer abstains instead of paying
    need(invoice_number, "the invoice number")
    return "schedule" if amount <= approval_limit else "escalate"

@cat.rule("days_left")
def days_left(due, today):
    d = (due - today).days
    return "overdue" if d < 0 else "under a week" if d < 7 else "a week or more"

QUESTIONS = [
    Question("pay", "Payment decision", Answer.choice(["schedule", "escalate", "reject"]),
             requires=["iban_checksum", "not_overdue"]),
    Question("days_left", "Time left to pay", Answer.choice(["overdue", "under a week", "a week or more"])),
]
`,
  docs: [
    {
      name: "German supplier, EUR, due next month",
      today: "2026-09-26",
      text: String.raw`Müller Präzisionsteile GmbH
Industriestraße 14 · 70565 Stuttgart · Deutschland
USt-IdNr.: DE 281 447 902 · Tel. +49 711 7894 220 · buchhaltung@mueller-praezision.de

INVOICE

Bill to:
Nordlicht Robotics B.V.
Attn: Accounts Payable
Keizersgracht 221, 1016 DV Amsterdam, Netherlands
VAT: NL 8634 12 790 B01

Invoice No.: MP-2026-04817
Invoice date: 15.09.2026
Customer No.: 10447
Your order: PO-NR-55120 of 02.09.2026
Delivery date: 12.09.2026
Payment terms: 30 days net, due on 15.10.2026

Pos  Article      Description                              Qty   Unit price      Amount
1    AL-6082-40   Aluminium flange, CNC milled, Ø 40 mm     120    € 18,40      € 2.208,00
2    ST-1.4301-M8 Stainless bracket M8, laser cut            300    € 4,75       € 1.425,00
3    SV-CAL       Calibration and 3D measurement report        1    € 390,00       € 390,00
4    PK-EUR       Packaging and pallet (EUR)                   2    € 28,00         € 56,00

Net amount                                                                     € 4.079,00
VAT 19 %                                                                         € 775,01
Total amount due                                                               € 4.854,01

Please transfer the total amount by the due date, quoting the invoice number.
Bank: Landesbank Baden-Württemberg
IBAN: DE89 3704 0044 0532 0130 00
BIC: SOLADEST600

Retention of title: the goods remain our property until paid in full.
Managing directors: Jürgen Müller, Anna Weiß · Amtsgericht Stuttgart HRB 734512
`,
    },
    {
      name: "UK agency, overdue",
      today: "2026-09-26",
      text: String.raw`BRIGHTWATER CREATIVE LTD
Unit 5, The Tannery, 91 Bermondsey Street, London SE1 3XF
Company no. 10938274 | VAT GB 274 5510 83

TAX INVOICE                                   Invoice number: BWC-1193
                                              Date of issue: 1 August 2026
                                              Payment due: 31 August 2026

To: Finance Department, Harbour & Vine Hotels plc, 12 Quayside, Bristol BS1 4RN

Project: Autumn brand campaign, phase 2 (PO 7731)

Description                                               Hours    Rate (£)    Amount (£)
Creative direction                                         18.0      120.00      2,160.00
Copywriting, 6 x hotel landing pages                       22.5       85.00      1,912.50
Photography, two-day shoot incl. retouching                 -           -        3,400.00
Social media assets (24 formats)                           16.0       75.00      1,200.00

Subtotal                                                                         8,672.50
VAT at 20%                                                                       1,734.50
TOTAL DUE                                                                     £ 10,407.00

Payment by bank transfer within 30 days of the date of issue.
Account name: Brightwater Creative Ltd
IBAN: GB82 WEST 1234 5698 7654 32   BIC: WESTGB22
Late payments may attract interest under the Late Payment of Commercial Debts (Interest) Act 1998.
`,
    },
    {
      name: "Invoice with a mistyped IBAN",
      today: "2026-09-26",
      text: String.raw`Atelier Lumière SARL
8 rue des Récollets, 75010 Paris, France · SIRET 812 334 567 00021 · TVA FR 41 812334567

FACTURE / INVOICE N° AL-26-0342
Date: 18/09/2026
Échéance / Due date: 18/10/2026

Client: Nordlicht Robotics B.V., Keizersgracht 221, 1016 DV Amsterdam

Désignation / Description                         Qté      P.U. HT        Total HT
Éclairage scénique, location 3 jours                1     1850,00 €       1850,00 €
Technicien lumière (jour)                           3       420,00 €       1260,00 €
Transport et installation                           1       310,00 €        310,00 €

Total HT                                                                 3420,00 €
TVA 20 %                                                                  684,00 €
Total TTC à payer                                                        4104,00 €

Règlement par virement / Payment by bank transfer:
IBAN : FR7630006000011234567890188
BIC : AGRIFRPP
Merci de votre confiance.
`,
    },
  ],
};
