"""Sample documents for the solvi documents Space. All companies, people and numbers are made up.

RECEIPTS: OCR-style receipt texts (Malaysian SROIE-like and Indonesian CORD-like), each with the "today" date and
reimbursement limit that make the demo questions interesting.
CONTRACTS: synthetic contracts (services agreement, NDA, supply agreement)."""

RECEIPTS = {
    "Malaysia: hardware store, GST tax invoice": {
        "today": "2018-03-05",
        "limit": 200,
        "currency": "RM",
        "text": """SENG HUAT HARDWARE & TRADING SDN BHD
(Co. Reg. No. 764521-P)
NO. 12, JALAN PERUSAHAAN 2,
TAMAN PERINDUSTRIAN BUKIT SERDANG,
43300 SERI KEMBANGAN, SELANGOR.
TEL : 03-8945 2231   FAX : 03-8945 2232
GST REG NO : 000812345664
TAX INVOICE
INVOICE NO : CS 18/02331
DATE : 24/02/2018        TIME : 10:42 AM
CASHIER : SITI
------------------------------------------
DESCRIPTION            QTY   PRICE   AMOUNT
PVC PIPE 1/2" X 10FT     4    6.50    26.00 SR
ELBOW 1/2" PVC          10    0.60     6.00 SR
SILICONE SEALANT CLEAR   2   12.90    25.80 SR
MASKING TAPE 24MM        3    2.80     8.40 SR
------------------------------------------
TOTAL QTY : 19
SUB TOTAL (EXCL GST)            62.45
GST @6%                          3.75
TOTAL (INCL GST)                66.20
ROUNDING                         0.00
TOTAL AMOUNT PAYABLE      RM    66.20
CASH                           100.00
CHANGE                          33.80
GST SUMMARY    AMOUNT(RM)   TAX(RM)
SR @ 6%           62.45        3.75
GOODS SOLD ARE NOT RETURNABLE. THANK YOU.
""",
    },
    "Malaysia: restaurant, paid by card": {
        "today": "2019-01-20",
        "limit": 30,
        "currency": "RM",
        "text": """RESTORAN MAJU JAYA NASI KANDAR
MAJU JAYA FOOD SDN BHD (1045562-W)
LOT 3, JALAN TUANKU ABDUL RAHMAN
50100 KUALA LUMPUR
SST ID : W10-1808-32000154
SIMPLIFIED TAX INVOICE
Table: 12        Pax: 3
Bill No: 00018842
15-01-2019 13:27:05   Cashier: AZMAN
1 NASI KANDAR AYAM            9.50
1 NASI KANDAR KAMBING        13.00
1 ROTI CANAI TELUR            2.80
2 TEH TARIK                   4.40
1 LIMAU AIS                   2.20
Sub Total                    31.90
Service Tax 6%                1.91
Rounding Adj                 -0.01
Total                        33.80
VISA **** **** **** 4417     33.80
Thank you. Please come again!
""",
    },
    "Malaysia: mini market, submitted late": {
        "today": "2018-06-15",
        "limit": 200,
        "currency": "RM",
        "text": """HENG HENG MINI MARKET
(002345678-U)
NO 8, JALAN PASAR BARU
81200 JOHOR BAHRU, JOHOR
TEL: 07-332 8812
RECEIPT #: 2-04413
Date: 03 Nov 2017   19:05
-----------------------------------
MILO 3IN1 18X33G        1    16.90
GARDENIA WHITE BREAD    1     3.80
DUTCH LADY UHT 1L       2     7.60
EGGS GRADE A 10S        1     5.20
-----------------------------------
ITEMS: 5
TOTAL (RM)                   33.50
GST ZR 0%                     0.00
CASH                         50.00
CHANGE                       16.50
*** THANK YOU, PLEASE COME AGAIN ***
""",
    },
    "Indonesia: coffee shop (CORD-like)": {
        "today": "2019-08-20",
        "limit": 250000,
        "currency": "IDR",
        "text": """KEDAI KOPI SENJA
Jl. Kemang Raya No. 45, Jakarta Selatan
17/08/2019 15:12        No: 0087
Kasir: DEWI
2 ES KOPI SUSU AREN          44,000
1 CAFE LATTE HOT             28,000
1 CROISSANT BUTTER           22,000
   x EXTRA CHEESE             5,000
SUBTOTAL                     99,000
PB1 10%                       9,900
TOTAL                       108,900
CASH                        150,000
CHANGE                       41,100
TERIMA KASIH
""",
    },
    "Indonesia: bakery, wrong change (CORD-like)": {
        "today": "2019-03-08",
        "limit": 250000,
        "currency": "IDR",
        "text": """ROTI MANIS BAKERY
Ruko Sentra Niaga Blok B/7
Bandung
Tgl 05-03-2019   Jam 08:31
--------------------------------------
Roti Sobek Coklat  1 x 18.500    18.500
Donat Gula         4 x  6.000    24.000
Bolu Pandan        1 x 32.000    32.000
--------------------------------------
Sub Total                        74.500
Pajak (incl.)                     6.773
Total                     Rp     74.500
Tunai                     Rp    100.000
Kembali                   Rp     26.500
""",
    },
}

CONTRACTS = {
    "Master services agreement": """MASTER SERVICES AGREEMENT

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
""",
    "Mutual non-disclosure agreement": """MUTUAL NON-DISCLOSURE AGREEMENT

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
""",
    "Supply agreement": """SUPPLY AGREEMENT

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
""",
}

CONTRACT_FIELDS = [
    ["effective_date", "the date when the contract was signed or entered into"],
    ["parties", "the names of the parties that enter into the agreement"],
    ["payment_terms", "the number of days within which invoices must be paid"],
    ["auto_renewal", "a clause saying that the agreement renews automatically"],
    ["confidentiality_survival", "how long the confidentiality obligations survive after termination"],
]
