export default {
  id: "receipt",
  title: "Receipt expense check",
  domain: "Expenses",
  altModel: "receipts",
  why: String.raw`An expense policy is arithmetic plus a few hard lines. The model only has to find four values on a noisy OCR receipt and cite them; plain Python then computes the change, checks the age of the receipt and applies the limit. A receipt paid by card has no cash line, so "is the change right?" abstains instead of inventing a number, and a receipt older than 90 days is refused by a hard check whatever the amount.`,
  fields: [
    ["receipt_date", "the date of the purchase"],
    ["total", "the total amount paid"],
    ["cash", "the cash amount given by the customer"],
    ["change", "What is the change?"],
  ],
  code: String.raw`# Fields above arrive as strings (empty when absent). Helpers: money, parse_date, days, found, need, mentions ...
INPUTS = {"limit": 200}                      # reimbursement limit, in the receipt's currency

@cat.fn
def amount(total):
    return money(total)

@cat.fn
def purchase_date(receipt_date):
    return parse_date(receipt_date)          # day first: 24/02/2018

@cat.check(hard=True, then={"reimburse": "no"})
def not_too_old(purchase_date, today):
    """hard: a receipt older than 90 days is never reimbursed"""
    return (today - purchase_date).days <= 90

@cat.rule("reimburse")
def reimburse(amount, limit, purchase_date):   # reads the date too: no date, no reimbursement decision
    return amount <= limit

@cat.rule("change_correct")
def change_correct(amount, cash, change):
    return abs(money(cash) - amount - money(change)) < 0.01

@cat.rule("weekend")
def weekend(purchase_date):
    return purchase_date.weekday() >= 5

QUESTIONS = [
    Question("reimburse", "Reimburse this receipt?", Answer.yes_no(), requires=["not_too_old"]),
    Question("change_correct", "Is the change right (cash - total)?", Answer.yes_no()),
    Question("weekend", "Bought on a weekend?", Answer.yes_no()),
]
`,
  docs: [
    {
      name: "Hardware store, paid cash",
      today: "2018-03-05",
      text: String.raw`SENG HUAT HARDWARE & TRADING SDN BHD
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
`,
    },
    {
      name: "Café, wrong change",
      today: "2018-05-02",
      text: String.raw`KOPI & KEK CAFE
KK F&B ENTERPRISE (SA0412345-X)
G-07, JALAN SS 15/4, SUBANG JAYA
47500 SELANGOR
TEL: 03-5612 8890
RECEIPT NO: KK-0098821
28/04/2018  16:05  TABLE 4
----------------------------------
2 FLAT WHITE           2 x 9.50   19.00
1 PANDAN CHIFFON SLICE            7.90
1 ICED LEMON TEA                  5.50
----------------------------------
SUBTOTAL                         32.40
SVC CHARGE 10%                    3.24
ROUNDING                         -0.04
TOTAL                            35.60
CASH                             50.00
CHANGE                            4.40
----------------------------------
THANK YOU & SEE YOU AGAIN
`,
    },
    {
      name: "Restaurant, paid by card (no cash line)",
      today: "2019-01-20",
      text: String.raw`RESTORAN MAJU JAYA NASI KANDAR
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
`,
    },
    {
      name: "Mini market, submitted too late",
      today: "2018-06-15",
      text: String.raw`HENG HENG MINI MARKET
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
`,
    },
  ],
};
