export default {
  id: "bol",
  title: "Bill of lading check",
  domain: "Logistics",
  why: String.raw`Freight forwarders read dense bills of lading to decide who pays for insurance, where the goods land and whether a shipment can be released. The consignee, port and Incoterm are quoted from the document; the Incoterm is mapped to responsibilities by a table in Python, and a restricted-port list is a hard check that holds the shipment whatever else is on the page. Honest note: all-caps, column-style B/Ls are hard for the general model; a few dozen labelled B/Ls fix that.`,
  fields: [
    ["consignee", "the name of the consignee"],
    ["port_of_discharge", "What is the port of discharge?"],
    ["incoterm", "What are the delivery terms?"],
    ["containers", "What is the container number?"],
  ],
  code: String.raw`TERMS = ["EXW", "FCA", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"]
INPUTS = {"restricted_ports": ["Bandar Abbas", "Latakia", "Nampo", "Sevastopol"]}

@cat.fn
def term(incoterm):
    need(incoterm, "an Incoterm")
    return next((t for t in TERMS if mentions(incoterm, t)), "unknown")

@cat.check(hard=True, then={"release": "hold"})
def port_allowed(port_of_discharge, restricted_ports):
    """hard: never release to a restricted port"""
    return not any(mentions(port_of_discharge, p) for p in restricted_ports)

@cat.rule("insurance_by")
def insurance_by(term):
    if term == "unknown":
        raise ValueError("Incoterm not recognised")
    return "seller" if term in ("CIF", "CIP", "DAP", "DPU", "DDP") else "buyer"

@cat.rule("release")
def release(consignee, containers):
    need(consignee, "the consignee")
    return "release" if found(containers) else "hold"

QUESTIONS = [
    Question("release", "Release the shipment?", Answer.choice(["release", "hold"]), requires=["port_allowed"]),
    Question("insurance_by", "Who arranges cargo insurance?", Answer.choice(["buyer", "seller"])),
]
`,
  docs: [
    {
      name: "Ocean B/L, Shanghai to Rotterdam, CIF",
      text: String.raw`BILL OF LADING FOR OCEAN TRANSPORT OR MULTIMODAL TRANSPORT                    B/L NO.: OOLU2715584930

Shipper: Ningbo Brightstar Housewares Co., Ltd., No. 88 Xinghai Road, Beilun District, Ningbo 315800, China

Consignee: Hollandia Home & Garden B.V., De Stelling 12, 3766 DB Soest, The Netherlands

Notify party: same as consignee, tel. +31 35 601 7788, attn. Mr. Pieter de Vries

Pre-carriage by: truck                         Place of receipt: Ningbo CFS
Vessel / voyage: OOCL Germany 102W              Port of loading: Shanghai, China
Port of discharge: Rotterdam, Netherlands       Place of delivery: Rotterdam CY

Container no.: OOLU8841207   Seal no.: CN4471829
  1,120 cartons stainless steel cookware sets, 11,480.00 kg, 66.500 cbm
Container no.: OOLU8841545   Seal no.: CN4471830
  960 cartons glass food storage containers, 9,215.00 kg, 64.200 cbm
2 x 40' HC containers said to contain 2,080 cartons
HS codes 7323.93 / 7013.49      Freight prepaid
Delivery terms: CIF Rotterdam (Incoterms 2020)   L/C no. ING-LC-2026-01184

Shipped on board: 11 Sep 2026                Place and date of issue: Shanghai, 11 Sep 2026
Number of original B/Ls: three (3)
Signed as carrier: Orient Overseas Container Line Ltd.
`,
    },
    {
      name: "Sea waybill to a restricted port, FOB",
      text: String.raw`SEA WAYBILL (NON-NEGOTIABLE)                                   WAYBILL NO.: MEDU-SW-4410962

Shipper: Anatolia Machine Parts San. Tic. A.S., Organize Sanayi Bolgesi 4. Cad. No: 17, Bursa, Turkey
Consignee: Golden Crescent Trading FZE, Warehouse 22, Jebel Ali Free Zone, Dubai, UAE
Notify: Same as consignee

Vessel / Voyage: MSC ARIANE / 638E
Port of loading: Gemlik, Turkey
Port of discharge: Bandar Abbas, Iran (via transhipment at Jebel Ali)
Place of delivery: Bandar Abbas CY

Marks and numbers: GCT/2026/77
Container: MSCU7719032 (40' standard), seal TR0881420
Goods: 18 crates of hydraulic pump spare parts and valve assemblies, 14,220 kg gross
Freight: collect
Delivery terms: FOB Gemlik (Incoterms 2020)

Shipped on board 19 September 2026 at Gemlik.
For the carrier: MSC Mediterranean Shipping Company S.A., as agent
`,
    },
  ],
};
