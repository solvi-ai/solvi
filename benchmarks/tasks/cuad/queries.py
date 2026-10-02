"""The 41 CUAD clause kinds: the question the model reads and the words retrieval searches by — the contract's own
wording (docs/best_practices.md, "For documents, give retrieval the document's words"). Written by hand and checked on
dev only, by whether a gold clause is among the sections read."""

# category → words a clause of this kind is usually written with
QUERY = {
    "Document Name": "agreement contract this agreement entered into made by and between dated",
    "Parties": "by and between parties party hereinafter referred to as corporation company inc llc ltd entered into",
    "Agreement Date": "dated as of made and entered into this day of date agreement signed executed in witness whereof",
    "Effective Date": "effective date effective as of commence commencement shall become effective date hereof begins",
    "Expiration Date": "term expire expiration terminate initial term period of years shall continue until remain in effect anniversary end",
    "Renewal Term": "renew renewal automatically renewed extend extension successive additional term periods unless",
    "Notice Period To Terminate Renewal": "notice of non-renewal renew renewal written notice days prior to expiration unless either party gives notice intent not to renew",
    "Governing Law": "governed by construed in accordance with laws of the state governing law jurisdiction without regard conflict",
    "Most Favored Nation": "most favored no less favorable terms than any other customer third party favourable pricing lower price offered",
    "Non-Compete": "compete competing competitive competition shall not directly or indirectly engage business territory restrict",
    "Exclusivity": "exclusive exclusively exclusivity sole solely only shall not appoint other sell third parties requirements",
    "No-Solicit Of Customers": "solicit customers clients divert shall not induce customer business relationship",
    "Competitive Restriction Exception": "notwithstanding foregoing except exclusive compete restriction shall not prevent prohibit nothing herein limit right exception",
    "No-Solicit Of Employees": "solicit employees hire employ personnel shall not recruit induce employee consultants",
    "Non-Disparagement": "disparage disparaging disparagement negative statements reputation defame goodwill detrimental",
    "Termination For Convenience": "terminate at any time without cause for convenience upon days prior written notice for any reason or no reason",
    "Rofr/Rofo/Rofn": "right of first refusal first offer first negotiation option to purchase offer bona fide third party match",
    "Change Of Control": "change of control merger consolidation acquisition sale of all or substantially all assets controlling ownership acquired",
    "Anti-Assignment": "assign assignment transfer delegate without prior written consent successors assigns not assignable",
    "Revenue/Profit Sharing": "royalty royalties percent of net sales revenue share commission percentage profits gross receipts pay",
    "Price Restrictions": "price increase prices shall not exceed raise adjust pricing reduce discount cap on price change",
    "Minimum Commitment": "minimum purchase commitment at least quantity guaranteed minimum annual order shall purchase no less than quota",
    "Volume Restriction": "exceed maximum limit number of users volume threshold additional fee in excess capacity up to",
    "Ip Ownership Assignment": "assigns hereby assign ownership all right title and interest intellectual property work made for hire inventions shall own property of",
    "Joint Ip Ownership": "jointly owned joint ownership jointly own co-owned shared developed jointly intellectual property",
    "License Grant": "grants hereby grant license non-exclusive right to use licensed sublicense licensee licensor",
    "Non-Transferable License": "non-transferable nontransferable non-assignable license may not transfer sublicense",
    "Affiliate License-Licensor": "affiliates license licensor and its affiliates grant intellectual property owned by affiliates",
    "Affiliate License-Licensee": "affiliates license licensee and its affiliates grant to affiliates sublicense subsidiaries",
    "Unlimited/All-You-Can-Eat-License": "unlimited enterprise license unlimited number of users copies unrestricted use",
    "Irrevocable Or Perpetual License": "irrevocable perpetual license perpetuity royalty-free fully paid-up worldwide",
    "Source Code Escrow": "source code escrow escrow agent deposit release bankruptcy",
    "Post-Termination Services": "upon termination expiration after following termination survive continue return transition wind-down sell-off remaining inventory period",
    "Audit Rights": "audit inspect examine books and records right to audit accountant inspection during normal business hours",
    "Uncapped Liability": "limitation of liability shall not apply except gross negligence willful misconduct indemnification confidentiality breach consequential damages",
    "Cap On Liability": "limitation of liability in no event liable shall not exceed aggregate liability consequential incidental indirect damages maximum total amount paid",
    "Liquidated Damages": "liquidated damages termination fee penalty pay upon termination early termination break-up fee",
    "Warranty Duration": "warranty warrants period days months from delivery defects free from defects workmanship warranty period",
    "Insurance": "insurance maintain policy coverage insured liability insurance certificate insurer additional insured",
    "Covenant Not To Sue": "not contest challenge validity ownership trademarks shall not sue covenant not to sue claim dispute attack rights",
    "Third Party Beneficiary": "third party beneficiary beneficiaries intended enforce no third party rights",
}


def details(question):
    return question.split("Details:")[-1].strip()


def task_of(row):
    """The question the model reads: the clause kind and CUAD's own explanation of it."""
    d = " ".join(details(row["question"]).split())
    return f'Which passage of the contract is the "{row["category"]}" clause a lawyer should review? ({d})'
