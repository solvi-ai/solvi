export default {
  id: "insurance",
  title: "Insurance claim vs policy",
  domain: "Insurance",
  why: String.raw`A first-notice-of-loss desk compares a claim with the policy wording: is the cause excluded, what is the deductible, what is the limit. The model quotes the exclusions list, the limits and the claim itself; whether the cause is excluded is a hard check over the quoted exclusions, and the payout is computed, not generated. When the claim amount is missing the payout question abstains and the claim goes to a person.`,
  fields: [
    ["deductible", "What is the deductible?"],
    ["coverage_limit", "What is the limit of insurance for the dwelling?"],
    ["exclusions", "What losses are excluded from coverage?"],
    ["claim_cause", "the cause of the loss or damage being claimed"],
    ["claim_amount", "the amount of money claimed"],
  ],
  code: String.raw`PERILS = ["flood", "earthquake", "subsidence", "wear and tear", "gradual", "mould", "war", "terrorism", "storm"]

@cat.fn
def cause_words(claim_cause):
    return [p for p in PERILS if mentions(claim_cause, p)]

@cat.check(hard=True, then={"covered": "no", "payout_band": "none"})
def cause_not_excluded(cause_words, exclusions):
    """hard: a cause named in the exclusions is never covered"""
    return not any(mentions(exclusions, p) for p in cause_words)

@cat.fn
def payout(claim_amount, coverage_limit, deductible):
    return max(0.0, min(money(claim_amount), money(coverage_limit)) - money(deductible))

@cat.rule("covered")
def covered(payout):
    return payout > 0

@cat.rule("payout_band")
def payout_band(payout, claim_amount):
    if payout <= 0:
        return "none"
    return "full" if payout >= money(claim_amount) else "partial"

QUESTIONS = [
    Question("covered", "Is the claim covered?", Answer.yes_no(), requires=["cause_not_excluded"]),
    Question("payout_band", "Payout", Answer.choice(["full", "partial", "none"]), requires=["cause_not_excluded"]),
]
`,
  docs: [
    {
      name: "Home policy + burst-pipe claim",
      text: String.raw`HEARTHSTONE MUTUAL INSURANCE COMPANY
HOMEOWNERS POLICY - DECLARATIONS AND WORDING (EXCERPT)

Policy number: HM-HO3-2210874          Policy period: 01/03/2026 to 28/02/2027
Named insured: Daniel and Rosa Ferreira
Insured location: 88 Juniper Lane, Asheville, NC 28803

SECTION I - PROPERTY COVERAGES AND LIMITS
Coverage A - Dwelling: up to $420,000 for direct physical loss or damage to the building.
Coverage B - Other structures: up to $42,000.
Coverage C - Personal property: up to $210,000.
Coverage D - Loss of use: up to $84,000.

DEDUCTIBLE
All covered losses under Section I are subject to a deductible of $2,500 per occurrence, which the insured bears before any
payment under this policy.

PERILS INSURED AGAINST
We insure against sudden and accidental direct physical loss to the property described in Coverages A and B, including
sudden and accidental discharge or overflow of water or steam from within a plumbing, heating or air-conditioning system.

EXCLUSIONS
We do not insure for loss caused directly or indirectly by: flood, surface water, waves or overflow of any body of water;
earthquake, landslide or subsidence; wear and tear, deterioration, rust or mould; gradual seepage or leakage of water over a
period of weeks or months; war, nuclear hazard or intentional loss.

CONDITIONS
The insured must give prompt notice of any loss, protect the property from further damage and allow us to inspect it.

---------------------------------------------------------------
FIRST NOTICE OF LOSS - CLAIM FORM
Claim reference: CL-2026-551093
Date of loss: 14/09/2026
Description of loss: A supply pipe under the upstairs bathroom burst suddenly overnight and water came through the kitchen
ceiling, damaging the ceiling, cabinets and flooring.
Cause of loss: burst pipe (sudden water discharge)
Amount claimed: $18,400 (contractor estimate attached)
Reported by: Rosa Ferreira, 15/09/2026
`,
    },
    {
      name: "Same policy + flood claim",
      text: String.raw`HEARTHSTONE MUTUAL INSURANCE COMPANY
HOMEOWNERS POLICY - DECLARATIONS AND WORDING (EXCERPT)

Policy number: HM-HO3-2210874          Policy period: 01/03/2026 to 28/02/2027
Named insured: Daniel and Rosa Ferreira
Insured location: 88 Juniper Lane, Asheville, NC 28803

SECTION I - PROPERTY COVERAGES AND LIMITS
Coverage A - Dwelling: up to $420,000 for direct physical loss or damage to the building.
Coverage C - Personal property: up to $210,000.

DEDUCTIBLE
All covered losses under Section I are subject to a deductible of $2,500 per occurrence, which the insured bears before any
payment under this policy.

EXCLUSIONS
We do not insure for loss caused directly or indirectly by: flood, surface water, waves or overflow of any body of water;
earthquake, landslide or subsidence; wear and tear, deterioration, rust or mould; gradual seepage or leakage of water over a
period of weeks or months; war, nuclear hazard or intentional loss.

---------------------------------------------------------------
FIRST NOTICE OF LOSS - CLAIM FORM
Claim reference: CL-2026-551347
Date of loss: 22/09/2026
Description of loss: After three days of heavy rain the Swannanoa River overflowed and water entered the basement and ground
floor to a depth of about 40 cm.
Cause of loss: river flood
Amount claimed: $61,750
Reported by: Daniel Ferreira, 24/09/2026
`,
    },
  ],
};
