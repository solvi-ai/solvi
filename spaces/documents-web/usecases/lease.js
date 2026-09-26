export default {
  id: "lease",
  title: "Residential lease",
  domain: "Real estate",
  why: String.raw`Tenants and letting agents check the same things in every lease: the rent, the deposit, how to get out, and whether the dog can come. Many places cap the deposit at a multiple of the rent; here that cap is a hard check computed from two quoted amounts, so a lease that breaks it is flagged "negotiate" no matter how the rest looks. The pets clause is read by the model and interpreted by three readable lines of Python.`,
  fields: [
    ["rent", "the monthly rent amount"],
    ["deposit", "the security deposit amount"],
    ["notice_period", "the notice period the tenant must give to end the lease"],
    ["pets", "the clause about whether pets or animals are allowed"],
  ],
  code: String.raw`@cat.fn
def monthly_rent(rent):
    return money(rent)

@cat.fn
def deposit_amount(deposit):
    return money(deposit)

@cat.check(hard=True, then={"sign": "negotiate"})
def deposit_within_cap(deposit_amount, monthly_rent):
    """hard: the deposit may not exceed two months' rent"""
    return deposit_amount <= 2 * monthly_rent

@cat.fn
def notice_days(notice_period):
    return days(notice_period)

@cat.rule("pets_allowed")
def pets_allowed(pets):
    if not found(pets):
        return "not stated"
    if mentions(pets, "no pets", "not permitted", "not allowed", "prohibited", "shall not keep"):
        return "no"
    return "yes"

@cat.rule("sign")
def sign(notice_days):
    return "sign" if notice_days <= 60 else "negotiate"     # tenant should be able to leave on two months' notice

QUESTIONS = [
    Question("sign", "Sign as is?", Answer.choice(["sign", "negotiate"]), checkpoints=["deposit_within_cap"]),
    Question("pets_allowed", "Are pets allowed?", Answer.choice(["yes", "no", "not stated"])),
]
`,
  docs: [
    {
      name: "Apartment lease, Portland",
      text: String.raw`RESIDENTIAL LEASE AGREEMENT

This Residential Lease Agreement ("Lease") is made on July 18, 2026 between Cedar Row Properties LLC ("Landlord") and
Jordan Ellis and Sam Okoro (together, "Tenant").

1. PREMISES. Landlord leases to Tenant the apartment located at 1420 NW Lovejoy Street, Unit 507, Portland, Oregon 97209,
together with one assigned parking space (P2-44) and one storage locker (the "Premises").

2. TERM. The Lease begins on August 1, 2026 and ends on July 31, 2027. After that date the tenancy continues from month to
month unless either party ends it as provided in Section 14.

3. RENT. Tenant shall pay monthly rent of $2,350.00, due in advance on the first day of each month, by bank transfer to the
account designated by Landlord. Rent paid after the fifth day of the month incurs a late fee of $50.00.

4. SECURITY DEPOSIT. On signing this Lease, Tenant shall pay a security deposit of $3,500.00. Landlord shall hold the deposit
as security for Tenant's performance and return it, less any lawful deductions itemized in writing, within thirty-one (31)
days after Tenant returns possession of the Premises.

5. UTILITIES. Tenant pays for electricity, internet and renter's insurance. Landlord pays for water, sewer and garbage
collection.

6. USE AND OCCUPANCY. The Premises shall be used only as a private residence for Tenant and Tenant's minor children. Guests
may not stay longer than fourteen (14) consecutive days without Landlord's written consent.

7. PETS. Tenant may keep up to two domestic cats or one dog under 40 lbs, subject to a one-time pet fee of $300 and proof of
current vaccinations. Tenant is responsible for any damage caused by the animals.

8. MAINTENANCE AND REPAIRS. Tenant shall keep the Premises clean and promptly report any water leak, mold or malfunction of
appliances. Landlord shall make necessary repairs within a reasonable time after notice.

9. ALTERATIONS. Tenant shall not paint, install fixtures or make alterations without Landlord's prior written consent.

10. ENTRY. Landlord may enter the Premises for inspection or repairs after giving at least twenty-four (24) hours' notice,
except in an emergency.

11. SMOKING. Smoking of any substance is prohibited inside the Premises and on balconies.

12. INSURANCE. Tenant shall maintain renter's insurance with liability coverage of at least $100,000 during the Lease.

13. SUBLETTING. Tenant may not sublet the Premises or assign this Lease without Landlord's written consent.

14. ENDING THE TENANCY. After the initial term, Tenant may end the tenancy by giving Landlord at least thirty (30) days'
written notice. Landlord may end a month-to-month tenancy only as permitted by Oregon law.

15. GOVERNING LAW. This Lease is governed by the laws of the State of Oregon.

LANDLORD: Cedar Row Properties LLC, by Maya Lindqvist, Property Manager
TENANT: Jordan Ellis          TENANT: Sam Okoro
`,
    },
    {
      name: "House lease with a high deposit",
      text: String.raw`ASSURED SHORTHOLD TENANCY AGREEMENT

Date: 2 September 2026
Landlord: Mr Peter Aldridge, of 9 Chestnut Grove, Harrogate HG1 2QT
Tenant: Ms Chloe Martin
Property: 27 Rosebank Road, Leeds LS6 3HT (a three-bedroom semi-detached house, part furnished)

1. The Landlord lets the Property to the Tenant for a fixed term of twelve months starting on 1 October 2026.

2. The rent is £1,150 per calendar month, payable in advance on the 1st of each month by standing order.

3. The Tenant shall pay a deposit of £3,450 before the start of the tenancy. The deposit will be protected in a government
approved tenancy deposit scheme within 30 days of receipt.

4. The Tenant shall keep the garden tidy and the lawn cut, and shall not remove any shrubs or trees.

5. The Tenant shall not keep any animals, birds or reptiles at the Property. No pets of any kind are permitted without the
prior written consent of the Landlord, which may be withheld at the Landlord's discretion.

6. The Tenant shall not smoke in the Property and shall not use paraffin or bottled gas heaters.

7. The Landlord shall keep in repair the structure and exterior of the Property and the installations for the supply of
water, gas, electricity and heating.

8. The Tenant may end the tenancy after the fixed term by giving the Landlord at least three months' written notice,
expiring on the last day of a rental period.

9. The Landlord may enter the Property to inspect its condition at reasonable times on giving at least 24 hours' written
notice.

Signed: P. Aldridge (Landlord)      Signed: C. Martin (Tenant)
`,
    },
  ],
};
