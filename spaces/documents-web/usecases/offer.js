export default {
  id: "offer",
  title: "Job offer letter",
  domain: "HR",
  why: String.raw`Before an offer goes out, HR checks it against the approved salary band and a few legal limits. Candidates do the same in reverse. The salary, start date, probation and any non-compete are quoted from the letter; the band is a two-line rule and the probation limit is a hard check, so a letter with a nine-month probation is sent back for revision even if everything else is fine.`,
  fields: [
    ["salary", "the annual base salary"],
    ["start_date", "the start date of employment"],
    ["probation", "the length of the probation period"],
    ["non_compete", "a clause that restricts the employee from working for a competitor after leaving"],
  ],
  code: String.raw`INPUTS = {"band_min": 95000, "band_max": 125000}      # approved band for the role

@cat.fn
def base_salary(salary):
    return money(salary)

@cat.fn
def probation_days(probation):
    return days(probation)

@cat.check(hard=True, then={"send": "revise"})
def probation_legal(probation_days):
    """hard: probation may not exceed six months"""
    return probation_days <= 183

@cat.rule("band")
def band(base_salary, band_min, band_max):
    return "below" if base_salary < band_min else "above" if base_salary > band_max else "in band"

@cat.rule("send")
def send(base_salary, band_min, band_max, non_compete):
    in_band = band_min <= base_salary <= band_max
    return "send" if in_band and not found(non_compete) else "revise"

@cat.rule("starts_within_60_days")
def starts_within_60_days(start_date, today):
    return (parse_date(start_date) - today).days <= 60

QUESTIONS = [
    Question("send", "Send the offer as is?", Answer.choice(["send", "revise"]), checkpoints=["probation_legal"]),
    Question("band", "Salary vs approved band", Answer.choice(["below", "in band", "above"])),
    Question("starts_within_60_days", "Starts within 60 days?", Answer.yes_no()),
]
`,
  docs: [
    {
      name: "Senior data engineer offer",
      today: "2026-09-26",
      text: String.raw`Tidewater Health Analytics, Inc.
600 Boylston Street, Suite 1200, Boston, MA 02116

September 22, 2026

Ms. Aisha Rahman
48 Elm Street, Apt 3
Somerville, MA 02144

Dear Aisha,

On behalf of Tidewater Health Analytics, Inc. (the "Company"), I am delighted to offer you the position of Senior Data
Engineer in our Platform team, reporting to Daniel Cho, Director of Data Engineering.

Start date. Your first day of employment will be Monday, November 2, 2026. Please report to our Boston office at 9:00 a.m.,
where our People team will take you through onboarding.

Compensation. You will receive an annual base salary of $118,000, paid in bi-weekly installments and subject to applicable
withholdings. You will also be eligible for an annual performance bonus with a target of 10% of your base salary, based on
individual and Company performance.

Equity. Subject to approval by the Board of Directors, you will be granted an option to purchase 6,000 shares of the
Company's common stock, vesting over four years with a one-year cliff.

Benefits. You will be eligible for the Company's medical, dental and vision plans from your first day, a 401(k) plan with a
4% Company match, and 20 days of paid time off per year in addition to Company holidays.

Probation. Your first three (3) months of employment will be an introductory period, during which your manager will meet with
you regularly to review your progress.

Confidentiality. As a condition of employment you will sign the Company's Employee Confidentiality and Invention Assignment
Agreement.

At-will employment. Your employment with the Company is at will, meaning that either you or the Company may end it at any
time, with or without cause or notice.

This offer is contingent on satisfactory verification of your eligibility to work in the United States. Please sign and
return this letter by October 2, 2026 to accept.

We are excited to have you join us.

Sincerely,
Laura Benedetti
Chief People Officer

Accepted: ______________________ Date: __________
`,
    },
    {
      name: "Sales manager offer with a non-compete",
      today: "2026-09-26",
      text: String.raw`VANTAGE INDUSTRIAL SUPPLY LTD
Registered office: 14 Canal Street, Manchester M1 3HE

Private and confidential
Mr Tomasz Nowak
22 Ashfield Road, Stockport SK4 2DP

18 September 2026

Offer of employment: Regional Sales Manager (North West)

Dear Tomasz,

Following your recent interviews, we are pleased to offer you employment with Vantage Industrial Supply Ltd on the terms below.
A full statement of terms will follow.

Commencement: your employment will commence on 5 January 2027.

Salary: your basic salary will be £82,500 per annum, payable monthly in arrears. You will also be eligible for commission
under the Sales Incentive Plan, currently on-target earnings of £25,000.

Company car: you will be provided with a company car in accordance with the Company's car policy, or a car allowance of
£6,000 per annum.

Probationary period: the first nine (9) months of your employment will be a probationary period. During this period either
party may terminate the employment by giving one week's notice.

Hours: 37.5 hours per week, Monday to Friday, with such additional hours as are necessary for the proper performance of your
duties.

Holiday: 25 working days per year plus bank holidays.

Restrictive covenants: for a period of twelve (12) months after the termination of your employment you shall not, within the
North West region, be employed by or provide services to any business that supplies industrial fasteners, bearings or
power-transmission products in competition with the Company.

Notice: after the probationary period, three months' notice by either party.

Please confirm your acceptance by signing and returning a copy of this letter.

Yours sincerely,
Rebecca Hale
HR Director
`,
    },
  ],
};
