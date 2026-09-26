export default {
  id: "dpa",
  title: "Data processing agreement",
  domain: "Privacy",
  why: String.raw`Privacy and procurement teams review every vendor's DPA for the same few terms: how fast they report a breach, how long they keep data, and who else touches it. GDPR expects breach notification to the controller fast enough to meet the 72-hour deadline, so "breach notice within 72 hours" is a hard check computed from the quoted clause ("five (5) business days" is 168 hours and fails). DPAs are long, so the model reads them in overlapping windows and the answer still cites the exact sentence. It is also the kind of document you would rather not upload anywhere.`,
  fields: [
    ["breach_notice", "the time within which the processor must notify the controller of a personal data breach"],
    ["retention", "how long personal data is kept after the end of the services"],
    ["subprocessors", "the list of subprocessors or third parties that process the personal data"],
    ["transfer_mechanism", "How are transfers of personal data outside the EEA protected?"],
  ],
  code: String.raw`@cat.fn
def breach_hours(breach_notice):
    return hours(breach_notice)              # "forty-eight (48) hours" -> 48, "five (5) business days" -> 168

@cat.check(hard=True, then={"approve": "reject"})
def breach_within_72h(breach_hours):
    """hard: breach notice must reach us within 72 hours"""
    return breach_hours <= 72

@cat.fn
def retention_days(retention):
    return days(retention)

@cat.rule("approve")
def approve(retention_days, subprocessors):
    if retention_days > 90 or not found(subprocessors):
        return "review"
    return "approve"

@cat.rule("transfers_safeguarded")
def transfers_safeguarded(transfer_mechanism):
    return mentions(transfer_mechanism, "standard contractual clauses", "SCC", "adequacy decision", "binding corporate rules",
                    "Data Privacy Framework")

QUESTIONS = [
    Question("approve", "Approve the vendor?", Answer.choice(["approve", "review", "reject"]),
             checkpoints=["breach_within_72h"]),
    Question("transfers_safeguarded", "Transfers outside the EEA safeguarded?", Answer.yes_no()),
]
`,
  docs: [
    {
      name: "SaaS vendor DPA (long, 4 pages)",
      text: String.raw`DATA PROCESSING AGREEMENT

This Data Processing Agreement ("DPA") forms part of the Subscription Agreement dated 1 June 2026 (the "Agreement") between
Fjordline Retail AS, Tollbugata 17, 0152 Oslo, Norway (the "Controller" or "Customer"), and Parcelwise Analytics Ltd, 40
Grand Canal Street Lower, Dublin 2, D02 X326, Ireland (the "Processor" or "Parcelwise"). It reflects the parties' agreement
with regard to the processing of Personal Data under the Agreement.

1. DEFINITIONS
1.1 "Data Protection Laws" means Regulation (EU) 2016/679 (the "GDPR"), the Norwegian Personal Data Act, and any other law
relating to the protection of personal data that applies to the processing under the Agreement.
1.2 "Personal Data", "Processing", "Data Subject", "Personal Data Breach" and "Supervisory Authority" have the meanings given
in the GDPR.
1.3 "Subprocessor" means any processor engaged by Parcelwise to process Personal Data on behalf of the Customer.
1.4 "Services" means the delivery-analytics platform and related support described in the Agreement.

2. SCOPE AND ROLES
2.1 The Customer is the controller and Parcelwise is the processor of the Personal Data processed in providing the Services.
2.2 Subject matter and duration: the processing of Personal Data for the provision of the Services, for the term of the
Agreement.
2.3 Nature and purpose: collection, storage, analysis and display of delivery events to measure carrier performance and to
notify the Customer's end customers about the status of their parcels.
2.4 Categories of Data Subjects: the Customer's end customers who receive parcels, and the Customer's employees who use the
Services.
2.5 Categories of Personal Data: name, delivery address, e-mail address, telephone number, order reference, delivery events
and, for employees, user name and log-in records. No special categories of Personal Data are processed.

3. INSTRUCTIONS
3.1 Parcelwise shall process Personal Data only on documented instructions from the Customer, including with regard to
transfers to a third country, unless required to do so by Union or Member State law; in that case Parcelwise shall inform the
Customer of that legal requirement before processing, unless the law prohibits such information.
3.2 The Agreement and this DPA are the Customer's complete instructions at the time of signature. Additional instructions
require a written amendment.
3.3 Parcelwise shall immediately inform the Customer if, in its opinion, an instruction infringes Data Protection Laws.

4. CONFIDENTIALITY
4.1 Parcelwise shall ensure that persons authorised to process the Personal Data have committed themselves to confidentiality
or are under an appropriate statutory obligation of confidentiality.

5. SECURITY
5.1 Parcelwise shall implement the technical and organisational measures described in Annex 2, including encryption of
Personal Data in transit (TLS 1.2 or higher) and at rest (AES-256), role-based access control with multi-factor
authentication, logging of administrative access, annual penetration tests by an independent firm and a documented incident
response plan.
5.2 Parcelwise may update the measures from time to time, provided that the overall level of security is not reduced.

6. SUBPROCESSORS
6.1 The Customer gives Parcelwise a general authorisation to engage Subprocessors. The Subprocessors approved at the date of
this DPA are: Amazon Web Services EMEA SARL (hosting, Frankfurt and Dublin regions); Twilio Ireland Limited (SMS
notifications); Zendesk Inc. (customer support ticketing); and Sentry (Functional Software, Inc.) (error monitoring).
6.2 Parcelwise shall inform the Customer of any intended addition or replacement of Subprocessors at least thirty (30) days in
advance by e-mail, giving the Customer the opportunity to object on reasonable data-protection grounds.
6.3 Parcelwise shall impose on each Subprocessor data-protection obligations no less protective than those in this DPA and
remains fully liable to the Customer for the performance of each Subprocessor's obligations.

7. INTERNATIONAL TRANSFERS
7.1 Personal Data transferred outside the European Economic Area, including to Zendesk Inc. and Functional Software, Inc. in
the United States, is protected by the EU Standard Contractual Clauses (Commission Implementing Decision (EU) 2021/914, Module
Three) or, where the recipient is certified, by the EU-U.S. Data Privacy Framework.

8. ASSISTANCE
8.1 Taking into account the nature of the processing, Parcelwise shall assist the Customer by appropriate technical and
organisational measures in responding to requests from Data Subjects exercising their rights, and shall forward to the
Customer, within five (5) business days, any such request received directly.
8.2 Parcelwise shall assist the Customer with data protection impact assessments and prior consultations with Supervisory
Authorities, where required, at the Customer's reasonable cost.

9. PERSONAL DATA BREACH
9.1 Parcelwise shall notify the Customer of a Personal Data Breach without undue delay and in any event within forty-eight (48)
hours after becoming aware of it.
9.2 The notification shall describe, to the extent then known, the nature of the breach, the categories and approximate number
of Data Subjects and records concerned, the likely consequences and the measures taken or proposed. Information that is not
available at first may be provided in phases.
9.3 Parcelwise shall not notify any Supervisory Authority or Data Subject of a breach concerning the Customer's Personal Data
without the Customer's prior written approval, unless required by law.

10. AUDITS
10.1 Parcelwise shall make available to the Customer all information necessary to demonstrate compliance with this DPA,
including its most recent ISO/IEC 27001 certificate and SOC 2 Type II report.
10.2 The Customer may conduct an audit, itself or through an independent auditor bound by confidentiality, once per calendar
year on at least thirty (30) days' written notice, or at any time following a Personal Data Breach.

11. RETURN AND DELETION
11.1 On termination or expiry of the Agreement, Parcelwise shall, at the Customer's choice, return all Personal Data to the
Customer in a commonly used machine-readable format, and shall delete all remaining copies within sixty (60) days after the
end of the Services, unless Union or Member State law requires storage of the Personal Data.
11.2 Back-up copies are overwritten in the normal back-up cycle, which does not exceed thirty-five (35) days.
11.3 On request, Parcelwise shall confirm the deletion in writing.

12. LIABILITY
12.1 Each party's liability under this DPA is subject to the limitations and exclusions of liability in the Agreement.

13. TERM, PRECEDENCE AND GOVERNING LAW
13.1 This DPA remains in force for as long as Parcelwise processes Personal Data on behalf of the Customer.
13.2 In case of conflict between this DPA and the Agreement, this DPA prevails with regard to the processing of Personal Data.
13.3 This DPA is governed by the laws of Norway, and the Oslo District Court has exclusive jurisdiction.

ANNEX 1 - CONTACTS
Customer data protection contact: personvern@fjordline-retail.example, +47 22 44 90 10.
Parcelwise data protection officer: Aoife Byrne, dpo@parcelwise.example, +353 1 555 0142.

ANNEX 2 - TECHNICAL AND ORGANISATIONAL MEASURES (SUMMARY)
Access control: single sign-on, multi-factor authentication for all staff, quarterly access reviews, least-privilege roles.
Encryption: TLS 1.2+ in transit; AES-256 at rest with keys managed in AWS KMS; customer-specific keys on request.
Availability: daily encrypted back-ups replicated across two availability zones; recovery point objective 24 hours,
recovery time objective 8 hours; tested twice a year.
Development: peer code review, dependency scanning, separation of production and test data (test data is synthetic).
Personnel: background checks where legally permitted, security training on hiring and annually.
Incident response: 24/7 on-call rota, documented playbooks, post-incident reviews shared with affected customers.

Signed for Fjordline Retail AS: Kari Solberg, Head of Legal, 1 June 2026
Signed for Parcelwise Analytics Ltd: Conor Walsh, Chief Executive Officer, 1 June 2026
`,
    },
    {
      name: "Marketing vendor DPA, slow breach notice",
      text: String.raw`DATA PROCESSING ADDENDUM

Between Bellweather Outdoor GmbH, Leopoldstraße 88, 80802 München ("Controller") and Clickmoor Media Inc., 500 Howard Street,
San Francisco, CA 94105, USA ("Processor").

1. Purpose. The Processor processes e-mail addresses, names, purchase histories and website interaction data of the
Controller's customers solely to send marketing e-mails and measure campaign performance on the Controller's instructions.

2. Security. The Processor maintains appropriate technical and organisational measures, including encryption at rest and
access restricted to named campaign staff.

3. Sub-processing. The Processor may use third-party service providers to deliver its services and will make a list of them
available on request.

4. Data breaches. The Processor will notify the Controller of any Personal Data Breach affecting Controller data without undue
delay and in any case within five (5) business days of confirming the breach.

5. International transfers. Data is hosted in the United States. Transfers from the EEA are governed by the Standard
Contractual Clauses approved by the European Commission, which are incorporated by reference.

6. Retention. The Processor retains Controller personal data for the duration of the services and for twenty-four (24) months
thereafter for analytics and reporting purposes, after which it is deleted or anonymised.

7. Audits. The Controller may request the Processor's most recent third-party security report once per year.

8. Governing law. This Addendum is governed by the laws of Germany.

Accepted electronically by both parties on 3 August 2026.
`,
    },
  ],
};
