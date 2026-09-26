export default {
  id: "blank",
  title: "Blank: your document, your fields",
  domain: "Any",
  why: String.raw`Paste any document and describe the fields you need in plain English, one per row. Each field is extracted immediately and cited; with no rules yet, every field gets an automatic "is it in the document?" question so it still goes through solvi (quote, trace, replay). Then write a rule or two below and press Run. When a use case works for you, it can join this library: see "Add a use case" in the footer.`,
  fields: [
    ["order_date", "the date of the order"],
    ["signed_by", "the name of the person who signed the letter"],
    ["balance", "What is the balance amount?"],
  ],
  code: String.raw`# Your rules. Every field in the table is a fact with the same name (a string; "" when absent).
# Example (uncomment and press Run):
#
# @cat.fn
# def value(balance):
#     return money(balance)
#
# @cat.rule("large")
# def large(value):
#     return value > 1000
#
# QUESTIONS = [Question("large", "Is the balance over 1000?", Answer.yes_no())]

QUESTIONS = []
`,
  docs: [
    {
      name: "Short letter (replace with your text)",
      text: String.raw`Dear Ms. Novak,

Thank you for your order of 12 September 2026. We confirm that the balance of EUR 1,840.00 for the conference catering will be
invoiced after the event.

Kind regards,
Tomas Berger
Alpine Events GmbH, Innsbruck
`,
    },
  ],
};
