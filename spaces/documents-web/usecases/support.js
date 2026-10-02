export default {
  id: "support",
  title: "Support email triage",
  domain: "Customer support",
  why: String.raw`Support inboxes are routed by a few facts buried in chatty emails: which order, what the customer wants, and by when. The model quotes those, Python maps the request to a queue and the deadline to an urgency, and an email without an order number is routed to "ask for the order number" by a hard check instead of being guessed onto the wrong ticket. Emails often contain personal data, and here they never leave the browser.`,
  fields: [
    ["order_number", "the order number"],
    ["requested_action", "What does the customer want the company to do?"],
    ["deadline", "When does the customer need it by?"],
  ],
  code: String.raw`@cat.fn
def request_kind(requested_action):
    need(requested_action, "the customer's request")
    if mentions(requested_action, "replace", "replacement", "exchange", "send another", "new one"):
        return "replacement"
    if mentions(requested_action, "refund", "money back", "reimburse"):
        return "refund"
    if mentions(requested_action, "cancel", "cancellation"):
        return "cancel"
    return "other"

@cat.check(hard=True, then={"route": "ask for order number"})
def has_order_number(order_number):
    """hard: without an order number nothing can be looked up"""
    return found(order_number)

@cat.rule("route")
def route(request_kind):
    return {"refund": "billing", "cancel": "billing", "replacement": "fulfilment"}.get(request_kind, "general")

@cat.rule("urgency")
def urgency(deadline, today):
    if not found(deadline):
        return "normal"
    return "high" if (parse_date(deadline) - today).days <= 3 else "normal"

QUESTIONS = [
    Question("route", "Which queue?", Answer.choice(["billing", "fulfilment", "general", "ask for order number"]),
             requires=["has_order_number"]),
    Question("urgency", "Urgency", Answer.choice(["high", "normal"])),
]
`,
  docs: [
    {
      name: "Broken espresso machine, needs it by Friday",
      today: "2026-09-26",
      text: String.raw`From: Léa Fontaine <lea.fontaine@example.fr>
To: support@brewcraft-shop.example
Date: Sat, 26 Sep 2026 09:14
Subject: Machine arrived broken – order BC-778120 😞

Hello BrewCraft team,

I received my Duetto II espresso machine yesterday (order number BC-778120, placed on 18 September). Unfortunately the
water tank is cracked along the bottom and it leaks as soon as I fill it. I attached three photos of the crack and of the box,
which was also dented on one corner.

I don't want a refund – I really like the machine. Could you please send me a replacement unit? I'm hosting a family lunch on
the 29th, so I would really need the new one to arrive by 29 September 2026 at the latest. I can hand the broken one to the
courier when they deliver the new machine.

Thanks a lot for your help 🙏

Léa Fontaine
12 rue Sainte-Catherine, 33000 Bordeaux
`,
    },
    {
      name: "Refund request without an order number",
      today: "2026-09-26",
      text: String.raw`From: Marcus Webb <m.webb@example.com>
To: help@brewcraft-shop.example
Subject: charged twice??

hi,

I bought a grinder from you guys last week and my card statement shows the payment twice, $249.00 on the 17th and again
$249.00 on the 18th. I only ordered one grinder!! I can't find the confirmation email, I think it went to my old address.

Please refund the duplicate charge to my card. This is really frustrating.

Marcus
`,
    },
  ],
};
