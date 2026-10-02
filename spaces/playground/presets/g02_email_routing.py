"""Email routing: sender + subject + body -> which team (billing, technical, sales, security, hr) and does a human need to look?

The routing table is a readable rule list learned in milliseconds from 200 synthetic labeled emails (at import, so it also
happens in the browser). Every line of the table that matches an email votes for its team with its support; a clear winner
routes the email, a split vote or no matching line makes `team` abstain (and `needs_human` say yes) instead of guessing.
A sender domain that imitates ours is a hard check: team=security, needs_human=yes, whatever the words say.
Try: change the sender to "billing@acme-io.com", or add "and SSO login fails with error 401" to the body."""
import random
import re

from solvi import Answer, Catalog, Question, Quote
from solvi.rulelist import RuleList, literals

cat = Catalog()
TEAMS = ["billing", "technical", "sales", "security", "hr"]
OUR_DOMAIN = "acme.io"
FREEMAIL = {"gmail.com", "outlook.com", "yahoo.com", "proton.me", "icloud.com", "hotmail.com"}
STOP = set("a an the i im i'm me my we our us you your it its is are was were be been to of for and or in on at with this that "
           "have has had do does did can could would will should please hi hello hey thanks thank regards best team just so how "
           "what why when who where which there want like need some any all from about into after before also get got very much "
           "more they them their re fw fwd dear kind".split())


def _stem(w):
    for suf in ("ation", "ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _edit1(a, b):
    """True if a and b differ by at most one substitution, insertion or deletion (or 'rn' written for 'm')."""
    a = a.replace("rn", "m")
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] or a[i:] == b[i + 1:] or a[i + 1:] == b[i:]


# ---------- facts
@cat.extract
def sender_domain(sender):
    m = re.search(r"@([\w.-]+\.[a-z]{2,})", sender, re.I)
    if not m:
        raise ValueError("no e-mail address in sender")
    return Quote(m.group(1).lower(), m.start(1), m.end(1), source="sender")


@cat.fn
def sender_kind(sender_domain):
    if sender_domain == OUR_DOMAIN:
        return "internal"
    return "freemail" if sender_domain in FREEMAIL else "external"


@cat.fn
def words(subject, body):
    """subject + body words for the learned table: stop words dropped, light stemming"""
    toks = re.findall(r"[a-z][a-z']+", (subject + " " + body).lower())
    return " ".join(dict.fromkeys(_stem(t) for t in toks if t not in STOP and len(t) > 2))


@cat.extract
def payment_change_ask(body):
    "asks to change where money goes (bank details, payee) — the classic invoice-fraud move"
    m = re.search(r"\b(?:update|change|new)\b[^.]{0,30}\b(?:bank (?:details|account)|account number|IBAN|payee|remittance details)\b",
                  body, re.I)
    return Quote(True, m.start(), m.end(), source="body") if m else Quote(False, 0, 0, source="body")


@cat.extract
def credential_ask(body):
    "asks for a password, a login code or to 'verify' an account through a link"
    m = re.search(r"\b(?:verify|confirm|re-?enter|validate)\b[^.]{0,30}\b(?:password|credentials|login|account|2FA code|MFA code)\b"
                  r"|\bgift cards?\b", body, re.I)
    return Quote(True, m.start(), m.end(), source="body") if m else Quote(False, 0, 0, source="body")


def team_votes(words):                  # registered at the end with model=ROUTING: its provenance is "learned"
    """every line of the learned table that matches votes for its team with its support"""
    lits = literals({"words": words}, ROUTING.facts)
    votes = {}
    for r in ROUTING.rules:
        if r["if"] in lits:
            votes[r["then"]] = votes.get(r["then"], 0) + r["support"]
    return dict(sorted(votes.items(), key=lambda kv: -kv[1]))


@cat.fn
def clear_winner(team_votes):
    """the top team has at least twice the votes of the runner-up"""
    v = list(team_votes.values()) + [0, 0]
    return v[0] > 0 and v[0] >= 2 * v[1]


# ---------- hard check
@cat.check(hard=True, then={"team": "security", "needs_human": "yes"})
def sender_not_lookalike(sender_domain):
    """a domain one edit away from ours, or ours with extra words (acme-io.com, acrne.io, acme.io-secure.net), is spoofing"""
    base, ours = sender_domain.rsplit(".", 1)[0], OUR_DOMAIN.rsplit(".", 1)[0]
    if sender_domain == OUR_DOMAIN:
        return True
    return not (_edit1(base, ours) or re.search(rf"(?:^|[.-]){ours}(?:[.-]|io\b)", base) is not None)


# ---------- answers
@cat.rule("team")
def team(team_votes, clear_winner):
    return next(iter(team_votes)) if clear_winner else None      # None is not a team: the question abstains


@cat.rule("needs_human")
def needs_human(clear_winner, payment_change_ask, credential_ask, sender_kind):
    return (not clear_winner) or credential_ask or (payment_change_ask and sender_kind != "internal")


QUESTIONS = [
    Question("team", "Which team handles this email?", Answer.choice(TEAMS), requires=["sender_not_lookalike"]),
    Question("needs_human", "Does a person need to look before it is routed?", Answer.yes_no(),
             requires=["sender_not_lookalike"]),
]


# ---------- 200 synthetic labeled emails -> the routing table
ASK = {
    "billing": ["please find attached invoice {inv} for {month}", "our payment of {amount} was declined", "can you resend the receipt for {month}",
                "when will the overdue invoice be paid", "please confirm the payment date for our invoice",
                "we were charged twice on the {month} statement", "please update the billing contact to {email}",
                "remittance advice for invoice {inv}", "invoice {inv} is now past due", "the credit note for {inv} is missing"],
    "technical": ["the API returns 503 errors since {time}", "SSO login fails with error {code}", "webhook deliveries are delayed by hours",
                  "the mobile app crashes on startup", "the data export times out", "we need help debugging the integration",
                  "the dashboard shows a blank page after the update"],
    "sales": ["we would like a demo for {n} seats", "could you send pricing for the Enterprise plan", "we are interested in upgrading to annual",
              "requesting a quote for {n} licenses", "is there a volume discount for {n} users", "our procurement team wants a proposal"],
    "security": ["I clicked a suspicious link and typed my password", "we received a phishing email impersonating you",
                 "please disable the account of a departed employee", "unusual login alerts from {country}",
                 "reporting a vulnerability in your login page", "our API key may have leaked on GitHub",
                 "a phishing link reached our finance staff", "I think my password was stolen", "someone accessed my mailbox"],
    "hr": ["I would like to request parental leave from {month}", "my payslip for {month} is missing overtime",
           "application for the backend engineer role", "question about the pension scheme", "can I get a reference letter",
           "sick leave note attached for this week", "when is the next payroll run"],
}
SUBJECT = {"billing": ["Invoice {inv}", "Payment", "Receipt request"], "technical": ["Bug report", "Outage?", "Integration issue"],
           "sales": ["Pricing", "Demo request", "Quote"], "security": ["Security concern", "Suspicious activity", "Urgent"],
           "hr": ["Leave request", "Payroll", "Application"]}
SHARED_SUBJECT = ["Quick question", "Following up", "Re: your message", "Hello", "Request"]
CONTEXT = ["", "", "Sorry for the long email.", "Copying my manager.", "We are a customer since {year}.", "See the attachment.",
           "This is for our {office} office."]
SENDER = {"billing": ["external"], "technical": ["external", "external", "freemail"], "sales": ["external", "freemail"],
          "security": ["internal", "external"], "hr": ["internal", "internal", "freemail"]}


def make_email(rng, team):
    f = dict(inv=f"INV-{rng.randint(1000, 9999)}", month=rng.choice(["March", "June", "September"]), amount=f"${rng.randint(90, 9000)}",
             email="ap@" + rng.choice(["kline.io", "orbit.co"]), time=rng.choice(["09:10", "yesterday"]), code=rng.choice(["401", "SAML-17"]),
             n=rng.choice([15, 40, 250]), country=rng.choice(["Brazil", "Vietnam", "Romania"]), year=rng.randint(2015, 2024),
             office=rng.choice(["Berlin", "Austin", "Lagos"]))
    kind = rng.choice(SENDER[team])
    domain = {"internal": OUR_DOMAIN, "freemail": rng.choice(sorted(FREEMAIL))}.get(kind) or rng.choice(["kline.io", "orbit.co", "hollow.dev"])
    subject = rng.choice(SUBJECT[team] + SHARED_SUBJECT).format(**f)
    body = " ".join(x for x in [rng.choice(["Hi,", "Hello,", "Dear team,", ""]), rng.choice(CONTEXT).format(**f),
                                (lambda a: a[0].upper() + a[1:])(rng.choice(ASK[team]).format(**f)) + ".", rng.choice(["Thanks,", "Best,", "Regards,"]),
                                rng.choice(["Sam", "Priya", "Lena", "Tom"])] if x)
    return {"sender": f"{rng.choice(['sam', 'priya', 'lena', 'tom'])}@{domain}", "subject": subject, "body": body}, team


def make_emails(n, seed):
    rng = random.Random(seed)
    return [make_email(rng, rng.choice(TEAMS)) for _ in range(n)]


TRAIN = make_emails(200, seed=3)
# the table System(cat, QUESTIONS).learn_rule("team", TRAIN, ["words"]) would learn, fitted on the one fact it reads
ROUTING = RuleList(["words"], min_support=3, min_precision=0.8).fit([{"words": words(s["subject"], s["body"])} for s, _ in TRAIN],
                                                                     [t for _, t in TRAIN])
# team_votes reads the learned table: the trace records it (type, fingerprint) and the audit counts it as learned
cat.fn(team_votes, model=ROUTING)
