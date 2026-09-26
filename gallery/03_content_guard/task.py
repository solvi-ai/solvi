"""Content guard: user text for an app -> allow / review / block, plus "sensitive data?" and "prompt injection?".

Every finding is an extract with a Quote, so the verdict cites the offending span: e-mail addresses, phone numbers, card numbers
(only if the Luhn checksum holds, so a 16-digit order number is not a card), IBANs (only if the mod-97 checksum holds), API keys
and private keys, and prompt-injection phrases (a phrase inside quotes is a mention, not an attack). A card number, a secret or
a direct injection into an LLM prompt are hard checks: the verdict is "block" whatever the risk points add up to.
`surface` says where the text goes: llm_prompt, support_chat or public_post (the same e-mail address is fine in a support chat
and a privacy risk in a public post). Try: change one digit of the card number, or put the injection phrase in quotes."""
import re

from solvi import Answer, Catalog, Question, Quote

cat = Catalog()
NONE = Quote(None, 0, 0, source="text")


def _luhn(digits):
    total = 0
    for i, c in enumerate(reversed(digits)):
        n = int(c) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def _iban_ok(s):
    s = s.replace(" ", "").upper()
    return int("".join(str(int(ch, 36)) for ch in s[4:] + s[:4])) % 97 == 1


def _quoted(text, start):
    """inside a quoted span: an odd number of quote marks before the position"""
    before = text[:start]
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”") or \
        len(re.findall(r"(?:^|\s)'", before)) > len(re.findall(r"'(?:\s|$|[.,?!])", before))


# ---------- findings, each with its quote
@cat.extract
def email_address(text):
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", text, re.I)
    return Quote(m.group(), m.start(), m.end(), source="text") if m else NONE


@cat.extract
def phone_number(text):
    "9-12 digits with separators, or an international +CC number (longer runs are card candidates)"
    for m in re.finditer(r"(?<![\w+])\+?\(?\d[\d ().-]{7,17}\d(?!\w)", text):
        n = len(re.sub(r"\D", "", m.group()))
        if (m.group().startswith("+") and 10 <= n <= 15) or (9 <= n <= 12 and re.search(r"[ ().-]", m.group())):
            return Quote(m.group(), m.start(), m.end(), source="text")
    return NONE


@cat.extract
def card_number(text):
    "13-19 digits that pass the Luhn checksum"
    for m in re.finditer(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", text):
        d = re.sub(r"\D", "", m.group())
        if _luhn(d):
            return Quote("card ending " + d[-4:], m.start(), m.end(), source="text")
    return NONE


@cat.extract
def iban(text):
    "an IBAN that passes the mod-97 checksum"
    for m in re.finditer(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", text):
        if _iban_ok(m.group()):
            return Quote(m.group()[:2] + " IBAN ending " + m.group().replace(" ", "")[-4:], m.start(), m.end(), source="text")
    return NONE


@cat.extract
def secret_token(text):
    for kind, p in [("API key", r"\bsk-[A-Za-z0-9_-]{20,}"), ("AWS access key", r"\bAKIA[0-9A-Z]{16}\b"),
                    ("GitHub token", r"\bghp_[A-Za-z0-9]{36}\b"), ("Slack token", r"\bxox[bpas]-[A-Za-z0-9-]{10,}"),
                    ("private key", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]:
        m = re.search(p, text)
        if m:
            return Quote(kind, m.start(), m.end(), source="text")
    return NONE


@cat.extract
def injection(text):
    "instructions aimed at the model; 'quoted' when the phrase is only mentioned inside quotes"
    for p in [r"\bignore (?:all |any )?(?:the |your )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)\b",
              r"\bdisregard (?:your|all|the|any) (?:rules|instructions|guidelines)\b",
              r"\b(?:reveal|print|show|repeat) (?:your|the) (?:system prompt|hidden instructions|initial instructions)\b",
              r"\byou are now (?:DAN|in developer mode|unrestricted)\b", r"</?system>",
              r"\b(?:an? )?AI (?:with|that has) no (?:rules|restrictions|filters)\b"]:
        m = re.search(p, text, re.I)
        if m:
            return Quote("quoted" if _quoted(text, m.start()) else "direct", m.start(), m.end(), source="text")
    return NONE


@cat.extract
def insult(text):
    m = re.search(r"\b(?:idiot|moron|stupid|shut up|pathetic|loser)s?\b", text, re.I)
    return Quote(m.group(), m.start(), m.end(), source="text") if m else NONE


@cat.fn
def link_count(text):
    return len(re.findall(r"https?://", text))


@cat.fn
def risk_points(email_address, phone_number, iban, injection, insult, link_count, surface):
    """soft signals and what they weigh on this surface"""
    public = surface == "public_post"
    pts = {"e-mail in a public post": 2 * (public and email_address is not None),
           "phone in a public post": 2 * (public and phone_number is not None),
           "IBAN": 2 * (iban is not None),
           "injection phrase mentioned": 2 * (injection == "quoted"),
           "injection phrase outside an LLM prompt": 3 * (injection == "direct" and surface != "llm_prompt"),
           "insult": 2 * (insult is not None),
           "3+ links": 2 * (link_count >= 3)}
    return {k: v for k, v in pts.items() if v}


# ---------- hard checks: False forces "block"
@cat.check(hard=True, then={"verdict": "block", "sensitive_data": "yes"})
def no_card_number(card_number):
    """card numbers never go to a model or a public page (PCI DSS)"""
    return card_number is None


@cat.check(hard=True, then={"verdict": "block", "sensitive_data": "yes"})
def no_secret(secret_token):
    return secret_token is None


@cat.check(hard=True, then={"verdict": "block"})
def no_direct_injection(injection, surface):
    return not (injection == "direct" and surface == "llm_prompt")


# ---------- answers
@cat.rule("verdict")
def verdict(risk_points):
    score = sum(risk_points.values())
    return "block" if score >= 5 else "review" if score >= 2 else "allow"


@cat.rule("sensitive_data")
def sensitive_data(email_address, phone_number, iban):
    return email_address is not None or phone_number is not None or iban is not None


@cat.rule("prompt_injection")
def prompt_injection(injection):
    return injection == "direct"


QUESTIONS = [
    Question("verdict", "Allow, send to review, or block?", Answer.choice(["allow", "review", "block"]),
             checkpoints=["no_card_number", "no_secret", "no_direct_injection"]),
    Question("sensitive_data", "Does the text carry personal data, payment data or secrets?", Answer.yes_no(),
             checkpoints=["no_card_number", "no_secret"]),
    Question("prompt_injection", "Does the text try to give the model instructions?", Answer.yes_no()),
]
