"""Guard.require_request: the intents' vocabulary (the verbs users write for an action) and register_intent, the public
way to add an intent or widen one."""
import pytest

from solvi.solutions.guard import INTENTS, Guard, register_intent


def outcome(intent, text, phrases=None):
    g = Guard()

    @g.tool
    def act(to: str, body: str) -> str:
        return "ok"
    g.require_request("act", intent, phrases=phrases)
    return g.check({"name": "act", "arguments": {"to": "bob", "body": "x"}}, [{"role": "user", "content": text}],
                   store=False).outcome


@pytest.mark.parametrize("intent, text", [
    ("send", "let Bob know the meeting moved"), ("send", "let my manager know I am late"),
    ("send", "dispatch an ambulance to the office"), ("send", "inform the team about the outage"),
    ("send", "contact the landlord about the leak"), ("send", "reach out to Alice"), ("send", "дай знать Боре"),
    ("pay", "withdraw 200 from my savings"), ("pay", "deposit the cheque"), ("pay", "top up my phone"),
    ("pay", "сними 200 со счёта"),
    ("share", "grant Bob access to the folder"), ("share", "unlock the door for the guest"),
    ("share", "give Alice access to the repo"),
    ("event", "book a call with Dana on Monday"), ("event", "put the dentist on my calendar"),
    ("event", "set up a call with the vendor"),
])
def test_the_verbs_users_write_count_as_asking(intent, text):
    assert outcome(intent, text) == "allow", (intent, text)


@pytest.mark.parametrize("intent, text", [
    ("send", "what did Bob say yesterday?"), ("pay", "what is my balance?"), ("share", "who has access to the folder?"),
    ("event", "what time is it in Tokyo?"), ("send", "let it be"),
])
def test_questions_that_ask_for_no_action_still_escalate(intent, text):
    assert outcome(intent, text) == "escalate", (intent, text)


def test_register_intent_adds_or_widens_an_intent():
    try:
        register_intent("launch", [r"\blaunch(es|ed|ing)?\b", r"\bdeploy\b"])
        assert "launch" in INTENTS and outcome("launch", "deploy the new build") == "allow"
        assert outcome("launch", "is the build green?") == "escalate"
        with pytest.raises(ValueError, match="exists"):
            register_intent("launch", [r"\bship\b"])
        register_intent("launch", [r"\bship\b"], extend=True)
        assert outcome("launch", "ship it") == "allow" and outcome("launch", "deploy now") == "allow"
        with pytest.raises(ValueError):
            register_intent("bad name!", [r"x"])
        with pytest.raises(ValueError):
            register_intent("broken", [r"(unclosed"])
        with pytest.raises(ValueError):
            register_intent("empty", [])
    finally:
        INTENTS.pop("launch", None)
