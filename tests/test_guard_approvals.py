"""Guard(user_verifier=): a person's approval of an escalated call is bound to who gave it and to this very call — an
approval token signed for the call's approval_key, checked for its signature, its reviewer and its age; the verified
identity is what the store records."""
import pytest

from solvi.solutions.guard import Guard, approval_verifier, sign_approval

KEYS = {"alice": b"alice-secret", "bob": b"bob-secret"}


def guard(tmp_path, verifier):
    g = Guard(storage=tmp_path / "calls.jsonl", user_verifier=verifier)
    paid = []

    @g.tool
    def pay(to: str, amount: float) -> str:
        paid.append((to, amount))
        return "paid"

    @g.policy("pay", on_fail="escalate")
    def small(amount: float) -> bool:
        """Payments of 100 or more go to a person."""
        return amount < 100
    return g, paid


def esc(g, amount=500):
    d = g.call({"name": "pay", "arguments": {"to": "Bob", "amount": amount}}, f"pay {amount} to Bob")
    assert d.outcome == "escalate"
    return d


def test_an_approval_needs_a_valid_token_for_this_call(tmp_path):
    now = [1000.0]
    g, paid = guard(tmp_path, approval_verifier(KEYS, max_age=300, clock=lambda: now[0]))
    e = esc(g)
    with pytest.raises(PermissionError, match="approval"):
        g.resolve(e, approve=True, reviewer="alice")                  # no token
    with pytest.raises(PermissionError):
        g.resolve(e, approve=True, approval="alice.1000.00ff")        # a forged signature
    with pytest.raises(PermissionError):                              # signed with another reviewer's key
        g.resolve(e, approve=True, approval=sign_approval(e, "alice", KEYS["bob"], now=1000))
    other = esc(g, 700)                                               # a token for another call does not cover this one
    with pytest.raises(PermissionError):
        g.resolve(e, approve=True, approval=sign_approval(other, "alice", KEYS["alice"], now=1000))
    with pytest.raises(PermissionError, match="old"):
        g.resolve(e, approve=True, approval=sign_approval(e, "alice", KEYS["alice"], now=600))
    with pytest.raises(PermissionError, match="reviewer"):            # the claimed reviewer is not the signer
        g.resolve(e, approve=True, reviewer="bob", approval=sign_approval(e, "alice", KEYS["alice"], now=1000))
    assert paid == [] and not e.resolved
    d = g.resolve(e, approve=True, approval=sign_approval(e, "alice", KEYS["alice"], now=990))
    assert d.outcome == "allow" and d.approved_by == "alice" and paid == [("Bob", 500)]
    c = g.storage.corrections()[-1]
    assert c["by"] == "alice" and g.storage.record(c["id"])["meta"]["guard"]["verified"] == "alice"
    r = g.resolve(other, approve=False, reviewer="carol")             # a rejection needs no token
    assert r.outcome == "deny"


def test_without_a_verifier_resolve_is_as_before(tmp_path):
    g, paid = guard(tmp_path, None)
    e = esc(g)
    assert g.resolve(e, approve=True, reviewer="alice").approved_by == "alice" and paid == [("Bob", 500)]
    with pytest.raises(TypeError):
        Guard(user_verifier="alice")
