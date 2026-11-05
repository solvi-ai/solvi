import pytest
from solvi.core.knowledge import KnowledgeStore
from solvi.solutions.guard import Guard, approval_verifier, sign_approval
from solvi.core.signed import HmacScheme, SignatureCache


def test_writer_registry_is_copied_reconstruction_controls_future_writes(tmp_path):
    path = tmp_path / "knowledge.jsonl"
    registry = {"alice": "outcome"}
    s = KnowledgeStore(path, writers=registry)
    old = s.add("fact", {"s": "old", "r": "v", "o": 1}, source="outcome", by="alice")
    registry.clear()
    assert s.add("fact", {"s": "copied", "r": "v", "o": 1}, source="outcome", by="alice")
    replacement = KnowledgeStore(path, writers={})
    assert replacement.add("fact", {"s": "new", "r": "v", "o": 1}, source="outcome", by="alice") is None
    assert replacement.usable(old) and replacement.verify()
    assert KnowledgeStore(gates=[]).add("fact", {"s": "model", "r": "v", "o": 1}, source="model") is None


def guard():
    g = Guard(user_verifier=approval_verifier({"alice": b"offline-fixture"}, clock=lambda: 1000))
    calls = []

    @g.tool
    def touch(x: int):
        calls.append(x)
        return x

    @g.policy("touch", on_fail="escalate")
    def manual(x: int):
        return False

    return g, calls


def test_declared_new_policy_invalidates_compiled_check():
    g, calls = guard()
    call = {"name": "touch", "arguments": {"x": 1}}
    assert g.check(call, "touch 1").outcome == "escalate"

    @g.policy("touch")
    def revoked(x: int):
        return False

    assert g.check(call, "touch 1").outcome == "deny" and calls == []


def test_new_verifier_rejects_revoked_reviewer_forgery_future_token():
    g, calls = guard()
    call = {"name": "touch", "arguments": {"x": 1}}
    d = g.check(call, "touch 1")
    assert d.outcome == "escalate"
    for token in ("alice.1000.00ff", sign_approval(d, "alice", b"offline-fixture", now=1100)):
        with pytest.raises(PermissionError):
            g.resolve(d, approve=True, approval=token)
    token = sign_approval(d, "alice", b"offline-fixture", now=1000)
    with pytest.raises(PermissionError, match="reviewer"):
        g.resolve(d, approve=True, approval=token, reviewer="forged")
    g.user_verifier = approval_verifier({}, clock=lambda: 1000)
    with pytest.raises(PermissionError, match="unknown reviewer"):
        g.resolve(d, approve=True, approval=token)
    assert not d.resolved and calls == []


def test_native_mac_revoke_invalidates_cached_authentication():
    scheme = HmacScheme({"alice": b"offline-fixture"})
    cache = SignatureCache()
    body = b"exact-offline-body"
    tag = scheme.sign("alice", body)
    assert scheme.verify_cached("alice", body, tag, cache)
    assert scheme.verify_cached("alice", body, tag, cache)
    scheme.revoke("alice")
    assert not scheme.verify_cached("alice", body, tag, cache)


def test_native_key_rotation_rejects_old_tag_even_with_warm_cache():
    scheme = HmacScheme({"alice": b"old-offline-fixture"})
    cache = SignatureCache()
    body = b"only-fixture"
    old = scheme.sign("alice", body)
    assert scheme.verify_cached("alice", body, old, cache)
    scheme.register("alice", b"new-offline-fixture")
    assert not scheme.verify_cached("alice", body, old, cache)
    new = scheme.sign("alice", body)
    assert scheme.verify_cached("alice", body, new, cache)
