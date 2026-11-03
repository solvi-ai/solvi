"""S321: simulate unsupported Dynamo while loading, without torch/models/downloads."""
import sys
from types import SimpleNamespace

import pytest

from solvi.core.deciders import backends


def loader(monkeypatch, *, retry_error=None):
    calls = []
    def compile(model=None, **kwargs):
        raise RuntimeError("Dynamo is not supported on Python 3.12+")
    torch = SimpleNamespace(compile=compile)
    configs = []
    def config(path):
        cfg = SimpleNamespace(reference_compile=True)
        configs.append(cfg)
        return cfg
    def build(cfg, **kwargs):
        calls.append((cfg.reference_compile, kwargs))
        assert torch.compile(lambda: "ok")() == "ok"
        if retry_error:
            raise retry_error
        return cfg
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=config), AutoModel=SimpleNamespace(from_config=build)))
    monkeypatch.delenv("SOLVI_NO_COMPILE", raising=False)
    return torch, compile, calls, configs


def test_unsupported_dynamo_retries_without_compile_and_restores_it(monkeypatch):
    torch, original, calls, configs = loader(monkeypatch)
    cfg = backends._load_torch_encoder(torch, "fake", block=True)
    assert calls == [(True, {"attn_implementation": "sdpa"}), (False, {"attn_implementation": "sdpa"})]
    assert len(configs) == 2 and not cfg.reference_compile and cfg._attn_implementation == "sdpa"
    assert torch.compile is original


def test_explicit_no_compile_does_not_attempt_normal_loading(monkeypatch):
    torch, original, calls, _ = loader(monkeypatch)
    monkeypatch.setenv("SOLVI_NO_COMPILE", "1")
    cfg = backends._load_torch_encoder(torch, "fake", block=False)
    assert calls == [(False, {})] and not cfg.reference_compile
    assert torch.compile is original and backends.no_compile()


def test_retry_failure_restores_compile_and_propagates(monkeypatch):
    failure = RuntimeError("encoder weights are corrupt")
    torch, original, calls, _ = loader(monkeypatch, retry_error=failure)
    with pytest.raises(RuntimeError) as exc:
        backends._load_torch_encoder(torch, "fake", block=False)
    assert exc.value is failure and len(calls) == 2
    assert torch.compile is original


def test_other_runtime_errors_are_not_retried(monkeypatch):
    torch, original, calls, _ = loader(monkeypatch)
    failure = RuntimeError("out of memory")
    def fail(cfg, **kwargs):
        calls.append("failed")
        raise failure
    sys.modules["transformers"].AutoModel.from_config = fail
    with pytest.raises(RuntimeError) as exc:
        backends._load_torch_encoder(torch, "fake", block=False)
    assert exc.value is failure and calls == ["failed"]
    assert torch.compile is original
