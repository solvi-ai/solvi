"""The `solvi[onnx]` extra covers the decider only: LongSpanExtractor without torch says so."""
import importlib

import pytest


def test_the_long_span_extractor_says_the_onnx_extra_does_not_cover_it(monkeypatch):
    real = importlib.import_module

    def no_torch(name, *a, **kw):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real(name, *a, **kw)
    monkeypatch.setattr(importlib, "import_module", no_torch)
    from solvi.core.extract import LongSpanExtractor
    with pytest.raises(ImportError, match=r"solvi\[onnx\] covers the decider, not this extractor"):
        LongSpanExtractor()
