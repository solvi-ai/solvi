"""Without torch, training a LongSpanExtractor names both extras: [model] to train, [onnx] to run a saved one."""
import importlib

import pytest


def test_the_long_span_extractor_without_torch_points_to_the_onnx_backend_for_prediction(monkeypatch):
    real = importlib.import_module

    def no_torch(name, *a, **kw):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real(name, *a, **kw)
    monkeypatch.setattr(importlib, "import_module", no_torch)
    from solvi.core.extract import LongSpanExtractor
    with pytest.raises(ImportError, match=r"solvi\[model\].*solvi\[onnx\] runs a saved extractor without torch: "
                                          r"LongSpanExtractor.load\(path_or_id, backend=\"onnx\"\)"):
        LongSpanExtractor()
