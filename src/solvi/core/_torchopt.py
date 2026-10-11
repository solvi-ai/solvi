"""The AdamW solvi trains with (the LoRA adapter, the long-span extractor): on the CPU it creates no CUDA context."""
from __future__ import annotations

__all__ = ["adamw"]


def adamw(params, device, **kw):
    """torch.optim.AdamW(params, **kw). On the CPU, torch's check for accelerator graph capture at step() is skipped: it
    asks the accelerator for its current stream, which creates a CUDA context even for CPU parameters (and fails with
    out of memory when the GPU is held by another job); CPU parameters cannot be graph-captured, so the check has
    nothing to find there. device: a torch.device or a device string."""
    import torch
    opt = torch.optim.AdamW(params, **kw)
    if torch.device(device).type == "cpu":
        for name in ("_accelerator_graph_capture_health_check", "_cuda_graph_capture_health_check"):
            if hasattr(opt, name):                 # the current name and its older one
                setattr(opt, name, lambda: None)
    return opt
