# LoRA training-time estimate

Before constructing the optimizer, training runs one unapplied cold forward/backward probe and three unapplied warm probes on the first training batch. It discards the cold duration and reports the median warm duration times the requested update count. Gradients are cleared after every probe, including when a probe fails. No probe performs an optimizer step; the actual update count and seeded minibatch order are unchanged.

The adapter's `info['estimate']` contains the discarded cold duration, three warm durations and the estimate basis. A caller's `notify(estimate_seconds, updates, per_update, device)` still runs before optimizer construction and may refuse training there. CPU probes do not synchronize or initialize CUDA. CUDA timing explicitly synchronizes the selected device; this path needs device-specific validation.

This is a diagnostic compute estimate, not a runtime budget guarantee. It excludes optimizer-step and gradient-clipping overhead, probe/preparation time and changes in batch lengths or machine load. A short training may spend more time in preflight than in updates. A tight wall-clock budget needs its own cancellation/deadline mechanism. At n=400, six epochs still request about 300 updates: stabilizing the estimate does not make a long CPU training cheap or establish a hardware-independent latency claim.
