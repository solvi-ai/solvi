# Solvi 1.5.0

Prepared for review; not published.

This release expands explicit decision, knowledge and checking APIs while preserving abstention when required evidence is missing. It includes priced answer checks, owner-declared evidence identities, clearer diagnostics, bounded asynchronous regime memory and tools for inspecting recorded decisions.

The knowledge system now handles pending later values conservatively: a writer's unsupported newer value can leave a gap instead of silently restoring that writer's obsolete direct value. Independent support from other writers remains eligible. Applications should review the migration notes before upgrading.

Ticket services separate request-producing and blind-oracle roles. The host must authenticate those roles and separate their transports and storage access. These views do not provide network authentication or prevent information leakage through arbitrary access to the underlying system.

Source reliability queries use a per-source history index. Repeated fingerprints of unchanged native registries reuse the same snapshot hash; supported writes invalidate it, while custom snapshot providers retain full hashing. Retained history still grows, and these changes do not establish a general latency or memory bound.

Transport settings reject invalid timeout, retry-count and backoff values before a request. Explicit batch-scoring and cache-refresh APIs expose their ownership and stability assumptions. Offline accounting examples distinguish purchased checks, wrong actions and abstentions using explicit prices and independently observed outcomes.

Experimental components retain their documented assumptions and supported subsets. This release does not establish automatic discovery of causal error costs, population-wide detection of hidden model changes, a universal statistical guarantee or autonomous improvement in quality. Deployment, publication and model-quality evaluation remain separate decisions.

See the [changelog](../CHANGELOG.md) and [migration guide](migration-1.5-candidate.md) for complete compatibility details.
