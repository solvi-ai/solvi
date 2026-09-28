# Security policy

## Supported versions

Security fixes go into the latest minor release (currently 0.5.x). solvi is alpha software; older releases are not patched.

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through
[GitHub security advisories](https://github.com/solvi-ai/solvi/security/advisories/new) or by e-mail to **hi@mxkuzn.dev**
with "solvi security" in the subject. Include the version, a minimal catalog / request that shows the problem, and what an
attacker gains. You will get an answer within 7 days; a fix and an advisory follow as soon as the problem is confirmed, and
you are credited unless you prefer otherwise.

## What counts

solvi's promises are about verification, so these are in scope:

- a trace that was altered but still passes `trace.replay` (a broken hash chain or a changed value that is not reported);
- a model output that is not literally grounded (a quote not at its offsets, an answer outside its options) but is accepted;
- a hard check that fails and does not force its answer;
- loading a response or trace from JSON (`Response.from_json`, `model_validate`) that executes code or reads files;
- code execution through a checkpoint's metadata (`solvi_decide.json`, `solvi_strategist.json`).
- `solvi serve`: a request that gets past the bearer token, the size / depth limits or the timeout, a response that
  carries a traceback or a server path, or request data that makes the server import or load something (see the
  guide's Serving chapter, Security).

Out of scope: running untrusted Python catalog code — a catalog is code and runs with your privileges (the playground Space
runs visitors' code only in their own browser); model accuracy (a wrong but grounded answer is a quality issue, please open a
normal issue). Checkpoints: the decider and the strategist load safetensors and ONNX only; the extractors' `span_head.pt` is
read with `torch.load(..., weights_only=True)` — a way around that is in scope.
