"""A trusted driver for tests/test_sandbox.py: calls the sandboxed module's `area` on each radius."""


def areas(ns, payload, signal):
    out = []
    for r in payload["rs"]:
        signal.alarm(2)
        try:
            out.append(ns["area"](r))
        finally:
            signal.alarm(0)
    return {"areas": out}
