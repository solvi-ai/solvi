"""Constructor domains and shared transport forwarding, with no network or real sleep."""

import io
import pytest
from solvi.core.deciders._remote import RemoteClient
from solvi.core.deciders.llm import LLMScorer, llm
from solvi.core.deciders.systemone import SystemOneScorer, systemone
from solvi.experimental.charts.propose import LLMProposer


def client(**kwargs):
    return RemoteClient("https://example.invalid", "test", **kwargs)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("timeout", 0),
        ("timeout", -1),
        ("timeout", True),
        ("timeout", float("nan")),
        ("timeout", float("inf")),
        ("timeout", None),
        ("timeout", []),
        ("timeout", 10**500),
        ("backoff", -1),
        ("backoff", True),
        ("backoff", float("nan")),
        ("backoff", float("inf")),
        ("backoff", None),
        ("retries", -1),
        ("retries", True),
        ("retries", 1.5),
        ("retries", 2.0),
        ("retries", "2"),
        ("retries", None),
        ("retries", float("inf")),
        ("retries", float("nan")),
    ],
)
def test_invalid_transport_configuration_refuses_before_callbacks(field, bad):
    calls = []
    with pytest.raises(ValueError, match=field):
        client(**{field: bad}, opener=lambda *_a, **_k: calls.append("network"), sleep=lambda _: calls.append("sleep"))
    assert calls == []


FACTORIES = [
    client,
    lambda **kw: LLMScorer("https://example.invalid", "test", **kw),
    lambda **kw: llm("https://example.invalid", "test", **kw).scorer,
    lambda **kw: LLMProposer("https://example.invalid", "test", **kw).client,
    lambda **kw: SystemOneScorer("https://example.invalid", "test", **kw),
    lambda **kw: systemone("https://example.invalid", "test", **kw).scorer,
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_explicit_options_reach_every_retry_without_network(factory):
    timeouts = []
    delays = []

    def opener(request, timeout=None):
        timeouts.append(timeout)
        if len(timeouts) <= 2:
            raise TimeoutError("local fake failure")
        return io.BytesIO(b'{"ok":true}')

    sc = factory(timeout="0.25", retries=2, backoff="0.5", opener=opener, sleep=delays.append)
    assert sc.send({}) == {"ok": True}
    assert timeouts == [0.25, 0.25, 0.25] and delays == [0.5, 1.0]
    assert (sc.timeout, sc.retries, sc.backoff) == (0.25, 2, 0.5)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("field,bad", [("timeout", -1), ("retries", 1.5), ("backoff", -1)])
def test_wrappers_do_not_discard_invalid_options(factory, field, bad):
    with pytest.raises(ValueError, match=field):
        factory(**{field: bad})


def test_defaults_and_zero_backoff_no_extra_retry():
    from solvi.core.deciders._remote import NoAnswer

    sc = client()
    assert (sc.timeout, sc.retries, sc.backoff) == (60.0, 2, 1.0)
    calls = []
    sleeps = []

    def down(*_a, **_k):
        calls.append(1)
        raise TimeoutError("fake")

    sc = client(timeout=1, retries=0, backoff=0, opener=down, sleep=sleeps.append)
    with pytest.raises(NoAnswer, match="1 attempts"):
        sc.send({})
    assert calls == [1] and sleeps == []


@pytest.mark.parametrize("field", ["timeout", "retries", "backoff"])
@pytest.mark.parametrize("value", [True, False])
def test_numpy_boolean_is_not_a_transport_number(field, value):
    import numpy as np

    with pytest.raises(ValueError, match=field):
        client(**{field: np.bool_(value)})


def test_numpy_numeric_decimal_and_numeric_strings_remain_supported():
    from decimal import Decimal
    import numpy as np

    for timeout, backoff in [(np.float64(0.25), np.int64(0)), (Decimal(".25"), Decimal("0")), (".25", "0")]:
        sc = client(timeout=timeout, retries=np.int64(2), backoff=backoff)
        assert (sc.timeout, sc.retries, sc.backoff) == (0.25, 2, 0.0)
