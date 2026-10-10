# `solvi.core.contract`

The contract of an answer: its type matches what the question asks, it is not the input it was read from, and an
applicability precondition is checked before the answerer is called (`Catalog.contract`, a slow path's `contract=`,
`Specialist.applicable`).

The echo clause's default (`echo=None`) follows the answer's kind: it is checked for free text (a span, a `str`
answer, `returns=str`) and not for a declared choice, yes / no, enumeration or number, which may rightly equal an
input field; an explicit `echo=True`, `False` or a list of fact names always overrides it.

::: solvi.core.contract
