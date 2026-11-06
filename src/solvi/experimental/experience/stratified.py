"""Paid label-aware proposal sampling; host labels confer no authority."""

from dataclasses import dataclass

from .competences import ExternalExample
from .observations import ObservationOption


@dataclass(frozen=True)
class StratifiedObservationPlan:
    names: tuple[str, ...]
    features: tuple[str, ...]
    thresholds: tuple[int, ...]
    reason: str


def plan_stratified_observations(examples, options, budget, *, limit=8, candidate_limit=64):
    """Plan unread rows from a bounded ordered pool without reading an oracle.

    Every phase is prepaid on the parent's ledger. Any debit or sink exception
    propagates unchanged; no partially completed plan is a successful result.
    Candidate metadata is checked throughout the pool, but only the bounded
    prefix has its inputs decoded. The host independently shuffles that pool.
    """
    charge = budget.charge
    charge("stratified_validate")
    if (
        type(limit) is not int
        or type(candidate_limit) is not int
        or not 1 <= limit <= 16
        or not limit <= candidate_limit <= 64
    ):
        raise ValueError("require 1 <= limit <= 16 and limit <= candidate_limit <= 64")
    if type(examples) not in (tuple, list) or type(options) not in (tuple, list):
        raise ValueError("bounded tuple/list inputs required")
    if not 1 <= len(examples) <= 64 or not limit <= len(options) <= 256:
        raise ValueError("require 1..64 examples and limit..256 options")
    charge("stratified_validate", len(examples) + len(options))
    context, events, names = None, set(), set()
    for example in examples:
        if type(example) is not ExternalExample or example.split != "proposal" or type(example.expected) is not bool:
            raise ValueError("boolean proposal ExternalExample required")
        current = tuple(getattr(example, key) for key in ("source", "mechanism", "schema", "scope", "task_id"))
        if context is not None and current != context or example.event_id in events:
            raise ValueError("one context and unique event ids required")
        context = current
        events.add(example.event_id)
    for option in options:
        if type(option) is not ObservationOption or option.name in names:
            raise ValueError("unique ObservationOption names required")
        names.add(option.name)
    prefix = options[:candidate_limit]
    charge("stratified_decode", sum(len(e._inputs) for e in examples) + sum(len(o._inputs) for o in prefix))
    rows = [e.inputs for e in examples]
    candidates = [o.inputs for o in prefix]
    p = len(rows[0])
    if not 1 <= p <= 128:
        raise ValueError("require 1..128 quantized features")
    features = tuple(f"f{i}" for i in range(p))
    charge("stratified_validate", (len(rows) + len(candidates)) * (p + 1))
    for row in rows + candidates:
        if set(row) != set(features) or any(type(v) is not int or not 0 <= v <= 7 for v in row.values()):
            raise ValueError("strict contiguous f0..fp-1 integer bins 0..7 required")
    charge("stratified_histogram", len(rows) * p + 8 * p)
    histograms = [[0] * 8 for _ in features]
    positives = [[0] * 8 for _ in features]
    for row, example in zip(rows, examples):
        for i, feature in enumerate(features):
            histograms[i][row[feature]] += 1
            positives[i][row[feature]] += example.expected
    charge("stratified_score", p * 16)
    scores = []
    for i, histogram in enumerate(histograms):
        cumulative = 0
        for threshold, count in enumerate(histogram):
            cumulative += count
            if cumulative > len(rows) // 2:
                break
        left = sum(histogram[:threshold])
        right = len(rows) - left
        if min(left, right) < 4:
            continue
        lp = sum(positives[i][:threshold])
        rp = sum(positives[i][threshold:])
        numerator = abs((lp + 1) * (right + 2) - (rp + 1) * (left + 2))
        if numerator:
            scores.append((i, threshold, numerator, (left + 2) * (right + 2)))
    charge("stratified_select", 4 * p + len(prefix) * 4 + limit * 4)

    def compare(a, b):
        difference = b[2] * a[3] - a[2] * b[3]
        return (difference > 0) - (difference < 0) if difference else a[0] - b[0]

    selected = []
    for score in scores:
        if not selected or compare(score, selected[0]) < 0:
            selected.insert(0, score)
        elif len(selected) == 1 or compare(score, selected[1]) < 0:
            selected.insert(1, score)
        del selected[2:]
    if not selected or len({e.expected for e in examples}) == 1:
        return StratifiedObservationPlan(tuple(o.name for o in prefix[:limit]), (), (), "random_prefix")
    queues = [[] for _ in range(1 << len(selected))]
    for option, row in zip(prefix, candidates):
        code = sum((row[features[entry[0]]] >= entry[1]) << bit for bit, entry in enumerate(selected))
        queues[code].append(option.name)
    result, positions = [], [0] * len(queues)
    while len(result) < limit:
        for code, queue in enumerate(queues):
            if positions[code] < len(queue):
                result.append(queue[positions[code]])
                positions[code] += 1
                if len(result) == limit:
                    break
    return StratifiedObservationPlan(
        tuple(result), tuple(features[s[0]] for s in selected), tuple(s[1] for s in selected), "stratified"
    )
