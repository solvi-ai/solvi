"""Packed host inputs, paid pilot labels and explicit feedback; no oracle."""
from solvi.experimental.experience import PackedObservationFrame, PackedStratifiedSampler, WorkBudget


def main():
    budget = WorkBudget(100000)
    records = [(f"pilot{i}", {"f0": 0 if i < 4 else 7}) for i in range(8)]
    records += [(f"candidate{i}", {"f0": i % 2 * 7}) for i in range(8)]
    frame = PackedObservationFrame(records, budget, source="toy-host")
    pilot = tuple((f"pilot{i}", i >= 4) for i in range(8))
    sampler = PackedStratifiedSampler(frame, pilot, budget, strategy="adaptive")
    # The host supplies outcomes explicitly; the sampler cannot fetch labels.
    supplied_labels = {f"candidate{i}": bool(i % 2) for i in range(8)}
    name = sampler.choose(budget)
    sampler.observe(name, supplied_labels[name], budget)
    restored = PackedStratifiedSampler.restore(frame, sampler.to_dict(), budget)
    while (name := restored.choose(budget)) is not None:
        restored.observe(name, supplied_labels[name], budget)
    print("observed:", tuple(restored.received))
    print("work units:", budget.spent)


if __name__ == "__main__":
    main()
