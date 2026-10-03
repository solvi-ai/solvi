"""How fast is the strategist on large catalogs? Random layered catalogs of N parts (functions over 1-3 facts from earlier layers,
checks on computed facts, 5 questions answered by rules over deep facts). Measures planning time, execution time, and how much
of the catalog the chosen flow actually runs compared with running every part.

Run:  uv run python benchmarks/strategist_scale.py"""
from __future__ import annotations

import random
import statistics
import time

from solvi import Answer, Catalog, Question, System
from solvi.core.plan.strategist import plan


def make_catalog(n_parts, n_inputs=20, layers=12, seed=0):
    rng = random.Random(seed)
    cat = Catalog()
    facts = [[f"x{i}" for i in range(n_inputs)]]
    per_layer = max(1, n_parts // layers)
    k = 0
    for layer in range(1, layers + 1):
        new = []
        pool = [f for lv in facts for f in lv]
        for _ in range(per_layer):
            args = rng.sample(pool, rng.randint(1, min(3, len(pool))))
            name = f"f{k}"
            k += 1
            if rng.random() < 0.15:
                src = f"def {name}({', '.join(args)}):\n    return sum(({' + '.join(args)},)) % 7 != 0\n"
                ns = {}
                exec(src, ns)
                cat.check(ns[name])
            else:
                src = f"def {name}({', '.join(args)}):\n    return ({' + '.join(args)}) % 1000 + {layer}\n"
                ns = {}
                exec(src, ns)
                cat.fn(ns[name])
                new.append(name)
        facts.append(new)
    deep = [f for lv in facts[-3:] for f in lv]
    qs = []
    for j in range(5):
        a, b = rng.sample(deep, 2)
        ns = {}
        exec(f"def r{j}({a}, {b}):\n    return ({a} + {b}) % 2 == 0\n", ns)
        cat.rule(f"q{j}")(ns[f"r{j}"])
        qs.append(Question(f"q{j}", f"question {j}", Answer.yes_no()))
    state = {f"x{i}": rng.randint(0, 100) for i in range(n_inputs)}
    return cat, qs, state


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


if __name__ == "__main__":
    print(f"{'parts':>7} {'plan ms':>9} {'ask ms':>9} {'flow steps':>11} {'of catalog':>11} {'run-all ms':>11}")
    for n in (100, 1000, 10000):
        cat, qs, state = make_catalog(n)
        system = System(cat, qs)
        reps = 20 if n <= 1000 else 5
        t_plan = timeit(lambda: plan(cat, qs, state.keys()), reps)
        t_ask = timeit(lambda: system.ask(state), reps)
        steps = len(system.ask(state).flow.steps)
        vals = dict(state)

        def run_all():
            for p in cat.parts.values():                   # parts are stored in dependency order
                vals[p.name] = p.func(**{x: vals[x] for x in p.inputs})
        t_all = timeit(run_all, reps)
        total = len(cat.parts) + len(cat.rules)
        print(f"{total:>7} {t_plan:>9.2f} {t_ask:>9.2f} {steps:>11} {steps / total:>10.1%} {t_all:>11.2f}")
