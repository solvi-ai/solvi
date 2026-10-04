"""A catalog that solvi 1.0.0 and later versions run unchanged, to prove that compact records written by 1.0.0 still
load, verify and replay (tests/test_store_1_0_0.py). make.py wrote the files in this folder with solvi 1.0.0 from PyPI.

system(): a hard check whose `then` is a function of facts (1.0) — a compact record of 1.0.0 does not keep the "then"
record it writes, so its replay says "not_kept" (1.0.1 keeps it and checks it) — and a hard check with a constant
`then`, a function and a rule."""
from typing import Literal

from solvi import Catalog, Question, System


def free_side(free: list, facing: str) -> str:
    """The step to take instead: the first free side that is not where we face."""
    return next((d for d in free if d != facing), free[0])


def system(storage=None):
    """A fresh System."""
    cat = Catalog()

    @cat.fn
    def free(walls: list) -> list:
        return [d for d in ("N", "E", "S", "W") if d not in walls]

    @cat.check(hard=True, then={"step": free_side})
    def not_looping(history: list) -> bool:
        return not (len(history) > 2 and len(set(history[-3:])) == 1)

    @cat.check(hard=True, then={"fight": "no"})
    def has_hearts(hearts: int) -> bool:
        return hearts > 0

    @cat.rule("step")
    def step(plan: str) -> Literal["N", "E", "S", "W"]:
        return plan

    @cat.rule("fight")
    def fight(enemies: int) -> bool:
        return enemies < 3
    return System(cat, [Question("step", "Which way?"), Question("fight", "Fight?")], storage=storage)


STATES = [
    {"plan": "N", "facing": "N", "walls": ["E"], "history": [4, 4, 4], "hearts": 3, "enemies": 1},   # then function
    {"plan": "W", "facing": "N", "walls": [], "history": [1, 2, 3], "hearts": 0, "enemies": 1},      # constant then
    {"plan": "E", "facing": "E", "walls": ["N"], "history": [5, 5, 5], "hearts": 0, "enemies": 4},   # both
    {"plan": "S", "facing": "S", "walls": [], "history": [1, 2], "hearts": 2, "enemies": 5},         # neither
]
THEN_FUNCTION = [True, False, True, False]   # which decisions a `then` function answered (not_kept in a 1.0.0 record)
