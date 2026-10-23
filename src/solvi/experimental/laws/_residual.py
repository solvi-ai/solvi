"""An opt-in residual check for a fitted numeric law: the χ² after a refit on all rows, as a `confirm=` closure."""
from __future__ import annotations

import math
from typing import Any, Callable

__all__ = ["residual_check"]


def residual_check(refit: Callable[[Any], Any], level: float = 0.995) -> Callable[[Any], bool]:
    """→ a `confirm(candidate)` closure for `LawSet(confirm=)` that keeps a fitted numeric law only while the χ² of its
    residuals after a refit is below the `level` quantile of χ² at n − k degrees of freedom. Opt-in; nothing changes
    unless you pass the closure.

        check = residual_check(refit)                       # level=0.995
        laws = LawSet(g, apply=..., tol=math.inf, confirm=check)

    refit(candidate) → (residuals, k), or None when the candidate cannot be fitted (refused). The refit is yours:
    fit the candidate's constants on the design rows and the held-out rows together, and return the residuals of every
    one of those rows in units of the measurement error, (observed − predicted) / σ, with k the number of fitted
    constants. The check is Σ r² ≤ χ²_level(n − k), n = len(residuals); n − k ≤ 0, a residual that is not finite, or
    None is a refusal. level: in (0, 1); 0.995 is 1 − α/2 at α = 0.01, the level measured.

    Use it with `tol=math.inf`: the row band is then off (every held-out row is a hit), `min_support` and net > 0 still
    apply, and the χ² decides. Under `tol=math.inf` the sized bound is vacuous — with every row a hit it no longer
    binds — so the false-accept control is the χ² level alone, and the level has no term for the number of candidates
    tested (four were tested where it was measured). With a finite `tol` the check is asked after the row test, on
    the candidates that passed it.

    Why. The row test counts held-out rows inside `tol`: a 4σ band has almost no power against a bias of 1–2σ, and when
    the law's constants are fitted it refuses the true law whose predictions carry a fit error larger than σ. Measured
    once, on synthetic physics worlds re-created from another group's description: on a pendulum whose formula list
    lacks the bob radius (200 worlds, 70 where the best wrong formula sits 1–2σ from the truth) the share of wrong laws
    accepted in the 1–2σ bin went 0.60 → 0.14 over those 70 worlds (rows 42 of 70, this check 10 of 70); on Kepler
    orbits at Tycho's precision (0.03°) Kepler was accepted 4 → 12 of 12 skies; where the truth is in the list nothing
    was lost (179 of 200 right and 0 wrong, the row rule 179 and 1). A sign test on the same residuals refused nothing
    in 415 cases at 12–14 rows and is not part of the check.

    Cautions. The refit χ² checks goodness of fit, not prediction: a flexible wrong model can pass it — an epicycle
    system with 25 constants did on 1 of 12 extra skies, where the row test on the held-out rows refused it; keep
    n − k large, and prefer a held-out check when the model is flexible. The level carries no candidate-count term.
    One task, physics only: a default change waits for a second task outside physics."""
    level = float(level)
    if not 0.0 < level < 1.0 or math.isnan(level):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    from scipy.stats import chi2

    def confirm(candidate) -> bool:
        out = refit(candidate)
        if out is None:
            return False
        residuals, k = out
        r = [float(x) for x in residuals]
        dof = len(r) - int(k)
        if dof <= 0 or not all(math.isfinite(x) for x in r):
            return False
        return bool(sum(x * x for x in r) <= chi2.ppf(level, dof))

    confirm.level = level                                  # type: ignore[attr-defined]
    return confirm
