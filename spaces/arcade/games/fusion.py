"""Card fusion lab: fuse two weapon cards with a fusion law, then let a solvi catalog balance the result.

Laws
  BLEND     order does not matter: shape of the stronger parent, numbers averaged in log space (geometric mean),
            effects stack (added).
  DOMINANT  "A in the style of B", order matters: shape of A, numbers 75% A / 25% B in log space, B's effects stack
            onto A's, then B's signature trait is applied.

Balance (solvi): hard checks slow <= 0.5, lifesteal <= 0.4, cooldown >= 0.3 s (a violation is clamped and reported);
soft check: power within [0.5 x median, 1.5 x max] of the base cards (outside -> damage is scaled and the factor reported)."""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, replace

from solvi import Answer, Catalog, Question, System

SLOW_CAP, LIFESTEAL_CAP, COOLDOWN_MIN = 0.5, 0.4, 0.3
NUMERIC = ["damage", "fire_rate", "area", "speed", "pierce"]


@dataclass(frozen=True)
class Card:
    name: str
    prefix: str          # used when this card gives its style to a fusion ("Frost ...")
    noun: str            # used when this card gives its shape ("... Whip")
    icon: str
    shape: str
    damage: float
    fire_rate: float     # shots per second (cooldown = 1 / fire_rate)
    area: float
    speed: float
    pierce: float
    slow: float = 0.0
    burn: float = 0.0    # extra damage over time, as a fraction of hit damage
    lifesteal: float = 0.0
    signature: str = ""  # trait applied when this card is the style (B) in DOMINANT

    @property
    def cooldown(self):
        return 1.0 / self.fire_rate


SIGNATURES = {  # name -> (description, function Card -> Card)
    "Whip": ("reach: area x1.3", lambda c: replace(c, area=c.area * 1.3)),
    "Magic Wand": ("homing: speed x1.5, pierce +1", lambda c: replace(c, speed=c.speed * 1.5, pierce=c.pierce + 1)),
    "Frost Orb": ("chill: slow +0.15", lambda c: replace(c, slow=c.slow + 0.15)),
    "Fire Staff": ("ignite: burn +0.4", lambda c: replace(c, burn=c.burn + 0.4)),
    "Vampire Fang": ("thirst: lifesteal +0.15", lambda c: replace(c, lifesteal=c.lifesteal + 0.15)),
    "Thunder Hammer": ("weight: damage x1.4, fire rate x0.8", lambda c: replace(c, damage=c.damage * 1.4, fire_rate=c.fire_rate * 0.8)),
    "Throwing Knives": ("flurry: fire rate x1.6", lambda c: replace(c, fire_rate=c.fire_rate * 1.6)),
    "Poison Cloud": ("spread: area x1.5, damage x0.7", lambda c: replace(c, area=c.area * 1.5, damage=c.damage * 0.7)),
}

BASE = [
    Card("Whip", "Lashing", "Whip", "🪢", "arc", 12, 1.2, 2.0, 4, 5, lifesteal=0.2),
    Card("Magic Wand", "Arcane", "Wand", "🪄", "bolt", 10, 2.0, 0.5, 9, 1),
    Card("Frost Orb", "Frost", "Orb", "🔮", "orb", 8, 0.8, 2.5, 3, 3, slow=0.35),
    Card("Fire Staff", "Blazing", "Staff", "🔥", "fireball", 14, 1.0, 1.5, 5, 1, burn=0.6),
    Card("Vampire Fang", "Vampiric", "Fang", "🦇", "bite", 12, 1.5, 0.6, 7, 1, lifesteal=0.25),
    Card("Thunder Hammer", "Thunder", "Hammer", "🔨", "shockwave", 30, 0.5, 3.0, 2, 8, slow=0.2),
    Card("Throwing Knives", "Swift", "Knives", "🗡️", "volley", 6, 3.0, 0.3, 12, 2),
    Card("Poison Cloud", "Toxic", "Cloud", "☠️", "cloud", 3, 1.0, 4.0, 1, 10, burn=1.0),
]
BASE = [replace(c, signature=SIGNATURES[c.name][0]) for c in BASE]
BY_NAME = {c.name: c for c in BASE}


def targets_of(pierce, area):
    """expected enemies hit per shot: area finds them, pierce caps them"""
    return min(pierce, 1 + area)


def power_of(damage, fire_rate, pierce, area, burn, slow, lifesteal):
    return damage * fire_rate * targets_of(pierce, area) * (1 + burn) * (1 + slow) * (1 + lifesteal)


def card_power(c: Card):
    return power_of(c.damage, c.fire_rate, c.pierce, c.area, c.burn, c.slow, c.lifesteal)


BASE_POWERS = tuple(round(card_power(c), 3) for c in BASE)


def _lmix(a, b, wa):
    return math.exp(wa * math.log(a) + (1 - wa) * math.log(b))


def fuse(a: Card, b: Card, law: str) -> Card:
    """Raw fusion (before balancing)."""
    law = law.upper()
    if law == "BLEND":
        pa, pb = card_power(a), card_power(b)
        strong, weak = (a, b) if (pa, a.name) >= (pb, b.name) else (b, a)
        nums = {k: _lmix(getattr(a, k), getattr(b, k), 0.5) for k in NUMERIC}
        name = f"Twin {strong.noun}" if a == b else f"{weak.prefix} {strong.noun}"
        shape_from, style_from = strong, weak
    elif law == "DOMINANT":
        nums = {k: _lmix(getattr(a, k), getattr(b, k), 0.75) for k in NUMERIC}
        name = f"Mighty {a.noun}" if a == b else f"{b.prefix} {a.noun}"
        shape_from, style_from = a, b
    else:
        raise ValueError(f"unknown law {law}")
    nums["pierce"] = float(max(1, round(nums["pierce"])))
    c = Card(name, style_from.prefix, shape_from.noun, shape_from.icon + (style_from.icon if style_from != shape_from else ""),
             shape_from.shape, slow=a.slow + b.slow if a != b else a.slow, burn=a.burn + b.burn if a != b else a.burn,
             lifesteal=a.lifesteal + b.lifesteal if a != b else a.lifesteal, **nums)
    if law == "DOMINANT":
        c = SIGNATURES[b.name][1](c)
        c = replace(c, pierce=float(max(1, round(c.pierce))))
    return replace(c, signature="")


# --------------------------------------------------------------------------------------------------------------------
# The balance catalog
# --------------------------------------------------------------------------------------------------------------------

cat = Catalog()


@cat.fn
def cooldown(fire_rate):
    """seconds between shots"""
    return round(1.0 / fire_rate, 4)


@cat.fn
def targets(pierce, area):
    return round(targets_of(pierce, area), 4)


@cat.fn
def raw_dps(damage, fire_rate, targets):
    """hit damage per second over all enemies hit, before effects"""
    return round(damage * fire_rate * targets, 3)


@cat.fn
def slow_final(slow):
    return min(slow, SLOW_CAP)


@cat.fn
def lifesteal_final(lifesteal):
    return min(lifesteal, LIFESTEAL_CAP)


@cat.fn
def fire_rate_final(fire_rate):
    return min(fire_rate, 1.0 / COOLDOWN_MIN)


@cat.fn
def power_after_clamps(raw_dps, fire_rate, fire_rate_final, burn, slow_final, lifesteal_final):
    """effective power = raw dps x (1 + burn) x (1 + slow) x (1 + lifesteal), with the hard limits applied"""
    return round(raw_dps * fire_rate_final / fire_rate * (1 + burn) * (1 + slow_final) * (1 + lifesteal_final), 3)


@cat.fn
def band_lo(base_powers):
    return round(0.5 * statistics.median(base_powers), 3)


@cat.fn
def band_hi(base_powers):
    return round(1.5 * max(base_powers), 3)


@cat.check
def power_in_band(power_after_clamps, band_lo, band_hi):
    """soft: power within [0.5 x median, 1.5 x max] of the base cards"""
    return band_lo <= power_after_clamps <= band_hi


@cat.fn
def damage_scale(power_after_clamps, band_lo, band_hi):
    """factor on damage that brings power back to the nearest edge of the band (1.0 when inside)"""
    if power_after_clamps > band_hi:
        return round(band_hi / power_after_clamps, 4)
    if power_after_clamps < band_lo:
        return round(band_lo / power_after_clamps, 4)
    return 1.0


@cat.check(hard=True, then={"slow": "clamp", "ship_as_is": "no"})
def slow_within_cap(slow):
    """hard: slow <= 0.5"""
    return slow <= SLOW_CAP + 1e-9


@cat.check(hard=True, then={"lifesteal": "clamp", "ship_as_is": "no"})
def lifesteal_within_cap(lifesteal):
    """hard: lifesteal <= 0.4"""
    return lifesteal <= LIFESTEAL_CAP + 1e-9


@cat.check(hard=True, then={"cooldown": "clamp", "ship_as_is": "no"})
def cooldown_ok(cooldown):
    """hard: cooldown >= 0.3 s"""
    return cooldown >= COOLDOWN_MIN - 1e-9


@cat.rule("slow")
def slow_rule(slow):
    return "keep"


@cat.rule("lifesteal")
def lifesteal_rule(lifesteal):
    return "keep"


@cat.rule("cooldown")
def cooldown_rule(cooldown):
    return "keep"


@cat.rule("power")
def power_rule(power_in_band, power_after_clamps, band_lo, band_hi, damage_scale):
    if power_in_band:
        return "in_band"
    return "scale_down" if power_after_clamps > band_hi else "scale_up"


@cat.rule("ship_as_is")
def ship_rule(power_in_band):
    return power_in_band


QUESTIONS = [
    Question("slow", "Keep the slow as fused?", Answer.choice(["keep", "clamp"]), checkpoints=["slow_within_cap"]),
    Question("lifesteal", "Keep the lifesteal as fused?", Answer.choice(["keep", "clamp"]), checkpoints=["lifesteal_within_cap"]),
    Question("cooldown", "Keep the cooldown as fused?", Answer.choice(["keep", "clamp"]), checkpoints=["cooldown_ok"]),
    Question("power", "Is the power in the base cards' band?", Answer.choice(["in_band", "scale_down", "scale_up"])),
    Question("ship_as_is", "Can the fused card ship unchanged?", Answer.yes_no(),
             checkpoints=["slow_within_cap", "lifesteal_within_cap", "cooldown_ok"]),
]
system = System(cat, QUESTIONS)


def card_state(c: Card):
    return {"damage": c.damage, "fire_rate": c.fire_rate, "area": c.area, "speed": c.speed, "pierce": c.pierce,
            "slow": c.slow, "burn": c.burn, "lifesteal": c.lifesteal, "base_powers": BASE_POWERS}


def balance(c: Card):
    """Run the balance catalog. Returns (final card, response, list of human-readable changes)."""
    resp = system.ask(card_state(c))
    v = resp.values
    changes = []
    out = c
    if resp["slow"].answer == "clamp":
        changes.append(f"slow {c.slow:.2f} → {SLOW_CAP:.2f} (hard check slow_within_cap)")
        out = replace(out, slow=v["slow_final"])
    if resp["lifesteal"].answer == "clamp":
        changes.append(f"lifesteal {c.lifesteal:.2f} → {LIFESTEAL_CAP:.2f} (hard check lifesteal_within_cap)")
        out = replace(out, lifesteal=v["lifesteal_final"])
    if resp["cooldown"].answer == "clamp":
        changes.append(f"cooldown {c.cooldown:.2f}s → {COOLDOWN_MIN:.2f}s (hard check cooldown_ok)")
        out = replace(out, fire_rate=v["fire_rate_final"])
    p = resp["power"].answer
    if p in ("scale_down", "scale_up"):
        s = v["damage_scale"]
        changes.append(f"power {v['power_after_clamps']:.1f} is {'above' if p == 'scale_down' else 'below'} the band "
                       f"[{v['band_lo']:.1f}, {v['band_hi']:.1f}] → damage x{s:.2f} ({(s - 1) * 100:+.0f}%)")
        out = replace(out, damage=c.damage * s)
    return out, resp, changes


def card_dict(c: Card):
    d = asdict(c)
    d["cooldown"] = c.cooldown
    d["power"] = card_power(c)
    return d
