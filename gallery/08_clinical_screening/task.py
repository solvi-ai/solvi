"""Clinical screening (DEMO ONLY, NOT MEDICAL ADVICE): vital signs + lactate -> NEWS2 risk band, sepsis screen, escalation.

NEWS2 (Royal College of Physicians, 2017) is an exact lookup table: each of seven parameters scores 0-3, SpO2 has a second
scale for patients with hypercapnic respiratory failure, and the aggregate plus "any single parameter scoring 3" gives the
clinical risk band. Each parameter is its own function here, so every point in the total can be checked. qSOFA (Sepsis-3)
counts respiratory rate >= 22, altered mentation and systolic BP <= 100. Two hard checks force an emergency whatever the
score: SpO2 below 85%, or systolic BP below 90 with altered consciousness. A missing vital sign is not guessed: the answers
that need it abstain (unless a hard check already decides).
This is a software demonstration on synthetic patients; it is not a clinical tool and must not guide care.
Try: spo2 83, or delete "resp_rate", or set consciousness to "C" (new confusion)."""
from __future__ import annotations

from solvi import Answer, Catalog, Question

cat = Catalog()


def prepare(state):
    """A vital sign sent as null is missing: drop it, so the strategist sees it cannot be computed. SpO2 scale defaults to 1."""
    state = {k: v for k, v in state.items() if v is not None}
    state.setdefault("spo2_scale", 1)
    return state


# ---------- NEWS2 parameter scores (RCP 2017 chart)
@cat.fn
def rr_score(resp_rate):
    """respiration rate per minute: <=8 -> 3, 9-11 -> 1, 12-20 -> 0, 21-24 -> 2, >=25 -> 3"""
    return 3 if resp_rate <= 8 else 1 if resp_rate <= 11 else 0 if resp_rate <= 20 else 2 if resp_rate <= 24 else 3


@cat.fn
def spo2_score(spo2, spo2_scale, on_oxygen):
    """scale 1: <=91 -> 3, 92-93 -> 2, 94-95 -> 1, >=96 -> 0.
    scale 2 (target 88-92%): <=83 -> 3, 84-85 -> 2, 86-87 -> 1, 88-92 (or >=93 on air) -> 0, on oxygen 93-94 -> 1, 95-96 -> 2, >=97 -> 3"""
    if spo2_scale == 1:
        return 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    if spo2 <= 83:
        return 3
    if spo2 <= 85:
        return 2
    if spo2 <= 87:
        return 1
    if spo2 <= 92 or not on_oxygen:
        return 0
    return 1 if spo2 <= 94 else 2 if spo2 <= 96 else 3


@cat.fn
def oxygen_score(on_oxygen):
    """supplemental oxygen -> 2, air -> 0"""
    return 2 if on_oxygen else 0


@cat.fn
def sbp_score(systolic_bp):
    """systolic BP mmHg: <=90 -> 3, 91-100 -> 2, 101-110 -> 1, 111-219 -> 0, >=220 -> 3"""
    s = systolic_bp
    return 3 if s <= 90 else 2 if s <= 100 else 1 if s <= 110 else 0 if s <= 219 else 3


@cat.fn
def pulse_score(pulse):
    """pulse per minute: <=40 -> 3, 41-50 -> 1, 51-90 -> 0, 91-110 -> 1, 111-130 -> 2, >=131 -> 3"""
    p = pulse
    return 3 if p <= 40 else 1 if p <= 50 else 0 if p <= 90 else 1 if p <= 110 else 2 if p <= 130 else 3


@cat.fn
def acvpu_score(consciousness):
    """ACVPU: A (alert) -> 0; C (new confusion), V, P, U -> 3"""
    if consciousness not in ("A", "C", "V", "P", "U"):
        raise ValueError(f"consciousness must be one of A, C, V, P, U, got {consciousness!r}")
    return 0 if consciousness == "A" else 3


@cat.fn
def temp_score(temperature):
    """temperature °C: <=35.0 -> 3, 35.1-36.0 -> 1, 36.1-38.0 -> 0, 38.1-39.0 -> 1, >=39.1 -> 2"""
    t = temperature
    return 3 if t <= 35.0 else 1 if t <= 36.0 else 0 if t <= 38.0 else 1 if t <= 39.0 else 2


@cat.fn
def news2(rr_score, spo2_score, oxygen_score, sbp_score, pulse_score, acvpu_score, temp_score):
    """the aggregate score and which parameters scored 3 (a single red parameter changes the response)"""
    parts = {"resp_rate": rr_score, "spo2": spo2_score, "oxygen": oxygen_score, "systolic_bp": sbp_score,
             "pulse": pulse_score, "consciousness": acvpu_score, "temperature": temp_score}
    return {"total": sum(parts.values()), "red": sorted(k for k, v in parts.items() if v == 3),
            "by_parameter": {k: v for k, v in parts.items() if v}}


@cat.fn
def qsofa(resp_rate, consciousness, systolic_bp):
    """Sepsis-3 quick SOFA: RR >= 22, altered mentation, SBP <= 100 (one point each)"""
    return int(resp_rate >= 22) + int(consciousness != "A") + int(systolic_bp <= 100)


@cat.fn
def lactate_high(lactate):
    """lactate >= 2 mmol/L"""
    return lactate >= 2.0


# ---------- hard safety checks (always in the escalation flow)
@cat.check(hard=True, then={"escalation": "emergency"})
def spo2_not_critical(spo2):
    """hard: SpO2 below 85% is an emergency, whatever the other numbers"""
    return spo2 >= 85


@cat.check(hard=True, then={"escalation": "emergency"})
def not_shocked(systolic_bp, consciousness):
    """hard: systolic below 90 with altered consciousness is an emergency"""
    return not (systolic_bp < 90 and consciousness != "A")


# ---------- answers
def _band(news2):
    if news2["total"] >= 7:
        return "high"
    if news2["total"] >= 5:
        return "medium"
    if news2["red"]:
        return "low-medium"
    return "low"


@cat.rule("news2_band")
def news2_band(news2):
    """RCP clinical risk: 0-4 low, a single parameter scoring 3 low-medium, 5-6 medium, >= 7 high"""
    return _band(news2)


@cat.rule("escalation")
def escalation(news2, qsofa):
    """high NEWS2 -> emergency; medium, low-medium or qSOFA >= 2 -> urgent review; otherwise routine"""
    band = _band(news2)
    if band == "high":
        return "emergency"
    if band in ("medium", "low-medium") or qsofa >= 2:
        return "urgent review"
    return "routine"


@cat.rule("sepsis_screen")
def sepsis_screen(qsofa, lactate_high):
    """positive: qSOFA >= 2, or qSOFA 1 with lactate >= 2"""
    return qsofa >= 2 or (qsofa == 1 and lactate_high)


QUESTIONS = [
    Question("escalation", "Escalation level", Answer.choice(["routine", "urgent review", "emergency"]),
             checkpoints=["spo2_not_critical", "not_shocked"]),
    Question("news2_band", "NEWS2 clinical risk band", Answer.choice(["low", "low-medium", "medium", "high"])),
    Question("sepsis_screen", "Sepsis screen positive?", Answer.yes_no()),
]
