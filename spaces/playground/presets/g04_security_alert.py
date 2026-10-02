"""Security alert: a login event plus the account's recent history -> is it suspicious, and what to do (allow / require_mfa /
lock_account)?

The numbers are computed, not guessed: great-circle distance (haversine) from the last successful login, the travel speed that
distance implies, the spike in failed logins against the account's 30-day hourly average, whether the device was seen before.
Hard checks: an admin account with impossible travel, or a login from a denylisted IP, is locked whatever else holds; the
rest of the analysis is then skipped (early exit). Missing coordinates make both answers abstain instead of guessing.
Try: set "is_admin" to false, or move the event's coordinates next to the last login."""
import math
from datetime import datetime

from solvi import Answer, Catalog, Question

cat = Catalog()
DENYLIST = {"198.51.100.23", "198.51.100.77", "203.0.113.200"}     # threat-intel feed (documentation IPs)
MAX_KMH = 900                                                       # faster than an airliner is impossible travel


def _dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def prepare(state):
    state["event"] = dict(state["event"], time=_dt(state["event"]["time"]))
    state["history"] = [dict(h, time=_dt(h["time"])) for h in state["history"]]
    return state


# ---------- computed facts
@cat.fn
def last_success(event, history):
    ok = [h for h in history if h["result"] == "success" and h["time"] < event["time"]]
    if not ok:
        raise ValueError("no earlier successful login")
    return max(ok, key=lambda h: h["time"])


@cat.fn
def distance_km(last_success, event):
    """great-circle distance between the last successful login and this one"""
    if None in (event.get("lat"), event.get("lon"), last_success.get("lat"), last_success.get("lon")):
        raise ValueError("location unknown")
    p1, p2 = math.radians(last_success["lat"]), math.radians(event["lat"])
    dp, dl = p2 - p1, math.radians(event["lon"] - last_success["lon"])
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * 6371 * math.asin(math.sqrt(a)), 1)


@cat.fn
def hours_between(last_success, event):
    return round((event["time"] - last_success["time"]).total_seconds() / 3600, 2)


@cat.fn
def travel_kmh(distance_km, hours_between):
    return round(distance_km / max(hours_between, 1 / 60), 1)


@cat.fn
def impossible_travel(distance_km, travel_kmh):
    return distance_km > 500 and travel_kmh > MAX_KMH


@cat.fn
def new_device(event, history):
    return event["device_id"] not in {h["device_id"] for h in history if h["result"] == "success"}


@cat.fn
def failed_spike_pct(failed_last_hour, failed_hourly_avg_30d):
    """failed logins in the last hour against the 30-day hourly average, in %"""
    return round((failed_last_hour - failed_hourly_avg_30d) / max(failed_hourly_avg_30d, 1) * 100, 1)


@cat.fn
def ip_denylisted(event):
    return event["ip"] in DENYLIST


# ---------- hard checks
@cat.check(hard=True, then={"suspicious": "yes", "action": "lock_account"})
def admin_travel_plausible(is_admin, impossible_travel):
    """an admin account never gets the benefit of the doubt on impossible travel"""
    return not (is_admin and impossible_travel)


@cat.check(hard=True, then={"suspicious": "yes", "action": "lock_account"})
def ip_not_denylisted(ip_denylisted):
    return not ip_denylisted


# ---------- answers
@cat.rule("suspicious")
def suspicious(impossible_travel, new_device, failed_spike_pct):
    return impossible_travel or (new_device and failed_spike_pct >= 150)


@cat.rule("action")
def action(impossible_travel, new_device, failed_spike_pct, event):
    if impossible_travel and new_device:
        return "lock_account"
    if impossible_travel or (new_device and (failed_spike_pct >= 150 or not event.get("mfa_passed"))):
        return "require_mfa"
    return "allow"


QUESTIONS = [
    Question("suspicious", "Is this login suspicious?", Answer.yes_no(),
             requires=["admin_travel_plausible", "ip_not_denylisted"]),
    Question("action", "What should happen to the session?", Answer.choice(["allow", "require_mfa", "lock_account"]),
             requires=["admin_travel_plausible", "ip_not_denylisted"]),
]
