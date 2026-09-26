"""Home: a smart thermostat picks heat / cool / off.
A comfort band around the target, eco mode when nobody is home, and a hard safety check:
if carbon monoxide is detected the answer is forced to "off", whatever the temperature.
Try: co_ppm 60, window_open true, or occupied false."""
from solvi import Answer, Catalog, Question

cat = Catalog()


@cat.fn
def effective_target(target_c, occupied, eco_offset_c):
    """when nobody is home, lower the target by the eco offset (saves energy)"""
    return target_c if occupied else target_c - eco_offset_c


@cat.fn
def band(effective_target, comfort_band_c):
    return (effective_target - comfort_band_c, effective_target + comfort_band_c)


@cat.fn
def deviation(indoor_c, effective_target):
    return round(indoor_c - effective_target, 2)


@cat.fn
def outdoor_helps(indoor_c, outdoor_c, effective_target):
    """opening a window would move the room toward the target"""
    return (indoor_c > effective_target and outdoor_c < indoor_c - 2) or (indoor_c < effective_target and outdoor_c > indoor_c + 2)


@cat.check(hard=True, then={"mode": "off", "fan": "yes"})
def air_safe(co_ppm):
    """hard: carbon monoxide above 35 ppm shuts the burner off and runs the fan"""
    return co_ppm < 35


@cat.check
def window_closed(window_open):
    return not window_open


@cat.check
def humidity_ok(humidity_pct):
    return 30 <= humidity_pct <= 60


@cat.fn
def energy_bill(kwh_history, tariff):           # no billing data: skipped
    return sum(kwh_history) * tariff


@cat.rule("mode")
def mode(indoor_c, band, window_closed):
    low, high = band
    if not window_closed:
        return "off"
    if indoor_c < low:
        return "heat"
    if indoor_c > high:
        return "cool"
    return "off"


@cat.rule("fan")
def fan(humidity_ok, deviation):
    return not humidity_ok or abs(deviation) > 3


@cat.rule("suggest_window")
def suggest_window(outdoor_helps, window_closed, deviation):
    return outdoor_helps and window_closed and abs(deviation) > 1


QUESTIONS = [
    Question("mode", "Heat, cool or off?", Answer.choice(["heat", "cool", "off"]), checkpoints=["air_safe"]),
    Question("fan", "Run the fan?", Answer.yes_no(), checkpoints=["air_safe"]),
    Question("suggest_window", "Suggest opening a window instead?", Answer.yes_no()),
]
