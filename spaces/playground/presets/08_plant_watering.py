"""Garden: water the plant today?
Each plant type has a moisture target; tomorrow's rain counts for outdoor pots. A pot without drainage is a hard
"do not water" (the roots would rot). Try: soil_moisture_pct 12 with plant "cactus", rain_mm_24h 8, or drainage false."""
from solvi import Answer, Catalog, Question

cat = Catalog()

TARGET = {"cactus": 10, "succulent": 15, "basil": 45, "fern": 55, "tomato": 50, "monstera": 35}
DRY_DAYS = {"cactus": 14, "succulent": 10, "basil": 2, "fern": 3, "tomato": 2, "monstera": 7}


@cat.fn
def target_moisture(plant):
    return TARGET.get(plant, 35)


@cat.fn
def moisture_gap(target_moisture, soil_moisture_pct):
    return target_moisture - soil_moisture_pct


@cat.fn
def rain_expected(rain_mm_24h, outdoors):
    return outdoors and rain_mm_24h >= 5


@cat.fn
def overdue(days_since_watered, plant):
    return days_since_watered >= DRY_DAYS.get(plant, 7)


@cat.fn
def heat_stress(temp_c, plant):
    return temp_c >= (38 if plant in ("cactus", "succulent") else 32)


@cat.check(hard=True, then={"water": "no", "amount": "none"})
def drains(drainage):
    """hard: never water a pot without drainage holes"""
    return drainage


@cat.check
def sensor_plausible(soil_moisture_pct):
    return 0 <= soil_moisture_pct <= 100


@cat.fn
def fertilizer_due(last_fed, feeding_schedule):   # no schedule given: skipped
    return False


@cat.rule("water")
def water(moisture_gap, rain_expected, overdue):
    return not rain_expected and (moisture_gap > 5 or (overdue and moisture_gap > 0))


@cat.rule("amount")
def amount(moisture_gap, rain_expected):
    if rain_expected or moisture_gap <= 5:
        return "none"
    return "sip" if moisture_gap <= 20 else "soak"


@cat.rule("move_to_shade")
def move_to_shade(heat_stress, outdoors):
    return heat_stress and outdoors


QUESTIONS = [
    Question("water", "Water today?", Answer.yes_no(), requires=["drains"]),
    Question("amount", "How much?", Answer.choice(["none", "sip", "soak"]), requires=["drains"]),
    Question("move_to_shade", "Move it to the shade?", Answer.yes_no()),
]
