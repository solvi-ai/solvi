"""Travel: what to pack, from the forecast and the plan.
Each item is a yes/no question with its own rule; the strategist runs only what those rules need.
Try: make the forecast rainy, add "hiking", or set the trip to 12 days."""
from solvi import Answer, Catalog, Question

cat = Catalog()


@cat.fn
def days(forecast):
    return len(forecast)


@cat.fn
def coldest(forecast):
    return min(d["low_c"] for d in forecast)


@cat.fn
def warmest(forecast):
    return max(d["high_c"] for d in forecast)


@cat.fn
def rainy_days(forecast):
    return sum(d["rain_pct"] >= 50 for d in forecast)


@cat.fn
def sunny_days(forecast):
    return sum(d["rain_pct"] < 20 and d["high_c"] >= 20 for d in forecast)


@cat.fn
def outfits_needed(days, laundry_available):
    return min(days, 4) if laundry_available else days


@cat.check
def forecast_sane(coldest, warmest):
    """the lows should not be above the highs"""
    return coldest <= warmest


@cat.fn
def visa_needed(passport_country, destination_country, visa_table):   # no visa data: skipped
    return visa_table.get((passport_country, destination_country), False)


@cat.rule("umbrella")
def umbrella(rainy_days):
    return rainy_days >= 1


@cat.rule("warm_jacket")
def warm_jacket(coldest, activities):
    return coldest <= 8 or "hiking" in activities


@cat.rule("sunscreen")
def sunscreen(sunny_days, activities):
    return sunny_days >= 2 or "beach" in activities or "hiking" in activities


@cat.rule("swimsuit")
def swimsuit(warmest, activities):
    return "beach" in activities or ("pool" in activities and warmest >= 18)


@cat.rule("bag")
def bag(outfits_needed, activities):
    if outfits_needed <= 3 and "business" not in activities:
        return "backpack"
    return "carry-on" if outfits_needed <= 6 else "checked suitcase"


QUESTIONS = [
    Question("umbrella", "Pack an umbrella?", Answer.yes_no()),
    Question("warm_jacket", "Pack a warm jacket?", Answer.yes_no()),
    Question("sunscreen", "Pack sunscreen?", Answer.yes_no()),
    Question("swimsuit", "Pack a swimsuit?", Answer.yes_no()),
    Question("bag", "Which bag?", Answer.choice(["backpack", "carry-on", "checked suitcase"])),
]
