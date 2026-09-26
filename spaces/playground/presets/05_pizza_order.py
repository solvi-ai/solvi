"""Fun: a pizza order sanity check.
Price and delivery time are computed; the budget is a hard check; the pineapple policy is a matter of principle.
Try: add "pineapple" without "ham", set the budget to 15, or make it 1:30 am."""
from solvi import Answer, Catalog, Question

cat = Catalog()

PRICES = {"margherita": 9.0, "pepperoni": 11.5, "quattro formaggi": 12.5, "hawaiian": 11.0, "veggie": 10.5}
TOPPING = 1.5
SIZE = {"small": 0.8, "medium": 1.0, "large": 1.3, "party": 2.2}


@cat.fn
def pizza_price(pizzas, size):
    return round(sum(PRICES.get(p, 12.0) for p in pizzas) * SIZE[size], 2)


@cat.fn
def extras_price(extra_toppings, pizzas):
    return round(len(extra_toppings) * TOPPING * len(pizzas), 2)


@cat.fn
def total(pizza_price, extras_price, delivery_fee):
    return round(pizza_price + extras_price + delivery_fee, 2)


@cat.fn
def eta_minutes(distance_km, orders_in_queue):
    """10 min baking + 4 min per queued order + 3 min per km"""
    return 10 + 4 * orders_in_queue + round(3 * distance_km)


@cat.fn
def has_pineapple(pizzas, extra_toppings):
    return "hawaiian" in pizzas or "pineapple" in extra_toppings


@cat.fn
def slices_per_person(pizzas, size, people):
    per_pizza = {"small": 6, "medium": 8, "large": 10, "party": 16}[size]
    return round(len(pizzas) * per_pizza / people, 1)


@cat.check(hard=True, then={"place_order": "no"})
def within_budget(total, budget):
    """hard: never spend more than the budget"""
    return total <= budget


@cat.check
def open_now(hour):
    return 11 <= hour < 24


@cat.check
def enough_food(slices_per_person):
    return slices_per_person >= 3


@cat.check
def hot_on_arrival(eta_minutes):
    return eta_minutes <= 45


@cat.fn
def loyalty_points(customer_id, loyalty_db):       # no loyalty data in this request: not run
    return loyalty_db.get(customer_id, 0)


@cat.rule("place_order")
def place_order(open_now, enough_food):
    return open_now and enough_food


@cat.rule("pineapple_verdict")
def pineapple_verdict(has_pineapple, extra_toppings, pizzas, guest_from_naples):
    if not has_pineapple:
        return "no pineapple, no drama"
    if guest_from_naples:
        return "diplomatic incident"
    with_ham = "ham" in extra_toppings or "hawaiian" in pizzas
    return "tolerated" if with_ham else "diplomatic incident"


@cat.rule("delivery")
def delivery(hot_on_arrival, distance_km):
    return "deliver" if hot_on_arrival else ("pick up" if distance_km < 5 else "cook at home")


QUESTIONS = [
    Question("place_order", "Place the order?", Answer.yes_no(), checkpoints=["within_budget"]),
    Question("pineapple_verdict", "Pineapple policy verdict?",
             Answer.choice(["no pineapple, no drama", "tolerated", "diplomatic incident"])),
    Question("delivery", "Deliver or pick up?", Answer.choice(["deliver", "pick up", "cook at home"])),
]
