"""NATURAL PLAN with solvi: search the plans through the checks instead of asking a model to propose one.

Three kinds of problem: a meeting slot that fits everyone's calendar, a day of meetings with friends across a city, a
trip through several cities by direct flights. Each problem is read into typed facts by rules (plans.py), and one small
solvi System per kind accepts a candidate only when every hard check passes: the day is allowed, nobody is busy, every
meeting can be reached in time, every city is visited once, the flights are direct, the days add up. `solvi.core.slow.search`
walks the candidates:

  calendar   every allowed day x every start on a 30-minute grid, earliest first; the first accepted slot wins
  meeting    orders of friends, one more friend per step; a prefix that cannot be walked is cut (`prune=`), and the
             order that meets the most friends wins (`objective=len`)
  trip       orders of cities, one more city per step; a prefix with a missing flight or a wrong event day is cut;
             `keep=2` says whether the accepted order is the only one

The answer text is rendered from the accepted candidate in the benchmark's format. No model is called: the search is
exact (`run.exact`) and every winner is stored as an ordinary decision whose trace replays.

    python naturalplan/solution.py [--split eval] [--n 10]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import options, parse, read_jsonl, write_jsonl  # noqa: E402
from plans import (KINDS, RULES, CalendarFacts, MeetingFacts, Slot, TripFacts, calendar_violations, hm, hm_str,  # noqa: E402
                   meeting_reachable, meeting_walk, problem_text, render_calendar, render_meeting, render_trip)
from score import D, score  # noqa: E402

from solvi import Answer, Catalog, Question, System  # noqa: E402
from solvi.core.slow.search import Tree, search  # noqa: E402

QUESTION = "accept"


def hard_check(cat, name):
    """A hard check over one family of violations: a non-empty list forces accept = "no"."""
    def check(violations: dict) -> bool:
        return not violations.get(name)
    check.__name__ = name
    cat.check(hard=True, then={QUESTION: "no"})(check)
    return name


def system_of(cat, kind, facts_type, text, checks):
    """The parts every kind shares: the problem read into facts by rules, the hard checks, the question."""
    @cat.fn
    def facts(problem: str) -> facts_type:
        return RULES[kind](problem)

    @cat.rule(QUESTION)
    def accept(violations: dict, answer_text: str) -> bool:
        return True                                   # "no" comes only from a failed hard check
    names = [hard_check(cat, c) for c in checks]
    return System(cat, [Question(QUESTION, text, Answer.yes_no(), requires=names)])


def calendar():
    cat = Catalog()

    @cat.fn
    def slot(facts: CalendarFacts, day: str, start: str) -> Slot:
        return Slot(day=day, start=start, end=hm_str(hm(start) + facts.duration_minutes))

    @cat.fn
    def violations(facts: CalendarFacts, slot: Slot) -> dict:
        return calendar_violations(facts, slot)

    @cat.fn
    def answer_text(slot: Slot) -> str:
        return render_calendar(slot)

    system = system_of(cat, "calendar_scheduling", CalendarFacts, "Does the slot keep every constraint?",
                       ["day_allowed", "inside_work_hours", "right_length", "nobody_busy", "wishes_kept"])

    def space(v):                                     # a dict of domains: every day x every start, earliest first
        f = v["facts"]
        return {"day": f.days, "start": [hm_str(a) for a in range(hm(f.work_start), hm(f.work_end) - f.duration_minutes + 1, 30)]}
    return system, space, {}


def meeting():
    cat = Catalog()

    @cat.fn
    def walk(facts: MeetingFacts, order: list) -> dict:
        steps, bad = meeting_walk(facts, order)
        return {"steps": steps, "bad": bad}

    @cat.fn
    def violations(facts: MeetingFacts, order: list, walk: dict) -> dict:
        return {"every_meeting_fits": walk["bad"],
                "meets_somebody": ["meets nobody"] if not order and meeting_reachable(facts) else []}

    @cat.fn
    def answer_text(facts: MeetingFacts, walk: dict) -> str:
        return render_meeting(facts, walk["steps"])

    system = system_of(cat, "meeting_planning", MeetingFacts, "Is the day of meetings possible?", ["every_meeting_fits", "meets_somebody"])

    def space(v):                                     # every prefix is a plan; nobody meets more than all the friends
        names = [x.name for x in v["facts"].friends]
        return Tree([], lambda o: [o + [n] for n in names if n not in o], complete=lambda o: True, bound=lambda o: len(names))
    return system, space, {"into": "order", "objective": len, "prune": ["every_meeting_fits"]}


def trip_walk(f: TripFacts, order):
    """Visit the cities of a (partial) order for their days -> (stays, violations by check). What a prefix breaks stays
    broken whatever comes next; "every city" and "the last day" are judged on a whole order only."""
    need = {s.city: s.days for s in f.stays}
    fly = {(x.a, x.b) for x in f.flights} | {(x.b, x.a) for x in f.flights if not x.one_way}
    v = {k: [] for k in ("cities_once", "direct_flights", "within_days", "events_kept", "all_cities", "fits_total")}
    v["cities_once"] = [f"{c} is not a city of this trip, or is visited twice" for i, c in enumerate(order)
                        if c not in need or c in order[:i]]
    if v["cities_once"]:
        return [], v
    v["direct_flights"] = [f"no direct flight from {a} to {b}" for a, b in zip(order, order[1:]) if (a, b) not in fly]
    stays, day = [], 1
    for c in order:
        stays.append((c, day, day + need[c] - 1))
        day += need[c] - 1                            # the day of a flight counts for both cities
    if day > f.total_days:
        v["within_days"] = [f"the stays end on day {day}, after day {f.total_days}"]
    at = {c: (a, b) for c, a, b in stays}
    v["events_kept"] = [f"{e.city} on days {at[e.city]}, not {e.first_day}-{e.last_day}" for e in f.events
                        if e.city in at and at[e.city] != (e.first_day, e.last_day)]
    v["all_cities"] = [c for c in need if c not in order]
    if not v["all_cities"] and day != f.total_days:
        v["fits_total"] = [f"the stays end on day {day}, the trip has {f.total_days} days"]
    return stays, v


def trip():
    cat = Catalog()

    @cat.fn
    def walk(facts: TripFacts, order: list) -> dict:
        stays, v = trip_walk(facts, order)
        return {"stays": stays, "violations": v}

    @cat.fn
    def violations(walk: dict) -> dict:
        return walk["violations"]

    @cat.fn
    def answer_text(facts: TripFacts, walk: dict) -> str:
        return render_trip(facts, walk["stays"])

    system = system_of(cat, "trip_planning", TripFacts, "Does the trip keep every constraint?",
                       ["cities_once", "direct_flights", "within_days", "events_kept", "all_cities", "fits_total"])

    def space(v):
        cities = [s.city for s in v["facts"].stays]
        return Tree([], lambda o: [o + [c] for c in cities if c not in o], complete=lambda o: len(o) == len(cities))
    return system, space, {"into": "order", "keep": 2, "prune": ["cities_once", "direct_flights", "within_days", "events_kept"]}


SETUP = {"calendar_scheduling": calendar, "meeting_planning": meeting, "trip_planning": trip}


def main():
    p = options(__doc__)
    p.add_argument("--split", default="eval")
    p.add_argument("--n", type=int, default=None, help="the first n problems of each kind")
    p.add_argument("--budget-asks", type=int, default=200_000, help="asks per problem at most")
    p.add_argument("--out", default=None)
    a = parse(p)
    rows_out, stats = [], {}
    for kind in KINDS:
        system, space, opts = SETUP[kind]()
        st, t0 = Counter(), time.perf_counter()
        for r in read_jsonl(D / f"prepared/{kind}_{a.split}.jsonl")[: a.n]:
            run = search(system, {"problem": problem_text(r)}, QUESTION, space, budget=a.budget_asks, **opts)
            found = run.best is not None
            rows_out.append({"id": r["id"], "text": run.response.values["answer_text"] if found else None,
                             "escalate": not found})
            st["found"] += found
            st["exact"] += run.exact
            st["asked"] += run.asked
            st["most asked"] = max(st["most asked"], run.asked)
            st["replayed"] += found and run.replay(system, objective=opts.get("objective"))["ok"]
        stats[kind] = {**st, "seconds": round(time.perf_counter() - t0, 1)}
        print(kind, stats[kind], flush=True)
    out = write_jsonl(a.out or Path(__file__).parent / "runs" / f"solution_{a.split}.jsonl", rows_out)
    print(json.dumps({"score": score(out, a.split), "search": stats}, indent=1))


if __name__ == "__main__":
    main()
