"""NATURAL PLAN in types: the problem read into typed facts by rules, the checks of a candidate plan against the facts
(each a list of violations in words), and the plan text in the benchmark's format rendered from a checked candidate.

Plain Python and pydantic, no solvi and no model here: solution.py wires these into solvi catalogs and searches.

    kind        a candidate                                the space
    calendar    (day, start)                               <= 5 days x 16 half-hour slots
    meeting     the order in which friends are met         <= 10! orders (pruned)
    trip        the order in which cities are visited      <= 10! orders (pruned)
"""
import re

from pydantic import BaseModel

KINDS = ("calendar_scheduling", "meeting_planning", "trip_planning")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def problem_text(row):
    """The problem alone (the 0-shot prompt without the instructions around it): what the rules read."""
    t = row["prompt_0shot"]
    if "TASK:" in t:
        t = t.split("TASK:")[-1]
    for cut in ("SOLUTION:", "Your response should start"):
        t = t.split(cut)[0]
    return t.strip()


# ------------------------------------------------------------------ times
def hm(s):
    """'9:30' / '13:00' -> minutes."""
    h, m = s.strip().split(":")
    return int(h) * 60 + int(m)


def hm_str(m):
    return f"{m // 60}:{m % 60:02d}"


def ampm(s):
    """'3:45PM' -> minutes."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*", s, re.I)
    if not m:
        raise ValueError(f"not a time: {s!r}")
    h = int(m.group(1)) % 12 + (12 if m.group(3).upper() == "PM" else 0)
    return h * 60 + int(m.group(2))


def ampm_str(m):
    h, mi = divmod(m, 60)
    return f"{(h - 1) % 12 + 1}:{mi:02d}{'AM' if h % 24 < 12 else 'PM'}"


# ------------------------------------------------------------------ typed facts
class Busy(BaseModel):
    person: str
    day: str
    start: str          # "9:30", as written
    end: str


class Avoid(BaseModel):
    person: str
    day: str
    before: str | None = None      # "not on <day> before 14:00": the meeting must not start before it
    after: str | None = None       # "not on <day> after 14:00": the meeting must not end after it


class CalendarFacts(BaseModel):
    people: list[str]
    days: list[str]
    duration_minutes: int
    work_start: str
    work_end: str
    busy: list[Busy]
    avoid: list[Avoid]
    earliest: bool


class Friend(BaseModel):
    name: str
    location: str
    start: str          # "3:45PM", as written
    end: str
    minutes: int


class Leg(BaseModel):
    a: str
    b: str
    minutes: int


class MeetingFacts(BaseModel):
    start_location: str
    start_time: str
    friends: list[Friend]
    travel: list[Leg]


class Stay(BaseModel):
    city: str
    days: int


class Event(BaseModel):
    city: str
    first_day: int
    last_day: int


class Flight(BaseModel):
    a: str
    b: str
    one_way: bool = False


class TripFacts(BaseModel):
    total_days: int
    stays: list[Stay]
    events: list[Event]
    flights: list[Flight]


# ------------------------------------------------------------------ facts by rules (regular expressions over the template)
_DUR = {"half an hour": 30, "one hour": 60, "an hour": 60, "two hours": 120}
_T = r"\d{1,2}:\d{2}"
_DAY = "|".join(DAYS)


def calendar_facts_rules(problem):
    head = re.search(rf"schedule a meeting for (.+?) for (.+?) between the work hours of ({_T}) to ({_T}) on (.+?)\.\s", problem)
    if not head:
        raise ValueError("no task sentence")
    people = [p for p in re.split(r",\s*|\s+and\s+", head.group(1)) if p]
    if head.group(2) not in _DUR:
        raise ValueError(f"unknown duration {head.group(2)!r}")
    days = re.findall(_DAY, head.group(5))
    body = problem[head.end():]
    if "existing schedules" not in body:
        raise ValueError("no schedules paragraph")
    sched, _, prefs = body.split("existing schedules", 1)[1].split("\n", 1)[1].partition("\n\n")
    busy = []
    for line in sched.split("\n"):
        line = line.strip()
        if not line:
            continue
        person = next((p for p in people if line.startswith(p)), None)
        if person is None:
            raise ValueError(f"a schedule line for nobody: {line[:40]!r}")
        for day, spans in re.findall(rf"({_DAY}) during ((?:{_T} to {_T}(?:, )?)+)", line):
            for a, b in re.findall(rf"({_T}) to ({_T})", spans):
                busy.append(Busy(person=person, day=day, start=a, end=b))
        if not re.search(_T, line) and not re.search(r"wide open|free the entire|no meetings", line):
            raise ValueError(f"a schedule line not understood: {line[:60]!r}")
    avoid, person = [], None
    earliest = bool(re.search(r"earl\w* availability", prefs))
    for sent in re.split(r"(?<=\.)\s+", prefs.strip()):
        m = re.match(r"(\w+) (?:can not meet|do not want to meet|would rather not meet|would like to avoid more meetings) on (.*)", sent)
        if m and m.group(1) in people:
            person, rest = m.group(1), m.group(2)
        elif person and re.match(rf"(?:{_DAY})\b", sent):
            rest = sent
        else:
            person = None
            if not re.search(r"earl\w* availability|Find a time", sent) and sent.strip():
                raise ValueError(f"a preference not understood: {sent[:60]!r}")
            continue
        m = re.match(rf"({_DAY})(?: (before|after) ({_T}))?\.?$", rest.strip())
        if not m:
            raise ValueError(f"a preference not understood: {sent[:60]!r}")
        avoid.append(Avoid(person=person, day=m.group(1), **({m.group(2): m.group(3)} if m.group(2) else {})))
    return CalendarFacts(people=people, days=days, duration_minutes=_DUR[head.group(2)], work_start=head.group(3),
                         work_end=head.group(4), busy=busy, avoid=avoid, earliest=earliest)


def meeting_facts_rules(problem):
    legs = [Leg(a=a.strip(), b=b.strip(), minutes=int(n)) for a, b, n in re.findall(r"^(.+?) to (.+?): (\d+)\.\s*$", problem, re.M)]
    cons = problem.split("CONSTRAINTS:")[1]
    m = re.search(r"You arrive at (.+?) at (\d{1,2}:\d{2}[AP]M)\.", cons)
    if not m or not legs:
        raise ValueError("no arrival sentence or no travel table")
    where = {n: (loc, a, b) for n, loc, a, b in re.findall(r"(\w[\w'-]*) will be at (.+?) from (\d{1,2}:\d{2}[AP]M) to (\d{1,2}:\d{2}[AP]M)\.", cons)}
    need = dict(re.findall(r"You'd like to meet (\w[\w'-]*) for a minimum of (\d+) minutes", cons))
    if set(where) != set(need) or not where:
        raise ValueError("the friends' places and the meeting lengths do not pair up")
    friends = [Friend(name=n, location=where[n][0], start=where[n][1], end=where[n][2], minutes=int(need[n])) for n in where]
    return MeetingFacts(start_location=m.group(1), start_time=m.group(2), friends=friends, travel=legs)


_EVENTS = [r"attend a wedding in (?P<c>\w+) between day (?P<a>\d+) and day (?P<b>\d+)",
           r"attend a workshop in (?P<c>\w+) between day (?P<a>\d+) and day (?P<b>\d+)",
           r"From day (?P<a>\d+) to day (?P<b>\d+), there is a annual show you want to attend in (?P<c>\w+)",
           r"meet a friend in (?P<c>\w+) between day (?P<a>\d+) and day (?P<b>\d+)",
           r"visit relatives in (?P<c>\w+) between day (?P<a>\d+) and day (?P<b>\d+)",
           r"meet your friends at (?P<c>\w+) between day (?P<a>\d+) and day (?P<b>\d+)",
           r"During day (?P<a>\d+) and day (?P<b>\d+), you have to attend a conference in (?P<c>\w+)"]
_STAYS = [r"You would like to visit (?P<c>\w+) for (?P<n>\d+) days", r"You want to spend (?P<n>\d+) days in (?P<c>\w+)",
          r"You plan to stay in (?P<c>\w+) for (?P<n>\d+) days"]


def trip_facts_rules(problem):
    m = re.search(r"visit (\d+) European cities for (\d+) days in total", problem)
    if not m or "direct flights:" not in problem:
        raise ValueError("no trip sentence or no flights")
    head, rest = problem.split("Here are the cities that have direct flights:")
    stays = [Stay(city=x.group("c"), days=int(x.group("n"))) for p in _STAYS for x in re.finditer(p, head)]
    events = [Event(city=x.group("c"), first_day=int(x.group("a")), last_day=int(x.group("b"))) for p in _EVENTS for x in re.finditer(p, head)]
    understood = sum(len(re.findall(p, head)) for p in _STAYS + _EVENTS)
    sentences = [s for s in re.split(r"(?<=\.)\s+", head.strip()) if s]
    if understood != len(sentences) - 2:                      # the first two sentences are the trip and "direct flights only"
        raise ValueError(f"{len(sentences) - 2} constraint sentences, {understood} understood")
    flights = []
    for piece in rest.strip().split("\n")[0].rstrip(". ").split(","):
        piece = piece.strip()
        one = re.fullmatch(r"from (\w+) to (\w+)", piece)
        two = re.fullmatch(r"(\w+) and (\w+)", piece)
        if not (one or two):
            raise ValueError(f"a flight not understood: {piece!r}")
        flights.append(Flight(a=(one or two).group(1), b=(one or two).group(2), one_way=bool(one)))
    if len(stays) != int(m.group(1)):
        raise ValueError(f"{m.group(1)} cities announced, {len(stays)} stays read")
    return TripFacts(total_days=int(m.group(2)), stays=stays, events=events, flights=flights)


RULES = {"calendar_scheduling": calendar_facts_rules, "meeting_planning": meeting_facts_rules, "trip_planning": trip_facts_rules}


# ------------------------------------------------------------------ a candidate meeting slot
class Slot(BaseModel):
    day: str
    start: str
    end: str


# ------------------------------------------------------------------ checks: a candidate against the facts -> violations in words
def calendar_violations(f: CalendarFacts, s: Slot):
    """-> {check name: [violations]}; the 'earliest' check needs the other slots, see calendar_earliest."""
    v = {"day_allowed": [], "inside_work_hours": [], "right_length": [], "nobody_busy": [], "wishes_kept": []}
    a, b = hm(s.start), hm(s.end)
    if s.day not in f.days:
        v["day_allowed"].append(f"{s.day} is not one of the allowed days ({', '.join(f.days)})")
    if a < hm(f.work_start) or b > hm(f.work_end):
        v["inside_work_hours"].append(f"{s.start} - {s.end} is outside the work hours {f.work_start} to {f.work_end}")
    if b - a != f.duration_minutes:
        v["right_length"].append(f"{s.start} - {s.end} is {b - a} minutes, the meeting must be {f.duration_minutes} minutes")
    for x in f.busy:
        if x.day == s.day and a < hm(x.end) and hm(x.start) < b:
            v["nobody_busy"].append(f"{x.person} is busy on {x.day} during {x.start} to {x.end}")
    for x in f.avoid:
        if x.day != s.day:
            continue
        if x.before is None and x.after is None:
            v["wishes_kept"].append(f"{x.person} does not meet on {x.day}")
        elif x.before is not None and a < hm(x.before):
            v["wishes_kept"].append(f"{x.person} does not meet on {x.day} before {x.before}")
        elif x.after is not None and b > hm(x.after):
            v["wishes_kept"].append(f"{x.person} does not meet on {x.day} after {x.after}")
    return v


def meeting_walk(f: MeetingFacts, order):
    """Meet the friends in this order, each as early as possible -> (steps, violations). A step is
    ("travel", place, minutes, arrive) / ("wait", until) / ("meet", name, minutes, from, to), times in minutes."""
    dist = {(x.a, x.b): x.minutes for x in f.travel}
    by = {x.name: x for x in f.friends}
    here, now, steps, bad, seen = f.start_location, ampm(f.start_time), [], [], set()
    for name in order:
        if name not in by:
            bad.append(f"{name} is not one of the friends")
            break
        if name in seen:
            bad.append(f"{name} is met twice")
            break
        seen.add(name)
        p = by[name]
        if p.location != here:
            if (here, p.location) not in dist:
                bad.append(f"no travel time from {here} to {p.location}")
                break
            now += dist[(here, p.location)]
            steps.append(("travel", p.location, dist[(here, p.location)], now))
            here = p.location
        if now < ampm(p.start):
            now = ampm(p.start)
            steps.append(("wait", now))
        if now + p.minutes > ampm(p.end):
            bad.append(f"{name}: you can be at {p.location} at {ampm_str(now)} at the earliest, and {p.minutes} minutes from then "
                       f"end at {ampm_str(now + p.minutes)}, after {name} leaves at {p.end}")
            break
        steps.append(("meet", name, p.minutes, now, now + p.minutes))
        now += p.minutes
    return steps, bad


def meeting_reachable(f: MeetingFacts):
    """The friends who can be met at all (going straight to them): an upper bound on any plan."""
    return [x.name for x in f.friends if not meeting_walk(f, [x.name])[1]]


# ------------------------------------------------------------------ the benchmark's plan text from a checked candidate
def render_calendar(s: Slot):
    return f"Here is the proposed time: {s.day}, {s.start} - {s.end} "


def render_meeting(f: MeetingFacts, steps):
    out = [f"You start at {f.start_location} at {f.start_time}."]
    for st in steps:
        if st[0] == "travel":
            out.append(f"You travel to {st[1]} in {st[2]} minutes and arrive at {ampm_str(st[3])}.")
        elif st[0] == "wait":
            out.append(f"You wait until {ampm_str(st[1])}.")
        else:
            out.append(f"You meet {st[1]} for {st[2]} minutes from {ampm_str(st[3])} to {ampm_str(st[4])}.")
    return "SOLUTION: " + " ".join(out)


def render_trip(f: TripFacts, stays):
    out = [f"Here is the trip plan for visiting the {len(stays)} European cities for {f.total_days} days:", ""]
    for i, (c, a, b) in enumerate(stays):
        if i:
            out.append(f"**Day {a}:** Fly from {stays[i - 1][0]} to {c}.")
        out.append(f"**Day {a}-{b}:** " + (f"Arriving in {c} and visit {c}" if i == 0 else f"Visit {c}") + f" for {b - a + 1} days.")
    return "\n".join(out)
