"""Mafia with a detective bot: a short single-player social deduction game.

Eight seats: the detective (the solvi bot, or you) and seven NPCs with hidden roles: 2 mafia, 1 doctor, 4 villagers. Night:
the mafia kill, the doctor protects one player, the detective checks one player's alignment. Day: the NPCs accuse, defend
and claim roles, then everybody votes and the player with the most votes is out (a tie: nobody). A dead player's role is
revealed. The town wins when both mafia are out; the mafia win when they are as many as the town.

NPCs follow noisy, seeded rules. Villagers keep private suspicions that move with what they hear. The mafia bandwagon on
town players, go after whoever accused their partner, defend their partner, lie about their role under pressure, and kill
at night whoever threatens them. None of these tells is certain: villagers also counter-accuse, defend and change their
story sometimes.

The detective bot is a solvi catalog: `evidence` turns the public record into concrete events per player ("day 2: Rosa voted
against Leo right after Leo accused Marco"), `suspicion` adds their weights (log-likelihood ratios measured on 3000
simulated games, see `measure_weights`) to the prior, and rules pick the top-2 accusation, the vote, whether to reveal, and
whom to check at night. The hard check `never_accuse_cleared` makes it impossible to accuse or vote for a player the
detective has verified as innocent.

Pure logic, no UI: `new_game`, `night`, `talk`, `vote`, `bot_night`, `bot_day`, `play_game`, `benchmark`."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from solvi import Answer, Catalog, Question, System

NPCS = ["Rosa", "Leo", "Marco", "Ivy", "Otto", "Nina", "Sam"]
AVATAR = {"Rosa": "👩‍🦰", "Leo": "🧔", "Marco": "👨‍🍳", "Ivy": "👩‍🎨", "Otto": "👴", "Nina": "👩‍🔬", "Sam": "🧑‍🌾",
          "Dex": "🕵️", "You": "🕵️"}
ROLES = ["mafia", "mafia", "doctor", "villager", "villager", "villager", "villager"]
MAX_DAYS = 7
RETALIATE, DEFEND = 0.5, 0.25  # how often a mafia jumps in when their partner is accused: hit back / defend

# Evidence weights: log-likelihood ratios (mafia vs town) per occurrence, measured by `measure_weights(3000)` on simulated
# games (seeds 10000+, the detective's view at the end of every day), rounded and capped at ±3; a false role claim is a
# certain lie and gets a fixed 6. Positive = mafia-like, negative = town-like.
WEIGHTS = {
    "retaliated_for_mafia": 0.82,  # went after someone right after that someone accused a (now known) mafia
    "retaliated": 0.44,            # accused someone in the very next statement after that someone accused a third player
    "defended_mafia": 2.28,        # defended a player who is now known mafia
    "defended_by_mafia": 3.0,      # was defended by a player who is now known mafia (measured 8.6, capped)
    "defended": 0.23,              # defended an accused player
    "accuser_killed": 1.58,        # accused this player, then was killed that night
    "voted_town_out": 0.83,        # voted for a player who was voted out and turned out to be town
    "voted_mafia_out": -2.26,      # voted for a player who was voted out and turned out to be mafia
    "accused_mafia": -2.35,        # accused a player before anyone knew that player was mafia
    "claim_changed": 0.67,         # claimed one role, later another
    "doctor_conflict": 1.19,       # claims doctor while someone else claims doctor too
    "false_claim": 6.0,            # claimed detective (the detective knows it is a lie) or doctor (the doctor is known)
    "inconsistent_vote": -0.92,    # accused one player, then voted for another nobody accused (a townie's habit here)
}
LABEL = {
    "retaliated_for_mafia": "went after an accuser of a known mafia",
    "retaliated": "went after an accuser",
    "defended_mafia": "defended a known mafia",
    "defended_by_mafia": "defended by a known mafia",
    "defended": "defended an accused player",
    "accuser_killed": "an accuser was killed that night",
    "voted_town_out": "voted out a townsperson",
    "voted_mafia_out": "voted out a mafia",
    "accused_mafia": "accused a mafia early",
    "claim_changed": "changed their role claim",
    "doctor_conflict": "contested doctor claim",
    "false_claim": "false role claim",
    "inconsistent_vote": "said one thing, voted another",
}


# ======================================================================================================================
# Game state and events
# ======================================================================================================================

@dataclass
class Event:
    i: int                  # position in the public record
    day: int
    phase: str              # night | talk | vote | result
    kind: str               # kill | saved | accuse | defend | claim | det_claim | counter_claim | vote | out | tie
    actor: str | None
    target: str | None = None
    info: str = ""          # role for kill/out, claimed role, check summary ...

    def text(self):
        a, t = self.actor, self.target
        k = self.kind
        if k == "kill":
            return f"{t} was killed in the night. {t} was {_a(self.info)}."
        if k == "saved":
            return "The mafia attacked, but nobody died: the doctor saved their target."
        if k == "accuse":
            return f"{a} accuses {t}."
        if k == "defend":
            return f"{a} defends {t}."
        if k == "claim":
            return f"{a} claims to be {_a(self.info)}."
        if k == "det_claim":
            return f"{a} claims to be the detective: “{self.info}”."
        if k == "counter_claim":
            return f"{a}: “No, I am the real detective, and {t} is lying!”"
        if k == "vote":
            return f"{a} votes against {t}."
        if k == "out":
            return f"{t} is voted out ({self.info.split('|')[1]} votes). {t} was {_a(self.info.split('|')[0])}."
        if k == "tie":
            return f"The vote is tied ({self.info}): nobody leaves."
        if k == "notebook":
            return f"{a}'s notebook is found: “{self.info}”."
        return f"{k} {a} {t} {self.info}"

    def key(self):
        return (self.i, self.day, self.phase, self.kind, self.actor, self.target, self.info)


def _a(role):
    return {"mafia": "mafia", "doctor": "the doctor", "villager": "a villager", "detective": "the detective"}.get(role, role)


@dataclass
class MafiaGame:
    seed: int
    det: str                                    # the detective's seat name: "Dex" (bot) or "You"
    roles: dict
    rng: random.Random
    alive: list
    day: int = 1
    phase: str = "night"                        # night | day | vote | over
    events: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)  # name → (night, "mafia" | "town"), the detective's private results
    flipped: dict = field(default_factory=dict) # dead player → role (public)
    beliefs: dict = field(default_factory=dict) # town NPC → {player: suspicion}
    claims: dict = field(default_factory=dict)  # player → [(day, role)]
    det_claimed: bool = False
    counter_claimer: str | None = None
    det_suspect: str | None = None              # the suspect the detective wrote in its notebook (the bot's top-1)
    winner: str | None = None
    history: list = field(default_factory=list) # [(day, bot top-2, vote)] for the UI / stats

    @property
    def players(self):
        return [self.det] + NPCS

    @property
    def mafia(self):
        return [p for p, r in self.roles.items() if r == "mafia"]

    def town_alive(self):
        return [p for p in self.alive if self.roles[p] != "mafia"]

    def mafia_alive(self):
        return [p for p in self.alive if self.roles[p] == "mafia"]

    def log(self, phase, kind, actor=None, target=None, info=""):
        e = Event(len(self.events), self.day, phase, kind, actor, target, info)
        self.events.append(e)
        return e

    def public(self):
        """the public record as plain tuples (what the detective bot reads)"""
        return [e.key() for e in self.events]


def new_game(seed=0, det="Dex"):
    rng = random.Random(seed)
    roles = dict(zip(NPCS, rng.sample(ROLES, len(ROLES))))
    roles[det] = "detective"
    g = MafiaGame(seed=seed, det=det, roles=roles, rng=rng, alive=[det] + NPCS)
    for p in NPCS:
        if roles[p] != "mafia":
            g.beliefs[p] = {q: rng.gauss(0, 0.35) for q in g.players if q != p}
    return g


def _winner(g):
    m, t = len(g.mafia_alive()), len(g.town_alive())
    if m == 0:
        return "town"
    if m >= t:
        return "mafia"
    return None


def _die(g, p):
    g.alive.remove(p)
    g.flipped[p] = g.roles[p]
    role = g.roles[p]
    if p == g.det:
        _notebook(g)
    # villagers learn a little from a revealed role: who voted for it, who defended it, who claimed to be it
    for i, b in g.beliefs.items():
        if i not in g.alive:
            continue
        for e in g.events:
            if e.target != p or e.actor in (None, i) or e.actor not in b:
                continue
            if role == "mafia":
                if e.kind == "defend":
                    b[e.actor] += 0.7
                elif e.kind in ("accuse", "vote"):
                    b[e.actor] -= 0.25
            elif e.kind == "vote":
                b[e.actor] += 0.15
        if role == "doctor":
            for q, cl in g.claims.items():
                if q != p and any(r == "doctor" for _, r in cl) and q in b:
                    b[q] += 2.5
    g.winner = _winner(g)
    if g.winner:
        g.phase = "over"


def _notebook(g):
    """The dead detective's notebook becomes public: the checks, and the suspect it had named (if any)."""
    parts = [f"{q} is {'MAFIA' if r == 'mafia' else 'town'}" for q, (_, r) in g.checks.items()]
    sus = g.det_suspect if g.det_suspect in g.alive and g.det_suspect not in g.checks else None
    if sus:
        parts.append(f"I suspect {sus}")
    g.log("result", "notebook", g.det, sus, "; ".join(parts) or "empty")
    for i, b in g.beliefs.items():
        if i not in g.alive:
            continue
        for q, (_, r) in g.checks.items():
            if q in b and q != i:
                b[q] += 2.5 if r == "mafia" else -2.5
        if sus and sus in b and sus != i:
            b[sus] += 1.2


def _trust(g, i, a):
    """how much town NPC i believes what player a says"""
    if a == g.det and g.det_claimed:
        return 1.2 if g.counter_claimer in g.alive else 3.0
    b = g.beliefs[i].get(a, 0.0)
    return 2.0 / (1.0 + math.exp(2.0 * b))


def _hear(g, kind, actor, target):
    """town NPCs update their suspicions after a public statement"""
    for i, b in g.beliefs.items():
        if i not in g.alive or i == actor:
            continue
        if kind == "accuse":
            if i == target:
                b[actor] += 0.8
            elif target in b:
                b[target] += 0.2 * _trust(g, i, actor)
        elif kind == "defend" and target in b and i != target:
            b[target] -= 0.2 * _trust(g, i, actor)
        elif kind == "claim_doctor" and actor in b:
            others = [q for q, cl in g.claims.items() if q != actor and q in g.alive and any(r == "doctor" for _, r in cl)]
            b[actor] += 0.4 if others else -0.7
            for q in others:
                if q in b:
                    b[q] += 0.4


# ======================================================================================================================
# Night
# ======================================================================================================================

def _threat(g, p):
    """how dangerous town player p is to the mafia: accusations and votes against mafia in the last two days"""
    s = 0.0
    for e in g.events:
        if e.actor == p and e.day >= g.day - 2 and e.target in g.mafia:
            s += {"accuse": 1.0, "vote": 0.7}.get(e.kind, 0.0)
        if e.actor == p and e.kind == "claim" and e.info == "doctor":
            s += 0.8
    return s


def night(g, check_target=None):
    """Resolve a night. check_target: whom the detective checks (ignored if the detective is dead)."""
    rng = g.rng
    events = []
    if g.det in g.alive and check_target in g.alive and check_target != g.det and check_target not in g.checks:
        g.checks[check_target] = (g.day, "mafia" if g.roles[check_target] == "mafia" else "town")
    town = g.town_alive()
    if g.det_claimed and g.det in g.alive and rng.random() < 0.75:
        victim = g.det
    elif rng.random() < 0.65:
        victim = max(town, key=lambda p: (_threat(g, p), rng.random()))
    else:
        victim = rng.choice(town)
    doctor = next((p for p in g.alive if g.roles[p] == "doctor"), None)
    protect = None
    if doctor:
        if g.det_claimed and g.det in g.alive and rng.random() < 0.6:
            protect = g.det
        elif rng.random() < 0.3:
            protect = doctor
        else:
            protect = rng.choice([p for p in g.alive if p != doctor])
    if victim == protect:
        events.append(g.log("night", "saved"))
    else:
        events.append(g.log("night", "kill", None, victim, g.roles[victim]))
        _die(g, victim)
    if g.phase != "over":
        g.phase = "day"
    return events


# ======================================================================================================================
# Day: talk, detective statement, vote
# ======================================================================================================================

def _accusers_today(g, p):
    return [e.actor for e in g.events if e.day == g.day and e.kind == "accuse" and e.target == p]


def _claim(g, p, role):
    g.claims.setdefault(p, []).append((g.day, role))
    ev = g.log("talk", "claim", p, None, role)
    if role == "doctor":
        _hear(g, "claim_doctor", p, None)
    return ev


def _interrupt(g, accuser, target):
    """Someone jumps in right after an accusation. The accused's mafia partner hits back at the accuser or defends the
    partner; now and then a villager who distrusts the accuser pushes back too (noise)."""
    rng = g.rng
    out = []
    if g.roles[target] == "mafia":
        partner = next((q for q in g.mafia_alive() if q != target), None)
        if partner and partner != accuser and partner != g.det:
            r = rng.random()
            if r < RETALIATE:
                out.append(g.log("talk", "accuse", partner, accuser))
                _hear(g, "accuse", partner, accuser)
            elif r < RETALIATE + DEFEND:
                out.append(g.log("talk", "defend", partner, target))
                _hear(g, "defend", partner, target)
            return out
    if rng.random() < 0.12:
        town = [q for q in g.beliefs if q in g.alive and q not in (accuser, target) and g.beliefs[q].get(accuser, 0) > 0.2]
        if town:
            w = rng.choice(town)
            out.append(g.log("talk", "accuse", w, accuser))
            _hear(g, "accuse", w, accuser)
    return out


def talk(g):
    """The NPCs' discussion: opening accusations, then reactions (defences, counter-accusations, role claims)."""
    rng = g.rng
    out = []
    npcs = [p for p in g.alive if p != g.det]
    for b in g.beliefs.values():                                  # a night of thinking: some drift
        for q in b:
            b[q] += rng.gauss(0, 0.2)
    # round 1: openings
    for p in rng.sample(npcs, len(npcs)):
        others = [q for q in g.alive if q != p]
        if g.roles[p] == "mafia":
            partner = [q for q in g.mafia_alive() if q != p]
            town = [q for q in others if g.roles[q] != "mafia"]
            r = rng.random()
            if r < 0.08 and partner:
                t = partner[0]                                    # distancing: a soft accusation of the partner
            elif r < 0.72:
                threats = [q for q in town if _threat(g, q) > 0]
                heat = {q: len(_accusers_today(g, q)) for q in town}
                if threats and rng.random() < 0.5:
                    t = rng.choice(threats)
                elif max(heat.values()) > 0 and rng.random() < 0.6:
                    t = max(town, key=lambda q: (heat[q], rng.random()))
                else:
                    t = rng.choice(town)
            else:
                continue
        else:
            b = g.beliefs[p]
            top = max(others, key=lambda q: (b[q], rng.random()))
            if b[top] > 0.3 and rng.random() < 0.75:
                t = top
            elif rng.random() < 0.3:
                t = rng.choice(others)
            else:
                continue
        out.append(g.log("talk", "accuse", p, t))
        _hear(g, "accuse", p, t)
        out += _interrupt(g, p, t)
    # round 2: reactions
    for p in rng.sample(npcs, len(npcs)):
        me_accused = _accusers_today(g, p)
        if g.roles[p] == "mafia":
            partner = next((q for q in g.mafia_alive() if q != p), None)
            if partner and _accusers_today(g, partner) and rng.random() < 0.2:
                out.append(g.log("talk", "defend", p, partner))
                _hear(g, "defend", p, partner)
            if me_accused:
                if rng.random() < 0.6:
                    t = rng.choice([a for a in me_accused if a in g.alive])
                    if g.roles.get(t) != "mafia":
                        out.append(g.log("talk", "accuse", p, t))
                        _hear(g, "accuse", p, t)
                doctor_dead = any(r == "doctor" for r in g.flipped.values())
                if len(set(me_accused)) >= 2 and not doctor_dead and rng.random() < 0.35:
                    out.append(_claim(g, p, "doctor"))
                elif rng.random() < 0.35:
                    out.append(_claim(g, p, "villager"))
        else:
            b = g.beliefs[p]
            if me_accused:
                if rng.random() < 0.5:
                    t = rng.choice([a for a in me_accused if a in g.alive])
                    out.append(g.log("talk", "accuse", p, t))
                    _hear(g, "accuse", p, t)
                if g.roles[p] == "doctor" and len(set(me_accused)) >= 2 and rng.random() < 0.6:
                    out.append(_claim(g, p, "doctor"))
                elif rng.random() < 0.3:
                    out.append(_claim(g, p, "villager"))
            accused = [q for q in g.alive if q != p and _accusers_today(g, q)]
            trusted = [q for q in accused if b.get(q, 0) < -0.2]
            if trusted and rng.random() < 0.25:
                t = min(trusted, key=lambda q: b[q])
                out.append(g.log("talk", "defend", p, t))
                _hear(g, "defend", p, t)
            elif accused and rng.random() < 0.15:
                t = max(accused, key=lambda q: (len(_accusers_today(g, q)), rng.random()))
                if t != p:
                    out.append(g.log("talk", "accuse", p, t))
                    _hear(g, "accuse", p, t)
    g.phase = "vote"
    return out


def detective_says(g, accuse=None, reveal=False):
    """The detective's statement before the vote: optionally claim and reveal the checks, then accuse."""
    out = []
    if g.det not in g.alive:
        return out
    if reveal and not g.det_claimed:
        g.det_claimed = True
        res = "; ".join(f"{p} is {'MAFIA' if r == 'mafia' else 'town'}" for p, (_, r) in g.checks.items())
        out.append(g.log("talk", "det_claim", g.det, None, f"I checked: {res}" if res else "I have no results yet"))
        found = [p for p, (_, r) in g.checks.items() if r == "mafia" and p in g.alive]
        # the accused mafia's partner may counter-claim detective
        partner = next((q for q in g.mafia_alive() if q not in found), None)
        if found and partner and g.rng.random() < 0.45:
            g.counter_claimer = partner
            g.claims.setdefault(partner, []).append((g.day, "detective"))
            out.append(g.log("talk", "counter_claim", partner, g.det))
            for i, b in g.beliefs.items():
                if i in g.alive and i != partner:
                    b[partner] += 0.3
        for p, (_, r) in g.checks.items():
            for i, b in g.beliefs.items():
                if i in g.alive and p in b and i != p:
                    b[p] += (3.0 if g.counter_claimer not in g.alive else 1.6) * (1 if r == "mafia" else -1)
    if accuse and accuse in g.alive and accuse != g.det:
        out.append(g.log("talk", "accuse", g.det, accuse))
        _hear(g, "accuse", g.det, accuse)
    return out


def vote(g, det_vote=None):
    """Everybody votes (random order); the most votes is out, a tie means nobody."""
    rng = g.rng
    out = []
    tally = {}
    order = rng.sample(g.alive, len(g.alive))
    accused_by = {}
    for e in g.events:
        if e.day == g.day and e.kind == "accuse":
            accused_by.setdefault(e.actor, []).append(e.target)
    n_voters = len(g.alive)
    for p in order:
        others = [q for q in g.alive if q != p]
        if p == g.det:
            t = det_vote if det_vote in others else None
            if t is None:
                continue
        elif g.roles[p] == "mafia":
            partner = next((q for q in g.mafia_alive() if q != p), None)
            town = [q for q in others if g.roles[q] != "mafia"]
            lead = max(tally.items(), key=lambda kv: kv[1], default=(None, 0))
            if partner and tally.get(partner, 0) >= n_voters // 2 and rng.random() < 0.5:
                t = partner                                             # the partner is lost anyway: bus them
            elif lead[0] in town and lead[1] >= 1 and rng.random() < 0.7:
                t = lead[0]
            else:
                mine = [q for q in accused_by.get(p, []) if q in town]
                t = mine[-1] if mine and rng.random() < 0.7 else max(
                    town, key=lambda q: (len(_accusers_today(g, q)), rng.random()))
        else:
            b = g.beliefs[p]
            lead = max(tally.items(), key=lambda kv: kv[1], default=(None, 0))
            r = rng.random()
            if r < 0.1:
                t = rng.choice(others)
            elif r < 0.35 and lead[1] >= 2 and lead[0] != p and b.get(lead[0], 0) > -0.2:
                t = lead[0]
            else:
                t = max(others, key=lambda q: (b[q], rng.random()))
        tally[t] = tally.get(t, 0) + 1
        out.append(g.log("vote", "vote", p, t))
    if tally:
        top = max(tally.values())
        leaders = [q for q, c in tally.items() if c == top]
    else:
        leaders = []
    if len(leaders) == 1:
        p = leaders[0]
        out.append(g.log("result", "out", None, p, f"{g.roles[p]}|{top}"))
        _die(g, p)
    else:
        out.append(g.log("result", "tie", None, None, " / ".join(f"{q} {tally[q]}" for q in leaders) or "no votes"))
    if g.phase != "over":
        g.day += 1
        g.phase = "night"
        if g.day > MAX_DAYS:
            g.winner, g.phase = "mafia", "over"
    return out


# ======================================================================================================================
# The detective's evidence (public record + own checks → concrete events per player)
# ======================================================================================================================

def _known(checks, flipped):
    """player → known role class: 'mafia' | 'town' (checked or revealed at death)"""
    k = {p: ("mafia" if r == "mafia" else "town") for p, r in flipped.items()}
    for p, (_, r) in checks.items():
        k[p] = r
    return k


def compute_evidence(events, me, alive, checks, flipped):
    """{player: [(kind, weight, citation)]} for every living player other than me, from the public record."""
    known = _known(checks, flipped)
    ev = {p: [] for p in alive if p != me}
    evs = [Event(*e) for e in events]
    public_at = {}                                   # mafia → record index from which everybody knows it
    for e in evs:
        if e.kind in ("kill", "out") and e.info.split("|")[0] == "mafia":
            public_at.setdefault(e.target, e.i)
        if e.kind == "det_claim":
            for p, (_, r) in checks.items():
                if r == "mafia":
                    public_at.setdefault(p, e.i)

    def add(p, kind, text, scale=1.0):
        if p in ev:
            ev[p].append((kind, round(WEIGHTS[kind] * scale, 2), text))

    by_day = {}
    for e in evs:
        by_day.setdefault(e.day, []).append(e)
    for d, day_ev in sorted(by_day.items()):
        accuses = [e for e in day_ev if e.kind == "accuse"]
        votes = [e for e in day_ev if e.kind == "vote"]
        # retaliation: X goes after Y right after Y accused Z (Z is not X): the very next statement, or with a vote
        # the same day when Z is known mafia
        talk_ev = [e for e in day_ev if e.phase == "talk"]
        pos = {e.i: k for k, e in enumerate(talk_ev)}
        for e in accuses + votes:
            prior = [a for a in accuses if a.actor == e.target and a.i < e.i and a.target not in (e.actor, e.target)]
            if not prior:
                continue
            a = prior[-1]
            soon = e.kind == "accuse" and pos[e.i] - pos[a.i] == 1
            verb = "accused" if e.kind == "accuse" else "voted against"
            txt = f"day {d}: {e.actor} {verb} {e.target} right after {e.target} accused {a.target}"
            if known.get(a.target) == "mafia" and (soon or e.kind == "vote"):
                add(e.actor, "retaliated_for_mafia", txt + f" ({a.target} is mafia)")
            elif soon and known.get(a.target) != "town":
                add(e.actor, "retaliated", txt)
        for e in day_ev:
            if e.kind == "defend":
                if known.get(e.target) == "mafia":
                    add(e.actor, "defended_mafia", f"day {d}: {e.actor} defended {e.target} ({e.target} is mafia)")
                else:
                    add(e.actor, "defended", f"day {d}: {e.actor} defended {e.target}")
                if known.get(e.actor) == "mafia":
                    add(e.target, "defended_by_mafia", f"day {d}: {e.target} was defended by {e.actor} ({e.actor} is mafia)")
            if (e.kind == "accuse" and known.get(e.target) == "mafia" and e.i < public_at.get(e.target, 10**9)
                    and e.actor != me):
                add(e.actor, "accused_mafia", f"day {d}: {e.actor} accused {e.target} before anyone knew they were mafia")
        # an accuser killed the following night
        kills = [e for e in by_day.get(d + 1, []) if e.kind == "kill"]
        for k in kills:
            for a in accuses:
                if a.actor == k.target and a.target != me:
                    add(a.target, "accuser_killed", f"day {d}: {a.actor} accused {a.target} — {a.actor} was killed that night")
        # votes on a player who was voted out, by their revealed role
        out = next((e for e in day_ev if e.kind == "out"), None)
        if out:
            role = out.info.split("|")[0]
            for v in votes:
                if v.target == out.target and v.actor != me:
                    if role == "mafia":
                        add(v.actor, "voted_mafia_out", f"day {d}: {v.actor} voted out {out.target}, who was mafia")
                    else:
                        add(v.actor, "voted_town_out", f"day {d}: {v.actor} voted out {out.target}, who was {_a(role)}")
        # said one thing, voted another
        for v in votes:
            said = [a.target for a in accuses if a.actor == v.actor]
            if said and v.target not in said and not any(a.target == v.target for a in accuses) and v.actor != me:
                add(v.actor, "inconsistent_vote", f"day {d}: {v.actor} accused {said[-1]} but voted against {v.target}")
    # role claims
    claims = {}
    for e in evs:
        if e.kind == "claim":
            claims.setdefault(e.actor, []).append((e.day, e.info))
        if e.kind == "counter_claim":
            add(e.actor, "false_claim", f"day {e.day}: {e.actor} claimed to be the detective — but I am the detective")
    doctor_known = next((p for p, r in flipped.items() if r == "doctor"), None)
    doctor_claimers = [p for p, cl in claims.items() if any(r == "doctor" for _, r in cl)]
    for p, cl in claims.items():
        roles = []
        for d, r in cl:
            if roles and r != roles[-1][1]:
                add(p, "claim_changed", f"day {roles[-1][0]}: {p} claimed {_a(roles[-1][1])}, day {d}: claimed {_a(r)}")
            roles.append((d, r))
        if any(r == "doctor" for _, r in cl):
            if doctor_known and doctor_known != p:
                add(p, "false_claim", f"day {cl[-1][0]}: {p} claimed to be the doctor — but {doctor_known} was the doctor")
            elif len([q for q in doctor_claimers if q in alive]) > 1:
                rivals = [q for q in doctor_claimers if q != p and q in alive]
                add(p, "doctor_conflict", f"day {cl[-1][0]}: {p} claims doctor, and so does {', '.join(rivals)}")
    return ev


# ======================================================================================================================
# The solvi catalog of the detective bot
# ======================================================================================================================

cat = Catalog()


@cat.fn
def cleared(checks, me):
    """players the detective verified as innocent"""
    return sorted(p for p, (_, r) in checks.items() if r == "town" and p != me)


@cat.fn
def known_mafia(checks, flipped, alive):
    """living players the detective verified as mafia"""
    return sorted(p for p, (_, r) in checks.items() if r == "mafia" and p in alive)


@cat.fn
def mafia_left(flipped):
    return 2 - sum(1 for r in flipped.values() if r == "mafia")


@cat.fn
def evidence(events, me, alive, checks, flipped):
    """concrete events per player, each with its weight (log-likelihood ratio mafia vs town)"""
    return compute_evidence(events, me, alive, checks, flipped)


@cat.fn
def behavior_score(evidence):
    """sum of evidence weights per player: public behaviour only, the checks are not used here"""
    return {p: round(sum(w for _, w, _ in items), 2) for p, items in evidence.items()}


@cat.fn
def suspicion(behavior_score, checks, mafia_left, me):
    """P(mafia) per living player: prior (mafia left among the unverified) + behaviour, normalised to mafia_left;
    verified players are 0 or 1"""
    out, unknown = {}, []
    for p, s in behavior_score.items():
        if p in checks:
            out[p] = 1.0 if checks[p][1] == "mafia" else 0.0
        else:
            unknown.append(p)
    m = mafia_left - sum(1 for p in out if out[p] == 1.0)
    if unknown and m > 0:
        prior = math.log(m / max(len(unknown) - m, 0.5))
        raw = {p: 1 / (1 + math.exp(-(prior + behavior_score[p]))) for p in unknown}
        z = sum(raw.values())
        for p in unknown:
            out[p] = round(min(0.99, raw[p] * m / z), 3)
    else:
        for p in unknown:
            out[p] = 0.0
    return out


@cat.fn
def ranking(suspicion):
    return sorted(suspicion, key=lambda p: (-suspicion[p], p))


@cat.fn
def top2(ranking, suspicion, cleared):
    """the two most likely mafia, never a verified innocent"""
    return [p for p in ranking if p not in cleared and suspicion[p] > 0][:2]


@cat.fn
def vote_pick(known_mafia, top2):
    """vote a verified mafia if there is one alive, else the top suspect"""
    return known_mafia[0] if known_mafia else (top2[0] if top2 else "skip")


@cat.fn
def check_pick(suspicion, checks, me):
    """whom to check tonight: the most suspicious player not checked yet"""
    cands = [p for p in suspicion if p not in checks and p != me]
    return max(cands, key=lambda p: (suspicion[p], p)) if cands else "none"


@cat.check(hard=True, then={"vote": "skip", "accuse": "nobody"})
def never_accuse_cleared(top2, vote_pick, cleared):
    """hard: never accuse or vote for a player the detective verified as innocent"""
    return not (set(top2) | {vote_pick}) & set(cleared)


@cat.rule("accuse")
def accuse_rule(top2):
    return " & ".join(top2) if top2 else "nobody"


@cat.rule("vote")
def vote_rule(vote_pick):
    return vote_pick


@cat.rule("reveal")
def reveal_rule(known_mafia, already_claimed):
    """claim detective and reveal the checks when a verified mafia is alive (the town then follows the detective)"""
    return bool(known_mafia) and not already_claimed


@cat.rule("investigate")
def investigate_rule(check_pick):
    return check_pick


def _options(det):
    ps = [det] + NPCS
    pairs = [f"{a} & {b}" for a in ps for b in ps if a != b]
    return ps, pairs + ps + ["nobody"]


_systems = {}


def get_system(det="Dex"):
    if det not in _systems:
        ps, acc = _options(det)
        qs = [Question("accuse", "Who are the two mafia?", Answer.choice(acc), requires=["never_accuse_cleared"]),
              Question("vote", "Whom to vote against today?", Answer.choice(ps + ["skip"]), requires=["never_accuse_cleared"]),
              Question("reveal", "Claim detective and reveal the checks today?", Answer.yes_no()),
              Question("investigate", "Whom to check tonight?", Answer.choice(ps + ["none"]))]
        _systems[det] = System(cat, qs)
    return _systems[det]


def bot_state(g):
    return {"events": g.public(), "me": g.det, "alive": list(g.alive), "checks": dict(g.checks), "flipped": dict(g.flipped),
            "already_claimed": g.det_claimed}


def bot_night(g):
    """The bot decides whom to check tonight. Returns (target, Response)."""
    resp = get_system(g.det).ask(bot_state(g), ["investigate"])
    t = resp["investigate"].answer
    return (None if t in (None, "none") else t), resp


def bot_day(g):
    """The bot's analysis after the talk: top-2 accusation, vote, reveal. Returns the Response."""
    return get_system(g.det).ask(bot_state(g), ["accuse", "vote", "reveal"])


# ======================================================================================================================
# Policies, self-play, benchmark
# ======================================================================================================================

def _random_check(g):
    c = [p for p in g.alive if p != g.det and p not in g.checks]
    return g.rng.choice(c) if c else None


def bot_turn_night(g):
    """The bot's night: whom to check, and the suspect it writes in its notebook. Returns (target, Response)."""
    t, r = bot_night(g)
    sus = {q: v for q, v in r.values.get("suspicion", {}).items() if q not in g.checks}
    g.det_suspect = max(sus, key=lambda q: (sus[q], q)) if sus else None
    return t, r


def bot_turn_day(g):
    """The bot's day decision after the talk. Returns (vote target or None, reveal?, Response)."""
    r = bot_day(g)
    g.det_suspect = (r.values.get("top2") or [None])[0]
    v = r["vote"].answer
    g.history.append((g.day, r["accuse"].answer, v))
    return (None if v in (None, "skip") else v), r["reveal"].answer == "yes", r


def policy_bot(g, phase):
    if phase == "night":
        return bot_turn_night(g)[0]
    v, rev, _ = bot_turn_day(g)
    return v, rev


def policy_checks_only(g, phase):
    """ablation: uses its checks and reveals, but ignores behaviour (random unverified suspect)"""
    if phase == "night":
        return _random_check(g)
    found = [p for p, (_, r) in g.checks.items() if r == "mafia" and p in g.alive]
    if found:
        return found[0], not g.det_claimed
    c = [p for p in g.alive if p != g.det and p not in g.checks]
    return (g.rng.choice(c) if c else None), False


def policy_random(g, phase):
    if phase == "night":
        return _random_check(g)
    return g.rng.choice([p for p in g.alive if p != g.det]), False


def policy_most_accused(g, phase):
    if phase == "night":
        return _random_check(g)
    counts = {}
    for e in g.events:
        if e.day == g.day and e.kind == "accuse" and e.target in g.alive and e.target != g.det:
            counts[e.target] = counts.get(e.target, 0) + 1
    if not counts:
        return g.rng.choice([p for p in g.alive if p != g.det]), False
    top = max(counts.values())
    return g.rng.choice(sorted(p for p, c in counts.items() if c == top)), False


POLICIES = {"solvi detective bot": policy_bot, "bot, checks only (no behaviour evidence)": policy_checks_only,
            "vote the most accused": policy_most_accused, "random voter": policy_random}


def play_game(seed, policy=policy_bot, audit=None):
    """One full game with the detective played by `policy`. audit: a list that collects (day, top2, vote, cleared) of the
    bot for checking the hard guarantee. Returns the game."""
    g = new_game(seed)
    while g.phase != "over":
        if g.phase == "night":
            t = policy(g, "night") if g.det in g.alive else None
            night(g, t)
        elif g.phase == "day":
            talk(g)
        elif g.phase == "vote":
            if g.det in g.alive:
                if policy is policy_bot and audit is not None:
                    r = bot_day(g)
                    v = r["vote"].answer
                    audit.append((g.seed, g.day, r.values.get("top2"), v, r.values.get("cleared"), r["vote"].status))
                    t, rev = (None if v in (None, "skip") else v), r["reveal"].answer == "yes"
                else:
                    t, rev = policy(g, "day")
                detective_says(g, t, rev)
                vote(g, t)
            else:
                vote(g, None)
    return g


def benchmark(n=200, seeds_from=0, policies=None):
    """Town win rate over n seeded games per detective policy (the same seeds and role deals for every policy)."""
    import time
    out = {}
    for name, pol in (policies or POLICIES).items():
        t0 = time.perf_counter()
        wins = days = 0
        for s in range(seeds_from, seeds_from + n):
            g = play_game(s, pol)
            wins += g.winner == "town"
            days += g.day
        out[name] = {"win_rate": wins / n, "avg_days": days / n, "seconds": time.perf_counter() - t0}
    return out


def top2_accuracy(n=200, day=2):
    """How often the bot's top-2 on `day` names both living mafia / at least one (only games still running that day)."""
    both = one = total = 0
    for s in range(n):
        g = new_game(s)
        while g.phase != "over" and g.day <= day:
            if g.phase == "night":
                night(g, policy_bot(g, "night") if g.det in g.alive else None)
            elif g.phase == "day":
                talk(g)
            else:
                if g.day == day and g.det in g.alive:
                    r = bot_day(g)
                    t2 = set(r.values["top2"])
                    m = set(g.mafia_alive())
                    total += 1
                    both += m <= t2
                    one += bool(m & t2)
                t, rev = policy_bot(g, "day") if g.det in g.alive else (None, False)
                detective_says(g, t, rev)
                vote(g, t)
    return {"games": total, "both": both / max(total, 1), "at_least_one": one / max(total, 1)}


def measure_weights(n=4000, seeds_from=10_000):
    """Estimate the evidence weights: per kind, log(rate among mafia / rate among town) over the detective's view at the end
    of every day of n games played by the bot (Poisson naive Bayes, +0.5 smoothing)."""
    cnt = {k: [0.5, 0.5] for k in WEIGHTS}
    n_m = n_t = 0
    for s in range(seeds_from, seeds_from + n):
        g = new_game(s)
        while g.phase != "over":
            if g.phase == "night":
                night(g, policy_bot(g, "night") if g.det in g.alive else None)
            elif g.phase == "day":
                talk(g)
            else:
                t, rev = policy_bot(g, "day") if g.det in g.alive else (None, False)
                detective_says(g, t, rev)
                vote(g, t)
                ev = compute_evidence(g.public(), g.det, g.alive, g.checks, g.flipped)
                for p, items in ev.items():
                    if p in g.checks:
                        continue
                    maf = g.roles[p] == "mafia"
                    n_m += maf
                    n_t += not maf
                    for k, _, _ in items:
                        cnt[k][0 if maf else 1] += 1
    return {k: round(math.log((a / max(n_m, 1)) / (b / max(n_t, 1))), 2) for k, (a, b) in cnt.items()}
