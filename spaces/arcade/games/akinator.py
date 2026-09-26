"""20 questions ("Akinator"): think of an animal, solvi asks yes/no questions and guesses.

The knowledge base is a plain Python table: each animal lists the attributes that are true for it; `attr?` marks "unknown
or it depends" and every other attribute is false. The player's answers are soft evidence: a wrong answer lowers an animal's
weight instead of removing it, so one mistake does not lose the game.

The solvi catalog computes, every turn, the posterior over the animals, the expected information gain of every question not
asked yet, and the move. The hard check `confident_enough` makes it impossible to guess before the top candidate reaches
GUESS_AT unless the questions ran out (or no question can still split the candidates), even when the player presses
"guess now". Learning a new animal is a direct update of the table (a new row, or vote counts on an existing row); nothing
is retrained.

Pure logic, no UI: `new_game`, `next_move`, `answer`, `guess_result`, `teach`, `play_auto`, `benchmark`."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from solvi import Answer, Catalog, Question, System

# ----------------------------------------------------------------------------------------------------------------------
# Attributes: key, question text
# ----------------------------------------------------------------------------------------------------------------------
ATTRS = [
    ("mammal", "Is it a mammal?"),
    ("bird", "Is it a bird?"),
    ("fish", "Is it a fish?"),
    ("reptile", "Is it a reptile?"),
    ("amphibian", "Is it an amphibian (a frog, newt ...)?"),
    ("insect", "Is it an insect?"),
    ("water", "Does it live mostly in water?"),
    ("sea", "Does it live in the sea?"),
    ("fly", "Can it fly?"),
    ("pet", "Do people keep it as a pet?"),
    ("farm", "Is it a farm animal?"),
    ("big", "Is it bigger than a person?"),
    ("small", "Is it smaller than a cat?"),
    ("meat", "Does it eat meat (or other animals)?"),
    ("plants", "Does it eat plants?"),
    ("fur", "Does it have fur or hair?"),
    ("scales", "Does it have scales?"),
    ("shell", "Does it have a shell or armour?"),
    ("stripes", "Does it have stripes?"),
    ("spots", "Does it have spots?"),
    ("horns", "Does it have horns, antlers or tusks?"),
    ("longneck", "Does it have a long neck?"),
    ("tail", "Does it have a noticeable tail?"),
    ("four_legs", "Does it walk on four legs?"),
    ("many_legs", "Does it have more than four legs (or arms)?"),
    ("no_legs", "Does it have no legs at all?"),
    ("night", "Is it active mostly at night?"),
    ("africa", "Does it live in Africa?"),
    ("australia", "Does it live in Australia or New Zealand?"),
    ("cold", "Does it live somewhere cold (snow, ice, high mountains)?"),
    ("jungle", "Does it live in forests or jungles?"),
    ("dangerous", "Can it be dangerous to people?"),
    ("venom", "Is it venomous or poisonous?"),
    ("eggs", "Does it lay eggs?"),
    ("jumps", "Is it known for jumping or hopping?"),
    ("climbs", "Does it climb trees?"),
    ("groups", "Does it live in groups (herd, pack, flock, colony)?"),
    ("black_white", "Is it black and white?"),
    ("colorful", "Is it brightly coloured?"),
    ("fast", "Is it known for being fast?"),
    ("endangered", "Is it endangered or rare?"),
]
ATTR_KEYS = [k for k, _ in ATTRS]
ATTR_TEXT = dict(ATTRS)
ATTR_INDEX = {k: i for i, k in enumerate(ATTR_KEYS)}

# ----------------------------------------------------------------------------------------------------------------------
# Knowledge base: name, emoji, true attributes ("x?" = unknown / it depends; everything not listed = no)
# ----------------------------------------------------------------------------------------------------------------------
ANIMALS = [
    # mammals
    ("Dog", "🐕", "mammal pet fur tail four_legs meat plants? groups? fast? farm?"),
    ("Cat", "🐈", "mammal pet fur tail four_legs meat night climbs jumps"),
    ("Horse", "🐎", "mammal farm pet? big fur tail four_legs plants fast groups"),
    ("Cow", "🐄", "mammal farm big fur tail four_legs plants groups spots black_white? horns?"),
    ("Pig", "🐖", "mammal farm tail four_legs plants meat"),
    ("Sheep", "🐑", "mammal farm fur four_legs plants groups horns?"),
    ("Goat", "🐐", "mammal farm fur tail four_legs plants horns climbs? jumps cold? pet?"),
    ("Donkey", "🐴", "mammal farm fur tail four_legs plants pet?"),
    ("Rabbit", "🐇", "mammal pet fur four_legs plants jumps fast? night? small? groups?"),
    ("Hare", "🐇", "mammal fur four_legs plants jumps fast night? small?"),
    ("Hamster", "🐹", "mammal pet small fur four_legs plants night"),
    ("Guinea pig", "🐹", "mammal pet small fur four_legs plants groups"),
    ("Mouse", "🐁", "mammal small fur tail four_legs plants night climbs? groups? pet?"),
    ("Rat", "🐀", "mammal small fur tail four_legs meat plants night groups climbs? dangerous? pet?"),
    ("Squirrel", "🐿️", "mammal small fur tail four_legs plants climbs jumps fast? jungle"),
    ("Bat", "🦇", "mammal small fur fly night groups meat? plants?"),
    ("Hedgehog", "🦔", "mammal small four_legs meat night pet? shell?"),
    ("Porcupine", "🦔", "mammal four_legs plants night climbs? dangerous? fur? tail?"),
    ("Mole", "🐾", "mammal small fur four_legs meat"),
    ("Fox", "🦊", "mammal fur tail four_legs meat plants? night jungle cold? fast?"),
    ("Wolf", "🐺", "mammal fur tail four_legs meat groups dangerous cold? jungle fast? night?"),
    ("Coyote", "🐺", "mammal fur tail four_legs meat night groups fast"),
    ("Brown bear", "🐻", "mammal big fur four_legs meat plants dangerous jungle cold? climbs?"),
    ("Polar bear", "🐻‍❄️", "mammal big fur four_legs meat dangerous cold water? sea endangered?"),
    ("Panda", "🐼", "mammal big? fur four_legs plants black_white endangered jungle climbs cold?"),
    ("Koala", "🐨", "mammal fur four_legs plants climbs australia night? jungle"),
    ("Kangaroo", "🦘", "mammal big? fur tail plants jumps australia groups fast"),
    ("Wombat", "🐾", "mammal fur four_legs plants australia night"),
    ("Platypus", "🦆", "mammal fur tail four_legs meat water eggs australia venom night?"),
    ("Tasmanian devil", "😈", "mammal fur tail four_legs meat night australia dangerous? endangered"),
    ("Lion", "🦁", "mammal big fur tail four_legs meat africa dangerous groups fast? night?"),
    ("Tiger", "🐅", "mammal big fur tail four_legs meat stripes dangerous jungle endangered night? climbs? water?"),
    ("Leopard", "🐆", "mammal fur tail four_legs meat spots africa climbs night dangerous fast? jungle?"),
    ("Cheetah", "🐆", "mammal fur tail four_legs meat spots africa fast dangerous? endangered?"),
    ("Jaguar", "🐆", "mammal fur tail four_legs meat spots jungle climbs? water dangerous endangered? night?"),
    ("Snow leopard", "🐆", "mammal fur tail four_legs meat spots cold endangered climbs? dangerous?"),
    ("Puma", "🐈", "mammal fur tail four_legs meat climbs jumps dangerous jungle? cold? night?"),
    ("Elephant", "🐘", "mammal big tail? four_legs plants africa groups endangered dangerous? horns"),
    ("Giraffe", "🦒", "mammal big tail four_legs plants spots longneck africa groups? horns?"),
    ("Zebra", "🦓", "mammal big fur tail four_legs plants stripes black_white africa groups fast?"),
    ("Hippo", "🦛", "mammal big four_legs plants water africa dangerous groups"),
    ("Rhino", "🦏", "mammal big four_legs plants horns africa endangered dangerous"),
    ("Gorilla", "🦍", "mammal big fur plants jungle africa groups endangered climbs?"),
    ("Chimpanzee", "🐒", "mammal fur plants meat climbs jungle africa groups endangered dangerous?"),
    ("Orangutan", "🦧", "mammal fur plants climbs jungle endangered"),
    ("Monkey", "🐒", "mammal fur tail climbs jumps plants jungle groups africa?"),
    ("Lemur", "🐒", "mammal fur tail climbs plants jungle groups endangered africa jumps? night? stripes?"),
    ("Sloth", "🦥", "mammal fur climbs plants jungle"),
    ("Camel", "🐫", "mammal big fur tail? four_legs plants farm? africa? longneck?"),
    ("Llama", "🦙", "mammal fur four_legs plants farm longneck groups cold? pet?"),
    ("Yak", "🐂", "mammal big fur tail four_legs plants horns cold farm groups"),
    ("Deer", "🦌", "mammal fur four_legs plants horns jungle fast jumps groups? tail? spots?"),
    ("Moose", "🦌", "mammal big fur four_legs plants horns cold jungle water?"),
    ("Reindeer", "🦌", "mammal fur four_legs plants horns cold groups"),
    ("Bison", "🦬", "mammal big fur four_legs plants horns groups dangerous cold? endangered?"),
    ("Gazelle", "🦌", "mammal fur tail four_legs plants horns africa fast jumps groups"),
    ("Hyena", "🐾", "mammal fur tail four_legs meat spots africa groups night dangerous"),
    ("Meerkat", "🐾", "mammal small fur tail four_legs meat africa groups stripes?"),
    ("Raccoon", "🦝", "mammal fur tail four_legs meat plants night climbs stripes? jungle?"),
    ("Skunk", "🦨", "mammal fur tail four_legs meat plants? night black_white stripes small?"),
    ("Badger", "🦡", "mammal fur four_legs meat night stripes black_white? jungle?"),
    ("Otter", "🦦", "mammal fur tail four_legs meat water groups? sea?"),
    ("Beaver", "🦫", "mammal fur tail four_legs plants water jungle?"),
    ("Dolphin", "🐬", "mammal water sea meat groups fast tail"),
    ("Blue whale", "🐋", "mammal big water sea tail endangered meat?"),
    ("Orca", "🐋", "mammal big water sea meat groups black_white dangerous? fast tail cold?"),
    ("Seal", "🦭", "mammal fur? water sea meat cold groups tail?"),
    ("Walrus", "🦭", "mammal big water sea meat cold groups horns"),
    ("Armadillo", "🐾", "mammal small? four_legs meat shell night tail"),
    ("Anteater", "🐾", "mammal fur tail four_legs meat jungle"),
    ("Warthog", "🐗", "mammal four_legs plants africa horns groups? fur? tail"),
    ("Wild boar", "🐗", "mammal fur four_legs plants meat? horns jungle dangerous? groups night?"),
    # birds
    ("Chicken", "🐔", "bird farm eggs plants meat? groups fly?"),
    ("Duck", "🦆", "bird farm? fly water eggs plants groups colorful?"),
    ("Swan", "🦢", "bird fly water eggs plants longneck"),
    ("Eagle", "🦅", "bird fly eggs meat fast cold? dangerous?"),
    ("Owl", "🦉", "bird fly eggs meat night jungle"),
    ("Parrot", "🦜", "bird fly eggs plants pet colorful jungle climbs groups?"),
    ("Penguin", "🐧", "bird eggs meat water sea cold groups black_white"),
    ("Ostrich", "🐦", "bird big eggs plants africa fast longneck groups?"),
    ("Emu", "🐦", "bird big eggs plants australia fast longneck?"),
    ("Kiwi", "🥝", "bird eggs small? night australia endangered meat? plants?"),
    ("Flamingo", "🦩", "bird fly eggs water longneck colorful groups"),
    ("Peacock", "🦚", "bird eggs plants colorful tail farm? fly?"),
    ("Pigeon", "🕊️", "bird fly eggs plants groups"),
    ("Sparrow", "🐦", "bird fly small eggs plants groups?"),
    ("Hummingbird", "🐦", "bird fly small eggs plants colorful fast jungle?"),
    ("Woodpecker", "🐦", "bird fly eggs meat jungle climbs colorful? small?"),
    ("Toucan", "🐦", "bird fly eggs plants jungle colorful"),
    ("Seagull", "🐦", "bird fly eggs meat sea water? groups"),
    ("Vulture", "🦅", "bird fly eggs meat africa groups? longneck?"),
    ("Canary", "🐤", "bird fly small eggs plants pet colorful"),
    ("Crow", "🐦‍⬛", "bird fly eggs meat plants groups? night?"),
    ("Pelican", "🐦", "bird fly eggs meat water sea groups longneck"),
    # fish
    ("Goldfish", "🐠", "fish water pet small eggs scales tail colorful no_legs"),
    ("Shark", "🦈", "fish water sea big meat dangerous fast tail no_legs eggs? scales?"),
    ("Salmon", "🐟", "fish water sea? eggs scales tail meat no_legs jumps"),
    ("Tuna", "🐟", "fish water sea eggs scales tail meat fast no_legs groups"),
    ("Swordfish", "🐟", "fish water sea big meat fast eggs scales tail no_legs"),
    ("Clownfish", "🐠", "fish water sea small eggs scales tail colorful stripes no_legs"),
    ("Pufferfish", "🐡", "fish water sea small? eggs venom no_legs spots?"),
    ("Piranha", "🐟", "fish water meat dangerous groups scales tail no_legs jungle eggs"),
    ("Catfish", "🐟", "fish water eggs tail no_legs meat night"),
    ("Electric eel", "⚡", "fish water tail no_legs meat night dangerous jungle"),
    ("Stingray", "🐟", "fish water sea meat venom tail no_legs dangerous? spots?"),
    ("Seahorse", "🐴", "fish water sea small eggs tail no_legs colorful?"),
    # reptiles
    ("Cobra", "🐍", "reptile scales eggs meat venom dangerous no_legs tail africa jungle"),
    ("Python", "🐍", "reptile scales eggs meat no_legs tail jungle dangerous climbs? big?"),
    ("Rattlesnake", "🐍", "reptile scales eggs meat venom dangerous no_legs tail night?"),
    ("Crocodile", "🐊", "reptile big scales eggs meat water dangerous tail four_legs africa"),
    ("Alligator", "🐊", "reptile big scales eggs meat water dangerous tail four_legs jungle?"),
    ("Sea turtle", "🐢", "reptile shell eggs water sea endangered plants? meat? four_legs"),
    ("Tortoise", "🐢", "reptile shell eggs plants four_legs pet? scales?"),
    ("Chameleon", "🦎", "reptile scales eggs meat small four_legs tail climbs colorful jungle"),
    ("Iguana", "🦎", "reptile scales eggs plants four_legs tail climbs pet? jungle"),
    ("Gecko", "🦎", "reptile small scales eggs meat four_legs tail climbs night pet?"),
    ("Komodo dragon", "🦎", "reptile big scales eggs meat four_legs tail dangerous venom endangered"),
    # amphibians
    ("Frog", "🐸", "amphibian water eggs jumps small meat four_legs"),
    ("Toad", "🐸", "amphibian eggs jumps small meat four_legs venom? night"),
    ("Poison dart frog", "🐸", "amphibian small eggs jumps meat colorful venom jungle four_legs"),
    ("Salamander", "🦎", "amphibian small eggs meat four_legs tail water? night? spots?"),
    ("Axolotl", "🦎", "amphibian water eggs meat four_legs tail pet? endangered"),
    # insects
    ("Bee", "🐝", "insect small many_legs eggs fly plants groups stripes venom"),
    ("Wasp", "🐝", "insect small many_legs eggs fly meat stripes venom groups? dangerous?"),
    ("Ant", "🐜", "insect small many_legs eggs groups meat? plants?"),
    ("Butterfly", "🦋", "insect small many_legs eggs fly plants colorful"),
    ("Moth", "🦋", "insect small many_legs eggs fly plants night"),
    ("Housefly", "🪰", "insect small many_legs eggs fly plants? meat?"),
    ("Mosquito", "🦟", "insect small many_legs eggs fly meat dangerous night? water?"),
    ("Ladybug", "🐞", "insect small many_legs eggs fly meat spots colorful shell?"),
    ("Grasshopper", "🦗", "insect small many_legs eggs jumps plants fly?"),
    ("Dragonfly", "🪽", "insect small many_legs eggs fly meat fast water? colorful?"),
    ("Beetle", "🪲", "insect small many_legs eggs shell plants? fly?"),
    ("Cockroach", "🪳", "insect small many_legs eggs night fast plants? meat? fly?"),
    ("Praying mantis", "🦗", "insect small many_legs eggs meat climbs?"),
    # other animals without a backbone
    ("Spider", "🕷️", "small many_legs eggs meat venom night? climbs"),
    ("Scorpion", "🦂", "small? many_legs eggs meat venom dangerous tail night africa?"),
    ("Crab", "🦀", "many_legs eggs meat? water sea shell small?"),
    ("Lobster", "🦞", "many_legs eggs water sea shell meat tail"),
    ("Octopus", "🐙", "water sea meat eggs many_legs night?"),
    ("Jellyfish", "🪼", "water sea venom dangerous? no_legs"),
    ("Starfish", "⭐", "water sea many_legs meat small? spots?"),
    ("Snail", "🐌", "small eggs shell plants no_legs night?"),
    ("Earthworm", "🪱", "small no_legs plants? night? eggs?"),
    ("Shrimp", "🦐", "many_legs eggs water sea small groups tail"),
]

GUESS_AT = 0.80            # the hard check: never guess below this confidence while questions remain
MAX_QUESTIONS = 20
MAX_GUESSES = 3
MIN_GAIN = 0.01            # a question expected to remove less than this many bits is useless
EPS = 0.10                 # how often the player is assumed to answer wrongly (soft evidence, not hard elimination)
PRIOR_VOTES = 3.0          # a table value counts as 3 votes; each taught game adds one vote per answered question

ANSWERS = {                # answer → (target value, strength); "don't know" carries no evidence
    "yes": (1.0, 1.0),
    "probably": (1.0, 0.5),
    "don't know": (0.5, 0.0),
    "probably not": (0.0, 0.5),
    "no": (0.0, 1.0),
}
ANSWER_LABELS = list(ANSWERS)


def _parse(tags):
    row = np.zeros(len(ATTR_KEYS))
    votes = np.full(len(ATTR_KEYS), PRIOR_VOTES)
    for t in tags.split():
        unknown = t.endswith("?")
        k = t.rstrip("?")
        if k not in ATTR_INDEX:
            raise ValueError(f"unknown attribute {k!r}")
        row[ATTR_INDEX[k]] = 0.5 if unknown else 1.0
        if unknown:
            votes[ATTR_INDEX[k]] = 0.0
    return row, votes


@dataclass
class KB:
    """The knowledge base: V[i, j] = how true attribute j is for animal i (0..1, 0.5 = unknown), N[i, j] = its votes."""
    names: list
    emoji: list
    V: np.ndarray
    N: np.ndarray
    learned: list = field(default_factory=list)   # [(name, "new" | "updated", ms)]

    def __repr__(self):                             # short repr: the trace hashes and shows it
        return f"KB({len(self.names)} animals × {len(ATTR_KEYS)} attributes, {len(self.learned)} learned)"

    def copy(self):
        return KB(list(self.names), list(self.emoji), self.V.copy(), self.N.copy(), list(self.learned))

    def value(self, i, attr):
        return float(self.V[i, ATTR_INDEX[attr]])


def default_kb():
    rows, votes = zip(*[_parse(t) for _, _, t in ANIMALS])
    return KB([n for n, _, _ in ANIMALS], [e for _, e, _ in ANIMALS], np.array(rows), np.array(votes))


# ----------------------------------------------------------------------------------------------------------------------
# The solvi catalog
# ----------------------------------------------------------------------------------------------------------------------

def _p_yes(v):
    """chance a player says yes to an attribute of value v (v = 1 true, 0 false, 0.5 unknown), with answer noise EPS"""
    return v * (1 - EPS) + (1 - v) * EPS


def _h(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def log_likelihood(kb, answers):
    ll = np.zeros(len(kb.names))
    for attr, ans in answers:
        t, s = ANSWERS[ans]
        if s == 0:
            continue
        py = _p_yes(kb.V[:, ATTR_INDEX[attr]])
        ll += s * np.log(py if t == 1.0 else 1 - py)
    return ll


cat = Catalog()


@cat.fn
def posterior(kb, answers, wrong_guesses):
    """P(animal | answers so far): uniform prior times soft likelihoods; animals already guessed wrong get 0"""
    ll = log_likelihood(kb, answers)
    ll[list(wrong_guesses)] = -np.inf
    p = np.exp(ll - ll.max())
    return p / p.sum()


@cat.fn
def live_candidates(posterior):
    """animals still in the running: at least 1% as likely as the leader (about one mismatch behind at most)"""
    return int((posterior >= posterior.max() * 0.01).sum())


@cat.fn
def shortlist(kb, posterior):
    """the 8 most likely animals with their probabilities"""
    idx = np.argsort(-posterior)[:8]
    return [(kb.names[i], round(float(posterior[i]), 4)) for i in idx]


@cat.fn
def top_candidate(kb, posterior):
    return kb.names[int(np.argmax(posterior))]


@cat.fn
def top_prob(posterior):
    return round(float(posterior.max()), 4)


@cat.fn
def question_gains(kb, posterior, answers):
    """expected information gain (bits) of every question not asked yet: H(answer) - H(answer | animal), plus how the
    live candidates split (yes / no / unsure in the table)"""
    asked = {a for a, _ in answers}
    live = posterior >= posterior.max() * 0.01
    out = []
    for j, k in enumerate(ATTR_KEYS):
        if k in asked:
            continue
        py = _p_yes(kb.V[:, j])
        p_yes = float(posterior @ py)
        bits = float(_h(np.array(p_yes)) - posterior @ _h(py))
        col = kb.V[live, j]
        out.append({"attr": k, "bits": round(bits, 4), "p_yes": round(p_yes, 3), "n_yes": int((col >= 0.75).sum()),
                    "n_no": int((col <= 0.25).sum()), "n_unsure": int(((col > 0.25) & (col < 0.75)).sum())})
    out.sort(key=lambda g: (-g["bits"], g["attr"]))
    return out


@cat.fn
def best_question(question_gains):
    return question_gains[0]["attr"] if question_gains else "none"


@cat.fn
def best_gain(question_gains):
    return question_gains[0]["bits"] if question_gains else 0.0


@cat.fn
def questions_left(q_count, max_questions):
    return max(0, max_questions - q_count)


@cat.fn
def proposed_move(top_prob, questions_left, best_gain, wants_guess):
    """guess when confident, when the questions ran out, when nothing is left to ask, or when the player asks for a guess"""
    if wants_guess or top_prob >= GUESS_AT or questions_left <= 0 or best_gain < MIN_GAIN:
        return "guess"
    return "ask"


@cat.check(hard=True, then={"move": "ask"})
def confident_enough(proposed_move, top_prob, questions_left, best_gain):
    """hard: a guess is allowed only when the top animal has at least GUESS_AT probability, or the questions ran out, or
    no question can still split the candidates"""
    return proposed_move != "guess" or top_prob >= GUESS_AT or questions_left <= 0 or best_gain < MIN_GAIN


@cat.rule("move")
def decide(proposed_move, top_prob, questions_left):
    return proposed_move


@cat.rule("ask_about")
def ask_about(best_question):
    return best_question


@cat.rule("guess")
def guess(top_candidate, top_prob, live_candidates, shortlist):
    """the most likely animal (the move decides whether to say it)"""
    return top_candidate


GUESS_OPTIONS = [n for n, _, _ in ANIMALS]           # grows when a new animal is learned
QUESTIONS = [
    Question("move", "Ask another question or guess?", Answer.choice(["ask", "guess"]), checkpoints=["confident_enough"]),
    Question("ask_about", "Which question has the highest expected information gain?", Answer.choice(ATTR_KEYS + ["none"])),
    Question("guess", "Which animal is the most likely?", Answer.choice(GUESS_OPTIONS)),
]
QUESTIONS[2].answer.options = GUESS_OPTIONS          # the same list object, so learned names become valid answers
system = System(cat, QUESTIONS)


# ----------------------------------------------------------------------------------------------------------------------
# The game
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class AkiGame:
    kb: KB
    answers: list = field(default_factory=list)          # [(attr, answer label)]
    path: list = field(default_factory=list)             # [dict] one per question asked, for the UI
    wrong: list = field(default_factory=list)            # indices of animals guessed wrong
    max_questions: int = MAX_QUESTIONS
    pending: tuple | None = None                         # ("ask", attr) | ("guess", name)
    over: bool = False
    result: str = ""                                     # "won" | "lost"
    resp: object = None                                  # the last solvi Response
    forced: int = 0                                      # how many times the hard check vetoed a guess
    solved: str = ""                                     # the animal guessed right

    @property
    def q_count(self):
        return len(self.answers)

    def state(self, wants_guess=False):
        return {"kb": self.kb, "answers": list(self.answers), "wrong_guesses": list(self.wrong), "q_count": self.q_count,
                "max_questions": self.max_questions, "wants_guess": bool(wants_guess)}


def new_game(kb=None, max_questions=MAX_QUESTIONS):
    g = AkiGame(kb=kb if kb is not None else default_kb(), max_questions=max_questions)
    next_move(g)
    return g


def next_move(game, wants_guess=False):
    """Ask solvi for the next move; sets game.pending. Returns the Response."""
    resp = system.ask(game.state(wants_guess))
    game.resp = resp
    if resp["move"].status == "forced":
        game.forced += 1
    if resp["move"].answer == "guess" or resp["ask_about"].answer in (None, "none"):
        if len(game.wrong) >= MAX_GUESSES or len(game.wrong) >= len(game.kb.names):
            game.over, game.result, game.pending = True, "lost", None
        else:
            game.pending = ("guess", resp.values["top_candidate"])
    else:
        game.pending = ("ask", resp["ask_about"].answer)
    return resp


def answer(game, label):
    """The player answers the pending question."""
    if game.over or not game.pending or game.pending[0] != "ask":
        return None
    attr = game.pending[1]
    v = game.resp.values
    g = next(x for x in v["question_gains"] if x["attr"] == attr)
    before = v["live_candidates"]
    game.answers.append((attr, label))
    resp = next_move(game)
    game.path.append({"attr": attr, "text": ATTR_TEXT[attr], "answer": label, "bits": g["bits"], "live_before": before,
                      "live_after": resp.values["live_candidates"], "split": (g["n_yes"], g["n_no"], g["n_unsure"])})
    return resp


def guess_result(game, correct):
    """The player says whether the pending guess is right."""
    if game.over or not game.pending or game.pending[0] != "guess":
        return None
    name = game.pending[1]
    if correct:
        game.solved = name
        game.over, game.result, game.pending = True, "won", None
        return None
    game.wrong.append(game.kb.names.index(name))
    return next_move(game)


def explain_question(resp):
    """One line: why this question (the split and the expected gain)."""
    v = resp.values
    attr = resp["ask_about"].answer
    g = next(x for x in v["question_gains"] if x["attr"] == attr)
    n = v["live_candidates"]
    exp_left = 2 ** max(0.0, math.log2(max(n, 1)) - g["bits"])
    unsure = f" / {g['n_unsure']} unsure" if g["n_unsure"] else ""
    return (f"asking “{ATTR_TEXT[attr]}” — splits the {n} live candidates into {g['n_yes']} yes / {g['n_no']} no{unsure} "
            f"(a yes is {g['p_yes'] * 100:.0f}% likely); expected to remove {g['bits']:.2f} bits of uncertainty "
            f"(≈ {n} → {exp_left:.0f} candidates)")


def teach(game_or_kb, name, answers, emoji="✨"):
    """Learn from a finished game: `name` was the right animal, `answers` the player's answers. An existing animal gets one
    vote per answered question (its table value moves towards the answer); a new animal gets a new row with the answered
    attributes and "unknown" for the rest. Direct knowledge-base update, nothing is retrained. Returns (kind, ms)."""
    kb = game_or_kb.kb if isinstance(game_or_kb, AkiGame) else game_or_kb
    t0 = time.perf_counter()
    name = name.strip()
    low = [n.lower() for n in kb.names]
    if name.lower() in low:
        i = low.index(name.lower())
        kind = "updated"
        for attr, lab in answers:
            t, s = ANSWERS[lab]
            if s == 0:
                continue
            j = ATTR_INDEX[attr]
            w = s                                       # "probably" counts as half a vote
            kb.V[i, j] = (kb.V[i, j] * kb.N[i, j] + t * w) / (kb.N[i, j] + w)
            kb.N[i, j] += w
    else:
        kind = "new"
        row, votes = np.full(len(ATTR_KEYS), 0.5), np.zeros(len(ATTR_KEYS))
        for attr, lab in answers:
            t, s = ANSWERS[lab]
            if s == 0:
                continue
            row[ATTR_INDEX[attr]] = t if s == 1 else 0.5 + (t - 0.5) * 0.6
            votes[ATTR_INDEX[attr]] = s
        kb.names.append(name)
        kb.emoji.append(emoji)
        if name not in GUESS_OPTIONS:
            GUESS_OPTIONS.append(name)
        kb.V = np.vstack([kb.V, row])
        kb.N = np.vstack([kb.N, votes])
    ms = (time.perf_counter() - t0) * 1000
    kb.learned.append((name, kind, ms))
    return kind, ms


# ----------------------------------------------------------------------------------------------------------------------
# Self-play: a simulated player who thinks of an animal
# ----------------------------------------------------------------------------------------------------------------------

def truthful_answer(kb_or_row, attr, i=None):
    v = kb_or_row.value(i, attr) if i is not None else kb_or_row[ATTR_INDEX[attr]]
    return "yes" if v >= 0.75 else "no" if v <= 0.25 else "don't know"


def play_auto(kb, secret_row, noise=0.0, rng=None, secret_name=None):
    """Play one game against a simulated player whose animal has table row `secret_row` (it may be missing from kb).
    noise: the chance each yes/no answer is flipped. Returns (won, questions asked, guesses made, game)."""
    rng = rng or random.Random(0)
    g = new_game(kb)
    guesses = 0
    while not g.over:
        kind, what = g.pending
        if kind == "ask":
            a = truthful_answer(secret_row, what)
            if a != "don't know" and rng.random() < noise:
                a = "no" if a == "yes" else "yes"
            answer(g, a)
        else:
            guesses += 1
            guess_result(g, what == secret_name)
    return g.result == "won", g.q_count, guesses, g


def benchmark(noise=0.0, seed=0, kb=None):
    """Every animal in the table, once: accuracy (right within MAX_GUESSES guesses), first-guess accuracy, average questions."""
    kb = kb or default_kb()
    rng = random.Random(seed)
    won = first = qs = 0
    t0 = time.perf_counter()
    for i, name in enumerate(kb.names):
        ok, q, guesses, _ = play_auto(kb, kb.V[i], noise, rng, name)
        won += ok
        first += ok and guesses == 1
        qs += q
    n = len(kb.names)
    return {"n": n, "accuracy": won / n, "first_guess": first / n, "avg_questions": qs / n,
            "seconds": time.perf_counter() - t0}


def distinct_pairs(kb=None, min_diff=1):
    """Pairs of animals whose rows differ in fewer than `min_diff` definite attributes (one says yes, the other no)."""
    kb = kb or default_kb()
    yes, no = kb.V >= 0.75, kb.V <= 0.25
    out = []
    for i in range(len(kb.names)):
        d = (yes[i] & no).sum(1) + (no[i] & yes).sum(1)
        for j in range(i + 1, len(kb.names)):
            if d[j] < min_diff:
                out.append((kb.names[i], kb.names[j], int(d[j])))
    return out
