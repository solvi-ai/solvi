"""solvi.core.knowledge — what a system knows across decisions, where it came from, and what rests on it.

    from solvi.core.knowledge import KnowledgeStore, ConservativeActionModel, Vocabulary, Agenda, FailureMemory

The pieces (each its own module; the names below are re-exported here):
- store (KnowledgeStore): a hash-chained journal of items — facts, rules, skills, actions, episodes — with provenance,
  versions, the source check, disputes to a person, the retraction cascade, staleness flags, snapshots for decisions;
- gates (WriteGate, SourceGate, ConsistencyGate): what lets an item in;
- actions (ActionModel, ConservativeActionModel, Vocabulary, Prediction): what an environment accepts and refuses,
  learned from outcomes over an explicit vocabulary;
- risk (RiskPolicy, Protect, RiskBudget, LearnedGate): protection by default, justified risk as an option;
- agenda (Agenda): goals with done checks in code, gates, order — in the store's journal;
- failures (FailureMemory): do not repeat a plan that failed recently — a hard check with an expiry and a bound;
- worldmap (WorldMap): a map an agent builds by acting, a view over "leads_to" fact items;
- episodes (Episode, EpisodeView, Chooser, LongMemory): an agent's episode as a decision's input; finished episodes as
  store records;
- memory (CorrectionMemory, attach): a memory of corrected cases — with a store, its cases are correction facts.

What the evidence behind each piece is, and its limits, is in each module's docstring and the guide ("Knowledge")."""
from .actions import MISSING, ActionModel, ConservativeActionModel, Prediction, Vocabulary
from .agenda import Agenda
from .failures import FailureMemory
from .gates import ConsistencyGate, SourceGate, WriteGate
from .risk import LearnedGate, Protect, RiskBudget, RiskDecision, RiskPolicy
from .store import KINDS, RECONFIRM_DEFAULTS, SOURCES, KnowledgeStore, Verdict
from .worldmap import WorldMap

__all__ = ["ActionModel", "Agenda", "ConservativeActionModel", "ConsistencyGate", "FailureMemory", "KINDS",
           "KnowledgeStore", "LearnedGate", "MISSING", "Prediction", "Protect", "RECONFIRM_DEFAULTS", "RiskBudget",
           "RiskDecision", "RiskPolicy", "SourceGate", "SOURCES", "Verdict", "Vocabulary", "WorldMap", "WriteGate"]
