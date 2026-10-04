"""solvi.core — the low level: the primitives and base classes the ready-made solutions are built from.

    from solvi.core import Catalog, Question, Answer, Quote, Decision   # the catalog and its value classes
    from solvi.core import SlowPath, TraceStorage, Strategist, Head      # the extension points (docs: Building blocks)
    from solvi.core.types import Maybe, Span                           # one area: solvi.core.<area>

The extension points — what you implement to rebuild one part, and what you then get for free — are exported here:
protocols (Scorer, Decider, Adapter, Head, Extractor, Strategist, Monitor, Proposer, Space, Environment; an object with
their methods is one, `isinstance` checks it), base classes with abstract methods (TraceStorage, SlowPath with AskPath,
RefinePath, SearchPath), the records and classes they meet (Thought, Outcome, DefaultStrategist, Fail) and the concrete
System, Response and Dispatcher. They load on first use (`solvi.core.SlowPath` imports solvi.core.dispatch then), so
importing solvi.core or one area does not import the others. `solvi.testing.conformance` checks your implementation.

The areas (solvi.core.<area>), bottom to top: the kernel — catalog (parts, questions, answers, quotes), types (typed
answers), runtime (flows, traces, replay), provenance (fingerprints, grounding), schema (export), textin (reading a
text), environment, primitives, sets, calibration and calibfile, costs, sources, chain; deciders (models as catalog
parts: the decider, LLMs, System One, cascades and votes, learned heads, rule lists), extract (long texts), plan
(strategists); guarantees (calibrated promises, drift, open-set, monitors); store (the decision store, audit, diff,
reports, signatures) and response; system (System); slow (generation, agreement, refine, search) and dispatch (the fast
and the slow path); knowledge (world map, episodes, memory). A module imports only its own area's tier or the ones
below it. In 0.9 `solvi.core` was the catalog module; its public names are still here."""
from .catalog import (NOT_STATED, NOT_STATED_KEY, PRIMITIVES, Answer, AnswerType, Catalog, Claim, Decision,
                      ExperimentalWarning, NotStated, Part, Question, Quote, Serial, Unknown, accept, accepts, bin_labels,
                      check_evidence, cuts_number, evidence_rows, find_quote, find_whole, ground, has_evidence, locate,
                      plain_json, question_data, unknown_key, unwrap, validated)

# The extension points, loaded on first use: name → the module that defines it. The table is the import (by name, so
# solvi.core — the kernel — imports no higher tier: tests/test_import_layers.py).
EXTENSION_POINTS = {
    # parts: models, deciders, adapters, heads, extractors, planners
    "Scorer": "solvi.core.deciders.protocols",
    "Decider": "solvi.core.deciders.protocols",
    "Adapter": "solvi.core.deciders.protocols",
    "Head": "solvi.core.deciders.protocols",
    "Extractor": "solvi.core.textin",
    "Strategist": "solvi.core.plan.strategist",
    "DefaultStrategist": "solvi.core.plan.strategist",
    # guarantees
    "Monitor": "solvi.core.guarantees.monitor",
    # records
    "TraceStorage": "solvi.core.store",
    "Response": "solvi.core.response",
    # the system
    "System": "solvi.core.system",
    # deliberate: the slow path and the dispatcher
    "Fail": "solvi.core.slow.refine",
    "Proposer": "solvi.core.slow.refine",
    "Space": "solvi.core.slow.search",
    "SlowPath": "solvi.core.dispatch",
    "AskPath": "solvi.core.dispatch",
    "RefinePath": "solvi.core.dispatch",
    "SearchPath": "solvi.core.dispatch",
    "Thought": "solvi.core.dispatch",
    "Dispatcher": "solvi.core.dispatch",
    # environments (the environment agent)
    "Environment": "solvi.core.environment",
    "Outcome": "solvi.core.environment",
    # knowledge (lane km10, solvi.core.knowledge): KnowledgeStore, WriteGate, ActionModel, Vocabulary, Agenda,
    # RiskPolicy — added here and to __all__ by that lane
}


def __getattr__(name):
    """An extension point (EXTENSION_POINTS: imported from its module on first use), or a name of the 0.9 module
    solvi.core that this package does not re-export (a private helper): found in solvi.core.catalog, with a
    SolviDeprecationWarning."""
    home = EXTENSION_POINTS.get(name)
    if home is not None:
        import importlib
        value = getattr(importlib.import_module(home), name)
        globals()[name] = value
        return value
    from . import catalog
    if not name.startswith("__") and hasattr(catalog, name):
        from .._deprecate import _warn_from_caller, moved_message
        _warn_from_caller(moved_message(f"solvi.core.{name}", f"solvi.core.catalog.{name}"), skip=1)
        return getattr(catalog, name)
    raise AttributeError(f"module 'solvi.core' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(EXTENSION_POINTS))


__all__ = ["ExperimentalWarning", "NOT_STATED", "accept", "accepts", "Answer", "AnswerType", "bin_labels", "Catalog",
           "check_evidence", "Claim", "cuts_number", "Decision", "evidence_rows", "find_quote", "find_whole", "ground",
           "has_evidence", "locate", "NOT_STATED_KEY", "NotStated", "Part", "plain_json", "PRIMITIVES", "Question",
           "question_data", "Quote", "Serial", "Unknown", "unknown_key", "unwrap", "validated",
           # the extension points (EXTENSION_POINTS)
           "Scorer", "Decider", "Adapter", "Head", "Extractor", "Strategist", "DefaultStrategist", "Monitor",
           "TraceStorage", "Response", "System", "Fail", "Proposer", "Space", "SlowPath", "AskPath", "RefinePath",
           "SearchPath", "Thought", "Dispatcher", "Environment", "Outcome",
           # knowledge (lane km10): KnowledgeStore, WriteGate, ActionModel, Vocabulary, Agenda, RiskPolicy
           ]
