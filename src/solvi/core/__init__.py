"""solvi.core — the low level: the primitives and base classes the ready-made solutions are built from.

    from solvi.core import Catalog, Question, Answer, Quote, Decision   # the catalog and its value classes
    from solvi.core.types import Maybe, Span                           # one area: solvi.core.<area>

The areas (solvi.core.<area>), bottom to top: the kernel — catalog (parts, questions, answers, quotes), types (typed
answers), runtime (flows, traces, replay), provenance (fingerprints, grounding), schema (export), textin (reading a
text), primitives, sets, calibration and calibfile, costs, sources, chain; deciders (models as catalog parts: the
decider, LLMs, System One, cascades and votes, learned heads, rule lists), extract (long texts), plan (strategists);
guarantees (calibrated promises, drift, open-set); store (the decision store, audit, diff, reports, signatures) and
response; system (System); slow (generation, agreement, refine, search) and dispatch (the fast and the slow path);
knowledge (world map, episodes, memory). A module imports only its own area's tier or the ones below it.
In 0.9 `solvi.core` was the catalog module; its public names are still here."""
from .catalog import (NOT_STATED, NOT_STATED_KEY, PRIMITIVES, Answer, AnswerType, Catalog, Claim, Decision,
                      ExperimentalWarning, NotStated, Part, Question, Quote, Serial, Unknown, accept, accepts, bin_labels,
                      check_evidence, cuts_number, evidence_rows, find_quote, find_whole, ground, has_evidence, locate,
                      plain_json, question_data, unknown_key, unwrap, validated)


def __getattr__(name):
    """A name of the 0.9 module solvi.core that this package does not re-export (a private helper): found in
    solvi.core.catalog, with a SolviDeprecationWarning."""
    from . import catalog
    if not name.startswith("__") and hasattr(catalog, name):
        from .._deprecate import _warn_from_caller, moved_message
        _warn_from_caller(moved_message(f"solvi.core.{name}", f"solvi.core.catalog.{name}"), skip=1)
        return getattr(catalog, name)
    raise AttributeError(f"module 'solvi.core' has no attribute {name!r}")


__all__ = ["ExperimentalWarning", "NOT_STATED", "accept", "accepts", "Answer", "AnswerType", "bin_labels", "Catalog",
           "check_evidence", "Claim", "cuts_number", "Decision", "evidence_rows", "find_quote", "find_whole", "ground",
           "has_evidence", "locate", "NOT_STATED_KEY", "NotStated", "Part", "plain_json", "PRIMITIVES", "Question",
           "question_data", "Quote", "Serial", "Unknown", "unknown_key", "unwrap", "validated"]
