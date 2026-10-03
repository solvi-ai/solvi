"""solvi — decision systems you can verify: a catalog of functions and checks, questions with typed answers, models as
parts whose every answer is grounded, a hash chain over stored decisions and independent replay.

The high level, ready to use and configure:

    solvi.build(question, examples, ...)   a decision system: System 1 fitted, its guarantee, a slow path, a store
    solvi.Guard(storage=...)                an agent's tool calls checked before they run
    solvi.models                            the models: decider("solvi-base"), llm(url, model), systemone(...)

with the vocabulary they share (Catalog, Question, Answer, System, Response, Quote, Claim, Decision, Fail, Unknown, the
typed answers Span, Maybe, Rank, Estimate, Scale, Bins, and Budget). The low level, to build or rebuild a part:
solvi.core and its areas (solvi.core.types, solvi.core.deciders, solvi.core.store, ...). What may still change:
solvi.experimental. Type hints on catalog functions are the facts' types (solvi.core.types), validated with pydantic."""
from . import _deprecate
from ._deprecate import SolviDeprecationWarning
from .core.catalog import Answer, Catalog, Claim, Decision, ExperimentalWarning, Question, Quote, Unknown
from .core.costs import Budget
from .core.slow.refine import Fail
from .core.system import Response, System
from .core.types import Bins, Estimate, Maybe, Rank, Scale, Span
from .solutions.decisions import build
from .solutions.guard import Guard

__version__ = "0.9.0"

_deprecate.install_old_paths()             # the 0.9 module paths import for one release, with a warning (CHANGELOG 1.0)


def __getattr__(name):
    """A name solvi exported in 0.9 and no longer does (`solvi.JSONLStorage`), or a moved module (`solvi.typed`): found
    at its 1.0 place, with a SolviDeprecationWarning, until 1.1."""
    return _deprecate.old_attribute(__name__, name)


# 21 names (LAYOUT §1; `Agent` and `Knowledge` join with the knowledge memory)
__all__ = ["build", "Guard", "Budget", "Catalog", "Question", "Answer", "System", "Response", "Quote", "Claim", "Decision",
           "Fail", "Unknown", "Span", "Maybe", "Rank", "Estimate", "Scale", "Bins", "SolviDeprecationWarning",
           "ExperimentalWarning"]
