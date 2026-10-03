"""solvi — a decision-system builder: a catalog of functions and checks, questions with typed answers, a strategist that
assembles the flow, execution with the computed state, a hash chain and independent replay, and an answer head that trains in milliseconds.
Type hints on catalog functions are the facts' types (solvi.core.types), validated with pydantic; untyped parts cost nothing."""
from . import _deprecate
from ._deprecate import SolviDeprecationWarning
from .core.catalog import Answer, AnswerType, Catalog, Claim, Decision, NotStated, Question, Quote, Unknown
from .core.store.diff import Shadow
from .core.slow.refine import Fail
from .core.runtime import MISSING, Record, Result, Trace
from .core.store import DuckDBStorage, JSONLStorage, PostgresStorage, SQLiteStorage, TraceStorage
from .core.system import Response, System
from .core.types import Bins, Estimate, FactTypeError, Maybe, Rank, Scale, Span

__version__ = "0.9.0"

_deprecate.install_old_paths()             # the 0.9 module paths import for one release, with a warning (CHANGELOG 1.0)


def __getattr__(name):
    return _deprecate.old_attribute(__name__, name)       # `solvi.typed` with only `import solvi`: the moved module


__all__ = ["MISSING", "Answer", "AnswerType", "Bins", "Catalog", "Claim", "Decision", "DuckDBStorage", "Estimate",
           "FactTypeError", "Fail", "JSONLStorage", "Maybe", "NotStated", "PostgresStorage", "Question", "Quote", "Rank",
           "Record", "Response", "Result", "SQLiteStorage", "Scale", "Shadow", "SolviDeprecationWarning", "Span", "System",
           "Trace", "TraceStorage", "Unknown"]
