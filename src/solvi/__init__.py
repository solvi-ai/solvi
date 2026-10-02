"""solvi — a decision-system builder: a catalog of functions and checks, questions with typed answers, a strategist that
assembles the flow, execution with the computed state, a hash chain and independent replay, and an answer head that trains in milliseconds.
Type hints on catalog functions are the facts' types (solvi.typed), validated with pydantic; untyped parts cost nothing."""
from ._deprecate import SolviDeprecationWarning
from .core import Answer, AnswerType, Catalog, Claim, Decision, NotStated, Question, Quote, Unknown
from .diff import Shadow
from .refine import Fail
from .runtime import MISSING, Record, Result, Trace
from .storage import DuckDBStorage, JSONLStorage, PostgresStorage, SQLiteStorage, TraceStorage
from .system import Response, System
from .typed import Bins, Estimate, FactTypeError, Maybe, Rank, Scale, Span

__version__ = "0.8.0"

__all__ = ["MISSING", "Answer", "AnswerType", "Bins", "Catalog", "Claim", "Decision", "DuckDBStorage", "Estimate",
           "FactTypeError", "Fail", "JSONLStorage", "Maybe", "NotStated", "PostgresStorage", "Question", "Quote", "Rank",
           "Record", "Response", "Result", "SQLiteStorage", "Scale", "Shadow", "SolviDeprecationWarning", "Span", "System",
           "Trace", "TraceStorage", "Unknown"]
