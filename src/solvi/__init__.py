"""solvi — a decision-system builder: a catalog of functions and checks, questions with typed answers, a strategist that
assembles the flow, execution with computed_state, a hash chain and independent replay, and an answer head that trains in milliseconds.
Type hints on catalog functions are the facts' types (solvi.typed), validated with pydantic; untyped parts cost nothing."""
from .core import Answer, AnswerType, Catalog, Claim, Decision, NotStated, Question, Quote, Unknown
from .storage import JSONLStorage, SQLiteStorage, TraceStorage
from .system import Response, System
from .typed import Bins, Estimate, FactTypeError, Maybe, Rank, Scale, Span

__version__ = "0.5.0"

__all__ = ["Answer", "AnswerType", "Bins", "Catalog", "Claim", "Decision", "Estimate", "FactTypeError", "JSONLStorage", "Maybe",
           "NotStated", "Question", "Quote", "Rank", "Response", "SQLiteStorage", "Scale", "Span", "System", "TraceStorage",
           "Unknown"]
