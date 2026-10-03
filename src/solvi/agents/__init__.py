"""solvi.agents (preview in 0.7) — guarding an agent's tool calls: the agent proposes a call, solvi checks it and makes it.

    from solvi.agents import Guard
    guard = Guard(storage="calls.db")
    @guard.tool(ground=["iban"])
    def send_payment(iban: str, amount: float) -> str: ...
    d = guard.call({"name": "send_payment", "arguments": {...}}, context=messages)   # allow / deny / escalate

A framework that runs the tools itself calls `guard.check` (the PydanticAI, LangGraph and OpenAI Agents SDK adapters were
removed in 1.0); `solvi serve --guard catalog.py:guard --upstream CMD` puts the guard in front of an MCP server
(solvi.agents.mcp)."""
from .guard import (AUTHORIZE_TASK, VERDICTS, Guard, GuardDecision, Session, Tool, ToolCall, arguments_model,
                    conversation, messages, model_from_json_schema, same_url, url_parts)
from .confirm import accepted_proposals, accepts
from .intents import INTENTS

__all__ = ["AUTHORIZE_TASK", "INTENTS", "VERDICTS", "Guard", "GuardDecision", "Session", "Tool", "ToolCall",
           "accepted_proposals", "accepts", "arguments_model", "conversation", "messages", "model_from_json_schema",
           "same_url", "url_parts"]
