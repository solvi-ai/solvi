"""solvi.agents (preview in 0.7) — guarding an agent's tool calls: the agent proposes a call, solvi checks it and makes it.

    from solvi.agents import Guard
    guard = Guard(storage="calls.db")
    @guard.tool(ground=["iban"])
    def send_payment(iban: str, amount: float) -> str: ...
    d = guard.call({"name": "send_payment", "arguments": {...}}, context=messages)   # allow / deny / escalate

Adapters (each imports its framework only when used): solvi.agents.pydantic_ai (a toolset wrapper),
solvi.agents.langgraph (a ToolNode with the guard around every call), solvi.agents.openai_agents (a tool input guardrail
plus needs_approval for escalations); `solvi serve --guard catalog.py:guard --upstream CMD` puts the guard in front of an
MCP server (solvi.agents.mcp)."""
from .guard import (AUTHORIZE_TASK, VERDICTS, Guard, GuardDecision, Session, Tool, ToolCall, arguments_model,
                    conversation, messages, model_from_json_schema)

__all__ = ["AUTHORIZE_TASK", "VERDICTS", "Guard", "GuardDecision", "Session", "Tool", "ToolCall", "arguments_model",
           "conversation", "messages", "model_from_json_schema"]
