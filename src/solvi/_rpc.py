"""What a request may be and how a refused one is answered, shared by solvi serve (HTTP and MCP over stdio) and the MCP
proxy (solvi.experimental.mcp): Limits, the request errors, JSON parsed within the limits, the incident message for a failure
of the server itself, and a line reader bounded by the limits. Moved out of solvi.serve in 1.0 (which re-exports every
name; the logger is still "solvi.serve"), so that the proxy does not import the server."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("solvi.serve")

# the guard proxy's session context (solvi.experimental.mcp): here, so that `solvi serve`'s options can show the defaults
# without importing the experimental module (and its warning) on every `solvi <command> -h`
CONTEXT_MESSAGES = 50            # the tool outputs the proxy's session keeps for checking
CONTEXT_CHARS = 100_000          # ... and their characters in all


@dataclass
class Limits:
    """What one request may be (see the module docs, Security). max_body: bytes of an HTTP body / an MCP message;
    max_depth: nesting of JSON objects and arrays in it; timeout: seconds a request may take (None: no limit) — HTTP
    answers 504 after it, an MCP tool call an error; an async System's parts are given 80% of it as System.aask's timeout
    (unless System(timeout=) or the part sets one), so a slow part makes its questions abstain (safeguard `timeout`) and
    the request still answers. max_questions / max_options: questions in one System One request, options (criteria) of
    one of them — more is refused (422). max_inflight: requests answered at once — by a worker thread (running or
    waiting for the System; a request whose thread timed out still counts until the thread ends) or, for an async
    System, on the event loop — more are refused at once
    with 503 "busy"; queue_timeout: seconds a request waits for the System while another is being answered (None: as
    long as it takes) — then 503 "busy", instead of piling up threads behind a slow one."""
    max_body: int = 1_000_000
    max_depth: int = 32
    timeout: Optional[float] = 60.0
    max_questions: int = 32
    max_options: int = 64
    max_inflight: int = 8
    queue_timeout: Optional[float] = 10.0


class RequestError(Exception):
    """A request the server refuses; its message is written by solvi (never an exception text from the catalog's code) and
    is safe to return to the client. `status`: the HTTP status."""
    status = 422

    def __init__(self, message, status=None):
        super().__init__(message)
        if status is not None:
            self.status = status


class NotFound(RequestError, LookupError):
    status = 404


class BadRequest(RequestError, ValueError, TypeError):
    status = 422


class Busy(RequestError):
    """The server is answering as many requests as it may (Limits.max_inflight), or the System stayed busy longer than
    Limits.queue_timeout: try again later."""
    status = 503


def too_deep(v, max_depth):
    """Does a JSON value nest objects / arrays deeper than max_depth? (Iterative: no recursion on hostile input.)"""
    stack = [(v, 1)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, dict):
            if d > max_depth:
                return True
            stack.extend((y, d + 1) for y in x.values())
        elif isinstance(x, (list, tuple)):
            if d > max_depth:
                return True
            stack.extend((y, d + 1) for y in x)
    return False


def parse_json(data, limits):
    """Bytes / text of a request → the JSON value, within the limits (BadRequest / RequestError 413 otherwise)."""
    if len(data) > limits.max_body:
        raise RequestError(f"the request is larger than {limits.max_body} bytes", 413)
    try:
        v = json.loads(data)
    except RecursionError:
        raise BadRequest(f"the request nests JSON deeper than {limits.max_depth} levels") from None
    except ValueError:
        raise BadRequest("the request is not JSON") from None
    if too_deep(v, limits.max_depth):
        raise BadRequest(f"the request nests JSON deeper than {limits.max_depth} levels")
    return v


def internal_error(where):
    """Log the exception being handled (with its traceback) on the server → the message for the client: an incident id,
    nothing of the exception (its text may carry paths, data or code)."""
    incident = uuid.uuid4().hex[:12]
    log.exception("solvi serve: %s failed (incident %s)", where, incident)
    return f"internal error (incident {incident}; the details are in the server log)"



def _readline(stdin, limit):
    """One line of at most `limit` characters → (line, too_long): a longer line is read to its end and dropped."""
    line = stdin.readline(limit + 1)
    if len(line) <= limit or line.endswith("\n"):
        return line, False
    while True:                                       # the rest of an over-long line: read and discard, a chunk at a time
        more = stdin.readline(65536)
        if not more or more.endswith("\n"):
            return "", True


__all__ = ["BadRequest", "Busy", "CONTEXT_CHARS", "CONTEXT_MESSAGES", "Limits", "NotFound", "RequestError", "internal_error", "parse_json", "too_deep"]
