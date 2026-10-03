"""What every remote model of solvi shares — solvi.core.deciders.llm (an OpenAI-compatible chat server), solvi.core.deciders.systemone (a System One
service), solvi.core.slow.generate (a generating model on an llm client) and the chart proposer: an http(s) endpoint, the API key
in the Authorization header only (never in a message, a trace or a repr), retries with backoff, token counts, and one
policy for errors:

- HTTP 401, 403, 404 and any other 4xx except 400 / 413 / 422 — a wrong key, model or URL: `RemoteError` is raised
  (llm's `LLMError` and systemone's `SystemOneError` are its subclasses), never an escalation that looks like an answer;
- 400, 413, 422 — the server refused this request's input: `Refused`; a decider escalates that one decision;
- 408, 409, 429, 5xx, network errors, timeouts, a broken connection — retried (backoff · 2^k seconds); after the
  retries `NoAnswer`: a decider escalates, and the decision is not cached (asked again next time).

`usage` counts tokens under one set of names whatever the server calls them: input_tokens (prompt_tokens),
output_tokens (completion_tokens), reasoning_tokens (completion_tokens_details.reasoning_tokens)."""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

WHY_CHARS = 300                  # the most characters of a server's error text kept in an escalation or an error
RETRY = (408, 409, 429)          # and every 5xx
REFUSED = (400, 413, 422)        # the request's input; any other 4xx is a configuration error (RemoteError)
USAGE = ("input_tokens", "output_tokens", "reasoning_tokens")


class RemoteError(RuntimeError):
    """The server refused the request itself (a wrong key, model or URL: HTTP 401, 403, 404 and other client errors
    except 400 / 413 / 422) — a configuration error, raised rather than escalated. The message has the endpoint, never
    the key."""


class Refused(Exception):
    """The server refused this request's input (HTTP 400 / 413 / 422): the reason, for the escalation."""

    def __init__(self, msg, code=None, why=""):
        super().__init__(msg)
        self.code, self.why = code, why


class NoAnswer(Exception):
    """The server did not answer after the retries: the reason, for the escalation."""


def endpoint(url):
    """A URL without credentials, query or fragment — what the trace may show."""
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ""
    if p.port:
        host += f":{p.port}"
    return urllib.parse.urlunsplit((p.scheme, host, p.path, "", ""))


def error_text(e):
    """The body of an HTTP error → its message: the JSON error's "message" (or a string "detail", as FastAPI and Jeeves
    send), else the text. A gateway that wraps the upstream provider's error (OpenRouter: "Provider returned error" with
    the real cause in `error.metadata.raw`) → the message and that cause. Whitespace runs collapsed, at most WHY_CHARS
    characters."""
    try:
        raw = e.read(16384).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — no body to read
        return ""
    try:
        j = json.loads(raw)
        err = j.get("error")
        msg = err.get("message") if isinstance(err, dict) else err or j.get("message")
        if not isinstance(msg, str) and isinstance(j.get("detail"), str):   # FastAPI / Jeeves: {"detail": "..."}
            msg = j["detail"]
        out = msg if isinstance(msg, str) else raw
        meta = err.get("metadata") if isinstance(err, dict) else None
        cause = meta.get("raw") if isinstance(meta, dict) else None
        if cause is not None and not isinstance(cause, str):
            cause = json.dumps(cause, ensure_ascii=False)
        if cause and cause.strip():
            who = meta.get("provider_name")
            out += f" ({who}: {cause})" if isinstance(who, str) and who else f" ({cause})"
        raw = out
    except (ValueError, AttributeError):
        pass
    return " ".join(raw.split())[:WHY_CHARS]


def tokens(u):
    """A reply's usage object → {input_tokens, output_tokens, reasoning_tokens} (the ones it gives), whatever the
    server names them."""
    if not isinstance(u, dict):
        return {}
    det = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
    pick = {"input_tokens": (u.get("input_tokens"), u.get("prompt_tokens")),
            "output_tokens": (u.get("output_tokens"), u.get("completion_tokens")),
            "reasoning_tokens": (u.get("reasoning_tokens"), det.get("reasoning_tokens"))}
    out = {}
    for k, vals in pick.items():
        v = next((x for x in vals if isinstance(x, int) and not isinstance(x, bool)), None)
        if v is not None:
            out[k] = v
    return out


class RemoteClient:
    """An http(s) endpoint with retries (see the module docs). Subclasses set `service` (the words an escalation uses:
    "the LLM server", "the System One service") and `error` (the RemoteError subclass raised)."""

    service = "the remote model"
    error = RemoteError

    def __init__(self, base_url, model, api_key=None, *, path="", timeout=60.0, retries=2, backoff=1.0, headers=None,
                 opener=None, sleep=None):
        sp = urllib.parse.urlsplit(str(base_url))
        if sp.scheme.lower() not in ("http", "https"):
            raise ValueError(f"{self.service} is an http(s):// URL, not {str(base_url)[:40]!r}")   # no file: / ftp:
        p = sp.path.rstrip("/")
        p = p if not path or p.endswith(path) else p + path
        self.url = urllib.parse.urlunsplit((sp.scheme, sp.netloc, p, sp.query, ""))   # a query stays a query
        self.endpoint = endpoint(self.url)            # what the trace, repr and fingerprint show: no credentials
        self.model = model
        self._key = api_key
        self.timeout = float(timeout)
        self.retries, self.backoff = max(0, int(retries)), float(backoff)
        self._headers = dict(headers or {})
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleep or time.sleep
        self.requests = 0
        self.usage = dict.fromkeys(USAGE, 0)
        self._lock = threading.Lock()

    def __repr__(self):                               # never the key
        return f"{type(self).__name__}({self.endpoint!r}, {self.model!r})"

    def _post(self, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        headers = {"content-type": "application/json", **self._headers}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(self.url, data=data, method="POST", headers=headers)  # noqa: S310 — http(s) only
        with self.opener(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def send(self, body, refused=None):
        """POST `body` with the retries → the response. `refused(code, why)`: called on HTTP 400 / 413 / 422 — return a
        new body to try instead (solvi.core.deciders.llm's reply-format ladder), or None to give up (Refused). RemoteError for a
        configuration error, NoAnswer after the retries."""
        attempt, last = 0, None
        while True:
            try:
                resp = self._post(body)
                with self._lock:
                    self.requests += 1
                return resp
            except urllib.error.HTTPError as e:
                code = e.code
                if code in REFUSED:
                    why = error_text(e)
                    again = refused(code, why) if refused is not None else None
                    if again is not None:
                        body = again
                        continue
                    raise Refused(f"{self.service} refused the request: HTTP {code}" + (f" — {why}" if why else ""),
                                  code, why) from None
                if code in RETRY or code >= 500:
                    why = error_text(e)
                    last = f"HTTP {code}" + (f" — {why}" if any(c.isalpha() for c in why) else "")
                else:
                    raise self.error(f"HTTP {code} from {self.endpoint} (model {self.model!r}): check the URL, the model "
                                     "name and the API key") from None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as e:
                # HTTPException: the connection broke mid-answer (IncompleteRead, RemoteDisconnected, BadStatusLine);
                # ValueError: a reply that is not JSON
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            attempt += 1
            if attempt > self.retries:
                raise NoAnswer(f"{self.service} did not answer after {attempt} attempts: {last}")
            self.sleep(self.backoff * 2 ** (attempt - 1))

    def count(self, usage):
        """Add a reply's usage to `self.usage` → the tokens it gives, by the shared names."""
        got = tokens(usage)
        with self._lock:
            for k, n in got.items():
                self.usage[k] += n
        return got


__all__ = ["NoAnswer", "Refused", "RemoteClient", "RemoteError", "USAGE", "WHY_CHARS", "endpoint", "error_text", "tokens"]
