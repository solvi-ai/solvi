"""Generation by an LLM: a text, or JSON checked against a pydantic model or a JSON schema — through the same
OpenAI-compatible client settings as solvi.llm, recorded in the trace as a model's output.

    from solvi.generate import generator
    writer = generator("http://127.0.0.1:8080/v1", "qwen2.5-7b-instruct", extra_body={"reasoning": {"effort": "low"}})
    g = writer.generate("Write one SQLite query that ...")            # g.value: the reply's text
    g = writer.generate(messages, schema=Plan)                         # g.value: a Plan (pydantic), validated
    g = writer.generate(messages, schema=Table, text=doc, quotes=["rows"])   # every row literally in doc
    g = writer.sample(messages, k=3, temperature=0.8)                  # g.value: 3 outputs (None where one failed)
    cat.fn(writer.part("sql", prompt, k=3))                            # a catalog part: the outputs recorded, not re-run

solvi.llm asks closed questions (options, yes / no, a span); this module is for the model that writes something — a
query, a plan, a JSON extraction of a table — which solvi's checks then judge. The model proposes; nothing here decides.

What is checked, and what escalates. A reply is accepted only when it is complete and well-formed: a refusal, a cut-off
reply (finish_reason "length"), an empty one, a `parse` function that raises, JSON that does not parse or does not
match the schema, and a quoted string that is not literally in the given text all raise `solvi.llm.InvalidOutput`
with the reason — never a repaired or guessed value. A server that does not answer after the retries, or refuses the
input (HTTP 400 / 413 / 422), raises `Unanswered`; a wrong key, model or URL raises `solvi.llm.LLMError`. In a catalog
any of these makes the part fail: the fact is missing, and the questions that need it abstain with the cause.

Quotes. `quotes=["rows", "items.*.source"]` names the strings of the value that must be copied from `text`: each is
looked up as written, as whole words and numbers (the rule of `Claim` evidence — "3" is not found in "30"). Ask the model
to copy table rows as the text writes them and parse them in code: a wrong number is then not in the text, where a
number copied into a field of its own may stand elsewhere in the text and pass. The quotes become the output's evidence
(`Claim.evidence`), so inside a System they are located again in the given text, recorded with their offsets and
checked on replay.

The trace. `generate` and `sample` return a `Generated` — a `Claim` whose `extra["generated"]` holds, per output, the
model id, the request's fingerprint (sha256 of the request body), temperature, seed, finish reason, tokens and, for a
structured or parsed reply, the reply's text. Returned from a catalog part, the value is the fact and the record keeps
that detail; `writer.part(...)` builds such a part with the model attached (`model=`), provenance `proposed`. An LLM's
output is not reproducible bit for bit, so replay does not call it again (`replay="rerun"` asks it to): it checks
the recorded output instead — the recorded reply read again through the same parse and schema must give the recorded
value, and the quotes must be in the recorded text. The API key is never recorded.

Not done here: no retries on an invalid reply (that is solvi.refine's loop: the reasons go back to the model), no
response caching (put a caching proxy in front of the server), no streaming, no tool calls."""
from __future__ import annotations

import hashlib
import http.client
import inspect
import json
import re
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .core import Claim, find_whole
from .llm import WHY_CHARS, InvalidOutput, LLMError, LLMScorer, _error_text

RESPONSE_FORMATS = ("prompt", "json_object", "json_schema")
FEEDBACK = "Your answer was checked by a program, and it does not work:\n{reasons}\nGive a corrected answer."


class Unanswered(RuntimeError):
    """The server did not answer after the retries (network, timeout, 408 / 409 / 429 / 5xx), or refused this request's
    input (HTTP 400 / 413 / 422): an escalation, not a configuration error. The message has the endpoint, never the key."""


@dataclass
class Generated(Claim):
    """What a generator returns: a Claim whose value is the output (a text, a validated schema value, or a list of K
    outputs with None where one failed), whose evidence is the quoted strings, and whose `extra["generated"]` is the
    record of the request(s) — one dict, or a list for K outputs. Returned from a catalog part, the fact is the value."""

    @property
    def meta(self):
        """The record of the request: {"model", "request", "temperature", "seed", "finish", "usage", ...} (a list for K)."""
        return (self.extra or {}).get("generated")

    @property
    def text(self):
        """The reply's text (a list for K outputs)."""
        m = self.meta
        if isinstance(m, list):
            return [x.get("text", v if isinstance(v, str) else None) for x, v in zip(m, self.value)]
        return (m or {}).get("text", self.value if isinstance(self.value, str) else None)


# ------------------------------------------------------------------------------------------------ schemas
_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list,
               "null": type(None)}
_IGNORED = {"title", "description", "default", "examples", "$schema", "$id", "$comment"}
_SUPPORTED = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "minimum", "maximum",
              "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "minLength", "maxLength", "anyOf",
              "prefixItems"} | _IGNORED


def _schema_keywords(s, path="$"):
    """A JSON schema → ValueError at the first keyword this module does not check ($ref, pattern, format, allOf, ...):
    a schema that is only partly checked would pass replies it describes as invalid."""
    if not isinstance(s, dict):
        raise ValueError(f"JSON schema at {path} is not an object")
    bad = sorted(set(s) - _SUPPORTED)
    if bad:
        raise ValueError(f"JSON schema keyword(s) {bad} at {path} are not checked by solvi.generate: pass a pydantic model "
                         "instead, or leave them out")
    for k, v in (s.get("properties") or {}).items():
        _schema_keywords(v, f"{path}.{k}")
    if isinstance(s.get("items"), dict):
        _schema_keywords(s["items"], f"{path}[]")
    for i, v in enumerate(s.get("prefixItems") or []):
        _schema_keywords(v, f"{path}[{i}]")
    if isinstance(s.get("additionalProperties"), dict):
        _schema_keywords(s["additionalProperties"], f"{path}.*")
    for i, v in enumerate(s.get("anyOf") or []):
        _schema_keywords(v, f"{path}|{i}")


def _is(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _JSON_TYPES[t])


def json_errors(s, v, path="$"):
    """Where a JSON value breaks a JSON schema (the keywords _SUPPORTED) → the first reason, or None."""
    if "anyOf" in s:
        if all(json_errors(x, v, path) for x in s["anyOf"]):
            return f"{path} matches none of anyOf"
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_is(v, t) for t in ts):
            return f"{path} is {type(v).__name__}, not {'/'.join(ts)}"
    if "enum" in s and v not in s["enum"]:
        return f"{path} = {v!r} is not one of {s['enum']}"
    if "const" in s and v != s["const"]:
        return f"{path} = {v!r} is not {s['const']!r}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        for k, bad in (("minimum", lambda b: v < b), ("maximum", lambda b: v > b), ("exclusiveMinimum", lambda b: v <= b),
                       ("exclusiveMaximum", lambda b: v >= b)):
            if k in s and bad(s[k]):
                return f"{path} = {v!r} breaks {k} {s[k]}"
    if isinstance(v, str):
        if len(v) < s.get("minLength", 0) or len(v) > s.get("maxLength", len(v)):
            return f"{path} has length {len(v)} outside [{s.get('minLength', 0)}, {s.get('maxLength', '∞')}]"
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", len(v)):
            return f"{path} has {len(v)} items, outside [{s.get('minItems', 0)}, {s.get('maxItems', '∞')}]"
        pre = s.get("prefixItems") or []
        for i, x in enumerate(v):
            sub = pre[i] if i < len(pre) else s.get("items")
            if isinstance(sub, dict):
                why = json_errors(sub, x, f"{path}[{i}]")
                if why:
                    return why
            elif sub is False and i >= len(pre):
                return f"{path} has more than {len(pre)} items"
    if isinstance(v, dict):
        props = s.get("properties") or {}
        for k in s.get("required") or []:
            if k not in v:
                return f"{path}.{k} is missing"
        for k, x in v.items():
            if k in props:
                why = json_errors(props[k], x, f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                why = f"{path}.{k} is not allowed"
            elif isinstance(s.get("additionalProperties"), dict):
                why = json_errors(s["additionalProperties"], x, f"{path}.{k}")
            else:
                why = None
            if why:
                return why
    return None


class Schema:
    """What a structured reply must be: a pydantic model (or any type pydantic validates: list[Row], dict[str, int], ...)
    or a JSON schema (a dict; the keywords listed in _SUPPORTED, else ValueError)."""

    def __init__(self, schema):
        self.raw = schema
        if isinstance(schema, dict):
            _schema_keywords(schema)
            self.adapter, self.json_schema = None, schema
            self.name = schema.get("title") or "the JSON schema"
        else:
            from pydantic import TypeAdapter
            self.adapter = TypeAdapter(schema)
            self.json_schema = self.adapter.json_schema()
            self.name = getattr(schema, "__name__", None) or str(schema)
        self.fp = hashlib.sha256(json.dumps(self.json_schema, sort_keys=True, ensure_ascii=False,
                                            default=str).encode()).hexdigest()[:16]

    def check(self, data):
        """Parsed JSON → the validated value; InvalidOutput with the first reason it is not."""
        if self.adapter is None:
            why = json_errors(self.json_schema, data)
            if why:
                raise InvalidOutput(f"the reply does not match {self.name}: {why}")
            return data
        from pydantic import ValidationError
        try:
            return self.adapter.validate_python(data)
        except ValidationError as e:
            err = e.errors()[0]
            where = ".".join(str(x) for x in err.get("loc", ())) or "$"
            raise InvalidOutput(f"the reply does not match {self.name}: {where}: {err.get('msg')}") from None


def _json_in(content):
    """The reply's JSON (an object or an array; a ```json fence or text around it tolerated) → the parsed value."""
    s = content.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    for a, b in (("{", "}"), ("[", "]")):
        i, j = s.find(a), s.rfind(b)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                continue
    raise InvalidOutput("the reply is not JSON")


def _pick(value, path):
    """The strings at a dotted path of a value ("rows", "items.*.source", "pairs.*.0"): dict keys, attributes of a
    model, list indexes or * for every item → a flat list. A missing key or attribute is an InvalidOutput (the quotes
    the caller asked for are not there); None stands for "not given" and gives nothing."""
    cur = [value]
    for step in path.split("."):
        nxt = []
        for v in cur:
            if v is None:
                continue
            if step == "*":
                if not isinstance(v, (list, tuple)):
                    raise InvalidOutput(f"quotes {path!r}: {type(v).__name__} is not a list")
                nxt += list(v)
            elif isinstance(v, dict):
                if step not in v:
                    raise InvalidOutput(f"quotes {path!r}: no {step!r} in the reply")
                nxt.append(v[step])
            elif isinstance(v, (list, tuple)) and step.lstrip("-").isdigit():
                nxt.append(v[int(step)] if -len(v) <= int(step) < len(v) else None)
            elif hasattr(v, step):
                nxt.append(getattr(v, step))
            else:
                raise InvalidOutput(f"quotes {path!r}: no {step!r} in the reply")
        cur = nxt
    out = []
    for v in cur:
        if isinstance(v, (list, tuple)):
            out += [x for x in v if x is not None]
        elif v is not None:
            out.append(v)
    bad = [x for x in out if not isinstance(x, (str, int, float)) or isinstance(x, bool)]
    if bad:
        raise InvalidOutput(f"quotes {path!r}: {bad[0]!r} is not a string")
    return [str(x) for x in out]


def quoted(value, quotes, text):
    """The strings at the `quotes` paths of a value, each checked to be in `text` as written (whole words and numbers)
    → the list of them; InvalidOutput naming the first one that is not there."""
    if not quotes:
        return []
    if not isinstance(text, str):
        raise ValueError("quotes= needs text=: the text the quoted strings must be in")
    out = []
    for path in quotes:
        for q in _pick(value, path):
            if not q.strip():
                raise InvalidOutput(f"an empty quote at {path!r}")
            if find_whole(q, text) < 0:
                raise InvalidOutput(f"the quote {q[:80]!r} ({path}) is not in the text")
            out.append(q)
    return out


def _messages(m):
    """A prompt → chat messages: a string is one user message; a list of {"role", "content"} is taken as it is."""
    if isinstance(m, str):
        return [{"role": "user", "content": m}]
    if isinstance(m, (list, tuple)) and all(isinstance(x, dict) and "role" in x and "content" in x for x in m) and m:
        return [dict(x) for x in m]
    raise ValueError("messages are a string or a non-empty list of {'role': ..., 'content': ...}")


def request_fp(body):
    """The fingerprint of a request: sha256 of its JSON body (keys sorted), 16 hex digits."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ the generator
class Generator:
    """A generating model over an OpenAI-compatible server. Build it with `generator(...)`, or share a decider's client
    with `Generator(llm_model.scorer)` / `Generator.of(llm_model)`. See the module docs."""

    deterministic = False                              # replay checks the recorded output instead of calling the model

    def __init__(self, client, *, max_tokens=1024, temperature=0.0, seed=None, response_format="prompt", workers=4):
        if not isinstance(client, LLMScorer):
            raise TypeError("Generator(client): an LLMScorer — `solvi.llm.llm(...).scorer`, or use generator(...)")
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}")
        self.client = client
        self.max_tokens, self.temperature, self.seed = int(max_tokens), float(temperature), seed
        self.response_format = response_format
        self.workers = max(1, int(workers))
        self.model_id = client.model_id
        self.calls = 0
        self._lock = threading.Lock()

    @classmethod
    def of(cls, model, **kw):
        """A Generator on the client of a decider from solvi.llm.llm(...) (or an LLMScorer): the same endpoint, key,
        headers, extra_body, retries and timeout."""
        return cls(getattr(model, "scorer", model), **kw)

    def __repr__(self):                                # never the key
        return f"Generator({self.client.endpoint!r}, {self.client.model!r})"

    def fingerprint(self):
        c = self.client
        fp = f"gen|{c.endpoint}|{c.model}|{self.max_tokens}|{self.temperature}|{self.seed}|{self.response_format}"
        if c.extra_body:
            fp += "|x:" + hashlib.sha256(json.dumps(c.extra_body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        return fp

    @property
    def usage(self):
        """Tokens used by this client (shared with a decider built on the same client)."""
        return self.client.usage

    # --- one request
    def body(self, messages, schema=None, temperature=None, seed=None, max_tokens=None):
        """The request body: extra_body, then the model, the messages, temperature and max_tokens (and seed when set,
        and the reply format when response_format asks for one)."""
        b = json.loads(json.dumps(self.client.extra_body)) if self.client.extra_body else {}
        b.update({"model": self.client.model, "messages": _messages(messages),
                  "temperature": float(self.temperature if temperature is None else temperature),
                  "max_tokens": int(self.max_tokens if max_tokens is None else max_tokens)})
        seed = self.seed if seed is None else seed
        if seed is not None:
            b["seed"] = int(seed)
        if schema is not None and self.response_format == "json_schema":
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": "solvi_output", "schema": schema.json_schema}}
        elif schema is not None and self.response_format == "json_object":
            b["response_format"] = {"type": "json_object"}
        return b

    def _send(self, body):
        """POST with the client's retries and backoff → the response dict; Unanswered / LLMError (see the module docs)."""
        c, attempt, last = self.client, 0, None
        while True:
            try:
                resp = c._post(body)
                with c._lock:
                    c.requests += 1
                with self._lock:
                    self.calls += 1
                return resp
            except urllib.error.HTTPError as e:
                if e.code in (400, 413, 422):
                    why = _error_text(e)
                    raise Unanswered(f"invalid input for the endpoint: HTTP {e.code}" + (f" — {why[:WHY_CHARS]}" if why else "")) from None
                if e.code in (408, 409, 429) or e.code >= 500:
                    last = f"HTTP {e.code}"
                else:
                    raise LLMError(f"HTTP {e.code} from {c.endpoint} (model {c.model!r}): check the URL, the model name and "
                                   "the API key") from None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as e:
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            attempt += 1
            if attempt > c.retries:
                raise Unanswered(f"no answer from {c.endpoint} after {attempt} attempts ({last})")
            c.sleep(c.backoff * 2 ** (attempt - 1))

    def _content(self, resp, meta):
        """A response → the reply's text; InvalidOutput for a refusal, a cut-off or empty reply, or no choices."""
        if not isinstance(resp, dict):
            raise InvalidOutput(f"the response is not a JSON object but a {type(resp).__name__}")
        chs = resp.get("choices")
        ch = chs[0] if isinstance(chs, list) and chs else None
        if not isinstance(ch, dict):
            raise InvalidOutput("the response has no choices")
        u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
        meta["usage"] = {k: u[k] for k in ("prompt_tokens", "completion_tokens") if isinstance(u.get(k), int)}
        with self.client._lock:
            for k, n in meta["usage"].items():
                self.client.usage[k] += n
        if resp.get("model") and resp.get("model") != self.client.model:
            meta["served_by"] = str(resp["model"])
        meta["finish"] = ch.get("finish_reason")
        msg = ch.get("message") or {}
        if not isinstance(msg, dict):
            raise InvalidOutput(f"the response's message is not an object but a {type(msg).__name__}")
        if msg.get("refusal"):
            raise InvalidOutput(f"the model refused: {str(msg['refusal'])[:120]}")
        if ch.get("finish_reason") == "length":
            raise InvalidOutput("the reply was cut off (max_tokens)")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise InvalidOutput("the reply is empty")
        return content

    def read(self, content, schema=None, parse=None, text=None, quotes=None):
        """A reply's text → (value, quotes): `parse` (text → value; raising or returning None rejects the reply), then
        the schema (JSON parsed from the text, or from what parse returned when it is a string), then the quotes.
        InvalidOutput with the reason when any step rejects it. Replay reads a recorded reply through this again."""
        value = content
        if parse is not None:
            try:
                value = parse(content)
            except Exception as e:  # noqa: BLE001 — the user's parser: whatever it raises rejects the reply
                raise InvalidOutput(f"the reply could not be read: {type(e).__name__}: {str(e)[:160]}") from None
            if value is None:
                raise InvalidOutput("nothing could be read from the reply")
        if schema is not None:
            value = schema.check(_json_in(value) if isinstance(value, str) else value)
        return value, quoted(value, quotes, text)

    def _one(self, messages, schema, parse, text, quotes, temperature, seed, max_tokens):
        """One request → (value, quotes, meta); raises InvalidOutput / Unanswered / LLMError."""
        body = self.body(messages, schema, temperature, seed, max_tokens)
        meta = {"model": self.model_id, "request": request_fp(body), "temperature": body["temperature"],
                "seed": body.get("seed")}
        if schema is not None:
            meta["schema"] = schema.fp
        content = self._content(self._send(body), meta)
        if schema is not None or parse is not None:
            meta["text"] = content
        try:
            value, qs = self.read(content, schema, parse, text, quotes)
        except InvalidOutput as e:
            e.reply = content                          # what the model said, for the next turn of a re-ask
            raise
        return value, qs, meta

    def generate(self, messages, *, schema=None, parse=None, text=None, quotes=None, source=None, temperature=None,
                 seed=None, max_tokens=None):
        """One reply → a Generated whose value is the text, `parse(text)`, or the JSON validated by `schema` (a pydantic
        model or type, or a JSON schema dict). text / quotes: the strings at the `quotes` paths of the value must be in
        `text` as written; they become the evidence (`source`: the given fact holding the text, for a catalog part).
        Raises InvalidOutput, Unanswered or LLMError (module docs) — an invalid reply is never repaired."""
        sch = _schema(schema)
        value, qs, meta = self._one(messages, sch, parse, text, quotes, temperature, seed, max_tokens)
        return Generated(value, evidence=qs, source=source, extra={"generated": meta})

    def sample(self, messages, k=3, *, temperature=0.8, schema=None, parse=None, text=None, quotes=None, source=None,
               max_tokens=None, first_greedy=True):
        """k replies to the same messages → a Generated whose value is the list of k outputs, None where a reply failed
        (its record says why). The first at the generator's own temperature (first_greedy=True: the reply `generate` gives),
        the others at `temperature` with seeds 1 .. k-1 (so a rerun asks the same requests). Raises only when every reply
        failed (the first failure's exception, with how many failed)."""
        if int(k) < 1:
            raise ValueError("k must be at least 1")
        sch = _schema(schema)
        k = int(k)

        def one(i):
            t, s = (None, None) if (i == 0 and first_greedy) else (temperature, i)
            try:
                return self._one(messages, sch, parse, text, quotes, t, s, max_tokens)
            except (InvalidOutput, Unanswered) as e:
                body = self.body(messages, sch, t, s, max_tokens)
                return None, [], {"model": self.model_id, "request": request_fp(body), "temperature": body["temperature"],
                                  "seed": body.get("seed"), "error": f"{type(e).__name__}: {e}"[:300], "exc": e}
        with ThreadPoolExecutor(min(self.workers, k)) as ex:
            outs = list(ex.map(one, range(k)))
        failed = [m for _, _, m in outs if "error" in m]
        if len(failed) == k:
            e = failed[0]["exc"]
            raise type(e)(f"all {k} replies failed; the first: {e}")
        metas = [{x: y for x, y in m.items() if x != "exc"} for _, _, m in outs]
        return Generated([v for v, _, _ in outs], evidence=[q for _, qs, _ in outs for q in qs], source=source,
                         extra={"generated": metas})

    # --- in a catalog, and in a loop
    def part(self, name, prompt, *, schema=None, parse=None, k=1, temperature=0.8, text=None, quotes=None,
             max_tokens=None, replay="trust"):
        """A catalog part that generates: `cat.fn(writer.part("sql", prompt))`. prompt: a function of facts by name (its
        parameters are the part's inputs) → the messages or one string. k > 1: `sample` (the fact is the list of k
        outputs). text: the name of the given fact the quotes must be in (one of prompt's parameters). replay: "trust"
        (default: the model is not called again; the recorded output is checked) or "rerun" (call it again and compare —
        only for a server that answers the same request the same way)."""
        if replay not in ("trust", "rerun"):
            raise ValueError('replay must be "trust" or "rerun"')
        sig = inspect.signature(prompt)
        if text is not None and text not in sig.parameters:
            raise ValueError(f"text={text!r} must be a parameter of the prompt function (the fact it reads)")
        if quotes and text is None:
            raise ValueError("quotes= needs text=: the given fact the quoted strings must be in")
        model = GenerationPart(self, prompt, _schema(schema), parse, int(k), temperature, quotes, max_tokens,
                               replay == "rerun")

        def run(**facts):
            msgs = prompt(**facts)
            src = facts.get(text) if text is not None else None
            if model.k == 1:
                return self.generate(msgs, schema=schema, parse=parse, text=src, quotes=quotes, source=text,
                                     max_tokens=max_tokens)
            return self.sample(msgs, model.k, temperature=temperature, schema=schema, parse=parse, text=src, quotes=quotes,
                               source=text, max_tokens=max_tokens)
        run.__name__ = run.__qualname__ = name
        run.__doc__ = prompt.__doc__
        run.__signature__ = sig.replace(return_annotation=inspect.Signature.empty)
        run.__annotations__ = {p: a for p, a in getattr(prompt, "__annotations__", {}).items() if p != "return"}
        if schema is not None and not isinstance(schema, (dict, Schema)):     # a typed fact: stored and restored as one
            from typing import Optional
            run.__annotations__["return"] = schema if model.k == 1 else list[Optional[schema]]
        run.__solvi_model__ = model
        run.__solvi_provenance__ = "proposed"
        return run

    def proposer(self, messages, *, schema=None, parse=None, text=None, quotes=None, first=None, template=FEEDBACK):
        """A proposer for solvi.refine.refine: (state, rounds) → a Generated. The first round asks `messages` (or
        `messages(state)` when it is a function); each later round asks the same messages followed, per earlier
        round, by its proposal as the assistant's turn (left out when empty) and its feedback as the user's turn — a text
        as it is, a list of reasons through `template` ("{reasons}": one "- reason" per line). first: another
        Generator for the first round (a stronger setting, say)."""
        def propose(state, rounds):
            msgs = _messages(messages(state) if callable(messages) else messages)
            for r in rounds:
                said = r.said
                if said:
                    msgs.append({"role": "assistant", "content": said})
                msgs.append({"role": "user", "content": render_feedback(r.feedback, template)})
            g = first if (first is not None and not rounds) else self
            return g.generate(msgs, schema=schema, parse=parse, text=text(state) if callable(text) else text,
                              quotes=quotes)
        propose.__solvi_model__ = self
        return propose


def render_feedback(feedback, template=FEEDBACK):
    """A round's feedback → the user's turn: a text as it is; a list of reasons through the template."""
    if isinstance(feedback, str):
        return feedback
    return template.format(reasons="\n".join(f"- {r}" for r in feedback or []))


def _schema(schema):
    return schema if (schema is None or isinstance(schema, Schema)) else Schema(schema)


class GenerationPart:
    """The model behind a part made by Generator.part: the generator, the prompt function's code, the schema and the
    settings — its fingerprint covers all of them, so a changed prompt or schema shows in a replay as a changed model.
    `check_record` re-reads a recorded output on replay (solvi.runtime calls it when the model is not re-run)."""

    def __init__(self, gen, prompt, schema, parse, k, temperature, quotes, max_tokens, rerun):
        self.gen, self.prompt, self.schema, self.parse = gen, prompt, schema, parse
        self.k, self.temperature, self.quotes, self.max_tokens = k, temperature, list(quotes or []), max_tokens
        self.model_id = gen.model_id
        self.deterministic = bool(rerun)

    def __repr__(self):
        return f"GenerationPart({self.gen!r}, k={self.k})"

    def fingerprint(self):
        from .provenance import code_fingerprint, digest
        return digest(self.gen.fingerprint(), code_fingerprint(self.prompt),
                      code_fingerprint(self.parse) if self.parse is not None else None,
                      self.schema.fp if self.schema is not None else None, self.k, self.temperature, self.quotes,
                      self.max_tokens)

    def check_record(self, r):
        """A recorded output, without calling the model → the reasons it is not what this part accepts: each recorded
        reply read again through parse and the schema must give the recorded value; a failed candidate must carry its
        error. (The quotes are checked in the text by the replay's grounding check, from the recorded evidence.)"""
        from .runtime import MISSING, vhash
        if r.value is MISSING or r.error is not None:
            return []
        metas = (r.extra or {}).get("generated")
        if metas is None:
            return ["no record of the generation (extra['generated'])"]
        single = self.k == 1
        metas, values = ([metas], [r.value]) if single else (metas, r.value)
        if not isinstance(metas, list) or not isinstance(values, (list, tuple)) or len(metas) != len(values):
            return ["the recorded outputs and their records do not match in number"]
        out = []
        for i, (m, v) in enumerate(zip(metas, values)):
            at = "" if single else f"output {i}: "
            if m.get("error"):
                if v is not None:
                    out.append(f"{at}recorded as failed ({m['error'][:60]}) but carries a value")
                continue
            if self.schema is None and self.parse is None:
                if not isinstance(v, str) or not v.strip():
                    out.append(f"{at}the recorded output is not a non-empty text")
                continue
            if "text" not in m:
                out.append(f"{at}the reply's text is not recorded")
                continue
            if self.schema is not None and m.get("schema") != self.schema.fp:
                out.append(f"{at}recorded against schema #{m.get('schema')}, the part now has #{self.schema.fp}")
                continue
            try:
                again, _ = self.gen.read(m["text"], self.schema, self.parse)
            except InvalidOutput as e:
                out.append(f"{at}the recorded reply is not accepted any more: {e}")
                continue
            if vhash(_plain(again)) != vhash(_plain(v)):
                out.append(f"{at}the recorded reply reads as {_short(again)}, not the recorded {_short(v)}")
        return out


def _plain(v):
    """A value as comparable data (a pydantic model as its dump)."""
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _short(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def generator(base_url, model, api_key=None, *, max_tokens=1024, temperature=0.0, seed=None, response_format="prompt",
              timeout=120.0, retries=2, backoff=1.0, headers=None, extra_body=None, workers=4, opener=None, sleep=None):
    """A Generator over an OpenAI-compatible chat-completions server (see the module docs). The connection settings are
    solvi.llm's (`llm(...)` takes the same: base_url, api_key — sent as a Bearer token, never recorded — timeout, retries
    and backoff for network errors, timeouts and 408 / 409 / 429 / 5xx, headers, extra_body — refused when it sets a field
    solvi sets: model, messages, temperature, max_tokens, seed, response_format —, opener for tests and proxies).
    max_tokens: per reply (reasoning tokens count against it on reasoning models). temperature: 0 by default; `sample`
    sets its own for the extra replies. seed: sent only when set. response_format: "prompt" (default: the contract is in
    your prompt, the request carries none), "json_object" or "json_schema" (the schema's JSON schema is sent) — for a
    structured reply only; whatever the server enforces, the reply is validated here."""
    client = LLMScorer(base_url, model, api_key, timeout=timeout, retries=retries, backoff=backoff, max_tokens=max_tokens,
                       seed=seed, headers=headers, extra_body=extra_body, opener=opener, sleep=sleep)
    return Generator(client, max_tokens=max_tokens, temperature=temperature, seed=seed, response_format=response_format,
                     workers=workers)


def several(generators, messages, **kw):
    """One reply from each of several generators (several models) to the same messages → a Generated whose value is
    the list of outputs, None where one failed — the candidates for solvi.agree. Raises only when every one failed."""
    gens = list(generators)
    if not gens:
        raise ValueError("several() needs at least one generator")

    def one(g):
        try:
            got = g.generate(messages, **kw)
            return got.value, list(got.evidence), got.meta
        except (InvalidOutput, Unanswered) as e:
            return None, [], {"model": g.model_id, "error": f"{type(e).__name__}: {e}"[:300], "exc": e}
    with ThreadPoolExecutor(len(gens)) as ex:
        outs = list(ex.map(one, gens))
    failed = [m for _, _, m in outs if "error" in m]
    if len(failed) == len(gens):
        e = failed[0]["exc"]
        raise type(e)(f"all {len(gens)} replies failed; the first: {e}")
    return Generated([v for v, _, _ in outs], evidence=[q for _, qs, _ in outs for q in qs], source=kw.get("source"),
                     extra={"generated": [{x: y for x, y in m.items() if x != "exc"} for _, _, m in outs]})
