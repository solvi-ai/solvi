"""What the CUAD solution adds before solvi checks a quote: a passage that is almost literal is cut to its longest
literal piece.

solvi.llm accepts a span answer only when it is literally in the text the model read (up to whitespace, typographic
quotes and letter case), and rejects the whole answer otherwise. An LLM asked for a clause often writes "..." in the
middle, stitches two passages, or adds a heading — the clause is there, the answer is rejected. The opener below sees
the reply before solvi does and replaces such an answer with its longest piece that is in the text; solvi then checks
the result as usual, so every quote that is kept is in the contract.
"""
import difflib
import json
import re
import urllib.request

from solvi.llm import locate


def literal_piece(answer, text, least=15):
    """(start, end) of the longest piece of `answer` that is literally in `text` — the answer cut at "..." / line
    breaks / sentence ends, else the longest common block (at least 30 characters) — or None."""
    best = None
    for sep in (r"\s*(?:\[\s*)?(?:\.{3,}|…)(?:\s*\])?\s*", r"\s*\n\s*", r"(?<=[.;:])\s+"):
        for piece in re.split(sep, answer):
            piece = piece.strip()
            if len(piece) >= least and (best is None or len(piece) > best[1] - best[0]):
                at = locate(piece, text)
                if at:
                    best = at
        if best:
            return best
    m = difflib.SequenceMatcher(None, text, answer, autojunk=False).find_longest_match(0, len(text), 0, len(answer))
    return (m.a, m.a + m.size) if m.size >= max(least, 30) else None


class _Reply:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.data


def opener(req, timeout=None):
    """urlopen for solvi.llm's `opener=`: a span answer that is not literally in the text the model read is replaced by
    its longest literal piece before solvi reads the reply. Anything this does not understand is passed on unchanged."""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        body, data = json.loads(req.data), json.loads(raw)
        msg = data["choices"][0]["message"]
        content = msg["content"]
        a, b = content.find("{"), content.rfind("}")                # the reply's JSON, also when the model wrote around it
        reply = json.loads(content[a:b + 1])
        user = body["messages"][-1]["content"]
        text = user[user.index("<text>\n") + 7: user.rindex("\n</text>")]
        ans = reply.get("answer")
        if isinstance(ans, str) and ans.strip() and locate(ans.strip(), text) is None:
            at = literal_piece(ans, text)
            if at:
                reply["answer"] = text[at[0]:at[1]]
                msg["content"] = content[:a] + json.dumps(reply, ensure_ascii=False) + content[b + 1:]
                raw = json.dumps(data, ensure_ascii=False).encode()
    except (ValueError, KeyError, IndexError, TypeError):
        pass
    return _Reply(raw)
