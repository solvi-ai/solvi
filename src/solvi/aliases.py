"""Name matching: when the catalog's parameter names do not match its facts —
parts written by different teams, each with its own naming style — a matcher model proposes aliases ("the parameter
`INVC_AMNT` is the fact `invoice_total`"), and deterministic code decides which to accept.

    from solvi.aliases import NameMatcher, propose, accept
    m = NameMatcher.load("path/to/strategist-checkpoint/matcher")
    props = propose(cat, questions, init_keys, m)                    # [Proposal(aliases={name: fact}, score)]
    got = accept(cat, questions, props, examples, probes=states, oracle=ask_person)   # labelled examples + targeted questions
    if got.aliases is not None:
        cat2 = apply(cat, got.aliases)                                # parts rewired to the facts; cat2.aliases lists them
        System(cat2, questions)                                       # plans by exact names again; the trace is unchanged

Acceptance: a proposal is accepted only if its answers match every labelled example AND no neighbouring wiring
(another proposal, one alias swapped for another candidate, two same-typed aliases exchanged) also matches the examples but
answers differently on the unlabelled probes. In the active mode, while such neighbours remain, solvi picks the probe on
which most of them disagree with the proposal and asks `oracle(state)` for its right answers (a person labels that case);
a proposal that contradicts an answer is dropped. No neighbour left → accepted; questions used up → not accepted
("ambiguous": ask a person). The model never decides alone: a wrong alias costs coverage (no aliases accepted), not a
silent wrong wiring — as far as the examples and probes can tell wirings apart.

Accepted aliases are applied by rewiring: every part that read an aliased name now reads the fact itself (its function is
wrapped, its docstring lists the aliases), and the new catalog lists them in `catalog.aliases`; planning, checks, quotes into
given texts, execution, the trace and replay behave exactly as for a catalog written with one naming.

The matcher model (`NameMatcher`) is experimental: no checkpoint is published — `NameMatcher.load` reads one you trained
yourself (docs/strategist.md has the format), and examples/17 uses a stand-in. `propose` takes any object with the same
methods; `accept` and `apply` are plain code."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re

import numpy as np

MAXC = 40
CH = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ")}
NCH = len(CH) + 2


def split_ident(s):
    """An identifier → lowercase words (snake, camel, UPPER): exactly the matcher's training split."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return " ".join(t for t in re.split(r"[_\s]+", s.lower()) if t)


def char_ids(names):
    out = np.zeros((len(names), MAXC), dtype=np.int64)
    for j, n in enumerate(names):
        if n is None:
            continue
        s = (" " + split_ident(n) + " ")[:MAXC]
        out[j, :len(s)] = [CH.get(c, NCH - 1) for c in s]
    return out


# ---------------------------------------------------------------------------------------------------------------- catalog
def _tname(t):
    if t is None:
        return "any"
    from .typed import type_name
    return type_name(t)


def readers(catalog):
    """name read → [(reader part, its declared type or None)] over parts (every producer), checks and rules."""
    from .strategy import alternatives
    out = {}
    for p in list(catalog.parts.values()) + list(catalog.rules.values()):
        for a in alternatives(p) if p.kind != "rule" else [p]:
            for x in a.inputs:
                out.setdefault(x, []).append((a, (a.types or {}).get(x)))
    return out


def unresolved(catalog, init_keys):
    """Names that parts read but that are neither given nor a fact → {name: [(reader part, type)]}."""
    init = set(init_keys)
    return {x: rs for x, rs in readers(catalog).items() if x not in init and x not in catalog.parts}


def _fits(catalog, fact, t, init_types):
    from .typed import compatible
    ft = catalog.types.get(fact, init_types.get(fact))
    if ft is None or t is None:
        return True
    try:
        return bool(compatible(ft, t))
    except Exception:  # noqa: BLE001
        return False


def sources(catalog, init_keys, name, rs, init_types=None):
    """Facts a name could be: given facts and facts made by parts other than its readers, of a fitting type."""
    init_types = init_types or {}
    own = {a.provides or a.name for a, _ in rs}
    out = [("given", x) for x in sorted(init_keys) if all(_fits(catalog, x, t, init_types) for _, t in rs)]
    out += [("fact", f) for f, p in catalog.parts.items()
            if f not in own and p.kind != "check" and all(_fits(catalog, f, t, init_types) for _, t in rs)]
    out += [("fact", f) for f, p in catalog.parts.items() if f not in own and p.kind == "check"
            and all(t is None or _fits(catalog, f, t, init_types) for _, t in rs) and any(t is bool for _, t in rs)]
    return out


def param_text(name, t, reader):
    return f"{split_ident(name)} ({_tname(t)}) | parameter of {split_ident(reader.name)}: {reader.doc or ''}"


def source_text(catalog, kind, f, init_types):
    if kind == "given":
        return f"{split_ident(f)} ({_tname(init_types.get(f))}) | given input"
    p = catalog.parts[f]
    doc = p.doc if not p.alternatives else " / ".join(a.doc for a in p.alternatives if a.doc)
    return f"{split_ident(f)} (returns {_tname(catalog.types.get(f))}) | {doc}"


# ---------------------------------------------------------------------------------------------------------------- matcher
class NameMatcher:
    """The link encoder: all-MiniLM-L6-v2 over texts (name, type, docstring), for arch "char" fused with a character CNN
    over the identifier (score = w_t·cos_text + w_c·cos_char), for arch "text" the text part alone; divided by T (0.05).
    Backends: "torch" (transformers) or "onnx" (onnxruntime + tokenizers).

    Experimental: no checkpoint is published — it reads one you trained yourself (docs/strategist.md has the format)."""

    def __init__(self, enc, meta, path, backend):
        self.enc, self.meta, self.path, self.backend = enc, meta, path, backend
        self.T = float(meta.get("T", 0.05))
        self._fp = None

    @classmethod
    def load(cls, path_or_id, backend="auto", threads=None):
        from .loader import optional
        from .strategy_model import _auto_backend
        if backend not in ("auto", "torch", "onnx"):
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        path = os.path.expanduser(str(path_or_id))
        if not os.path.isdir(path):
            path = optional("huggingface_hub", "onnx", "NameMatcher.load of a Hugging Face id").snapshot_download(path_or_id)
        mf = os.path.join(path, "solvi_matcher.json")
        if not os.path.isfile(mf):
            raise FileNotFoundError(f"{path} has no solvi_matcher.json: not a solvi name matcher checkpoint")
        meta = json.load(open(mf))
        onnx_file = os.path.join(path, "onnx", "matcher.onnx")
        if backend == "auto":
            backend = _auto_backend(os.path.isfile(onnx_file), "NameMatcher")
        enc = _OnnxEnc(path, onnx_file, threads, meta) if backend == "onnx" else _TorchEnc(path, meta, threads)
        m = cls(enc, meta, path, backend)
        m._files = [os.path.join(path, f) for f in ("solvi_matcher.json", "tokenizer.json")] + \
                   [onnx_file if backend == "onnx" else os.path.join(path, "model.safetensors")]
        return m

    @property
    def fingerprint(self):
        if self._fp is None:
            from .decide import _file_fingerprint
            self._fp = _file_fingerprint(self._files)
        return self._fp

    def info(self):
        return {"type": "NameMatcher", "id": self.meta.get("name", os.path.basename(self.path.rstrip("/"))),
                "fp": self.fingerprint}

    def embed(self, texts, names, bs=128):
        out = [self.enc(texts[s:s + bs], names[s:s + bs]) for s in range(0, len(texts), bs)]
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


class _TorchEnc:
    def __init__(self, path, meta, threads=None):
        from .loader import optional
        torch = optional("torch", "model", 'NameMatcher (backend="torch")')
        load_file = optional("safetensors.torch", "model", 'NameMatcher (backend="torch")').load_file
        tf = optional("transformers", "model", 'NameMatcher (backend="torch")')
        AutoTokenizer, BertConfig, BertModel = tf.AutoTokenizer, tf.BertConfig, tf.BertModel
        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.text = BertModel(BertConfig(**{k: v for k, v in cfg.items() if k != "architectures"}), add_pooling_layer=False)
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.text.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("text.")}, strict=False)
        self.text.eval()
        self.char = None                              # arch "text": the ablation without the character CNN
        if meta.get("arch", "char") == "char":
            self.char = _char_cnn(torch)
            self.char.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("char.")})
            self.char.eval()
            self.w = sd["logw"].exp().float()

    def __call__(self, texts, names):
        torch = self.torch
        with torch.inference_mode():
            b = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = self.text(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            et = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)
            if self.char is None:
                return et.float().numpy()
            ec = self.char(torch.from_numpy(char_ids(names)))
            has = torch.tensor([n is not None for n in names], dtype=ec.dtype).unsqueeze(1)
            return torch.cat([et * self.w[0].sqrt(), ec * has * self.w[1].sqrt()], 1).float().numpy()


class _OnnxEnc:
    def __init__(self, path, file, threads=None, meta=None):
        from .loader import optional
        ort = optional("onnxruntime", "onnx", 'NameMatcher (backend="onnx")')
        Tokenizer = optional("tokenizers", "onnx", 'NameMatcher (backend="onnx")').Tokenizer
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.s = ort.InferenceSession(file, so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.tok.enable_truncation(128)
        self.tok.enable_padding()
        self.char = (meta or {}).get("arch", "char") == "char"

    def __call__(self, texts, names):
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tt = np.zeros_like(ids)
        if not self.char:
            return self.s.run(["emb"], {"input_ids": ids, "attention_mask": am, "token_type_ids": tt})[0]
        has = np.array([[n is not None] for n in names], dtype=np.float32)
        return self.s.run(["emb"], {"input_ids": ids, "attention_mask": am, "token_type_ids": tt,
                                    "chars": char_ids(names), "has_char": has})[0]


def _char_cnn(torch, dim=256, emb=48, ch=128, widths=(2, 3, 4, 5)):
    nn = torch.nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(NCH, emb, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(emb, ch, k, padding=k // 2) for k in widths])
            self.proj = nn.Sequential(nn.Linear(ch * len(widths), dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        def forward(self, ids):
            x = self.emb(ids).transpose(1, 2)
            mask = (ids > 0).unsqueeze(1)
            hs = []
            for c in self.convs:
                h = torch.relu(c(x))[:, :, : ids.shape[1]]
                h = h.masked_fill(~mask, -1e4)
                hs.append(h.max(dim=2).values)
            return torch.nn.functional.normalize(self.proj(torch.cat(hs, 1)), dim=-1)
    return CharCNN()


# ---------------------------------------------------------------------------------------------------------------- proposals
@dataclasses.dataclass
class Proposal:
    aliases: dict                 # name → fact (or given)
    score: float                  # sum of log-probabilities of the chosen links
    links: dict = dataclasses.field(default_factory=dict)   # name → [(source, log-probability)], the top candidates


def _log_softmax(x):
    m = np.max(x)
    return x - m - math.log(np.exp(x - m).sum())


def link_table(catalog, questions, init_keys, matcher, k=4, init_types=None):
    """For each unresolved name its top-k sources with log-probabilities → {name: [(source, logp)]}."""
    init_types = init_types or {}
    un = unresolved(catalog, init_keys)
    names = sorted(un)
    if not names:
        return {}
    cand = {x: sources(catalog, init_keys, x, un[x], init_types) for x in names}
    srcs = sorted({s for cs in cand.values() for s in cs})
    anchors = []
    for x in names:
        rd, t = un[x][0]
        anchors.append((param_text(x, t, rd), x))
    S = matcher.embed([source_text(catalog, kd, f, init_types) for kd, f in srcs], [f for _, f in srcs])
    A = matcher.embed([t for t, _ in anchors], [n for _, n in anchors])
    row = {s: j for j, s in enumerate(srcs)}
    out = {}
    for i, x in enumerate(names):
        cs = cand[x]
        if not cs:
            out[x] = []
            continue
        sc = np.array([float(A[i] @ S[row[c]]) for c in cs]) / matcher.T
        lp = _log_softmax(sc)
        top = np.argsort(-lp)[:k]
        out[x] = [(cs[j][1], float(lp[j])) for j in top]
    return out


def _cyclic(catalog, aliases):
    """Would these aliases make a fact depend on itself?"""
    from .strategy import alternatives
    dep = {}
    for f, p in catalog.parts.items():
        dep[f] = {aliases.get(x, x) for a in alternatives(p) for x in a.inputs}
    color = {}

    def dfs(f):
        color[f] = 1
        for y in dep.get(f, ()):
            c = color.get(y, 0)
            if c == 1 or (c == 0 and y in dep and dfs(y)):
                return True
        color[f] = 2
        return False
    return any(color.get(f, 0) == 0 and dfs(f) for f in dep)


def _same_source_twice(catalog, aliases):
    from .strategy import alternatives
    for p in list(catalog.parts.values()) + list(catalog.rules.values()):
        for a in alternatives(p) if p.kind != "rule" else [p]:
            got = [aliases.get(x, x) for x in a.inputs]
            if len(set(got)) != len(got):
                return True
    return False


def propose(catalog, questions, init_keys, matcher, k=4, beam=16, max_props=24, prune=4.6, init_types=None, table=None):
    """Joint alias proposals by beam search over the unresolved names (best first): each name takes one of its top-k
    sources; a part never reads one fact twice; no cycles. → [Proposal] by score."""
    table = table if table is not None else link_table(catalog, questions, init_keys, matcher, k, init_types)
    names = sorted(table, key=lambda x: -(table[x][0][1] - table[x][1][1]) if len(table[x]) > 1 else 0)
    names = [x for x in names if table[x]]
    states = [(0.0, {})]
    for x in names:
        nxt = []
        best = table[x][0][1]
        opts = list(table[x]) + [(None, table[x][-1][1] - 2.0)]   # None: leave the name unresolved (e.g. an input that
        for sc, al in states:                                     # nobody gives); the examples decide whether it matters
            for s, lp in opts:
                if s is not None and lp < best - prune:
                    continue
                nxt.append((sc + lp, {**al, x: s} if s is not None else dict(al)))
        nxt.sort(key=lambda t: -t[0])
        keep, seen = [], set()
        for sc, al in nxt:
            if len(keep) >= beam:
                break
            key = tuple(sorted(al.items()))
            if key in seen or _same_source_twice(catalog, al) or _cyclic(catalog, al):
                continue
            seen.add(key)
            keep.append((sc, al))
        states = keep
        if not states:
            return []
    out = [Proposal(al, sc, table) for sc, al in states if not _cyclic(catalog, al)]
    return out[:max_props]


# ---------------------------------------------------------------------------------------------------------------- apply
_ATTRS = ("__solvi_model__", "__solvi_provenance__", "__solvi_options__", "__solvi_decision__")


def _rewired(func, part, aliases):
    """func with its parameters renamed through the aliases (the part now reads the facts themselves)."""
    import functools
    import inspect
    old = list(part.inputs)
    new = [aliases.get(x, x) for x in old]
    if new == old:
        return func
    back = dict(zip(new, old))

    @functools.wraps(func)
    def w(**kw):
        return func(**{back[n]: v for n, v in kw.items()})
    w.__signature__ = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in new])
    ann = {aliases.get(k, k): v for k, v in (getattr(func, "__annotations__", None) or {}).items()}
    w.__annotations__ = ann
    for a in _ATTRS:
        if hasattr(func, a):
            setattr(w, a, getattr(func, a))
    return w


def _validate_rewired(v, aliases):
    if v is None:
        return None
    import inspect
    names = list(inspect.signature(v).parameters)[1:]
    new = [aliases.get(x, x) for x in names]
    if new == names:
        return v

    def vv(value, **kw):
        return v(value, **{o: kw[n] for n, o in zip(new, names) if n in kw})
    vv.__signature__ = inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)] +
                                         [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in new])
    return vv


def apply(catalog, aliases):
    """A new catalog in which every part reads the aliased facts under their own names (`name → fact`): the result is the
    catalog as if the teams had used one naming — planning, checks on computed facts, quotes into given texts, types and
    the trace behave exactly as they would, and every declaration of a part (cost, timeout, blocking, validate, …) is
    kept. The accepted aliases are listed in `catalog.aliases` (name → fact) and in the
    rewired parts' docstrings."""
    from .core import Catalog
    from .strategy import alternatives
    base = getattr(catalog, "aliases", None) or {}
    new = Catalog()
    new.aliases = {**base, **aliases}
    for f, p in catalog.parts.items():
        for a in alternatives(p):
            fn = a.func.__wrapped__ if getattr(a, "source", None) and hasattr(a.func, "__wrapped__") else a.func
            w = _rewired(fn, a, aliases)
            got = [x for x in a.inputs if x in aliases]
            if got and w is not fn:
                w.__doc__ = ((a.doc or "") + " [aliases: " + ", ".join(f"{x} = {aliases[x]}" for x in got) + "]").strip()
            kw = {}
            for k in ("cost", "model", "min_confidence", "timeout"):
                if getattr(a, k) is not None:
                    kw[k] = getattr(a, k)
            if a.blocking:
                kw["blocking"] = True
            if a.provenance is not None and a.kind != "check":
                kw["provenance"] = a.provenance
            if p.alternatives is not None:
                kw["provides"] = f
                kw["validate"] = _validate_rewired(a.validate, aliases)
            if a.kind == "extract":
                if a.exact is not None:
                    kw["exact"] = a.exact
                if a.source is not None:
                    kw["source"] = aliases.get(a.source, a.source)
                new.extract(w, **kw)
            elif a.kind == "fn":
                if a.options is not None:
                    kw["options"] = a.options
                if a.validate is not None and p.alternatives is None:
                    kw["validate"] = _validate_rewired(a.validate, aliases)
                new.fn(w, **kw)
            elif a.kind == "check":
                kw.pop("min_confidence", None)
                new.check(hard=a.hard, then=dict(a.then or {}), **kw)(w)
            else:
                raise ValueError(f"cannot rewire a {a.kind} part")
        if p.features is not None and p.alternatives is not None:
            new.features(f)(_rewired(p.features, _FakePart(p.features), aliases))
    for q, r in catalog.rules.items():
        w = _rewired(r.func, r, aliases)
        new.rule(q, **{k: getattr(r, k) for k in ("model", "provenance", "timeout") if getattr(r, k) is not None},
                 blocking=r.blocking or None)(w)
    for c in catalog.constraints.values():
        new.constraint(c.func)
    return new


class _FakePart:
    def __init__(self, f):
        import inspect
        self.inputs = list(inspect.signature(f).parameters)


# ---------------------------------------------------------------------------------------------------------------- accept
@dataclasses.dataclass
class Acceptance:
    aliases: dict | None          # the accepted aliases, or None (not accepted: ask a person)
    why: str
    labels: int                   # labelled examples used (given + asked)
    asked: list = dataclasses.field(default_factory=list)   # probes whose answers were asked (the active mode)
    proposal: int | None = None   # index of the accepted proposal


class _Answers:
    def __init__(self, catalog, questions, strategist=None):
        from .strategy import ModelStrategist
        from .system import System
        self.catalog, self.questions, self.cache, self.systems = catalog, questions, {}, {}
        self.System = System
        self.strategist = strategist if strategist is not None else ModelStrategist()   # dead ends do not block

    def system(self, al):
        key = tuple(sorted(al.items()))
        if key not in self.systems:
            try:
                s = self.System(apply(self.catalog, al), copy.deepcopy(self.questions), strategist=self.strategist)
            except Exception as e:  # noqa: BLE001  (a type conflict: the wiring is rejected)
                s = e
            self.systems[key] = s
        return self.systems[key]

    def __call__(self, al, states, tag):
        key = (tuple(sorted(al.items())), tag)
        if key not in self.cache:
            s = self.system(al)
            out = []
            for st in states:
                if isinstance(s, Exception):
                    out.append("<error>")
                    continue
                try:
                    r = s.ask(copy.deepcopy(st))
                    out.append(tuple((q.name, "<abstain>" if r[q.name].status == "abstain" else _plain(r[q.name].answer))
                                     for q in self.questions))
                except Exception:  # noqa: BLE001
                    out.append("<error>")
            self.cache[key] = out
        return self.cache[key]


def _plain(a):
    return tuple(a) if isinstance(a, (list, tuple)) else a


def _key(answers):
    return tuple(sorted(answers.items())) if isinstance(answers, dict) else answers


def neighbours(proposals, i, catalog, max_alt=4):
    """Wirings next to proposal i: the other proposals, one alias replaced by another of its top candidates, two aliases of
    fitting types exchanged."""
    p = proposals[i]
    out, seen = [], {tuple(sorted(p.aliases.items()))}

    def add(al):
        key = tuple(sorted(al.items()))
        if key not in seen and not _same_source_twice(catalog, al) and not _cyclic(catalog, al):
            seen.add(key)
            out.append(al)
    for j, q in enumerate(proposals):
        if j != i:
            add(q.aliases)
    for x, s in p.aliases.items():
        for s2, _ in p.links.get(x, [])[:max_alt]:
            if s2 != s:
                add({**p.aliases, x: s2})
    xs = sorted(p.aliases)
    for a in range(len(xs)):
        for b in range(a + 1, len(xs)):
            x, y = xs[a], xs[b]
            if p.aliases[y] in [s for s, _ in p.links.get(x, [])] and p.aliases[x] in [s for s, _ in p.links.get(y, [])]:
                add({**p.aliases, x: p.aliases[y], y: p.aliases[x]})
    return out


def _minimal(ans, al, lab_s, lab_y, probes):
    """Drop every alias the answers do not need (same answers on the examples and the probes without it): a name that is
    legitimately not given (a dead end's input) stays unresolved instead of being wired to some look-alike."""
    ref = ans(al, probes, "probe")
    for x in sorted(al):
        al2 = {k: v for k, v in al.items() if k != x}
        if ans(al2, lab_s, "lab") == lab_y and ans(al2, probes, "probe") == ref:
            al = al2
    return al


def accept(catalog, questions, proposals, examples, probes=(), k=5, oracle=None, active=None, check_neighbours=True,
           minimal=True, strategist=None, confirm=True):
    """Deterministic acceptance of alias proposals.
    examples: [(state, {question: answer})] labelled cases (the first k are used); probes: unlabelled states (targeted
    questions and the distinguishability check); oracle(state) → {question: answer} (a person) for the active mode;
    active=(k0, m): k0 labelled examples + up to m targeted questions (instead of k examples). minimal: drop the accepted
    aliases that change no answer on the examples and probes (they stay unresolved). confirm (active mode): spend the whole
    question budget even when no neighbour disagrees any more (the extra questions go where the other proposals disagree
    most), so a wiring is never accepted on the k0 random labels alone. strategist: plans each candidate wiring
    (default ModelStrategist(): the deterministic plan with dead ends dropped)."""
    if not proposals:
        return Acceptance(None, "no proposal", 0)
    ans = _Answers(catalog, list(questions), strategist)
    probes = list(probes)
    if active is None:
        lab = [(s, _key({q: a[q] for q in a})) for s, a in examples[:k]]
        lab_s = [s for s, _ in lab]
        lab_y = [_want(y, questions) for _, y in lab]
        pick = next((i for i, p in enumerate(proposals) if ans(p.aliases, lab_s, "lab") == lab_y), None)
        if pick is None:
            return Acceptance(None, "no proposal matches the examples", len(lab))
        if not check_neighbours:
            return Acceptance(proposals[pick].aliases, "matches the examples", len(lab), proposal=pick)
        ref = ans(proposals[pick].aliases, probes, "probe")
        amb = [al for al in neighbours(proposals, pick, catalog)
               if ans(al, lab_s, "lab") == lab_y and ans(al, probes, "probe") != ref]
        if amb:
            return Acceptance(None, f"ambiguous: {len(amb)} other wirings fit the examples but answer differently",
                              len(lab), proposal=pick)
        al = _minimal(ans, proposals[pick].aliases, lab_s, lab_y, probes) if minimal else proposals[pick].aliases
        return Acceptance(al, "matches the examples; no other wiring fits them", len(lab), proposal=pick)
    k0, m = active
    min_q = m if confirm else 0
    lab = examples[:k0]
    lab_s = [s for s, _ in lab]
    lab_y = [_want(_key(y), questions) for _, y in lab]
    known, excluded, asked = {}, set(), []

    def fits(al):
        if ans(al, lab_s, "lab") != lab_y:
            return False
        pa = ans(al, probes, "probe")
        return all(pa[j] == y for j, y in known.items())
    while True:
        pick = next((i for i, p in enumerate(proposals) if i not in excluded and fits(p.aliases)), None)
        if pick is None:
            return Acceptance(None, "no proposal matches the examples", k0 + len(asked), asked)
        ref = ans(proposals[pick].aliases, probes, "probe")
        A = [al for al in neighbours(proposals, pick, catalog) if fits(al) and ans(al, probes, "probe") != ref]
        while True:
            if not A and len(asked) < min_q and oracle is not None:
                # no neighbour disagrees, but the question budget is not spent: confirm on the probe where the other
                # proposals disagree most with this one (else the next unasked probe) — a proposal is never accepted on
                # the k0 random labels alone
                free = [j for j in range(len(probes)) if j not in known]
                if free:
                    others = [p.aliases for i, p in enumerate(proposals) if i != pick and i not in excluded and fits(p.aliases)]
                    j = max(free, key=lambda j: (sum(ans(al, probes, "probe")[j] != ref[j] for al in others), -j))
                    y = _want(_key(oracle(probes[j])), questions)
                    known[j] = y
                    asked.append(j)
                    if ref[j] != y:
                        excluded.add(pick)
                        break
                    continue
            if not A:
                al = proposals[pick].aliases
                if minimal:
                    ls = lab_s + [probes[j] for j in known]
                    ly = lab_y + [known[j] for j in known]
                    al = _minimal(ans, al, ls, ly, probes)
                return Acceptance(al, f"no other wiring fits {k0} examples and {len(asked)} answers", k0 + len(asked),
                                  asked, pick)
            if len(asked) >= m or oracle is None:
                return Acceptance(None, f"ambiguous after {len(asked)} questions ({len(A)} wirings left)", k0 + len(asked),
                                  asked, pick)
            free = [j for j in range(len(probes)) if j not in known]
            if not free:
                return Acceptance(None, "no probe left to ask", k0 + len(asked), asked, pick)
            j = max(free, key=lambda j: sum(ans(al, probes, "probe")[j] != ref[j] for al in A))
            y = _want(_key(oracle(probes[j])), questions)
            known[j] = y
            asked.append(j)
            if ref[j] != y:
                excluded.add(pick)
                break
            A = [al for al in A if ans(al, probes, "probe")[j] == y]


def _want(y, questions):
    d = dict(y) if not isinstance(y, dict) else y
    return tuple((q.name, _plain(d.get(q.name, "<abstain>")) if d.get(q.name) is not None else "<abstain>")
                 for q in questions)


def match_names(catalog, questions, init_keys, matcher, examples, probes=(), oracle=None, active=(3, 7), k=5,
                init_types=None):
    """Everything in one call: propose, accept (active mode when an oracle is given), apply. → (catalog, Acceptance) —
    the catalog with the accepted aliases, or the original one when nothing was accepted (or nothing was unresolved)."""
    if not unresolved(catalog, init_keys):
        return catalog, Acceptance({}, "every name resolves", 0)
    props = propose(catalog, questions, init_keys, matcher, init_types=init_types)
    got = accept(catalog, questions, props, examples, probes, k=k, oracle=oracle, active=active if oracle else None)
    return (apply(catalog, got.aliases) if got.aliases else catalog), got


__all__ = ["NameMatcher", "Proposal", "Acceptance", "unresolved", "link_table", "propose", "accept", "apply", "match_names",
           "split_ident"]
