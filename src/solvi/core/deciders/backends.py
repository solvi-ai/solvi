"""The encoders: ONNX and torch scorers, tokenizing with block masks, file fingerprints. (Part of solvi.core.deciders, which
re-exports every name.)"""
from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

from .kinds import MARKERS
from .wire import pass_prompt, prompt
from .capabilities import capabilities


class LongInputWarning(UserWarning):
    """long="full": a checkpoint forced to read long inputs it was not trained on, or whole long texts read on a CPU.
    Without long=: an input that does not fit the pass was read cut (the decision's extra["truncated"] says how much)."""


LONG_CPU_TOKENS = 2048               # long="full" on a CPU warns (once per model) when it reads a text longer than this


SECTION_TOKENS = 170                 # top_k=None: sections of about this many tokens (top_k = budget / 170, at least 3)


class _Encoder:
    """Tokenization with the checkpoint's tokenizer.json (the `tokenizers` library).

    encode (one question, or the "concat" layout): [CLS] questions [SEP] input [SEP], only the input is truncated;
    encode_block (the "block" layout): [CLS] input [SEP] question 1 [SEP] question 2 [SEP] …, with position ids (each block
    continues the input's positions) and a block id per token (−1 input, j block j) for the attention mask.
    Both return the token ids and, per question, the position of its mode marker and of its option markers."""

    def __init__(self, path, max_len, markers=None):
        Tokenizer = need("tokenizers", "onnx", "a decider checkpoint's tokenizer").Tokenizer
        f = os.path.join(path, "tokenizer.json")
        self.tok = Tokenizer.from_file(f)
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len, strategy="only_second")
        self.raw = Tokenizer.from_file(f)             # no truncation: the block layout budgets the input itself
        self.raw.no_padding()
        self.raw.no_truncation()
        self.max_len = max_len
        self._path, self._at, self._at_lock = f, {max_len: self.tok}, threading.Lock()
        self._qt = {}                                 # question segment → its tokens (see question_tokens)
        self.markers = {**MARKERS, **(markers or {})}
        self.opt_id = self.tok.token_to_id(self.markers["option"])
        pad = next((self.tok.token_to_id(t) for t in ("[PAD]", "<pad>") if self.tok.token_to_id(t) is not None), 0)
        self.pad_id = pad
        self.cls_id = next((self.tok.token_to_id(t) for t in ("[CLS]", "<s>") if self.tok.token_to_id(t) is not None), None)
        self.sep_id = next((self.tok.token_to_id(t) for t in ("[SEP]", "</s>") if self.tok.token_to_id(t) is not None), None)
        if self.opt_id is None:
            raise ValueError(f"the tokenizer has no {self.markers['option']} token: not a solvi-decide checkpoint")
        self.mode_ids = {i for m, t in self.markers.items() if m != "option" for i in [self.tok.token_to_id(t)]
                         if i is not None}

    def _groups(self, ids, keep, items):
        groups = []
        for i, t in enumerate(ids):
            if not keep(i):
                continue
            if t in self.mode_ids:
                groups.append((i, []))
            elif t == self.opt_id and groups:
                groups[-1][1].append(i)
        if len(groups) != len(items) or any(len(g[1]) != len(it.options) for g, it in zip(groups, items)):
            raise ValueError(f"found {sum(len(g[1]) for g in groups)} option markers for {sum(len(it.options) for it in items)} "
                             f"options (an option contains {self.markers['option']}, or the options do not fit in "
                             f"{self.max_len} tokens)")
        return groups

    def tok_at(self, n):
        """The tokenizer that truncates the input so that the sequence has at most n tokens (the default: max_len;
        long="full" reads at the checkpoint's long-input length)."""
        n = int(n or self.max_len)
        with self._at_lock:
            tok = self._at.get(n)
            if tok is None:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(self._path)
                tok.no_padding()
                tok.enable_truncation(max_length=n, strategy="only_second")
                self._at[n] = tok
            return tok

    def question_tokens(self, items):
        """The tokens a pass spends before the input: the questions' segment and the special tokens around the pair."""
        p = pass_prompt(items, self.markers)
        with self._at_lock:
            k = self._qt.get(p)
        if k is None:
            k = len(self.raw.encode(p, add_special_tokens=False).ids) + self.raw.num_special_tokens_to_add(True)
            with self._at_lock:
                if len(self._qt) > 4096:
                    self._qt.clear()
                self._qt[p] = k
        return k

    def read(self, items, text):
        """How much of an input one sequence reads (the full layout: only the input is cut) → None when all of it, else
        {"input_tokens", "read_tokens", "question_tokens", "max_len"}. A text of fewer bytes than the tokens left for it
        fits whatever its tokens are (a token is at least one byte), so short inputs are not tokenized again."""
        n = max((getattr(it, "max_len", 0) or 0) for it in items) or self.max_len
        q = self.question_tokens(items)
        left = n - q
        text = text or " "
        if len(text) <= left and len(text.encode()) <= left:
            return None
        total = len(self.raw.encode(text, add_special_tokens=False).ids)
        if total <= left:
            return None
        return {"input_tokens": total, "read_tokens": max(0, left), "question_tokens": q, "max_len": n}

    def encode(self, items, text):
        p = pass_prompt(items, self.markers)
        n = max((getattr(it, "max_len", 0) or 0) for it in items) or self.max_len
        try:
            enc = self.tok_at(n).encode(p, text or " ")
        except Exception:  # noqa: BLE001  (the questions alone are longer than max_len)
            k, opts = self.question_tokens(items), sum(len(it.options) for it in items)
            raise ValueError(f"task and options do not fit in {n} tokens: the question ({opts} option(s) with their "
                             f"descriptions) takes {k} tokens and the checkpoint reads {n} per pass, so nothing is left "
                             "for the input — fewer options (choose among a shortlist first), shorter descriptions, or a "
                             "larger max_len") from None
        seq = enc.sequence_ids
        groups = self._groups(enc.ids, lambda i: seq[i] == 0, items)
        if any(it.pointer for it in items):           # the input's tokens and their character offsets (for the pointer)
            pos = [i for i, x in enumerate(seq) if x == 1]
            return enc.ids, groups, None, None, (pos, [tuple(enc.offsets[i]) for i in pos])
        return enc.ids, groups

    def encode_block(self, items, text, max_len):
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("the tokenizer has no [CLS] / [SEP] token: no block layout")
        blocks = [self.raw.encode(prompt(it.task, it.options, it.descriptions, mode=it.mode, markers=self.markers),
                                  add_special_tokens=False).ids + [self.sep_id] for it in items]
        budget = max_len - sum(len(b) for b in blocks) - 2
        if budget < 8:
            raise ValueError(f"{len(items)} questions do not fit in {max_len} tokens together")
        st = self.raw.encode(text or " ", add_special_tokens=False).ids[:budget]
        ids = [self.cls_id] + st + [self.sep_id]
        Ls = len(ids)
        pids, blk = list(range(Ls)), [-1] * Ls
        for j, b in enumerate(blocks):
            ids += b
            pids += list(range(Ls, Ls + len(b)))
            blk += [j] * len(b)
        groups = self._groups(ids, lambda i: blk[i] >= 0, items)
        return ids, groups, pids, blk


def block_masks(blk, pids, window):
    """blk [B, L] (−1 input, j ≥ 0 block j, −2 padding), pids [B, L] → (full, sliding) boolean masks [B, 1, L, L] (True =
    may attend): the input sees only the input, block j sees the input and itself, blocks do not see each other; the sliding
    (local) layers also need |pos_i − pos_j| ≤ window. Padding rows see themselves only."""
    qb, kb = blk[:, :, None], blk[:, None, :]
    see = ((kb == -1) & (qb != -2)) | ((qb == kb) & (qb >= 0))
    eye = np.eye(blk.shape[1], dtype=bool)[None]
    see = see | (eye & (qb == -2))
    near = np.abs(pids[:, :, None] - pids[:, None, :]) <= window
    return see[:, None], (see & (near | eye))[:, None]


class BlockUnsupported(ValueError):
    """The scorer cannot run the block layout at all (e.g. an ONNX export without its inputs): one question per sequence."""


def _batches(encs, bs):
    order = sorted(range(len(encs)), key=lambda i: len(encs[i][0]))
    for b in range(0, len(order), bs):
        yield order[b:b + bs]


class _NetScorer:
    """Shared by the ONNX and torch backends: encode, batch by length, read the logits at the markers (and the act logit at
    each question's mode marker)."""

    def _setup(self, path, max_len, bs, caps):
        caps = caps or capabilities({})
        self.enc = _Encoder(path, max_len, caps["markers"])
        self.bs = bs
        self.act_col = caps["act"]["column"] if caps["act"] is not None else None
        self.mq = caps["multi_question"]
        unk, ptr = caps.get("unknown"), caps.get("pointer")
        self.unk_col = unk["column"] if unk else None
        self.ptr_cols = (ptr["start"], ptr["end"]) if ptr else None

    def logits(self, items):
        return [o[0] for o in self._run([self.enc.encode((it,), it.text) for it in items])]

    def logits_pass(self, passes):
        if self.mq is not None and self.mq["layout"] == "block":
            return self._run([self.enc.encode_block(p.items, p.text, self.mq["max_len"]) for p in passes], block=True)
        return self._run([self.enc.encode(p.items, p.text) for p in passes])

    def pack(self, encs, block=False):
        """Encoded sequences → padded arrays (ids, attention mask, position ids, block masks); the last two None outside
        the block layout. Shared by scoring and adapter training (solvi.lora)."""
        L = max(len(e[0]) for e in encs)
        ids = np.full((len(encs), L), self.enc.pad_id, dtype=np.int64)
        att = np.zeros((len(encs), L), dtype=np.int64)
        pids = np.zeros((len(encs), L), dtype=np.int64) if block else None
        blk = np.full((len(encs), L), -2, dtype=np.int64) if block else None
        for r, e in enumerate(encs):
            n = len(e[0])
            ids[r, :n], att[r, :n] = e[0], 1
            if block:
                pids[r, :n], blk[r, :n] = e[2], e[3]
        return ids, att, pids, (block_masks(blk, pids, self.mq["window"]) if block else None)

    def _run(self, encs, block=False):
        out = [None] * len(encs)
        for ch in _batches(encs, self.bs):
            ids, att, pids, masks = self.pack([encs[i] for i in ch], block)
            lg = self._forward(ids, att, pids, masks) if block else self._forward(ids, att)
            act = self.act_col is not None and self.act_col < lg.shape[-1]
            unk = self.unk_col is not None and self.unk_col < lg.shape[-1]
            ptr = self.ptr_cols is not None and max(self.ptr_cols) < lg.shape[-1]
            for r, i in enumerate(ch):
                row = []
                for mpos, opos in encs[i][1]:
                    if not (act or unk):
                        row.append(lg[r, opos])
                        continue
                    o = {"logits": lg[r, opos]}
                    if act:
                        o["act"] = float(lg[r, mpos, self.act_col])
                    if unk:
                        o["unknown"] = float(lg[r, mpos, self.unk_col])
                    if ptr and len(encs[i]) > 4:          # full layout with the input's tokens: the pointer (l14g)
                        pos, offs = encs[i][4]
                        a, b = self.ptr_cols
                        o["pointer"] = {"start": lg[r, pos, a], "end": lg[r, pos, b],
                                        "null": [float(lg[r, mpos, a]), float(lg[r, mpos, b])], "offsets": offs}
                    row.append(o)
                out[i] = row
        return out


def need(module, extra, what):
    """Import an optional dependency → the module; ImportError naming the extra to install when it is missing (a
    dependency that fails inside it is raised as it is)."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:
        if e.name is not None and e.name.split(".")[0] != module.split(".")[0]:
            raise
        raise ImportError(f'{what} needs {module}: pip install "solvi[{extra}]"') from None


class OnnxScorer(_NetScorer):
    """onnxruntime session over onnx/model_*.onnx (inputs input_ids, attention_mask; output logits [B, L, C]). The block
    layout needs an export with the inputs position_ids, full_attention_mask and sliding_attention_mask ([B, 1, L, L] bool);
    without them several questions fall back to one per pass."""

    def __init__(self, path, max_len=512, device=None, onnx_file=None, bs=16, caps=None):
        ort = need("onnxruntime", "onnx", 'backend="onnx"')
        self._setup(path, max_len, bs, caps)
        self.file = onnx_file or _onnx_file(path, block=self.mq is not None and self.mq["layout"] == "block")
        if self.file is None:
            raise FileNotFoundError(f"no ONNX model in {path}/onnx")
        providers = ["CPUExecutionProvider"]
        if device and str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        self.sess = ort.InferenceSession(self.file, providers=providers)
        self.names = [i.name for i in self.sess.get_inputs()]
        self.tag = "onnx:" + os.path.basename(self.file)

    def logits_pass(self, passes):
        if self.mq is not None and self.mq["layout"] == "block" and not {"position_ids", "full_attention_mask",
                                                                          "sliding_attention_mask"} <= set(self.names):
            raise BlockUnsupported("this ONNX export has no block-layout inputs")
        return super().logits_pass(passes)

    def _forward(self, ids, att, pids=None, masks=None):
        feed = {"input_ids": ids, "attention_mask": att}
        if masks is not None:
            feed.update(position_ids=pids, full_attention_mask=masks[0], sliding_attention_mask=masks[1])
        return self.sess.run(None, {k: v for k, v in feed.items() if k in self.names})[0].astype(np.float32)


class TorchScorer(_NetScorer):
    """The checkpoint's encoder (transformers AutoModel from config.json) + the option head, weights from model.safetensors.
    The block layout passes per-layer-type attention masks and position ids (sdpa attention)."""

    def __init__(self, path, max_len=512, device=None, bs=16, caps=None):
        torch = need("torch", "model", 'backend="torch"')
        need("transformers", "model", 'backend="torch"')
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._setup(path, max_len, bs, caps)
        sd = load_file(os.path.join(path, "model.safetensors"))
        cfg = AutoConfig.from_pretrained(path)
        if self.mq is not None and self.mq["layout"] == "block":
            cfg._attn_implementation = "sdpa"
            encoder = AutoModel.from_config(cfg, attn_implementation="sdpa")
        else:
            encoder = AutoModel.from_config(cfg)
        h = encoder.config.hidden_size
        n_out = sd["head.3.weight"].shape[0] if "head.3.weight" in sd else 2
        head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.LayerNorm(h), torch.nn.Linear(h, n_out))

        class Decider(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc, self.head = encoder, head

            def forward(self, input_ids, attention_mask, position_ids=None):
                return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask,
                                          position_ids=position_ids).last_hidden_state)

        self.model = Decider()
        self.model.load_state_dict({k: v.float() for k, v in sd.items()})
        self.model.eval().to(self.device)
        self.tag = "torch"

    def inputs(self, ids, att, pids=None, masks=None, device=None):
        """Packed arrays (see pack) → the network's inputs as tensors on `device` (default: the scorer's)."""
        torch = self.torch
        dev = device or self.device
        mask = torch.from_numpy(att).to(dev)
        pos = None
        if masks is not None:
            mask = {"full_attention": torch.from_numpy(masks[0]).to(dev), "sliding_attention": torch.from_numpy(masks[1]).to(dev)}
            pos = torch.from_numpy(pids).to(dev)
        return torch.from_numpy(ids).to(dev), mask, pos

    def _forward(self, ids, att, pids=None, masks=None):
        torch = self.torch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
            return self.model(*self.inputs(ids, att, pids, masks)).float().cpu().numpy()


def _onnx_file(path, block=False):
    """The ONNX export to load. block: the checkpoint scores in the block layout (several questions per pass) — then the
    export with the block layout's inputs (onnx/model_block*.onnx) when there is one; the plain export otherwise."""
    d = os.path.join(path, "onnx")
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".onnx"))
    plain = ("model_fp16.onnx", "model.onnx", "model_fp32.onnx")
    for pref in (("model_block_fp16.onnx", "model_block.onnx", "model_block_fp32.onnx") if block else ()) + plain:
        if pref in files:
            return os.path.join(d, pref)
    return os.path.join(d, files[0]) if files else None


def _file_fingerprint(paths, chunks=32, size=4096):
    """Names, sizes and evenly spaced 4 KB chunks of the checkpoint files: ~ms, and any retrained checkpoint differs."""
    h = hashlib.sha256()
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        n = os.path.getsize(p)
        h.update(f"{os.path.basename(p)}:{n}".encode())
        with open(p, "rb") as fh:
            if n <= chunks * size:
                h.update(fh.read())
            else:
                for k in range(chunks):
                    fh.seek(k * (n - size) // (chunks - 1))
                    h.update(fh.read(size))
    return h.hexdigest()[:16]


__all__ = ["block_masks", "BlockUnsupported", "LongInputWarning", "OnnxScorer", "TorchScorer"]
