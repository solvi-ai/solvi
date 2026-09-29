"""long="full": a checkpoint trained on long inputs (`max_len_long` in solvi_decide.json) reads a text that does not fit
max_len whole, up to max_len_long tokens, and retrieves within max_len_long beyond that. A tiny word-level ModernBERT
decider with a pointer, built in a temporary folder (no downloads): max_len 64, max_len_long 1024."""
import json
import shutil
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import solvi.decide as sd  # noqa: E402
from solvi import Catalog, Span, System  # noqa: E402
from solvi.decide import DecideModel, LongInputWarning  # noqa: E402

TEAMS = ["billing", "technical"]
BILLING = ["invoice", "refund", "charged", "payment", "card", "price", "bill", "money"]
TECH = ["crash", "error", "bug", "login", "freeze", "screen", "update", "slow"]
FILLER = ["the", "my", "is", "please", "help", "today", "again", "with", "it", "a"]
NEEDLE = "zebra"
META = {"format": "solvi_decide v2", "subformat": "l14g typed v2", "max_len": 64, "max_len_long": 1024,
        "modes": ["single", "multi", "score", "noul", "rank", "number", "span"]}


@pytest.fixture(scope="module", autouse=True)
def _threads():
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A tiny l14g decider: word-level tokenizer with the solvi markers, a 2-layer ModernBERT (positions up to 2048),
    the six-column head (options, multi, act, not stated, pointer start / end)."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoConfig, AutoModel
    d = tmp_path_factory.mktemp("tiny_long_decider")
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"[unused{i}]" for i in range(8)]
    words = sorted(set(BILLING + TECH + FILLER + TEAMS + [NEEDLE, "which", "team", "where", "?", ".", "yes", "no",
                                                          "true", "false"]))
    vocab = {t: i for i, t in enumerate(special + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(special)
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                       special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok.save(str(d / "tokenizer.json"))
    cfg = AutoConfig.for_model("modernbert", vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=2, pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2,
                               eos_token_id=3, global_attn_every_n_layers=1, local_attention=16,
                               max_position_embeddings=2048)
    cfg.save_pretrained(str(d))
    torch.manual_seed(0)
    enc = AutoModel.from_config(cfg)
    h = cfg.hidden_size
    head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.LayerNorm(h), torch.nn.Linear(h, 6))
    sdict = {**{f"enc.{k}": v.contiguous() for k, v in enc.state_dict().items()},
             **{f"head.{k}": v.contiguous() for k, v in head.state_dict().items()}}
    save_file(sdict, str(d / "model.safetensors"))
    json.dump(META, open(d / "solvi_decide.json", "w"))
    return str(d)


@pytest.fixture(scope="module")
def short_ckpt(ckpt, tmp_path_factory):
    """The same weights, but the checkpoint declares no long-input length (trained on max_len only)."""
    d = tmp_path_factory.mktemp("tiny_short_decider")
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        shutil.copy(f"{ckpt}/{f}", d / f)
    json.dump({k: v for k, v in META.items() if k != "max_len_long"}, open(d / "solvi_decide.json", "w"))
    return str(d)


def doc(n_sections, needle_at=None):
    """Sections of 25 filler words (+ a full stop); the needle in section `needle_at`, once in the whole text."""
    parts = []
    for i in range(n_sections):
        words = (FILLER * 3)[:25]
        if i == needle_at:
            words = words[:12] + [NEEDLE] + words[12:]
        parts.append(" ".join(words) + " .")
    return "\n\n".join(parts)


def pointing(m):
    """Wrap the network: the pointer points at the needle token (a large start and end score there, low elsewhere and
    at the null span), so a quote must come back as the needle's characters in the document; record each pass's
    length."""
    needle, lengths = m.scorer.enc.tok.token_to_id(NEEDLE), []
    fwd = m.scorer._forward

    def forward(ids, att, pids=None, masks=None):
        lengths.append(int(att.sum(axis=1).max()))
        lg = fwd(ids, att, pids, masks) if masks is not None else fwd(ids, att)
        hit = ids == needle
        for c in (4, 5):
            lg[..., c] = np.where(hit, 40.0, -20.0)
        return lg
    m.scorer._forward = forward
    return lengths


def load(path, **kw):
    return DecideModel.load(path, backend="torch", device="cpu", **kw)


# --------------------------------------------------------------------------------------------------- reading
def test_full_reads_the_whole_text(ckpt):
    m = load(ckpt)
    assert m.long_len == 1024 and m.long_declared and m.max_len == 64
    lengths = pointing(m)
    text = doc(10)                                           # ~260 tokens: over max_len, within max_len_long
    n = m.count_tokens(text)
    assert 64 < n < 1024
    cut = m.decision("team", "Which team?", "email", TEAMS)
    d0 = cut(email=text)
    assert lengths[-1] == 64 and "long" not in d0.extra      # the default: truncated at max_len
    part = m.decision("team", "Which team?", "email", TEAMS, long="full")
    d = part(email=text)
    assert lengths[-1] > n and lengths[-1] <= 1024           # the whole text, in one pass
    assert d.extra["long"] == {"mode": "full", "tokens": n, "max_len": 1024}
    assert part.budget() > 900 and part.budget() < 1024
    short = "my invoice is wrong"
    assert "long" not in part(email=short).extra and lengths[-1] < 64     # a text that fits: as before, nothing recorded
    assert sum(d.probs.values()) == pytest.approx(1.0)


def test_past_max_len_long_it_retrieves_within_it(ckpt):
    m = load(ckpt)
    lengths = pointing(m)
    text = doc(60, needle_at=47)                             # ~1600 tokens: over max_len_long
    n = m.count_tokens(text)
    assert n > 1024
    part = m.decision("where", "Where is the zebra?", "email", Span[str], long="full")
    assert part.sections_k() == round(part.budget() / 170)    # top_k=None: sections of ~170 tokens
    d = part(email=text)
    lg = d.extra["long"]
    assert lg["mode"] == "full" and lg["fallback"] == "retrieve" and lg["tokens"] == n and lg["max_len"] == 1024
    assert lg["read"] == part.sections_k() and lg["of"] > lg["read"]
    assert 64 < lengths[-1] <= 1024                          # the window, read in one long pass
    assert any(a <= text.index(NEEDLE) < b for a, b, _, _ in lg["sections"])
    assert d.value.value == NEEDLE and text[d.value.start:d.value.end] == NEEDLE
    assert d.value.start == text.index(NEEDLE)


def test_quote_offsets_point_at_the_text_in_full_mode(ckpt):
    m = load(ckpt)
    pointing(m)
    text = doc(10, needle_at=8)                              # the needle ~220 tokens in: beyond what 64 tokens reach
    at = text.index(NEEDLE)
    assert m.count_tokens(text[:at]) > 64
    span = m.decision("where", "Where is the zebra?", "email", Span[str], long="full")(email=text)
    assert span.value.value == NEEDLE and (span.value.start, span.value.end) == (at, at + len(NEEDLE))
    ev = m.decision("paid", "Is it paid?", "email", bool, long="full", evidence=True)(email=text)
    assert ev.evidence and all(text[e.start:e.end] == e.value for e in ev.evidence)
    assert ev.evidence[0].start == at and ev.evidence[0].source == "email"
    cut = m.decision("where", "Where is the zebra?", "email", Span[str])(email=text)
    assert cut.value is None or cut.value.start != at        # truncated at 64 tokens, the needle is not read


def test_a_checkpoint_without_max_len_long_refuses_full(short_ckpt):
    m = load(short_ckpt)
    assert m.long_len is None and not m.long_declared
    with pytest.raises(ValueError, match='long="retrieve"'):
        m.decision("team", "Which team?", "email", TEAMS, long="full")
    m.decision("team", "Which team?", "email", TEAMS, long="retrieve")                   # retrieve still works
    with pytest.raises(ValueError, match="long"):
        m.decision("team", "Which team?", "email", TEAMS, long="whole")


def test_forcing_full_on_an_untrained_checkpoint_warns_once(short_ckpt, ckpt):
    m = load(short_ckpt, max_len_long=512)
    with pytest.warns(LongInputWarning, match="not trained on inputs that long"):
        part = m.decision("team", "Which team?", "email", TEAMS, long="full")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongInputWarning)
        m.decision("paid", "Is it paid?", "email", bool, long="full")                   # once per model
    assert part(email=doc(10)).extra["long"]["max_len"] == 512
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongInputWarning)
        load(ckpt).decision("team", "Which team?", "email", TEAMS, long="full")         # declared: no warning
        load(ckpt, max_len_long=512).decision("team", "Which team?", "email", TEAMS, long="full")   # less: no warning
    with pytest.warns(LongInputWarning):
        load(ckpt, max_len_long=2048).decision("team", "Which team?", "email", TEAMS, long="full")  # beyond training
    with pytest.raises(ValueError, match="larger than max_len"):
        load(short_ckpt, max_len_long=64)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        load(short_ckpt, max_len_long=4096)


def test_cpu_cost_warning_once_per_model(ckpt, monkeypatch):
    m = load(ckpt)
    assert m.on_cpu() is True
    part = m.decision("team", "Which team?", "email", TEAMS, long="full")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongInputWarning)
        part(email=doc(10))                                  # ~260 tokens: under the 2k-token line, no warning
    monkeypatch.setattr(sd, "LONG_CPU_TOKENS", 100)
    with pytest.warns(LongInputWarning, match="31x"):
        part(email=doc(11))
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongInputWarning)
        part(email=doc(12))
        m.decision("paid", "Is it paid?", "email", bool, long="full")(email=doc(13))


# --------------------------------------------------------------------------------------------------- identity
def test_fingerprint_has_the_mode_and_the_length_and_replay_rechecks(ckpt):
    m = load(ckpt)
    q = dict(name="team", task="Which team?", text_fact="email", options=TEAMS)
    full, ret, cut = (m.decision(**q, long="full"), m.decision(**q, long="retrieve"), m.decision(**q))
    assert len({full.fingerprint(), ret.fingerprint(), cut.fingerprint()}) == 3
    assert full.long_key() == ("full", 1024, full.sections_k(), False)
    m512 = load(ckpt, max_len_long=512)
    assert m512.decision(**q, long="full").fingerprint() != full.fingerprint()
    assert m512.weights_fingerprint() != m.weights_fingerprint()
    # top_k=None resolves to 3 at max_len 64: the same fingerprint as an explicit top_k=3 (retrieve hashes as before)
    assert ret.sections_k() == 3 and ret.fingerprint() == m.decision(**q, long="retrieve", top_k=3).fingerprint()

    cat = Catalog()
    qq = m.decision("team", "Which team?", "email", TEAMS, long="full").question(cat)
    s = System(cat, [qq])
    text = doc(10)
    res = s.ask({"email": text})
    rec = next(x for x in res.trace.records if x.name == "answer:team")
    assert rec.extra["long"]["mode"] == "full"
    assert "read whole" in str(res.audit("team"))
    rep = res.trace.replay(s)
    assert rep["ok"], rep["mismatches"]

    cat2 = Catalog()                                         # the same question read at another length
    s2 = System(cat2, [m512.decision("team", "Which team?", "email", TEAMS, long="full").question(cat2)])
    assert not res.trace.replay(s2)["ok"]

    import dataclasses

    from solvi.runtime import vhash
    forged = dataclasses.replace(rec, extra={**rec.extra, "long": {**rec.extra["long"], "tokens": 64}})
    forged.hash = vhash(forged.body())                       # a consistent record that claims less was read
    res.trace.records[res.trace.records.index(rec)] = forged
    assert any("long" in why for *_, why in res.trace.replay(s)["mismatches"])


def test_calibration_reads_what_a_decision_reads(ckpt):
    m = load(ckpt)
    part = m.decision("team", "Which team?", "email", TEAMS, long="full")
    texts = [doc(10), doc(60, needle_at=3), "my invoice"]
    for t in texts:
        z, _ = part._read([t])[0]
        assert np.allclose(z, part.model._raw_full([(part.spec, part.long_input(t) if part._too_long(t) else t)],
                                                   read_len=1024 if part._too_long(t) else 0)[0][0])
    assert part.long_input(texts[0]) == texts[0] and part.long_input(texts[1]) != texts[1]
    part.adapt(texts)                                        # adapt on long inputs: the logits a decision sees
    assert part.adaptation is not None


def test_lora_refuses_full(ckpt):
    pytest.importorskip("peft")
    from solvi.lora import check
    part = load(ckpt).decision("team", "Which team?", "email", TEAMS, long="full")
    with pytest.raises(ValueError, match='long="retrieve"'):
        check(part)


def test_retrieve_top_k_scales_with_max_len(ckpt):
    q = dict(name="team", task="Which team?", text_fact="email", options=TEAMS, long="retrieve")
    small, big = load(ckpt).decision(**q), load(ckpt, max_len=512).decision(**q)
    assert small.sections_k() == 3 and big.sections_k() == round(big.budget() / 170) == 3
    wide = load(ckpt, max_len=1024).decision(**q)
    assert wide.sections_k() == round(wide.budget() / 170) == 6
    assert load(ckpt, max_len=1024).decision(**q, top_k=2).sections_k() == 2          # an explicit top_k wins


# --------------------------------------------------------------------------------------------------- ONNX
def test_onnx_backend_reads_the_whole_text_like_torch(ckpt, tmp_path):
    """The ONNX export has a dynamic sequence length, so the same encoder reads max_len_long tokens there too."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import torch
    d = tmp_path / "onnx_ckpt"
    shutil.copytree(ckpt, d)
    tm = load(ckpt)
    net = tm.scorer.model.eval()

    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, input_ids, attention_mask):
            return self.net(input_ids, attention_mask)

    (d / "onnx").mkdir()
    ids = torch.ones((1, 80), dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(Plain(), (ids, ids), str(d / "onnx" / "model.onnx"), input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "length"}, "attention_mask": {0: "batch", 1: "length"},
                                        "logits": {0: "batch", 1: "length"}})
    om = DecideModel.load(str(d), backend="onnx")
    assert om.backend.startswith("onnx") and om.long_len == 1024 and om.on_cpu() is True
    text = doc(12)
    q = dict(name="team", task="Which team?", text_fact="email", options=TEAMS, long="full")
    a, b = tm.decision(**q)(email=text), om.decision(**q)(email=text)
    assert a.extra["long"] == b.extra["long"] == {"mode": "full", "tokens": tm.count_tokens(text), "max_len": 1024}
    for k in TEAMS:
        assert a.probs[k] == pytest.approx(b.probs[k], abs=1e-3)
