"""solvi.lora.adapt_lora (experimental): a LoRA adapter per question on a tiny random ModernBERT decider built in a
temporary folder (no downloads). Skipped without torch, transformers, tokenizers and peft (`uv sync --group lora`)."""
import json
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from solvi import Catalog, System  # noqa: E402
from solvi.decide import DecideModel  # noqa: E402
from solvi.learning import ExperimentalWarning  # noqa: E402
from solvi.lora import LoraWarning, adapt_lora, remove_lora  # noqa: E402

TEAMS = ["billing", "technical"]
BILLING = ["invoice", "refund", "charged", "payment", "card", "price", "bill", "money"]
TECH = ["crash", "error", "bug", "login", "freeze", "screen", "update", "slow"]
FILLER = ["the", "my", "is", "please", "help", "today", "again", "with", "it", "a"]


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        team = TEAMS[i % 2]
        words = list(rng.choice(BILLING if team == "billing" else TECH, 2)) + list(rng.choice(FILLER, 4))
        rng.shuffle(words)
        out.append((" ".join(words), team))
    return out


@pytest.fixture(scope="module", autouse=True)
def _threads():
    """Two threads: a tiny network on many threads is slower (and a thread count is part of determinism)."""
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A tiny decider checkpoint: word-level tokenizer with the solvi markers, a 2-layer ModernBERT, the option head."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoConfig, AutoModel
    d = tmp_path_factory.mktemp("tiny_decider")
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"[unused{i}]" for i in range(8)]
    words = sorted(set(BILLING + TECH + FILLER + TEAMS + ["which", "team", "?", "yes", "no", "true", "false", "angry"]))
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
                               max_position_embeddings=256)
    cfg.save_pretrained(str(d))
    torch.manual_seed(0)
    enc = AutoModel.from_config(cfg)
    h = cfg.hidden_size
    head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.LayerNorm(h), torch.nn.Linear(h, 3))
    sd = {**{f"enc.{k}": v.contiguous() for k, v in enc.state_dict().items()},
          **{f"head.{k}": v.contiguous() for k, v in head.state_dict().items()}}
    save_file(sd, str(d / "model.safetensors"))
    json.dump({"format": "solvi_decide v2", "max_len": 64, "modes": ["single", "multi", "score", "noul"],
               "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "act": 2}, "act": None},
              open(d / "solvi_decide.json", "w"))
    return str(d)


def _part(ckpt, name="team"):
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    return m, m.decision(name, "Which team?", "email", TEAMS)


def _adapt(part, examples, **kw):
    kw.setdefault("lr", 1e-2)
    kw.setdefault("epochs", 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = adapt_lora(part, examples, **kw)
    return rep, w


def _acc(part, data):
    return float(np.mean([part.decide(t).value == y for t, y in data]))


def test_training_changes_predictions_fingerprint_trace_and_rollback(ckpt):
    m, part = _part(ckpt)
    train, test = _toy(120, 1), _toy(40, 2)
    z0 = np.array([m.logits(t, "Which team?", TEAMS)["billing"] for t, _ in test])
    fp0, mfp0, acc0 = part.fingerprint(), m.fingerprint(), _acc(part, test)
    part.act_guard(_toy(40, 5), max_risk=0.2)                      # a calibration that the adapter makes stale
    snap = (part.escalate_below, part.guarantee)
    fp_cal = part.fingerprint()
    rep, w = _adapt(part, train, holdout=0.25)
    assert rep["experimental"] and rep["k"] == 90 and rep["updates"] == 113 and rep["size_mb"] > 0
    assert rep["holdout"]["n"] == 30 and rep["holdout"]["act_guard"]["signal"] == "confidence"
    assert "escalate_below" in rep["cleared"] and part.guarantee["method"] == "crc"
    assert rep["estimate_seconds"] is not None
    assert any(issubclass(x.category, LoraWarning) and "updates on cpu" in str(x.message) for x in w)
    assert part.lora is not None and part.lora.hash == rep["adapter"]
    assert part.fingerprint() not in (fp0, fp_cal) and m.fingerprint() != mfp0
    z1 = np.array([m.logits(t, "Which team?", TEAMS)["billing"] for t, _ in test])
    assert np.max(np.abs(z1 - z0)) > 1e-3                      # the adapter answers this question
    assert _acc(part, test) > max(acc0, 0.8)
    assert rep["holdout"]["accuracy_after"] > rep["holdout"]["accuracy_before"]
    d = part.decide(test[0][0])
    assert d.extra["lora"] == {"adapter": rep["adapter"], "experimental": True}
    cat = Catalog()
    s = System(cat, [part.question(cat)])
    res = s.ask({"email": test[0][0]})
    rec = next(r for r in res.trace.records if r.name == "answer:team")
    assert rec.extra["lora"]["adapter"] == rep["adapter"] and res.trace.replay(cat)["ok"]
    assert any(x["adapter"] == rep["adapter"] for x in m.metadata()["loras"])
    # rollback: the checkpoint's logits to the bit, the fingerprint and the calibration as before
    assert remove_lora(part) == rep["adapter"] and part.lora is None and remove_lora(part) is None
    z2 = np.array([m.logits(t, "Which team?", TEAMS)["billing"] for t, _ in test])
    assert np.array_equal(z2, z0)
    assert part.fingerprint() == fp_cal and m.fingerprint() == mfp0
    assert (part.escalate_below, part.guarantee) == snap


def test_other_questions_are_not_affected(ckpt):
    m, part = _part(ckpt)
    other = m.decision("angry", "angry ?", "email", kind="noul")
    texts = [t for t, _ in _toy(12, 3)]
    before = [other.decide(t).probs["yes"] for t in texts]
    fp_other = other.fingerprint()
    _adapt(part, _toy(40, 1))
    assert [other.decide(t).probs["yes"] for t in texts] == before and other.fingerprint() == fp_other
    assert "lora" not in other.decide(texts[0]).extra
    ds = m.decide_pass(texts[0], [part, other])                  # a shared pass keeps them apart too
    assert "lora" in ds[0].extra and "lora" not in ds[1].extra
    remove_lora(part)


def test_small_k_warning_no_holdout_warning_and_experimental_marker(ckpt):
    import solvi.lora as L
    L._warned[0] = False
    m, part = _part(ckpt)
    rep, w = _adapt(part, _toy(24, 1))
    msgs = [(x.category, str(x.message)) for x in w]
    assert any(c is ExperimentalWarning for c, _ in msgs)
    assert any(c is LoraWarning and "below about 100" in t for c, t in msgs)
    assert any(c is LoraWarning and "not calibrated" in t for c, t in msgs)
    assert rep["holdout"] is None
    _, w2 = _adapt(part, _toy(24, 1))
    assert not any(x.category is ExperimentalWarning for x in w2)           # once per process
    remove_lora(part)


def test_deterministic_for_a_seed(ckpt):
    hs = []
    for seed in (0, 0, 1):
        m, part = _part(ckpt)
        hs.append(_adapt(part, _toy(40, 1), seed=seed)[0]["adapter"])
    assert hs[0] == hs[1] != hs[2]


def test_save_load_round_trip_and_calibration_file(ckpt, tmp_path):
    m, part = _part(ckpt)
    data, test = _toy(60, 1), _toy(20, 2)
    rep, _ = _adapt(part, data, holdout=20)
    probs = [part.decide(t).probs for t, _ in test]
    fp = part.fingerprint()
    f = part.save_lora(tmp_path / "team.lora.safetensors")
    part.save_calibration(tmp_path / "team.calib.json")
    rec = json.load(open(tmp_path / "team.calib.json"))
    assert rec["lora"] == {"hash": rep["adapter"], "file": "team.calib.lora.safetensors"}
    assert (tmp_path / "team.calib.lora.safetensors").exists()
    m2, p2 = _part(ckpt)                                           # a fresh process: the adapter from its file
    p2.load_lora(f)
    assert p2.lora.hash == rep["adapter"] and [p2.decide(t).probs for t, _ in test] == probs
    m3, p3 = _part(ckpt)                                           # the calibration loads its adapter first
    p3.load_calibration(tmp_path / "team.calib.json")
    assert p3.fingerprint() == fp and p3.guarantee == part.guarantee
    assert [p3.decide(t).probs for t, _ in test] == probs
    remove_lora(p3)                                               # no snapshot from before: the thresholds are cleared
    assert p3.guarantee is None and p3.escalate_below is None
    other = m2.decision("x", "Which team?", "email", ["billing", "technical", "shipping"])
    with pytest.raises(ValueError, match="another question"):
        other.load_lora(f)
    m4, p4 = _part(ckpt)
    with pytest.raises(ValueError, match="calibrated for fingerprint|another model"):
        _calib_without_lora(p4, tmp_path)                          # made with the adapter, loaded without it
    remove_lora(part)


def _calib_without_lora(part, tmp_path):
    rec = json.load(open(tmp_path / "team.calib.json"))
    rec.pop("lora")
    json.dump(rec, open(tmp_path / "nolora.calib.json", "w"))
    part.load_calibration(tmp_path / "nolora.calib.json")


def test_refusals(ckpt, monkeypatch):
    import solvi.lora as L
    from solvi.llm import LLMScorer
    m, part = _part(ckpt)
    with pytest.raises(ValueError, match="needs at least 8"):
        adapt_lora(part, _toy(4, 1))
    with pytest.raises(ValueError, match="holdout"):
        adapt_lora(part, _toy(40, 1), holdout=40)
    rank = m.decision("r", "Which team?", "email", TEAMS, kind="rank")
    with pytest.raises(ValueError, match="rank questions"):
        adapt_lora(rank, _toy(40, 1))
    monkeypatch.setattr(L, "MAX_HIDDEN", 16)                      # the tiny model plays solvi-large
    with pytest.raises(ValueError, match="adapt_lora_gpu.py"):
        adapt_lora(part, _toy(40, 1))
    monkeypatch.undo()

    class Fake:
        def logits(self, items):
            return [np.zeros(len(it.options)) for it in items]
    for scorer, why in ((Fake(), 'backend="torch"'), (object.__new__(LLMScorer), "LLM decider")):
        p = DecideModel(scorer).decision("team", "Which team?", "email", TEAMS)
        with pytest.raises(ValueError, match=why):
            adapt_lora(p, _toy(40, 1))
    onnx = type("OnnxScorer", (), {"logits": Fake.logits})
    with pytest.raises(ValueError, match="runs on ONNX"):
        adapt_lora(DecideModel(onnx()).decision("team", "Which team?", "email", TEAMS), _toy(40, 1))
    from solvi.lora import adapt
    with pytest.raises(TypeError, match="DecisionPart"):
        adapt(lambda email: "billing", _toy(40, 1))


def test_missing_peft_is_a_clear_error(ckpt, monkeypatch):
    import builtins
    real = builtins.__import__

    def no_peft(name, *a, **kw):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("no peft")
        return real(name, *a, **kw)
    m, part = _part(ckpt)
    monkeypatch.setattr(builtins, "__import__", no_peft)
    with pytest.raises(ImportError, match=r"solvi\[lora\]"):
        adapt_lora(part, _toy(40, 1))


@pytest.mark.filterwarnings("ignore::solvi.lora.LoraWarning")
def test_offline_script_trains_an_adapter_the_part_loads(ckpt, tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    import solvi.lora as L
    spec = importlib.util.spec_from_file_location("adapt_lora_gpu", Path(__file__).resolve().parents[1] / "tools" /
                                                  "adapt_lora_gpu.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rows = tmp_path / "labels.jsonl"
    rows.write_text("".join(json.dumps({"text": t, "label": y}) + "\n" for t, y in _toy(60, 1)))
    out = tmp_path / "team.lora.safetensors"
    monkeypatch.setattr(L, "MAX_HIDDEN", 16)                      # it trains what adapt_lora refuses in-process
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tool.main([ckpt, str(rows), "--task", "Which team?", "--options", *TEAMS, "--fact", "email",
                          "--holdout", "20", "--device", "cpu", "--lr", "1e-2", "--epochs", "10",
                          "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "40 examples" in printed and "held out 20" in printed
    m, part = _part(ckpt)
    part.load_lora(out)
    assert part.lora.hash in printed and part.decide(_toy(2, 9)[0][0]).extra["lora"]["adapter"] == part.lora.hash


def test_the_part_has_an_adapter_slot_not_the_training_calls(ckpt):
    """1.0: training and rolling back an adapter left the part (a stable class does not import an experimental module):
    part.adapt_lora / remove_lora raise an AttributeError naming the function; the adapter itself is a solvi Adapter."""
    m, part = _part(ckpt)
    with pytest.raises(AttributeError, match=r"adapt_lora\(\) was removed in 1.0: use solvi.lora.adapt_lora\(part"):
        part.adapt_lora(_toy(40, 1))
    with pytest.raises(AttributeError, match=r"use solvi.lora.remove_lora\(part\)"):
        part.remove_lora()
    rep, _ = _adapt(part, _toy(40, 1))
    ad = part.lora
    assert ad.kind == "lora" and ad.fingerprint() == ad.hash == rep["adapter"]
    assert remove_lora(part) == rep["adapter"] and part.lora is None
