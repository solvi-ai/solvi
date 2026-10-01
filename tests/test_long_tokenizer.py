"""Long texts with a real tokenizer: counting tokens of a text longer than the checkpoint's max_len must not hit the
encoder's truncation (a tiny word-level decider built in a temporary folder, no downloads)."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from solvi.decide import DecideModel  # noqa: E402

TEAMS = ["billing", "technical"]
BILLING = ["invoice", "refund", "charged", "payment", "card", "price", "bill", "money"]
TECH = ["crash", "error", "bug", "login", "freeze", "screen", "update", "slow"]
FILLER = ["the", "my", "is", "please", "help", "today", "again", "with", "it", "a"]


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



def _long_text(n_sections=12):
    parts = []
    for i in range(n_sections):
        words = (BILLING if i == 7 else FILLER) * 3
        parts.append(" ".join(words) + ".")
    return "\n\n".join(parts)


def test_count_tokens_of_a_text_longer_than_max_len(ckpt):
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    text = _long_text()
    n = m.count_tokens(text)
    assert n == len(text.replace(".", " .").split()) and n > 64


def test_retrieve_reads_a_long_text(ckpt):
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    part = m.decision("team", "Which team?", "email", TEAMS, long="retrieve", top_k=2)
    assert 0 < part.budget() < 64
    r = part.decide(_long_text())
    assert r.value in TEAMS + [None]


def test_an_input_read_cut_is_marked_in_the_decision_and_warned_once(ckpt):
    """Without long=, an input that does not fit the pass is cut by the tokenizer. That was silent: the model answered as
    sure as ever about a text whose end it had not read. The decision now says how much was read."""
    import warnings
    from solvi import Catalog, System
    from solvi.decide import LongInputWarning, pass_prompt
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    part = m.decision("team", "Which team?", "email", TEAMS)
    enc = m.scorer.enc
    q = enc.question_tokens((m._item(part.spec, "x"),))
    fits, over = " ".join(["help"] * (64 - q)), " ".join(["help"] * (64 - q + 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d_fit, d_over, d_long = part.decide(fits), part.decide(over), part.decide(_long_text())
    assert "truncated" not in d_fit.extra
    assert d_over.extra["truncated"] == {"input_tokens": 64 - q + 1, "read_tokens": 64 - q, "question_tokens": q, "max_len": 64}
    n = m.count_tokens(_long_text())
    assert d_long.extra["truncated"]["input_tokens"] == n and d_long.extra["truncated"]["read_tokens"] == 64 - q
    for text, cut in ((fits, False), (over, True)):          # the mark is what the encoder really did
        e = enc.tok_at(64).encode(pass_prompt((m._item(part.spec, text),), enc.markers), text)
        assert sum(1 for x in e.sequence_ids if x == 1) == 64 - q and (m.count_tokens(text) > 64 - q) is cut
    mine = [x for x in w if issubclass(x.category, LongInputWarning)]
    assert len(mine) == 1 and "the rest was cut" in str(mine[0].message)          # once per part, not per decision
    retr = m.decision("team_r", "Which team?", "email", TEAMS, long="retrieve", top_k=2).decide(_long_text())
    assert "truncated" not in retr.extra and "long" in retr.extra                 # read by its sections: nothing is cut
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plain = m.decide(_long_text(), "Which team?", TEAMS)                      # the model's own decide: marked as well
        m.decide("help please", "Which team?", TEAMS)
    assert plain.extra["truncated"]["input_tokens"] == n and len(w) == 1
    cat = Catalog()
    s = System(cat, [part.question(cat)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = s.ask({"email": over})
    assert f"read {64 - q} of {64 - q + 1} input tokens (the rest was cut)" in str(res.audit("team"))
    assert "остальное отрезано" in res.audit("team").render(lang="ru")
    assert res.trace.replay(s)["ok"]


def test_options_that_do_not_fit_say_how_many_tokens_they_take(ckpt):
    m = DecideModel.load(ckpt, backend="torch", device="cpu")
    with pytest.raises(ValueError) as e:
        m.decide("help", "Which team?", [f"billing {i}" for i in range(40)])
    msg = str(e.value)
    assert "do not fit in 64 tokens" in msg and "40 option(s)" in msg and "nothing is left for the input" in msg
    assert "Truncation error" not in msg
