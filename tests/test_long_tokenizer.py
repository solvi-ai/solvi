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
