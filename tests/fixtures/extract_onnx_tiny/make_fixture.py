"""Writes this folder: a tiny LongSpanExtractor checkpoint for the ONNX backend (tests/test_extract_onnx.py).

A word-level tokenizer.json and onnx/model.onnx with the published export's interface (input_ids, attention_mask int64
[batch, length] -> logits float [batch, length, 2]): each token's start and end scores are looked up in a table —
"TOTAL" starts an answer, "EUR" ends it, [CLS] ("no answer") scores 2 on both — and padding is multiplied by 0.
Needs onnx and tokenizers: python tests/fixtures/extract_onnx_tiny/make_fixture.py"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "TOTAL", "EUR", "the", "total", "amount", "invoice", "due", "thanks",
         "15.50", "42"] + [f"w{i}" for i in range(200)]
vocab = {w: i for i, w in enumerate(words)}
tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.save(os.path.join(HERE, "tokenizer.json"))
json.dump({"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "unk_token": "[UNK]"},
          open(os.path.join(HERE, "tokenizer_config.json"), "w"), indent=1)
json.dump({"kind": "long", "max_len": 32, "stride": 8, "max_span": 16, "thr_default": 0.5, "thr": {"iban": 0.9},
           "base_model": "tiny"}, open(os.path.join(HERE, "solvi_extract.json"), "w"), indent=1)

table = np.zeros((len(words), 2), np.float32)
table[vocab["TOTAL"], 0] = 8.0
table[vocab["EUR"], 1] = 8.0
table[vocab["[CLS]"]] = 2.0
g = helper.make_graph(
    [helper.make_node("Gather", ["table", "input_ids"], ["scores"], axis=0),
     helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
     helper.make_node("Unsqueeze", ["maskf", "axis"], ["mask3"]),
     helper.make_node("Mul", ["scores", "mask3"], ["logits"])],
    "tiny_span",
    [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "length"]),
     helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "length"])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "length", 2])],
    [numpy_helper.from_array(table, "table"), numpy_helper.from_array(np.array([-1], np.int64), "axis")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
onnx.checker.check_model(m)
os.makedirs(os.path.join(HERE, "onnx"), exist_ok=True)
onnx.save(m, os.path.join(HERE, "onnx", "model.onnx"))
