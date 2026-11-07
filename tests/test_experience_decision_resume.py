"""Crash boundary regression: JSON payload and its commit terminator differ."""
import json

import pytest

from benchmarks.decision_workflows import _completed_trials


def row(seed):
    return json.dumps({"direction": "28", "seed": seed, "arm": "decision", "limit": 20}).encode()


@pytest.mark.parametrize("tail", [b'{"direction":', row(2), b'{"incomplete_utf8":"\xd0'])
def test_resume_preserves_committed_rows_and_replays_unterminated_tail(tmp_path, tail):
    raw = tmp_path / "raw.jsonl"
    completed = row(1) + b"\n"
    raw.write_bytes(completed + tail)
    assert _completed_trials(raw) == {("28", 1, "decision", 20)}
    assert raw.read_bytes() == completed
    # A resumed writer must produce two separate JSON documents, not concatenation.
    with raw.open("ab") as stream:
        stream.write(row(2) + b"\n")
    assert [json.loads(line)["seed"] for line in raw.read_bytes().splitlines()] == [1, 2]
    assert len(_completed_trials(raw)) == 2


def test_completed_corruption_is_never_discarded(tmp_path):
    raw = tmp_path / "raw.jsonl"
    data = row(1) + b"\nnot-json\n"
    raw.write_bytes(data)
    with pytest.raises(ValueError, match="corrupt completed"):
        _completed_trials(raw)
    assert raw.read_bytes() == data


def test_duplicate_completed_trial_is_not_resumed_as_success(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes((row(1) + b"\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        _completed_trials(raw)


def test_embedded_carriage_return_never_truncates_completed_corruption(tmp_path):
    raw = tmp_path / "raw.jsonl"
    data = row(1) + b"\nmalformed\rrest\n" + row(2) + b"\n"
    raw.write_bytes(data)
    with pytest.raises(ValueError, match="corrupt completed"):
        _completed_trials(raw)
    assert raw.read_bytes() == data


def test_carriage_return_whitespace_and_crlf_are_valid_completed_json(tmp_path):
    raw = tmp_path / "raw.jsonl"
    data = row(1) + b"\r\n" + row(2).replace(b'"seed":', b'\r"seed":') + b"\n"
    raw.write_bytes(data)
    assert len(_completed_trials(raw)) == 2
    assert raw.read_bytes() == data
