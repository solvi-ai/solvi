"""The two files that let the stand rerun offline anywhere: the model's replies the published run read, and the prepared
splits of the sets whose licence allows passing them on.

    python benchmarks/tasks/replies.py pack --cache DIR --used USED [--leave-out USED2] [--name NAME]
    python benchmarks/tasks/replies.py unpack --cache DIR                # packed/replies*.jsonl.xz -> DIR/chat, DIR/raw
    python benchmarks/tasks/replies.py pack-data                         # $STAND_DATA/<task>/prepared -> packed/prepared.tar.xz
    python benchmarks/tasks/replies.py unpack-data                       # packed/prepared.tar.xz -> $STAND_DATA

`pack` takes only the replies a run read (the list `STAND_USED` makes, see common/llm.py), once each, keyed by the hash of
the request: the requests themselves are not in it. Each reply keeps what the scripts and solvi read — the text, the
reasoning, tool calls, token log-probabilities, the finish reason, the token counts — and drops the rest (the
provider's generation id, timing, the account's billing details). A reply that holds something like a key or an e-mail address stops the pack.

Abt-Buy's replies go into a pack of their own, `packed/replies_abtbuy.jsonl.xz`, which is not published: the model
quotes the offers it compares (names, model numbers), and Abt-Buy states no licence. Without it the stand runs Abt-Buy's
solution (no model in it) and skips the four steps that read a model (stand.py PACKS).

`pack-data` leaves out Abt-Buy (no licence is stated for it: it is downloaded at run time and never redistributed) and
writes a NOTICE with each set's source and licence into the archive.
"""
import argparse
import io
import json
import lzma
import os
import re
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common.llm import DATA  # noqa: E402

PACKED = HERE / "packed"
REPLIES = PACKED / "replies.jsonl.xz"                   # published: every task but Abt-Buy
PREPARED = PACKED / "prepared.tar.xz"

# the sets whose prepared splits may be passed on, with what their licence asks (attribution; BIRD: share alike)
LICENCES = {
    "taubench": ("τ-bench, Sierra Research, https://github.com/sierra-research/tau-bench", "MIT"),
    "cuad": ("CUAD v1, The Atticus Project, https://github.com/TheAtticusProject/cuad", "CC BY 4.0"),
    "ragtruth": ("RAGTruth, ParticleMedia (processed copy wandb/RAGTruth-processed), "
                 "https://github.com/ParticleMedia/RAGTruth", "MIT"),
    "banking77": ("Banking77, PolyAI, https://github.com/PolyAI-LDN/task-specific-datasets", "CC BY 4.0"),
    "credit": ("Statlog (German Credit Data), H. Hofmann, UCI Machine Learning Repository, "
               "https://archive.ics.uci.edu/dataset/144", "CC BY 4.0"),
    "bird": ("BIRD mini-dev, https://github.com/bird-bench/mini_dev", "CC BY-SA 4.0"),
    "naturalplan": ("NATURAL PLAN, Google DeepMind, https://github.com/google-deepmind/natural-plan", "Apache-2.0"),
    "nab": ("Numenta Anomaly Benchmark, Numenta, https://github.com/numenta/NAB", "MIT"),
}
NOT_PASSED_ON = {"abtbuy": "Abt-Buy states no licence: downloaded at run time by fetch.sh, never redistributed"}

KEEP_MESSAGE = ("role", "content", "refusal", "reasoning", "reasoning_content", "tool_calls")
# a key, a bearer token, or an e-mail address: τ-bench's made-up customers live at example.com, and the simulated
# customer, asked for an address it does not have, sometimes invents one at email.com; any other address stops the pack
SECRET = re.compile(r"sk-or-v1-|sk-[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9]|[A-Za-z0-9._%+-]+@(?!(?:example|email)\.com\b)[A-Za-z0-9-]+\.(?:com|org|net|ru|io)\b")


def slim_raw(r):
    """A whole chat-completions response -> the parts solvi reads: the message, the token log-probabilities (its
    confidence when the server gives them), the finish reason, the model and the token counts. Not the generation id,
    the provider or the billing details."""
    out = {"model": r.get("model"), "choices": []}
    for ch in r.get("choices") or []:
        msg = {k: v for k, v in (ch.get("message") or {}).items() if k in KEEP_MESSAGE and v is not None}
        details = (ch.get("message") or {}).get("reasoning_details")
        if details and not msg.get("reasoning"):          # the only sign the model thought (solvi.core.deciders.llm looks for one)
            msg["reasoning_details"] = details
        c = {"index": ch.get("index", 0), "message": msg, "finish_reason": ch.get("finish_reason")}
        if ch.get("logprobs") is not None:
            c["logprobs"] = ch["logprobs"]
        out["choices"].append(c)
    u = r.get("usage")
    if isinstance(u, dict):
        keep = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "reasoning_tokens",
                "cost")
        out["usage"] = {k: u[k] for k in keep if k in u}
        det = u.get("completion_tokens_details")
        if isinstance(det, dict) and "reasoning_tokens" in det:
            out["usage"]["completion_tokens_details"] = {"reasoning_tokens": det["reasoning_tokens"]}
    return out


def slim_chat(r):
    """What common.llm.chat stored -> the same without the provider's name."""
    return {k: r.get(k) for k in ("content", "reasoning", "tool_calls", "finish", "model", "usage")}


def _pairs(path):
    return {tuple(x.split()[:2]) for x in Path(path).read_text().splitlines() if x.strip()}


def pack(a):
    cache = Path(a.cache)
    used = sorted(_pairs(a.used) - (_pairs(a.leave_out) if a.leave_out else set()))
    rows, flagged = [], []
    for kind, h in used:
        r = json.loads((cache / kind / h[:2] / f"{h}.json").read_text())
        r = slim_raw(r) if kind == "raw" else slim_chat(r)
        text = json.dumps(r, ensure_ascii=False, sort_keys=True)
        if SECRET.search(text):
            flagged.append((kind, h, SECRET.search(text).group(0)[:12]))
        rows.append(json.dumps({"kind": kind, "key": h, "reply": r}, ensure_ascii=False, sort_keys=True))
    if flagged:
        sys.exit(f"{len(flagged)} replies hold something like a key or an address, not packed: {flagged[:5]}")
    out = PACKED / (f"replies_{a.name}.jsonl.xz" if a.name else REPLIES.name)
    PACKED.mkdir(exist_ok=True)
    data = ("\n".join(rows) + "\n").encode()
    out.write_bytes(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
    kinds = {k: sum(1 for x, _ in used if x == k) for k in ("chat", "raw")}
    print(f"{len(rows)} replies ({kinds}), {len(data) / 1e6:.1f} MB -> {out} {out.stat().st_size / 1e6:.1f} MB")


def unpack(a):
    cache = Path(a.cache)
    packs = sorted(PACKED.glob("replies*.jsonl.xz")) if a.replies is None else [Path(a.replies)]
    names = []
    for f in packs:
        n = 0
        for line in lzma.decompress(f.read_bytes()).decode().splitlines():
            r = json.loads(line)
            p = cache / r["kind"] / r["key"][:2] / f"{r['key']}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r["reply"], ensure_ascii=False))
            n += 1
        names.append(f.name[len("replies_"):-len(".jsonl.xz")] if f.name.startswith("replies_") else "stand")
        print(f"{f.name}: {n} replies -> {cache}")
    listed = cache / "packs.json"                     # stand.py skips a step whose pack is not listed here
    old = json.loads(listed.read_text()) if listed.exists() else []
    listed.write_text(json.dumps(sorted(set(old) | set(names))))


def notice():
    lines = ["The prepared splits of the solvi task stand (benchmarks/tasks/prepare.py, seed 20261002), made from these",
             "sets and passed on under their licences. Each file is a selection and reformatting of the original; the",
             "files of a set are under that set's licence, not under solvi's. BIRD's prepared files are shared under",
             "CC BY-SA 4.0 (share alike).", ""]
    lines += [f"{t}/prepared/: {src}. Licence: {lic}." for t, (src, lic) in LICENCES.items()]
    lines += ["", "Not included: " + "; ".join(f"{t} ({why})" for t, why in NOT_PASSED_ON.items()), ""]
    return "\n".join(lines)


def pack_data(a):
    PACKED.mkdir(exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
        add("NOTICE", notice().encode())
        manifest = json.loads((DATA / "manifest.json").read_text())
        add("manifest.json", json.dumps({t: manifest[t] for t in LICENCES}, indent=1).encode())
        for t in LICENCES:
            for f in sorted((DATA / t / "prepared").rglob("*")):
                if f.is_file():
                    add(str(f.relative_to(DATA)), f.read_bytes())
    raw = buf.getvalue()
    PREPARED.write_bytes(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    print(f"{', '.join(LICENCES)}: {len(raw) / 1e6:.1f} MB -> {PREPARED} {PREPARED.stat().st_size / 1e6:.1f} MB")


def unpack_data(a):
    DATA.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(PREPARED.read_bytes())), mode="r") as tar:
        tar.extractall(DATA, filter="data")
    mf = DATA / "manifest.json"
    have = json.loads(mf.read_text()) if mf.exists() and mf.stat().st_size else {}
    print(f"prepared splits of {', '.join(sorted(have))} -> {DATA} (licences: {DATA / 'NOTICE'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pack", help="the replies a run read -> packed/replies.jsonl.xz")
    s.add_argument("--cache", required=True)
    s.add_argument("--used", required=True, help="the file STAND_USED pointed at during the run")
    s.add_argument("--leave-out", default=None, help="a second such file: its replies are not packed")
    s.add_argument("--name", default=None, help="packed/replies_NAME.jsonl.xz instead of packed/replies.jsonl.xz")
    s = sub.add_parser("unpack", help="packed/replies*.jsonl.xz -> a cache directory")
    s.add_argument("--cache", default=os.environ.get("STAND_CACHE", str(HERE / "cache")))
    s.add_argument("--replies", default=None, help="one pack (default: every packed/replies*.jsonl.xz there is)")
    sub.add_parser("pack-data", help="the prepared splits that may be passed on -> packed/prepared.tar.xz")
    sub.add_parser("unpack-data", help="packed/prepared.tar.xz -> $STAND_DATA")
    a = p.parse_args()
    {"pack": pack, "unpack": unpack, "pack-data": pack_data, "unpack-data": unpack_data}[a.cmd](a)
