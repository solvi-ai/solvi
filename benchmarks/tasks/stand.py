"""Reruns the stand from cached model answers and checks every published number against one file, results.json.

    python benchmarks/tasks/stand.py run --cache DIR [--tasks cuad,bird] [--fresh]   # every script, offline -> runs/stand.json
    python benchmarks/tasks/stand.py check [--measured runs/stand.json] [--require taubench,cuad]
    python benchmarks/tasks/stand.py measure                                         # runs/stand.json again from the outputs

`run` starts `common/proxy.py --offline` on a free port and runs every baseline, every solution and the scripts behind
the further numbers of the README, all offline: a request that is not in the cache is an error, never sent, so no key
and no money are needed. What each script printed is kept in `<task>/runs/<step>.out`, and its numbers go to
`runs/stand.json`: the scorer's output and the few numbers a script prints in a line of its own, flattened to keys
such as `cuad/solution:accuracy`. A script that fails, or a request the cache does not hold, fails the run.

`check` compares results.json, the one place the stand's published numbers are kept, with two things:
  1. the run (`--measured`): every number of results.json must be what the run printed (timings are shown, not compared);
  2. the docs: every passage listed under "quoted", with its numbers filled in from results.json, must be in its file
     as written (whitespace and line breaks aside). A number changed in a doc and not in results.json, or the other way
     round, fails here. Without `--measured` only this part runs: no data, no run, a second.
Exit status 1 on any difference.

A task whose data is not here is skipped with the reason, and its numbers are not compared; so is a step whose model
replies are not in the cache (Abt-Buy's: see replies.py). `--require` names the tasks that must not be skipped.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common.llm import DATA  # noqa: E402

RESULTS = HERE / "results.json"
MEASURED = HERE / "runs" / "stand.json"

# (task, step, script and arguments). In a task's order: abtbuy's repair reads the baseline's answers. `{llm}` is the
# offline proxy's endpoint for the task, `{runs}` the task's runs/ folder.
STEPS = [
    ("taubench", "baseline", ["baseline.py"]),
    ("taubench", "solution", ["solution.py"]),
    ("cuad", "baseline", ["baseline.py"]),
    ("cuad", "solution", ["solution.py", "--llm", "{llm}"]),
    ("ragtruth", "baseline", ["baseline.py"]),
    ("ragtruth", "solution", ["solution.py", "--llm", "{llm}"]),
    ("ragtruth", "reply_format_dev", ["reply_format.py", "--llm", "{llm}", "--split", "dev"]),
    ("ragtruth", "reply_format_eval", ["reply_format.py", "--llm", "{llm}", "--split", "eval"]),
    ("banking77", "baseline", ["baseline.py"]),
    ("banking77", "solution", ["solution.py"]),
    ("banking77", "solution_plain", ["solution.py", "--plain"]),
    ("credit", "baseline", ["baseline.py"]),
    ("credit", "solution", ["solution.py"]),
    ("bird", "baseline", ["baseline.py"]),
    ("bird", "solution", ["solution.py", "--llm", "{llm}"]),
    ("abtbuy", "solution", ["solution.py"]),
    ("abtbuy", "baseline", ["baseline.py"]),
    ("abtbuy", "repair", ["solution.py", "--repair", "{runs}/baseline.jsonl", "--out", "{runs}/solution_repair.jsonl"]),
    ("abtbuy", "llm_pair", ["llm_pair.py", "--llm", "{llm}"]),
    ("abtbuy", "llm_pair_schema", ["llm_pair.py", "--llm", "{llm}", "--response-format", "json_schema",
                                   "--out", "{runs}/llm_pair_schema.jsonl"]),
    ("naturalplan", "baseline", ["baseline.py"]),
    ("naturalplan", "solution", ["solution.py"]),
    ("nab", "baseline", ["baseline.py"]),
    ("nab", "solution", ["solution.py"]),
]
TASKS = list(dict.fromkeys(t for t, *_ in STEPS))

# the scripts that call a model themselves (common/llm.py's options): they get --offline and the cache too
OFFLINE = {"taubench/baseline.py", "taubench/solution.py", "cuad/baseline.py", "cuad/solution.py", "ragtruth/baseline.py",
           "ragtruth/solution.py", "ragtruth/reply_format.py", "bird/baseline.py", "abtbuy/baseline.py",
           "naturalplan/baseline.py", "naturalplan/solution.py"}

# steps whose model replies come in a pack of their own (replies.py): pack name, and why it may be absent
PACKS = {name: "abtbuy" for name in ("abtbuy/baseline", "abtbuy/repair", "abtbuy/llm_pair", "abtbuy/llm_pair_schema")}
PACK_WHY = {"abtbuy": "the model's replies quote Abt-Buy's offers, and Abt-Buy states no licence: they are not "
                      "published (replies.py)"}

# what a task needs from $STAND_DATA beyond its prepared files (downloaded by fetch.sh)
NEEDS = {"taubench": ["taubench/repo"], "naturalplan": ["naturalplan/repo"], "nab": ["nab/repo/data"],
         "bird": ["bird/minidev"], "abtbuy": ["abtbuy/prepared/pairs_eval_test.jsonl"]}

# numbers a script prints in a line of its own, not in its final JSON: step -> [(key, regex)], group 1 is the number
LINES = {
    "cuad/solution": [("dev_auroc_llm_confidence", r"the LLM's confidence as the signal:.*AUROC ([\d.]+)"),
                      ("dev_answered_llm_confidence", r"the LLM's confidence as the signal: \{[^}]*'answered': ([\d.]+)"),
                      ("dev_auroc_trust", r"AUROC of trust ([\d.]+)")],
    "abtbuy/solution": [("dev_pairs_fitted", r"fitted on (\d+) dev-train pairs")],
    "abtbuy/llm_pair": [("cut_off", r"escalated:.*the reply was cut off \(max_token[^:]*: (\d+)")],
    "nab/solution": [("ms_per_decision", r'"ms_per_decision": ([\d.]+)')],
}

# numbers the docs quote that are sums or shares of what a script prints: step -> {key: f(numbers of the step)}
DERIVED = {
    "cuad/baseline": {"answered_alone": lambda g: 1 - g["escalated"] / g["n"]},
    "cuad/solution": {"answered_alone": lambda g: 1 - g["escalated"] / g["n"]},
    "ragtruth/baseline": {"wrong": lambda g: 1 - g["all.accuracy"]},
    "ragtruth/solution": {"wrong_among_answered_alone": lambda g: 1 - g["answered_alone.all.accuracy"],
                          "answered_alone_and_wrong": lambda g: g["answered_alone.all.share"] * (1 - g["answered_alone.all.accuracy"])},
    "banking77/solution": {"flagged_after_shift": lambda g: g["drift_flag_at"] - g["before.n"]},
    "credit/baseline": {"equal_to_policy": lambda g: g["1 correctness.v1.equal_to_reference"] + g["1 correctness.v2.equal_to_reference"],
                        "decisions": lambda g: g["1 correctness.v1.equal_to_reference/of"] + g["1 correctness.v2.equal_to_reference/of"]},
    "credit/solution": {"equal_to_policy": lambda g: g["1 correctness, 8 cost.v1.equal_to_reference"]
                        + g["1 correctness, 8 cost.v2.equal_to_reference"],
                        "decisions": lambda g: g["1 correctness, 8 cost.v1.equal_to_reference/of"]
                        + g["1 correctness, 8 cost.v2.equal_to_reference/of"]},
}
# machine-dependent: measured and shown, never compared (a slower runner is not a wrong number)
TIMINGS = {"nab/solution:ms_per_decision"}


def missing_data(task):
    """Why a task cannot run here (its data is not in $STAND_DATA), or None."""
    gone = [w for w in [f"{task}/prepared"] + NEEDS.get(task, []) if not (DATA / w).exists()]
    if not gone:
        return None
    how = "fetch.sh abtbuy; prepare.py abtbuy — Abt-Buy states no licence, so it is downloaded, never redistributed" \
        if task == "abtbuy" else "replies.py unpack-data and fetch.sh, or fetch.sh and prepare.py"
    return f"not in {DATA}: {', '.join(gone)} (run {how})"


def missing_pack(name, cache):
    """Why a step's replies are not in the cache, or None. A cache filled by replies.py unpack lists its packs."""
    pack, listed = PACKS.get(name), cache / "packs.json"
    if pack is None or not listed.exists() or pack in json.loads(listed.read_text()):
        return None
    return f"{PACK_WHY[pack]}; not in {cache}"


def last_json(text):
    """The last JSON object a script printed from the start of a line (the scorer's output), or {}."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except json.JSONDecodeError:
                continue
    return {}


def flatten(obj, pre=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[pre] = obj
    elif isinstance(obj, str) and re.fullmatch(r"\d+ of \d+", obj):          # "243 of 304": the count and its total
        n, of = obj.split(" of ")
        out[pre], out[pre + "/of"] = int(n), int(of)
    return out


def measure(name, text):
    """A step's name and what it printed -> {key: number}."""
    got = flatten(last_json(text))
    for key, rx in LINES.get(name, []):
        m = re.search(rx, text)
        if m:
            got[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    for key, f in DERIVED.get(name, {}).items():
        try:
            got[key] = round(f(got), 6)
        except KeyError:                     # a number it is made of is missing: the comparison reports the gap
            pass
    return got


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(a):
    cache = Path(a.cache).resolve()
    tasks = a.tasks.split(",") if a.tasks else TASKS
    out = Path(a.out)
    measured = json.loads(out.read_text()) if out.exists() and a.tasks else {}
    skipped = measured.setdefault("skipped", {})
    for t in tasks:
        runs = HERE / t / "runs"
        if runs.exists() and any(runs.iterdir()):
            if not a.fresh:
                sys.exit(f"{runs} is not empty: the scripts resume from or reuse what is there (--fresh clears it)")
            shutil.rmtree(runs)
    (cache / "misses.jsonl").unlink(missing_ok=True)
    port = free_port()
    proxy = subprocess.Popen([sys.executable, str(HERE / "common/proxy.py"), "--offline", "--cache", str(cache),
                              "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failed, t_all = [], time.time()
    try:
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        for task, step, argv in STEPS:
            if task not in tasks:
                continue
            name = f"{task}/{step}"
            for k in (task, name):
                skipped.pop(k, None)
            why = missing_data(task) or missing_pack(name, cache)
            if why:
                print(f"SKIP {name}: {why}", flush=True)
                skipped[name] = why
                measured.pop(name, None)
                continue
            runs = HERE / task / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            args = [x.format(llm=f"http://127.0.0.1:{port}/{task}/v1", runs=runs) for x in argv]
            if f"{task}/{argv[0]}" in OFFLINE:
                args += ["--offline", "--cache", str(cache)]
            t0 = time.time()
            p = subprocess.run([sys.executable, str(HERE / task / args[0]), *args[1:]], cwd=ROOT, capture_output=True,
                               text=True)
            secs = round(time.time() - t0, 1)
            (runs / f"{step}.out").write_text(p.stdout + ("\n--- stderr\n" + p.stderr if p.stderr else ""))
            got = measure(name, p.stdout)
            measured[name] = {**got, "_seconds": secs}
            ok = p.returncode == 0 and got
            print(f"{name:28s} {secs:7.1f} s  {'ok' if ok else f'FAILED (exit {p.returncode})'}", flush=True)
            if not ok:
                failed.append(name)
                print("\n".join((p.stderr or p.stdout).splitlines()[-15:]), flush=True)
    finally:
        proxy.terminate()
        proxy.wait()
    misses = cache / "misses.jsonl"
    measured["_misses"] = len(misses.read_text().splitlines()) if misses.exists() else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measured, indent=1, sort_keys=True))
    print(f"{len(tasks)} tasks in {time.time() - t_all:.0f} s -> {out}")
    if measured["_misses"]:
        print(f"{measured['_misses']} requests to the proxy were not in the cache ({misses}): their decisions escalated")
    if failed or measured["_misses"]:
        sys.exit(1)


def remeasure(a):
    """The numbers of a run again from the outputs it kept (<task>/runs/<step>.out), e.g. after a change to LINES."""
    out = Path(a.out)
    measured = json.loads(out.read_text()) if out.exists() else {}
    for task, step, _ in STEPS:
        f = HERE / task / "runs" / f"{step}.out"
        if f.exists():
            secs = measured.get(f"{task}/{step}", {}).get("_seconds")
            text = f.read_text().split("\n--- stderr\n")[0]
            measured[f"{task}/{step}"] = {**measure(f"{task}/{step}", text), "_seconds": secs}
    out.write_text(json.dumps(measured, indent=1, sort_keys=True))
    print(f"-> {out}")


def number(v, spec):
    """format(v, spec), rounding half up as a person reads a printed number (0.625 -> "0.63", not "0.62")."""
    m = re.fullmatch(r"(,?)\.(\d+)([f%])", spec)
    if not m:
        return format(v, spec)
    d = Decimal(repr(v)) * (100 if m.group(3) == "%" else 1)
    d = d.quantize(Decimal(1).scaleb(-int(m.group(2))), rounding=ROUND_HALF_UP)
    return format(d, f"{m.group(1)}.{m.group(2)}f") + ("%" if m.group(3) == "%" else "")


def fill(template, values):
    """A passage with {step:key} or {step:key|format} placeholders -> the passage with the numbers of `values`."""
    def one(m):
        key, _, spec = m.group(1).partition("|")
        if key not in values:
            raise KeyError(f"{key} is quoted but not in results.json")
        return number(values[key], spec) if spec else str(values[key])
    return re.sub(r"\{([^{}]+)\}", one, template)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def check(a):
    res = json.loads(RESULTS.read_text())
    values, bad = res["numbers"], []
    required = {t for t in a.require.split(",") if t}
    if a.measured:
        got = json.loads(Path(a.measured).read_text())
        skipped = got.get("skipped", {})
        compared = 0
        for key, want in sorted(values.items()):
            step = key.partition(":")[0]
            task = step.split("/")[0]
            if step in skipped or task in skipped:
                continue
            have = got.get(step, {}).get(key.partition(":")[2])
            if key in TIMINGS:
                print(f"not compared, a timing: {key} is {want} in results.json, {have} here")
                continue
            compared += 1
            if have is None or abs(float(have) - float(want)) > 1e-9:
                bad.append(f"{key}: {want} in results.json, {have} in the run")
        for name, why in sorted(skipped.items()):
            n = sum(k.startswith(name + ("/" if "/" not in name else ":")) for k in values)
            print(f"not compared ({n} numbers), {name} was skipped: {why}")
            if name.split("/")[0] in required:
                bad.append(f"{name} was skipped, and --require names {name.split('/')[0]}")
        if got.get("_misses"):
            bad.append(f"the run met {got['_misses']} requests that were not in the cache")
        print(f"run: {compared} of {len(values)} numbers of results.json compared with {a.measured}")
    for q in res["quoted"]:
        text = norm((ROOT / q["file"]).read_text(encoding="utf-8"))
        try:
            want = norm(fill(q["text"], values))
        except KeyError as e:
            bad.append(f"{q['file']}: {e}")
            continue
        if want not in text:
            bad.append(f"{q['file']}: this passage, with the numbers of results.json, is not in the file:\n    {want}")
    print(f"docs: {len(res['quoted'])} passages in {len({q['file'] for q in res['quoted']})} files checked")
    for b in bad:
        print("DIFFERS", b)
    if bad:
        sys.exit(1)
    print("every published number matches results.json" + (" and the run" if a.measured else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="every script of the stand, offline, from the cache")
    r.add_argument("--cache", default=os.environ.get("STAND_CACHE", str(HERE / "cache")))
    r.add_argument("--tasks", default=None, help="comma-separated (default: all nine)")
    r.add_argument("--fresh", action="store_true", help="clear each task's runs/ first (the scripts resume from it)")
    r.add_argument("--out", default=str(MEASURED))
    m = sub.add_parser("measure", help="the numbers again from the outputs a run kept in <task>/runs/")
    m.add_argument("--out", default=str(MEASURED))
    c = sub.add_parser("check", help="results.json against the docs, and against a run with --measured")
    c.add_argument("--measured", default=None, help="a run's runs/stand.json (without it: the docs only)")
    c.add_argument("--require", default="", help="comma-separated tasks that must not have been skipped")
    a = p.parse_args()
    {"run": run, "measure": remeasure, "check": check}[a.cmd](a)
