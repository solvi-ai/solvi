"""Where a training runs (experimental; the adapter (LoRA) loop, stage 3): a runner calls a training function — a
target named "module:function" — with a payload and a context, and returns the dict the function returns.

    from solvi.experimental.stream.runner import SubprocessRunner, ThreadRunner

    out = SubprocessRunner().run("pkg.mod:train", payload, name="train_3.1", workdir=dir, budget=budget)

- `SubprocessRunner` (the default): a fresh Python process (`python -m solvi.experimental.stream.runner job.pkl`),
  `nice` 10, its torch / BLAS threads capped at `budget.cpu_threads`, the device picked by `pick_device` (budget
  `device="auto"`: cuda when a GPU has `free_gpu_mb` free, else the CPU; a training that runs out of GPU memory is the
  target's to retry on the CPU, as the LoRA target does). It writes `<name>.log` (a line per phase and a heartbeat every
  `heartbeat_s`, 30 s), the payload as `<name>.job.pkl` and its result as `<name>.result.json`; the process's own
  output goes to `<name>.err`. It isolates the thread count and a crash, and the slice's memory cap applies to it.
- `ThreadRunner`: the same target in a thread of this process (tests, tiny data): torch threads are process-global, so
  a training slows live scoring, and nothing is isolated; the device is picked the same way.

The context the target gets (`RunContext`): `device` (picked), `threads`, `budget`, `log(message)` (a line of the
log), `progress(done, total, **extra)` (what the heartbeat says), `estimate(seconds)` — the target calls it with its
estimate of the training's time before training; above `budget.max_train_s` it raises `TrainingRefused`, which the
runner returns as a refusal.

What a run can end with, as the caller sees it: the target's dict (done); `TrainingRefused` (the estimate is above
`max_train_s`); `TrainingFailed` (the target raised, or ran past its estimate + `grace_s`, counted from the estimate
— before the target estimates, `max_train_s` + `grace_s` from the start — and was stopped: a subprocess is killed, a
thread cannot be and is reported when it ends); `RunnerError` (the runner itself failed: the
process died — a signal, the memory cap, an exit without a result — or wrote no log line for `stall_s`). The adapter
loop records the first three in its "train" event (the served version keeps serving) and stops on a `RunnerError`
(status "error", replay exact).

While a run goes on, `status` (a dict the caller passes; the loop's `status()["job"]`) is updated in place: "runner",
"pid", "log", "device", "eta" (the wall clock the training should end at, once estimated), "last" (the log's last
line) and "last_t" (when it was written). `cancel=` (a threading.Event): once set, a subprocess is killed and
`TrainingCancelled` raised; a thread runner stops waiting and raises it (the thread ends on its own). Unmeasured."""
from __future__ import annotations

import importlib
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


class TrainingRefused(RuntimeError):
    """The training was not started: its estimated time is above the budget's max_train_s."""


class TrainingFailed(RuntimeError):
    """The training ran and failed (the target raised, or it ran past its time and was stopped)."""


class TrainingCancelled(RuntimeError):
    """The caller cancelled the training (its `cancel` event was set): a subprocess is killed; a thread cannot be
    stopped and is left to end on its own."""


class RunnerError(RuntimeError):
    """The runner itself failed: the training process died without a result, or went silent."""


# ───────────────────────────────────────────────────────────── the device
def gpu_free_mb():
    """Free memory of each GPU in MB, read with nvidia-smi (no CUDA context is created) → [mb, ...]; [] when there is
    no GPU or no nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],  # noqa: S603,S607
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    got = []
    for line in out.stdout.splitlines():
        try:
            got.append(int(float(line.strip())))
        except ValueError:
            continue
    return got


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a broken driver is "no GPU"
        return False


def _visible(env=None):
    """CUDA_VISIBLE_DEVICES as nvidia-smi indices in torch's ordinal order → None (not set: torch's ordinal i is
    nvidia-smi's GPU i), [] (no GPU visible) or [smi index, ...]; "uuid" when it names GPUs another way (UUIDs, MIG)."""
    raw = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    got = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            break
        try:
            i = int(tok)
        except ValueError:
            return "uuid"
        if i < 0:                                   # CUDA stops at the first invalid entry
            break
        got.append(i)
    return got


def _torch_free_mb():
    """Free memory of each GPU torch sees, in torch's ordinals (creates a CUDA context) → [mb, ...]."""
    import torch
    return [int(torch.cuda.mem_get_info(i)[0] // 2 ** 20) for i in range(torch.cuda.device_count())]


def pick_device(device="auto", free_gpu_mb=3000, *, probe=None, cuda=None, env=None, torch_probe=None) -> tuple:
    """The device a training runs on → (device, why). "cpu": the CPU; "cuda" / "cuda:<i>": that GPU (as asked);
    "auto": the first GPU with at least free_gpu_mb MB free when torch sees CUDA, else the CPU (the reason is said).
    The device is torch's ordinal: under CUDA_VISIBLE_DEVICES nvidia-smi's index is mapped to it (when that variable
    names GPUs by UUID, torch is asked for the free memory instead). probe: () → free MB per GPU by nvidia-smi's index
    (default gpu_free_mb); cuda: () → whether torch sees CUDA (default: torch's); env: the environment (default
    os.environ); torch_probe: () → free MB per torch ordinal (default: torch.cuda.mem_get_info)."""
    if device == "cpu" or (isinstance(device, str) and device.startswith("cuda")):
        return device, "asked"
    if device != "auto":
        raise ValueError(f"device is 'auto', 'cpu', 'cuda' or 'cuda:<i>', not {device!r}")
    if not (cuda or _cuda_available)():
        return "cpu", "auto: no CUDA device"
    vis = _visible(env)
    if vis == "uuid":
        free = (torch_probe or _torch_free_mb)()
        cands = list(enumerate(free))
    else:
        smi = (probe or gpu_free_mb)()
        order = range(len(smi)) if vis is None else [i for i in vis if i < len(smi)]
        cands = [(smi_i, smi[smi_i]) for smi_i in order]
        free = [mb for _, mb in cands]
    for ordinal, (idx, mb) in enumerate(cands):
        if mb >= free_gpu_mb:
            where = f"GPU {idx}" if vis in (None, "uuid") else f"GPU {idx} (nvidia-smi; torch's cuda:{ordinal})"
            return f"cuda:{ordinal}", f"auto: {where} has {mb} MB free (≥ {free_gpu_mb})"
    best = max(free) if free else None
    return "cpu", (f"auto: no GPU with {free_gpu_mb} MB free (most: {best} MB)" if best is not None
                   else "auto: GPU memory unknown (no nvidia-smi)" if vis is None or vis == "uuid" or smi == []
                   else "auto: no GPU visible (CUDA_VISIBLE_DEVICES)")


# ───────────────────────────────────────────────────────────── the context a target gets
class RunContext:
    """See the module docstring."""

    def __init__(self, budget, device, why, log_path, status=None, heartbeat_s=30.0):
        self.budget, self.device, self.device_why = budget, device, why
        self.threads = int(getattr(budget, "cpu_threads", 4))
        self.log_path = Path(log_path)
        self.status = status if status is not None else {}
        self.heartbeat_s = float(heartbeat_s)
        self.in_process = False                    # True in a ThreadRunner: process-wide settings are not the target's
        self.phase, self.done, self.total, self.extra = "start", 0, None, {}
        self.estimate_s = self.estimate_t = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beat = None

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        with self._lock, open(self.log_path, "a") as f:
            f.write(line + "\n")
        self.status["last"], self.status["last_t"] = line, time.time()

    def progress(self, done, total=None, **extra):
        self.done, self.total, self.extra = done, total, extra

    def set_phase(self, phase):
        self.phase = phase
        self.log(f"phase {phase}")

    def estimate(self, seconds):
        """The training's estimated time, before it trains: logged; above budget.max_train_s → TrainingRefused."""
        self.estimate_s, self.estimate_t = float(seconds), time.monotonic()
        self.status["eta"] = time.time() + self.estimate_s
        self.log(f"estimate {self.estimate_s:.1f} s")
        cap = float(getattr(self.budget, "max_train_s", float("inf")))
        if self.estimate_s > cap:
            raise TrainingRefused(f"estimated {self.estimate_s:.0f} s > max_train_s {cap:g}")

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_s):
            tot = "" if self.total is None else f"/{self.total}"
            ex = "".join(f" {k}={v}" for k, v in self.extra.items())
            self.log(f"heartbeat {self.phase} {self.done}{tot}{ex}")

    def start(self):
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()

    def stop(self):
        self._stop.set()


def _target(name):
    mod, _, fn = str(name).partition(":")
    if not fn:
        raise ValueError(f"a target is 'module:function', not {name!r}")
    return getattr(importlib.import_module(mod), fn)


def _call(target, payload, ctx) -> dict:
    """Run the target; a refusal and a failure become results (the caller raises them)."""
    ctx.start()
    try:
        ctx.log(f"start device={ctx.device} ({ctx.device_why}) threads={ctx.threads}")
        out = _target(target)(payload, ctx)
        out = dict(out or {})
        out.setdefault("state", "done")
        ctx.log("done")
    except TrainingRefused as e:
        ctx.log(f"refused: {e}")
        out = {"state": "refused", "why": str(e)}
    except Exception as e:  # noqa: BLE001 — the target's failure is the caller's to record
        ctx.log(f"failed: {type(e).__name__}: {e}")
        out = {"state": "failed", "why": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    finally:
        ctx.stop()
    out.setdefault("device", ctx.device)
    out.setdefault("estimate_s", ctx.estimate_s)
    return out


def _result(out):
    if out.get("state") == "refused":
        raise TrainingRefused(out.get("why") or "refused")
    if out.get("state") == "failed":
        raise TrainingFailed(out.get("why") or "failed")
    return out


# ───────────────────────────────────────────────────────────── runners
class ThreadRunner:
    """See the module docstring. heartbeat_s: seconds between heartbeat lines; grace_s: past the estimate (max_train_s
    before one) before a training is reported as over its time. In this runner the LoRA trainer copies the network in
    the loop's process while holding the scorer's network gate alone, so every decision on that model (every question,
    not only the loop's) waits while the copy is made; unmeasured. `SubprocessRunner`, the default, copies in the
    child and is unaffected."""
    kind = "thread"

    def __init__(self, *, heartbeat_s=30.0, grace_s=120.0):
        self.heartbeat_s, self.grace_s = float(heartbeat_s), float(grace_s)

    def run(self, target, payload, *, name, workdir, budget, status=None, cancel=None) -> dict:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        log = workdir / f"{name}.log"
        dev, why = pick_device(budget.device, budget.free_gpu_mb)
        status = status if status is not None else {}
        status.update(runner=self.kind, pid=os.getpid(), log=str(log), device=dev)
        ctx = RunContext(budget, dev, why, log, status, self.heartbeat_s)
        ctx.in_process = True
        box = {}
        t0 = time.monotonic()
        th = threading.Thread(target=lambda: box.update(out=_call(target, payload, ctx)), daemon=True,
                              name=f"solvi-train-{name}")
        th.start()
        while th.is_alive():
            th.join(0.05)
            if cancel is not None and cancel.is_set():
                raise TrainingCancelled(f"{name}: cancelled (the training thread cannot be stopped; it ends on its own)")
        out = box.get("out") or {"state": "failed", "why": "the training thread ended without a result"}
        if ctx.estimate_s is not None:             # the same deadline as the subprocess runner's
            over = time.monotonic() - ctx.estimate_t - ctx.estimate_s - self.grace_s
        else:
            over = time.monotonic() - t0 - float(budget.max_train_s) - self.grace_s
        if out.get("state") == "done" and over > 0:
            raise TrainingFailed(f"ran {over:.0f} s past its time limit (estimate or max_train_s) + grace "
                                 f"{self.grace_s:g} s "
                                 "(a thread cannot be stopped; use the subprocess runner)")
        return _result(out)


class SubprocessRunner:
    """See the module docstring. nice: the process's niceness; heartbeat_s: seconds between heartbeat lines; grace_s: on
    top of the target's estimate (counted from the estimate line; before it, on top of max_train_s from the start)
    before the process is killed — it must cover what the estimate does not: loading the checkpoint, calibration,
    saving; stall_s: no log line for this long → killed, a RunnerError (None: max(10 heartbeats, 300 s)); python: the
    interpreter (default this one); pythonpath: extra import paths for the target's module (the folder solvi is imported
    from always comes first); poll_s: how often the process is checked."""
    kind = "subprocess"

    def __init__(self, *, nice=10, heartbeat_s=30.0, grace_s=120.0, stall_s=None, python=None, pythonpath=(),
                 poll_s=0.5):
        self.nice, self.heartbeat_s, self.grace_s = int(nice), float(heartbeat_s), float(grace_s)
        self.stall_s = float(stall_s) if stall_s is not None else max(10 * self.heartbeat_s, 300.0)
        self.python = python or sys.executable
        self.pythonpath, self.poll_s = [str(p) for p in pythonpath], float(poll_s)

    def _env(self, budget):
        env = dict(os.environ)
        n = str(int(budget.cpu_threads))
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = n
        here = str(Path(__file__).resolve().parents[3])          # the folder `solvi` is imported from
        paths = [here, *self.pythonpath] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        return env

    def run(self, target, payload, *, name, workdir, budget, status=None, cancel=None) -> dict:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        log, job, res, err = (workdir / f"{name}{x}" for x in (".log", ".job.pkl", ".result.json", ".err"))
        for f in (res,):
            f.unlink(missing_ok=True)
        spec = {"target": target, "payload": payload, "budget": budget, "log": str(log), "result": str(res),
                "nice": self.nice, "heartbeat_s": self.heartbeat_s}
        with open(job, "wb") as f:
            pickle.dump(spec, f)
        status = status if status is not None else {}
        status.update(runner=self.kind, log=str(log), device=None)
        with open(err, "ab") as ef:
            p = subprocess.Popen([self.python, "-m", "solvi.experimental.stream.runner", str(job)], env=self._env(budget),  # noqa: S603
                                 stdout=ef, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        status["pid"] = p.pid
        t0 = time.monotonic()
        # The kill deadline: max_train_s + grace from the start until the target logs its estimate; from then on the
        # estimate + grace from the moment the estimate line is seen (a later estimate, e.g. after a GPU → CPU
        # fallback, moves it again). So a training that stays within its estimate is never killed for the loading
        # and calibration the estimate does not cover; grace_s covers those.
        deadline_at = t0 + float(budget.max_train_s) + self.grace_s
        n_est = 0
        why = None
        while True:
            code = p.poll()
            if code is not None:
                break
            now = time.monotonic()
            try:
                seen = log.stat().st_mtime
                age = time.time() - seen
                status["last_t"] = seen
                status["last"] = _last_line(log)
                ests = _estimates(log)
                if len(ests) > n_est:
                    n_est, est = len(ests), ests[-1]
                    status["eta"] = seen + est
                    deadline_at = now + est + self.grace_s
                if status.get("device") is None:
                    status["device"] = _device(log)
            except FileNotFoundError:
                age = now - t0
            if cancel is not None and cancel.is_set():
                why = ("cancel", "cancelled: stopped")
            elif now > deadline_at:
                why = ("over", (f"ran past its estimate + grace {self.grace_s:g} s: stopped" if n_est else
                                f"ran past max_train_s {budget.max_train_s:g} + grace {self.grace_s:g} s with no "
                                "estimate: stopped"))
            elif age > self.stall_s:
                why = ("stall", f"no log line for {age:.0f} s (stall_s {self.stall_s:g}): stopped")
            if why is not None:
                _kill(p)
                break
            time.sleep(self.poll_s)
        if why is not None:
            if why[0] == "over":
                raise TrainingFailed(why[1])
            if why[0] == "cancel":
                raise TrainingCancelled(f"the training process {p.pid}: {why[1]}")
            raise RunnerError(f"the training process {p.pid}: {why[1]}; log {log}")
        if not res.exists():
            tail = _tail(err)
            cap = " (killed: the memory cap?)" if code in (-signal.SIGKILL, 137) else ""
            raise RunnerError(f"the training process {p.pid} exited with {code}{cap} and wrote no result; "
                              f"log {log}{': ' + tail if tail else ''}")
        with open(res) as f:
            out = json.load(f)
        return _result(out)


def _kill(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()
    p.wait()


def _last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode(errors="replace").strip().splitlines()
        return lines[-1] if lines else None
    except FileNotFoundError:
        return None


def _scan(path, word):
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[1] == word:
                    return parts[2:]
    except FileNotFoundError:
        pass
    return None


def _estimates(path):
    """Every estimate line's seconds, in order → [s, ...]."""
    got = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[1] == "estimate":
                    try:
                        got.append(float(parts[2]))
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    return got


def _device(path):
    got = _scan(path, "start")
    return got[0].split("=", 1)[1] if got and got[0].startswith("device=") else None


def _tail(path, n=3):
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
    except FileNotFoundError:
        return ""
    return " | ".join(lines[-n:])


def make_runner(runner):
    """"subprocess" / "thread" / a runner object (with run(target, payload, *, name, workdir, budget, status), and
    optionally cancel=) → a runner."""
    if runner == "subprocess":
        return SubprocessRunner()
    if runner == "thread":
        return ThreadRunner()
    if callable(getattr(runner, "run", None)):
        return runner
    raise ValueError(f"runner is 'subprocess', 'thread' or a runner object, not {runner!r}")


def _main(job_path):
    """The training process: read the job, lower its priority, cap its threads, run the target, write the result."""
    with open(job_path, "rb") as f:
        spec = pickle.load(f)  # noqa: S301 — the job file the parent process just wrote
    try:
        os.nice(int(spec.get("nice", 0)))
    except OSError:
        pass
    budget = spec["budget"]
    dev, why = pick_device(budget.device, budget.free_gpu_mb)
    ctx = RunContext(budget, dev, why, spec["log"], None, spec.get("heartbeat_s", 30.0))
    out = _call(spec["target"], spec["payload"], ctx)
    tmp = spec["result"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, default=str)
    os.replace(tmp, spec["result"])


if __name__ == "__main__":                     # run as the module solvi imports, so its exception classes are the ones
    from solvi.experimental.stream.runner import _main as _run   # a target raises
    _run(sys.argv[1])


__all__ = ["RunContext", "RunnerError", "SubprocessRunner", "ThreadRunner", "TrainingCancelled", "TrainingFailed",
           "TrainingRefused",
           "gpu_free_mb", "make_runner", "pick_device"]
