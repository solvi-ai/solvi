#!/usr/bin/env bash
# Colab driver for the 100 000-turn lookahead health run (seed 5, --no-tracemalloc, as the value-head variant's 100k run).
# Colab VMs die after ~1 h, so the run goes in segments: a fresh CPU VM, the code + the latest exact checkpoint uploaded,
# sim.py --resume for at most WALL seconds (it pauses at the next checkpoint, exit 3), checkpoint + logs downloaded
# (also every few minutes while it runs), the VM released; repeat until the run completes (exit 0).
# Local side: only this loop (sleeps and CLI calls). Session name distinct from other agents' sessions.
#   nohup spaces/realms/results/lookahead/colab/driver.sh > /dev/null 2>&1 &     (from the solvi repo)
set -u
ROOT=/home/exactor/Projects/Prototypes/new_kelly/solvi
W=$ROOT/spaces/realms/results/lookahead/colab
SES=${REALMS_SES:-realms_100k}
CFG=$HOME/.config/colab-cli/new_kelly_sessions.json
C="colab --config $CFG"
PY=$HOME/.local/share/uv/tools/google-colab-cli/bin/python
GUARD=/home/exactor/Projects/Prototypes/new_kelly/exps_v2/colab/colab_guard.py
REFRESH=/home/exactor/Projects/Prototypes/new_kelly/exps_v2/colab/refresh_session.py
SEED=5; TURNS=100000; WALL=${REALMS_WALL:-2700}
CK=$W/ckpt_$SEED.json; OUT=$W/endless_$SEED.json
cd "$W" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$W/driver.log"; }
turn_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['game']['turn'])" "$1" 2>/dev/null || echo -1; }

# bundle: realms code (working copy) + solvi as committed (git HEAD: stable while the library is being edited)
make_bundle() {
  rm -rf "$W/stage" && mkdir -p "$W/stage/spaces/realms" || return 1
  (cd "$ROOT" && git archive HEAD src/solvi) | tar -x -C "$W/stage" || return 1
  cp -r "$ROOT/spaces/realms/realms" "$ROOT/spaces/realms/sim.py" "$W/stage/spaces/realms/" || return 1
  find "$W/stage" -name __pycache__ -prune -exec rm -rf {} + ; tar -czf "$W/bundle.tgz" -C "$W/stage" . && rm -rf "$W/stage"
}
up() { for t in 1 2 3 4 5; do timeout 600 $C upload -s $SES "$1" "$2" 2>&1 | grep -q Uploaded && return 0; sleep 15; done; return 1; }
vm() { timeout 180 $C exec -s $SES 2>&1 | grep "^\[vm\]"; }   # python code on stdin; only [vm] lines are kept
fetch_ckpt() {   # download the VM's checkpoint; keep it only if it is valid and not older than the local one
  timeout 600 $C download -s $SES /content/r/out/ckpt_$SEED.json "$W/ckpt.part" > /dev/null 2>&1 || return 1
  local a b; a=$(turn_of "$W/ckpt.part"); b=$(turn_of "$CK")
  if [ "$a" -ge "$b" ] && [ "$a" -gt 0 ]; then mv "$W/ckpt.part" "$CK"; log "checkpoint turn $a"; else rm -f "$W/ckpt.part"; fi
}
stop_vm() { touch "$W/guard.stop"; timeout 300 $C stop -s $SES >> "$W/driver.log" 2>&1; }

make_bundle || { log "bundle failed"; exit 1; }
log "driver start: session $SES, seed $SEED, $TURNS turns, $WALL s per segment; local checkpoint turn $(turn_of "$CK")"
fails=0; seg=0
while [ ! -f "$W/FINISHED" ] && [ $fails -lt 12 ]; do
  seg=$((seg + 1))
  machine=""; [ $fails -ge 3 ] && machine="--gpu T4"          # CPU runtime first; a T4 only if CPU VMs keep failing
  timeout 300 $C stop -s $SES > /dev/null 2>&1
  log "segment $seg: new VM ${machine:-CPU}"
  if ! timeout 900 $C new -s $SES $machine >> "$W/driver.log" 2>&1; then fails=$((fails + 1)); log "new failed ($fails)"; sleep 120; continue; fi
  $PY "$REFRESH" $SES "$CFG" >> "$W/driver.log" 2>&1
  up "$W/bundle.tgz" /content/bundle.tgz || { fails=$((fails + 1)); log "upload failed"; stop_vm; continue; }
  RES=""
  if [ "$(turn_of "$CK")" -gt 0 ]; then
    up "$CK" /content/ckpt_in.json || { fails=$((fails + 1)); log "checkpoint upload failed"; stop_vm; continue; }
    RES="--resume /content/r/out/ckpt_$SEED.json"
  fi
  vm <<PY
import os, subprocess
os.makedirs("/content/r/out", exist_ok=True)
subprocess.run(["tar", "-xzf", "/content/bundle.tgz", "-C", "/content/r"], check=True)
if os.path.exists("/content/ckpt_in.json"):
    os.replace("/content/ckpt_in.json", "/content/r/out/ckpt_$SEED.json")
for f in ("EXIT",):
    p = "/content/r/out/" + f
    if os.path.exists(p): os.remove(p)
cmd = ("cd /content/r && PYTHONPATH=/content/r/src OPENBLAS_NUM_THREADS=1 python spaces/realms/sim.py --variant lookahead "
       "--seed $SEED --turns $TURNS --no-tracemalloc --out /content/r/out/endless_$SEED.json "
       "--checkpoint /content/r/out/ckpt_$SEED.json --max-wall $WALL $RES >> /content/r/out/sim.log 2>&1; "
       "echo \$? > /content/r/out/EXIT")
subprocess.Popen(["nohup", "bash", "-c", cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
import platform, numpy
print("[vm] started", platform.python_version(), "numpy", numpy.__version__, os.cpu_count(), "cpus")
PY
  rm -f "$W/guard.stop"
  nohup $PY "$GUARD" "$CFG" $SES "$W/guard.stop" "$W/guard.log" > /dev/null 2>&1 &
  # poll: every 2 min status, every ~6 min a checkpoint download; the VM job pauses itself after WALL seconds
  lost=0; polls=0; code=""
  while true; do
    sleep 120; polls=$((polls + 1))
    $PY "$REFRESH" $SES "$CFG" >> "$W/driver.log" 2>&1
    st=$(vm <<'PY'
import os
p = "/content/r/out/"
code = open(p + "EXIT").read().strip() if os.path.exists(p + "EXIT") else ""
tail = ""
if os.path.exists(p + "sim.log"):
    lines = [l for l in open(p + "sim.log").read().splitlines() if l.strip()]
    tail = lines[-1][:90] if lines else ""
print("[vm]", "EXIT=" + code, "|", tail)
PY
)
    if [ -z "$st" ]; then lost=$((lost + 1)); log "no answer from VM ($lost)"; [ $lost -ge 5 ] && break; continue; fi
    lost=0
    [ $((polls % 3)) -eq 0 ] && fetch_ckpt
    code=$(echo "$st" | sed -n 's/.*EXIT=\([0-9]*\).*/\1/p')
    [ -n "$code" ] && { log "segment $seg ended: exit $code | $st"; break; }
  done
  fetch_ckpt
  timeout 600 $C download -s $SES /content/r/out/endless_$SEED.json "$W/out.part" > /dev/null 2>&1 && mv "$W/out.part" "$OUT"
  timeout 600 $C download -s $SES /content/r/out/sim.log "$W/sim_seg$seg.log" > /dev/null 2>&1
  stop_vm
  case "$code" in
    0) touch "$W/FINISHED"; log "run complete at turn $(turn_of "$CK")";;
    3) fails=0;;
    *) fails=$((fails + 1)); log "segment $seg failed or VM lost (exit '${code}'); resuming from turn $(turn_of "$CK")";;
  esac
done
log "driver end (finished: $([ -f "$W/FINISHED" ] && echo yes || echo no))"
