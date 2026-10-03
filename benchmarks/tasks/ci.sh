#!/usr/bin/env bash
# The stand's published numbers, checked from the packed replies: no key, no model call, nothing paid. What the weekly
# workflow (.github/workflows/stand.yml) runs, and the same on any machine:
#   ci.sh data   the prepared splits from packed/prepared.tar.xz, and the downloads the scripts read at run time:
#                τ-bench's environment, NATURAL PLAN's evaluator, NAB's series, BIRD's databases (needs the network);
#                Abt-Buy is downloaded and prepared (never redistributed) — if that fails, its steps are skipped
#   ci.sh run    packed/replies*.jsonl.xz into a cache, every script offline (stand.py run), then stand.py check
# Data: $STAND_DATA (default benchmarks/tasks/data); cache: $STAND_CACHE (default benchmarks/tasks/cache).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=(uv run --with numpy --with scikit-learn --with scipy python)
CACHE="${STAND_CACHE:-$HERE/cache}"
REQUIRED=taubench,cuad,ragtruth,banking77,credit,bird,naturalplan,nab          # every task but Abt-Buy must run
case "${1:-}" in
  data)
    "${PY[@]}" "$HERE/replies.py" unpack-data
    bash "$HERE/fetch.sh" taubench naturalplan nab bird
    if bash "$HERE/fetch.sh" abtbuy && "${PY[@]}" "$HERE/prepare.py" abtbuy; then
      echo "Abt-Buy downloaded and prepared"
    else
      echo "::warning::Abt-Buy could not be downloaded or prepared: its steps are skipped and its numbers not compared"
    fi
    ;;
  run)
    "${PY[@]}" "$HERE/replies.py" unpack --cache "$CACHE"
    status=0
    "${PY[@]}" "$HERE/stand.py" run --cache "$CACHE" --fresh || status=$?
    "${PY[@]}" "$HERE/stand.py" check --measured "$HERE/runs/stand.json" --require "$REQUIRED" || status=1
    exit $status
    ;;
  *)
    echo "usage: ci.sh data | run" >&2
    exit 2
    ;;
esac
