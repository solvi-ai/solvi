#!/usr/bin/env bash
# Downloads the public data of the nine tasks into $STAND_DATA (default: benchmarks/tasks/data; about 1.7 GB on disk,
# 0.9 GB of traffic). Nothing here is redistributed by solvi: every file comes from its own source. Then: prepare.py
#   fetch.sh                 every task
#   fetch.sh bird nab        only these
set -euo pipefail
DATA="${STAND_DATA:-$(cd "$(dirname "$0")" && pwd)/data}"
mkdir -p "$DATA" && cd "$DATA"
TASKS="${*:-taubench naturalplan nab banking77 cuad ragtruth abtbuy credit bird}"
want() { [[ " $TASKS " == *" $1 "* ]] && mkdir -p "$1"; }
get() { [ -s "$1" ] || curl -sfL --retry 3 -o "$1" "$2"; }
clone() { [ -d "$1" ] || git clone -q --depth 1 "${@:3}" "$2" "$1"; }
if want taubench; then clone taubench/repo https://github.com/sierra-research/tau-bench; fi
if want naturalplan; then clone naturalplan/repo https://github.com/google-deepmind/natural-plan; fi
if want nab; then
  clone nab/repo https://github.com/numenta/NAB --filter=blob:none --sparse && git -C nab/repo sparse-checkout set data labels >/dev/null
fi
if want banking77; then clone banking77/repo https://github.com/PolyAI-LDN/task-specific-datasets; fi
if want cuad; then get cuad/data.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip && (cd cuad && unzip -q -o data.zip); fi
if want ragtruth; then
  for f in test train; do get ragtruth/$f.parquet https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/main/data/$f-00000-of-00001.parquet; done
fi
if want abtbuy; then
  for f in tableA tableB test train valid; do get abtbuy/$f.csv https://huggingface.co/datasets/matchbench/Abt-Buy/resolve/main/$f.csv; done
fi
if want credit; then
  get credit/german.zip "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip" && (cd credit && unzip -q -o german.zip -d german)
fi
if want bird && [ ! -d bird/minidev ]; then                   # 800 MB; only the SQLite databases are unpacked (1.4 GB)
  get bird/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
  (cd bird && unzip -q -o minidev.zip 'minidev/MINIDEV/mini_dev_sqlite.json' 'minidev/MINIDEV/dev_tables.json' 'minidev/MINIDEV/dev_databases/*' && rm minidev.zip)
fi
echo "data in $DATA; next: python prepare.py"
