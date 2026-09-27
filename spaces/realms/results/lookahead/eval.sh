#!/bin/bash
# pre-registered evaluation of the lookahead variant (realms/lookahead.py B1-B5): seeds 1-4 x 20 000 turns, default sim.py
# settings (tracemalloc, save/load every 5 000), all 4 at once; plus the B5 audit (2 000 turns of seed 1)
cd /home/exactor/Projects/Prototypes/new_kelly/solvi
for s in 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 uv run python spaces/realms/sim.py --variant lookahead --seed $s --turns 20000 > spaces/realms/results/lookahead/logs/sim_$s.log 2>&1 &
done
OPENBLAS_NUM_THREADS=1 uv run python spaces/realms/results/lookahead/b5_check.py 1 2000 > spaces/realms/results/lookahead/logs/b5.log 2>&1 &
wait
echo "DONE eval"
