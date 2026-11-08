#!/usr/bin/env bash
# D7: n_jobs study subset on WSL2 (block A, V1, reduced B-E, Sarimax repro).
# Usage: bash run_wsl.sh pilot|full
set -u
MODE="${1:?pilot or full}"
DST="$HOME/github/skforecast_njobs_wsl"
P="$HOME/miniconda3/envs/skforecast_27_py13_wsl/bin/python"
cd "$DST/dev/n_jobs_review"
export PYTHONPATH="$DST"
export PYTHONIOENCODING=utf-8
if [ "$MODE" = pilot ]; then
  export NJOBS_RESULTS_DIR="$DST/dev/n_jobs_review/results/wsl_pilot"
  PILOT=(--pilot); REPS=1
else
  export NJOBS_RESULTS_DIR="$DST/dev/n_jobs_review/results/wsl"
  PILOT=(); REPS=5
fi
mkdir -p "$NJOBS_RESULTS_DIR"
step() { echo "STEP $1 START $(date +%T)"; }
done_() { echo "STEP $1 END rc=$2 $(date +%T)"; }

T0=$(date +%s)
step machine
"$P" -c "from common import machine_info, dump_json, RESULTS_DIR; import skforecast; print(skforecast.__file__); dump_json(machine_info(), RESULTS_DIR / 'machine.json')"
done_ machine $?

step probe
"$P" probe_workers.py; done_ probe $?

step A
"$P" bench_spawn.py "${PILOT[@]}"; done_ A $?

step V1
"$P" bench_cold.py --reps "$REPS"; done_ V1 $?

for B in B C D E; do
  step "$B"
  "$P" bench.py --block "$B" --subset --lean "${PILOT[@]}"; done_ "$B" $?
done

if [ "$MODE" = full ]; then
  step sarimax
  "$P" exp_sarimax_parallel.py > "$NJOBS_RESULTS_DIR/exp_sarimax_parallel.txt" 2>&1
  done_ sarimax $?
fi
echo "ALL DONE $(( ($(date +%s) - T0) / 60 )) min"
