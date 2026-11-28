#!/usr/bin/env bash
# First real run of a script of the forecaster finder, as a new user would do
# it: a new environment, the install line of the script, the script with its
# sample data, and the same script when Hugging Face cannot be reached.
#
# It measures the time and the size of the install, the first run (it
# downloads the weights of Chronos-2) and the second one. It installs about
# 6 GB (PyTorch with the CUDA packages, the default of pip on Linux).
#
# Usage (from this folder, after `python -I build.py`):
#     bash check_clean_install.sh [answers]
# Set VENV to reuse an environment instead of creating one.
# `answers` is the key of a combination of results/review/finder.json
# (default: number-one-monthly-no, the script with the three models).
set -euo pipefail

key="${1:-number-one-monthly-no}"
work="$(mktemp -d)"
venv="${VENV:-$work/venv}"
script="$work/script.py"

# The script that the page copies: the sample data block and the snippet
python -I - "$key" "$script" <<'PY'
import json
import sys

finder = json.load(open("results/review/finder.json"))
combo = next(c for c in finder["combos"] if c["key"] == sys.argv[1])
code = finder["snippets"][combo["snippet"]]["code"]
open(sys.argv[2], "w").write(combo["sample"] + "\n" + code + "\n")
PY
install="$(grep -m1 '^# pip install' "$script" | sed 's/^# //')"

[ -d "$venv" ] || python -m venv "$venv"
start=$(date +%s)
eval "\"$venv/bin/python\" -m $install -q"
echo "install: $(( $(date +%s) - start )) s, $(du -sh "$venv" | cut -f1)"
"$venv/bin/python" -m pip list 2>/dev/null | grep -i -E "^(skforecast|pandas|numpy|torch|lightgbm|chronos-forecasting|statsmodels) "

export HF_HOME="$work/hf"
for run in first second; do
    start=$(date +%s)
    "$venv/bin/python" "$script" > "$work/$run.log" 2>&1
    echo "$run run: $(( $(date +%s) - start )) s, $(du -sh "$HF_HOME" | cut -f1) of weights"
done
tail -8 "$work/second.log"

# Hugging Face blocked (a company network): no cache and an endpoint that
# does not answer
start=$(date +%s)
if HF_HOME="$work/hf_empty" HF_ENDPOINT="http://127.0.0.1:9" \
    "$venv/bin/python" "$script" > "$work/blocked.log" 2>&1; then
    echo "blocked: the script ran"
else
    echo "blocked: failed after $(( $(date +%s) - start )) s with"
    tail -2 "$work/blocked.log"
fi
echo "logs in $work"
