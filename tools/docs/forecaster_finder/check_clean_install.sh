#!/usr/bin/env bash
# First real run of a script of the forecaster finder, as a new user would do
# it: a new environment, the install line of the script, the script with its
# sample data, and the same script when Hugging Face cannot be reached.
#
# It measures the time and the size of the install, the first run (it
# downloads the weights of Chronos-2) and the second one. It installs about
# 6 GB (PyTorch with the CUDA packages, the default of pip on Linux).
#
# Usage (from the root of the repository):
#     bash tools/docs/forecaster_finder/check_clean_install.sh [answers]
# `answers` is the key of a combination of docs/javascripts/finder-data.json
# (default: number-one-monthly-no, the script with the three models).
# Set VENV to reuse an environment instead of creating one.
set -euo pipefail

key="${1:-number-one-monthly-no}"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
venv="${VENV:-$work/venv}"
script="$work/script.py"

# The script as the page gives it: it includes its sample data
python -I - "$key" "$script" "$here" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads((Path(sys.argv[3]).parents[2] / "docs/javascripts/finder-data.json").read_text())
combo = data["combos"][sys.argv[1]]
Path(sys.argv[2]).write_text(data["snippets"][combo["snippet"]])
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
