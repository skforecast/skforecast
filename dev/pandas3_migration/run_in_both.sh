#!/usr/bin/env bash
# Run a Python script in the pandas 2 and the pandas 3 environments, one after
# the other (never in parallel), and store the stdout+stderr of each run in
# results/<script>_pandas<version>.txt. Extra arguments are passed through.
#
#   bash dev/pandas3_migration/run_in_both.sh experiments/exp_cow_basics.py [args...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
P2="${SKF_P2:-/root/.venvs/skforecast/bin/python}"
P3="${SKF_P3:-/root/.venvs/skforecast-p3/bin/python}"
script="$1"; shift
name="$(basename "$script" .py)"
mkdir -p "$HERE/results"
for py in "$P2" "$P3"; do
    ver="$(PYTHONPATH="$REPO" "$py" -c 'import pandas; print(pandas.__version__)')"
    out="$HERE/results/${name}_pandas${ver}.txt"
    echo "== pandas $ver -> $out"
    (cd "$REPO" && PYTHONPATH="$REPO" "$py" "$HERE/$script" "$@") > "$out" 2>&1
    cat "$out"
done
