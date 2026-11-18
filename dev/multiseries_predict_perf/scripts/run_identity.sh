#!/usr/bin/env bash
# Identity batteries of one tree in the three environments. Results in
# $MS_WORK/out/identity/<label>_<battery>_<env>.pkl
#
# Usage: run_identity.sh <label> <tree: base|new> [compare]
#   With `compare`, every result is compared with the one of the label `base`
#   (run `run_identity.sh base base` once before).
# Environments: p2 p3 p3nopa, or the ones in ENVS. Batteries: ms trims, or the
# ones in BATTERIES (ms: identity_ms.py; trims: identity_trims.py, the battery
# of the previous block, which covers the other forecasters; dtypes:
# identity_dtypes.py, last windows with columns that are not float; exog:
# identity_exog.py, exog with columns that are not float; the last two only
# when given in BATTERIES).
source "$(dirname "$0")/env.sh"
LABEL="$1"; select_tree "$2"; COMPARE="$3"
mkdir -p "$OUT/identity"
cd "$SCRIPTS" || exit 1
# PYTHONHASHSEED: some warnings print a set of names, whose order depends on it.
export PYTHONPATH="$TREE_DIR" PYTHONPYCACHEPREFIX="$MS_WORK/pycache_$2" PYTHONHASHSEED=0
for env in ${ENVS:-p2 p3 p3nopa}; do
  select_env "$env"
  for battery in ${BATTERIES:-ms trims}; do
    result="$OUT/identity/${LABEL}_${battery}_${env}.pkl"
    NO_PYARROW=$NOPA "$PY" "identity_${battery}.py" "$result" 2>/dev/null | tail -1
    if [ "$COMPARE" = "compare" ]; then
      NO_PYARROW=$NOPA "$PY" identity_compare_ms.py \
        "$OUT/identity/base_${battery}_${env}.pkl" "$result" | cut -c1-400
    fi
  done
done
echo DONE "$LABEL"
