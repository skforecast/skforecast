#!/usr/bin/env bash
# Shared settings of the runners. Source it, do not run it.
#
# Required environment variables:
#   MS_WORK   work folder outside the repository: results, exported trees and
#             bytecode go there (nothing is written into the repository)
#   MS_PY_P2  python of the pandas 2 environment
#   MS_PY_P3  python of the pandas 3 environment (with pyarrow installed)
# Optional:
#   MS_BASE_TREE  folder that contains the `skforecast` package of the base
#                 commit (default: $MS_WORK/base_tree), created with
#                 git archive <commit> skforecast | tar -x -C "$MS_WORK/base_tree"
#   MS_NEW_TREE   tree compared with the base (default: the repository)
#   MS_CPU        logical processor the timed runs are pinned to (default: 12)
: "${MS_WORK:?set MS_WORK to a work folder outside the repository}"
: "${MS_PY_P2:?set MS_PY_P2 to the python of the pandas 2 environment}"
: "${MS_PY_P3:?set MS_PY_P3 to the python of the pandas 3 environment}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/../../.." && pwd)"
BASE_TREE="${MS_BASE_TREE:-$MS_WORK/base_tree}"
NEW_TREE="${MS_NEW_TREE:-$REPO}"
OUT="$MS_WORK/out"
mkdir -p "$OUT"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Sets PY and NOPA for an environment name: p2, p3 (pyarrow) or p3nopa.
select_env() {
  case $1 in
    p2) PY="$MS_PY_P2"; NOPA=0 ;;
    p3) PY="$MS_PY_P3"; NOPA=0 ;;
    p3nopa) PY="$MS_PY_P3"; NOPA=1 ;;
    *) echo "unknown environment $1" >&2; exit 1 ;;
  esac
}

# Sets TREE_DIR for a tree name: base or new.
select_tree() {
  case $1 in
    base) TREE_DIR="$BASE_TREE" ;;
    new) TREE_DIR="$NEW_TREE" ;;
    *) echo "unknown tree $1" >&2; exit 1 ;;
  esac
}
