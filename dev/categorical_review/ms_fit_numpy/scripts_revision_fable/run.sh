#!/bin/bash
# usage: run.sh <base|branch> <p2|p3> script.py [args...]
W="<scratchpad>/indep_review"
tree=$1; pd=$2; shift 2
if [ "$tree" = "base" ]; then PP="$W/base"; elif [ "$tree" = "alt" ]; then PP="$W/alt"; else PP="<repo>"; fi
if [ "$pd" = "p2" ]; then ENV=skforecast_27_py13; else ENV=skforecast_27_py13_p3; fi
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$W/numba_cache_$pd" PYTHONPATH="$PP" PYTHONHASHSEED=0 TREE=$tree PDV=$pd
exec <conda>/envs/$ENV/python.exe "$@"
