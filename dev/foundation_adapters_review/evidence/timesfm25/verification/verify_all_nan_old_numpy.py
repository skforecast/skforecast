"""
Behaviour of the timesfm 2.5 NaN helpers (strip_leading_nans,
linear_interpolation) on an all-NaN context with the installed numpy. The two
functions are extracted with ast from the timesfm_2p5_base.py file given as
argument, so that numpy versions older than the main review environment can
be tested without installing timesfm.

Usage (from the repository root):
    python verify_all_nan_old_numpy.py <path to timesfm_2p5_base.py>
"""

import ast
import sys
from importlib.metadata import version

import numpy as np

with open(sys.argv[1]) as fh:
    src = fh.read()
tree = ast.parse(src)
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ("strip_leading_nans", "linear_interpolation")]
ns = {"np": np}
exec(compile(ast.Module(body=funcs, type_ignores=[]), "helpers", "exec"), ns)
print(f"numpy {version('numpy')}")
try:
    out = ns["linear_interpolation"](ns["strip_leading_nans"](np.full(6, np.nan)))
    print("all-NaN context ->", out)
except Exception as exc:
    print(f"all-NaN context -> {type(exc).__name__}: {exc}")
