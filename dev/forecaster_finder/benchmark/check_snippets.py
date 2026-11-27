"""
History that the snippets of the forecaster finder need.

build.py runs every snippet on a comfortable history. This script runs each
kind of snippet (target, number of series and frequency) with the shortest
history it must work with (`min_history` in finder_tree.FREQ) and with
shorter ones, and writes results/review/snippet_min_history.csv. A snippet
passes if it raises no error and no warning.

Usage:
    python -I check_snippets.py
"""
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402
import finder_tree as tree  # noqa: E402

FRACTIONS = [0.5, 0.75, 1.0, 1.5]  # Of the `min_history` of the frequency

rows = []
for target, series in [("number", "one"), ("number", "several"), ("category", None)]:
    for freq, f in tree.FREQ.items():
        a = {"target": target, "freq": freq, "extra": "no"}
        if series:
            a["series"] = series
        code = tree.render(a, tree.decide(a))
        row = {"target": target, "series": series or "", "freq": freq,
               "min_history": f["min_history"]}
        for fraction in FRACTIONS:
            n = int(f["min_history"] * fraction)
            with contextlib.redirect_stdout(io.StringIO()):
                error, seen = build.run(code, a, n)
            row[f"x{fraction}"] = "ok" if error is None and not seen else (error or seen[0])[:70]
        row["passes"] = row["x1.0"] == "ok" and row["x1.5"] == "ok"
        rows.append(row)
        print(row, flush=True)
out = pd.DataFrame(rows)
out.to_csv(build.RES / "snippet_min_history.csv", index=False)
print(out[["target", "series", "freq", "min_history", "x0.5", "x0.75", "x1.0", "passes"]]
      .to_string(index=False))
sys.exit(0 if out["passes"].all() else 1)
