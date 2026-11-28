"""
History that the scripts of the forecaster finder need.

build.py runs every script on a comfortable history. This one runs each kind
of script (target, number of series and frequency) with the shortest history
it must work with (`min_history` in finder.FREQ, the figure the page shows
under the script), with half of it and with one and a half times it. A script
passes if it raises no error and no warning with the minimum history or more.

Usage:
    python tools/docs/forecaster_finder/check_snippets.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402
import finder  # noqa: E402

FRACTIONS = [0.5, 1.0, 1.5]  # Of the `min_history` of the frequency

failed = 0
for target, series in [("number", "one"), ("number", "several"), ("category", None)]:
    for freq, f in finder.FREQ.items():
        a = {"target": target, "freq": freq, "extra": "no"}
        if series:
            a["series"] = series
        status = {}
        for fraction in FRACTIONS:
            code = finder.render(a, finder.decide(a), n=int(f["min_history"] * fraction))
            with contextlib.redirect_stdout(io.StringIO()):
                error, seen = build.run(code)
            status[fraction] = "ok" if error is None and not seen else (error or seen[0])[:60]
        passes = status[1.0] == "ok" and status[1.5] == "ok"
        failed += not passes
        print(f"{target:<9}{series or '':<9}{freq:<10}{f['min_history']:>5} values  "
              f"{'pass' if passes else 'FAIL'}  half: {status[0.5]}  "
              f"minimum: {status[1.0]}  x1.5: {status[1.5]}", flush=True)
print(f"{failed} kinds of script fail with their minimum history")
sys.exit(1 if failed else 0)
