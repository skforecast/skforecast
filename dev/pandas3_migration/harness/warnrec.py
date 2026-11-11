"""
pytest plugin that writes every warning captured by pytest as one JSON line
(test id, category, file, line and message) to the file named by the
`WARNREC_OUT` environment variable.

Usage (the harness directory must be importable):
    WARNREC_OUT=warnings.jsonl PYTHONPATH=<repo root>:<this directory> \
        python -m pytest skforecast -q -p warnrec
"""

from __future__ import annotations

import json
import os


def pytest_warning_recorded(warning_message, when, nodeid, location):
    path = os.environ.get("WARNREC_OUT")
    if not path:
        return
    record = {
        "nodeid": nodeid,
        "when": when,
        "category": warning_message.category.__name__,
        "filename": warning_message.filename,
        "lineno": warning_message.lineno,
        "message": str(warning_message.message).splitlines()[0][:300],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
