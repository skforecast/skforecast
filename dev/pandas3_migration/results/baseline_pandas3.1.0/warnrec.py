"""pytest plugin: dump every recorded warning as JSON lines."""
import json
import os
import warnings

_OUT = os.environ.get("WARNREC_OUT", "warnings.jsonl")
_fh = None


def pytest_configure(config):
    global _fh
    _fh = open(_OUT, "w")


def pytest_warning_recorded(warning_message, when, nodeid, location):
    rec = {
        "nodeid": nodeid,
        "when": when,
        "category": warning_message.category.__name__,
        "module": warning_message.category.__module__,
        "message": str(warning_message.message)[:600],
        "filename": warning_message.filename,
        "lineno": warning_message.lineno,
    }
    _fh.write(json.dumps(rec) + "\n")


def pytest_unconfigure(config):
    if _fh:
        _fh.close()
