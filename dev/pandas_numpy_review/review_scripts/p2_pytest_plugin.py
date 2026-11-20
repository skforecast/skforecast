"""
pytest plugin that applies the prototype of P2 (p2_proto.proposal) during a
whole test session, to list the existing tests the change would break:

    PYTHONPATH=/home/user/sk_perf:<this folder> python -m pytest -p p2_pytest_plugin <test files> -q

Set P2_CONDITION=strict to use `p2_proto.strict_condition`.
"""
import os
import p2_proto as P

_cond = P.strict_condition if os.environ.get("P2_CONDITION") == "strict" else P.condition
_ctx = P.proposal(_cond)
_ctx.__enter__()


def pytest_report_header(config):
    return f"P2 prototype active (condition: {_cond.__name__})"


def pytest_terminal_summary(terminalreporter):
    mixed = [c for c in P.calls if len(c[1]) > 1 or c[1][0] not in ("float64", "int64")]
    terminalreporter.write_line(
        f"P2 prototype: {len(P.calls)} exog conversions, {len(mixed)} of them on "
        f"frames that are not all float64 or all int64; dtype sets: "
        f"{sorted({c[1] for c in mixed})}"
    )
