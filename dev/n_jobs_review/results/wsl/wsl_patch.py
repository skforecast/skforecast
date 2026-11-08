"""Adapt the n_jobs study scripts of the WSL copy to Linux (WSL copy only).

- common.py: Linux topology from /sys, CPU name from /proc/cpuinfo, no
  powercfg, lazy wintypes import, results directory from NJOBS_RESULTS_DIR.
- bench.py: --subset runs the reduced cells (those measured with affinity on
  Windows) on the whole machine.
"""

import sys
from pathlib import Path

root = Path(sys.argv[1]) / "dev" / "n_jobs_review"


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (path.name, old)
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {path.name}")


LINUX_TOPOLOGY = '''    if sys.platform.startswith("linux"):
        # The WSL2 guest does not expose the hybrid layout (P and E cores):
        # every core gets efficiency class 0.
        groups: dict[tuple[int, int], list[int]] = {}
        base = Path("/sys/devices/system/cpu")
        for d in base.glob("cpu[0-9]*"):
            topo = d / "topology"
            if not topo.exists():
                continue
            key = (
                int((topo / "physical_package_id").read_text()),
                int((topo / "core_id").read_text()),
            )
            groups.setdefault(key, []).append(int(d.name[3:]))
        return [
            {
                "core": i,
                "efficiency_class": 0,
                "logical": sorted(cpus),
                "smt": len(cpus) > 1,
            }
            for i, (_, cpus) in enumerate(sorted(groups.items()))
        ]
    if sys.platform != "win32":
        raise RuntimeError("cpu_topology is only implemented for Windows and Linux.")
    from ctypes import wintypes
'''

patch(
    root / "common.py",
    [
        ("from ctypes import wintypes\n", ""),
        (
            'RESULTS_DIR = HERE / "results"',
            'RESULTS_DIR = Path(os.environ.get("NJOBS_RESULTS_DIR", HERE / "results"))',
        ),
        (
            '    if sys.platform != "win32":\n'
            '        raise RuntimeError("cpu_topology is only implemented for Windows.")\n',
            LINUX_TOPOLOGY,
        ),
        (
            '        "cpu": cpu_name,',
            '        "cpu": cpu_name if sys.platform == "win32" else _linux_cpu_name(),',
        ),
        (
            '        "power_scheme": _run(["powercfg", "/getactivescheme"]),',
            '        "power_scheme": (\n'
            '            _run(["powercfg", "/getactivescheme"])\n'
            '            if sys.platform == "win32"\n'
            '            else None\n'
            '        ),\n'
            '        "kernel": platform.release(),',
        ),
        (
            "def machine_info() -> dict:",
            "def _linux_cpu_name() -> str:\n"
            "    for line in Path(\"/proc/cpuinfo\").read_text().splitlines():\n"
            "        if line.startswith(\"model name\"):\n"
            "            return line.split(\":\", 1)[1].strip()\n"
            "    return \"unknown\"\n\n\n"
            "def machine_info() -> dict:",
        ),
    ],
)

patch(
    root / "bench.py",
    [
        (
            '    parser.add_argument("--pilot", action="store_true")',
            '    parser.add_argument("--pilot", action="store_true")\n'
            '    parser.add_argument(\n'
            '        "--subset", action="store_true", help="reduced cells, whole machine"\n'
            "    )",
        ),
        (
            "reduced=args.affinity is not None)",
            "reduced=args.affinity is not None or args.subset)",
        ),
    ],
)
