"""
Shared helpers for the n_jobs review benchmarks.

Import this module before joblib or skforecast and call `set_affinity` first,
so that the process (and the loky workers it spawns, which inherit the mask)
only sees the selected logical processors.

Windows, Linux and macOS. The affinity (emulated smaller machines) is only
available on Windows; on Linux and macOS the scripts run on the whole machine.
Environment variables: `NJOBS_RESULTS_DIR` (output folder, default `results/`)
and `NJOBS_REPO_ROOT` (repository that skforecast must be imported from,
default the repository that contains this folder).
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("NJOBS_REPO_ROOT", HERE.parents[1])).resolve()
RESULTS_DIR = Path(os.environ.get("NJOBS_RESULTS_DIR", HERE / "results"))

N_REPS = 5
N_WARMUP = 1


# Topology
# ==============================================================================
def cpu_topology() -> list[dict]:
    """
    Physical cores of the machine with their efficiency class and logical
    processors. Windows: `GetLogicalProcessorInformationEx`. Linux: `/sys`
    (no efficiency classes: every core gets class 0). macOS: `sysctl`
    performance levels (level 0, the performance cores, gets the highest
    class); the logical ids are not known, so they are numbered in order.

    Returns
    -------
    cores : list of dict
        One dict per physical core: `core`, `efficiency_class` (higher is
        faster on hybrid CPUs), `logical` (list of logical processor ids)
        and `smt` (bool).
    """

    if sys.platform.startswith("linux"):
        return _cpu_topology_linux()
    if sys.platform == "darwin":
        return _cpu_topology_macos()
    if sys.platform != "win32":
        raise RuntimeError(f"cpu_topology is not implemented for {sys.platform}.")

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    func = kernel32.GetLogicalProcessorInformationEx
    func.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    func.restype = wintypes.BOOL

    relation_processor_core = 0
    length = wintypes.DWORD(0)
    func(relation_processor_core, None, ctypes.byref(length))
    buffer = ctypes.create_string_buffer(length.value)
    if not func(relation_processor_core, buffer, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())

    raw = buffer.raw[: length.value]
    cores = []
    offset = 0
    while offset < len(raw):
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        flags = raw[offset + 8]
        efficiency_class = raw[offset + 9]
        group_count = int.from_bytes(raw[offset + 30 : offset + 32], "little")
        logical = []
        for g in range(group_count):
            base = offset + 32 + g * 16
            mask = int.from_bytes(raw[base : base + 8], "little")
            group = int.from_bytes(raw[base + 8 : base + 10], "little")
            logical.extend(64 * group + bit for bit in range(64) if mask >> bit & 1)
        cores.append(
            {
                "core": len(cores),
                "efficiency_class": efficiency_class,
                "logical": logical,
                "smt": bool(flags & 1),
            }
        )
        offset += size

    return cores


def _cpu_topology_linux() -> list[dict]:
    groups: dict[tuple[int, int], list[int]] = {}
    for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        topo = d / "topology"
        if not topo.exists():
            continue
        key = (
            int((topo / "physical_package_id").read_text()),
            int((topo / "core_id").read_text()),
        )
        groups.setdefault(key, []).append(int(d.name[3:]))

    return [
        {"core": i, "efficiency_class": 0, "logical": sorted(cpus), "smt": len(cpus) > 1}
        for i, (_, cpus) in enumerate(sorted(groups.items()))
    ]


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _cpu_topology_macos() -> list[dict]:
    try:
        n_levels = int(_sysctl("hw.nperflevels"))
    except ValueError:
        n_levels = 0
    levels = []
    if n_levels > 0:
        for level in range(n_levels):
            levels.append(
                (
                    n_levels - 1 - level,
                    int(_sysctl(f"hw.perflevel{level}.physicalcpu")),
                    int(_sysctl(f"hw.perflevel{level}.logicalcpu")),
                )
            )
    else:
        levels.append(
            (0, psutil.cpu_count(logical=False), psutil.cpu_count(logical=True))
        )

    cores = []
    next_logical = 0
    for efficiency_class, n_physical, n_logical in levels:
        per_core = max(1, n_logical // max(1, n_physical))
        for _ in range(n_physical):
            logical = list(range(next_logical, next_logical + per_core))
            next_logical += per_core
            cores.append(
                {
                    "core": len(cores),
                    "efficiency_class": efficiency_class,
                    "logical": logical,
                    "smt": per_core > 1,
                }
            )

    return cores


def affinity_set(n_logical: int | None) -> list[int] | None:
    """
    Logical processors that emulate a smaller machine on this hybrid CPU.

    The emulated machines are typical laptops with hyperthreading: `n_logical`
    logical processors on `n_logical // 2` performance cores, both threads of
    each core. The highest-numbered performance cores are used to keep away
    from core 0, which serves most system interrupts.

    Parameters
    ----------
    n_logical : int, None
        Number of logical processors. `None` (or the total number of logical
        processors) means no restriction.

    Returns
    -------
    cpus : list of int, None
        Logical processor ids, or `None` for no restriction.
    """

    if n_logical is None or n_logical >= psutil.cpu_count(logical=True):
        return None

    cores = cpu_topology()
    top_class = max(c["efficiency_class"] for c in cores)
    p_cores = [c for c in cores if c["efficiency_class"] == top_class and c["smt"]]
    n_cores = n_logical // 2
    if n_logical % 2 or n_cores > len(p_cores):
        raise ValueError(
            f"Cannot build {n_logical} logical processors from "
            f"{len(p_cores)} performance cores with 2 threads each."
        )
    selected = p_cores[-n_cores:]

    return sorted(cpu for c in selected for cpu in c["logical"])


def set_affinity(n_logical: int | None) -> list[int]:
    """
    Restrict the current process to `affinity_set(n_logical)`. Must be called
    before importing joblib, so that the CPU count and the workers see it.

    Returns
    -------
    cpus : list of int
        Logical processors the process can run on after the call.
    """

    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        # macOS: psutil cannot read or set the affinity.
        if n_logical is not None and n_logical < psutil.cpu_count(logical=True):
            raise RuntimeError(f"The affinity cannot be set on {sys.platform}.")
        return list(range(psutil.cpu_count(logical=True)))

    cpus = affinity_set(n_logical)
    if cpus is not None:
        proc.cpu_affinity(cpus)

    return proc.cpu_affinity()


def physical_cores_in_affinity() -> int:
    """
    Number of physical cores with at least one logical processor in the
    affinity mask of the current process.
    """

    proc = psutil.Process()
    if not hasattr(proc, "cpu_affinity"):
        return psutil.cpu_count(logical=False)
    allowed = set(proc.cpu_affinity())

    return sum(1 for c in cpu_topology() if allowed & set(c["logical"]))


# Machine information
# ==============================================================================
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _cpu_freq() -> dict | None:
    # Not available on some platforms (for example, Apple silicon).
    try:
        freq = psutil.cpu_freq()
    except Exception:  # noqa: BLE001
        return None
    return None if freq is None else freq._asdict()


def machine_info() -> dict:
    """
    CPU, topology, memory, operating system, power settings and versions of
    the packages involved.
    """

    import importlib

    cores = cpu_topology()
    classes = {}
    for c in cores:
        key = f"class_{c['efficiency_class']}"
        classes.setdefault(key, {"cores": 0, "logical": 0})
        classes[key]["cores"] += 1
        classes[key]["logical"] += len(c["logical"])

    if sys.platform == "win32":
        cpu_name = _run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        power_scheme = _run(["powercfg", "/getactivescheme"])
    elif sys.platform == "darwin":
        cpu_name = _sysctl("machdep.cpu.brand_string")
        power_scheme = _run(["pmset", "-g", "custom"])
    else:
        cpu_name = "unknown"
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
        power_scheme = None
    battery = psutil.sensors_battery()
    versions = {"python": platform.python_version()}
    for name in [
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "joblib",
        "numba",
        "lightgbm",
        "xgboost",
        "catboost",
        "statsmodels",
        "psutil",
        "threadpoolctl",
        "skforecast",
    ]:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"missing: {exc}"

    import joblib

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu_name,
        "platform": platform.platform(),
        "os_version": platform.version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "joblib_cpu_count": joblib.cpu_count(),
        "joblib_cpu_count_physical": joblib.cpu_count(only_physical_cores=True),
        "efficiency_classes": classes,
        "topology": cores,
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "kernel": platform.release(),
        "power_scheme": power_scheme,
        "ac_power": None if battery is None else battery.power_plugged,
        "battery_percent": None if battery is None else battery.percent,
        "cpu_freq_mhz": _cpu_freq(),
        "versions": versions,
    }


def check_repo_skforecast() -> None:
    """
    Abort if skforecast is not imported from the repository (the conda env
    has a stale non-editable copy in site-packages).
    """

    import skforecast

    path = Path(skforecast.__file__).resolve()
    if REPO_ROOT not in path.parents:
        raise SystemExit(
            f"skforecast imported from {path}, not from {REPO_ROOT}. "
            f"Set PYTHONPATH to the repository root."
        )


# Results
# ==============================================================================
class CsvLog:
    """
    Append-only CSV writer that flushes every row, so an interrupted run
    keeps its data, and knows which cells are already complete (resume).
    """

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.path = Path(path)
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # A cell ends with its stored predictions or a timeout.
                    if row.get("phase") in ("preds", "timeout", "done"):
                        self.done.add(row["cell_id"])
        new = not self.path.exists()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fields, extrasaction="ignore")
        if new:
            self._w.writeheader()
            self._f.flush()

    def write(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def shuffled(items: list, seed: int = 20261107) -> list:
    """
    Seeded shuffle of the cells of a block, so slow drift (temperature,
    background load) does not line up with one configuration.
    """

    items = list(items)
    random.Random(seed).shuffle(items)

    return items


def n_jobs_values(cpu_count: int, physical: int, extra: tuple = ()) -> list[tuple]:
    """
    Values of `n_jobs` compared in every cell, as `(label, value)`: 1,
    'auto', cpu_count - 1, -1 and the number of physical cores, plus `extra`
    (pairs). A value equal to a previous numeric one is skipped; 'auto' is
    always measured on its own.
    """

    values = [
        ("1", 1),
        ("auto", "auto"),
        ("cpu-1", max(1, cpu_count - 1)),
        ("-1", -1),
        ("physical", physical),
        *extra,
    ]
    out, seen = [], set()
    for label, value in values:
        key = cpu_count if value == -1 else value
        if value == "auto" or key not in seen:
            out.append((label, value))
            if value != "auto":
                seen.add(key)

    return out


def md_table(df, index: bool = True) -> str:
    """
    Markdown table of a DataFrame without the optional `tabulate` dependency
    of `DataFrame.to_markdown`.
    """

    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v != v else str(v) for v in row) + " |")

    return "\n".join(lines)


def env_summary() -> dict:
    return {
        "affinity": len(psutil.Process().cpu_affinity()),
        "pid": os.getpid(),
    }


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
