"""
Check that a foundation model adapter works with its backend library.

Creates a virtual environment with uv, installs skforecast from this checkout
together with the backend of one adapter, runs the tests in
`tools/foundation_compat/tests` inside it and writes the outcome to
`<results-dir>/<adapter>.json`.

The backend is installed as a user installs it today (`backend_package` in
`adapters.toml`); skforecast does not pin its backends. The installed version
is compared with the latest one on PyPI: if the resolver held the backend
back, the run fails, because it would be testing an old version.

Two levels of tests:

- `contract`: the backend still exposes the symbols and arguments the adapter
  uses. No weights are downloaded.
- `smoke`: forecasts with the real weights through the public API. Skipped
  when the contract fails or when a secret the adapter needs is not set.

It only needs the standard library (Python >=3.11) and `uv` on the PATH, so
the same command reproduces locally what the workflow
`.github/workflows/foundation-backends.yml` does.

Usage:
    python tools/foundation_compat/run.py --adapter t0
    python tools/foundation_compat/run.py --adapter moirai --level contract

Exits with code 1 if the installation or a test fails, or if the backend was
held back.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

COMPAT_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMPAT_DIR.parents[1]
TESTS_DIR = COMPAT_DIR / "tests"
PYPI_URL = "https://pypi.org/pypi/{package}/json"
LEVELS = ("contract", "smoke")
# Packages whose installed version is reported next to the one of the backend.
REPORTED_PACKAGES = ("torch", "numpy", "scipy", "pandas", "huggingface_hub")
MAX_ERROR_LENGTH = 300


def load_adapters() -> dict:
    """
    Return the content of `adapters.toml`.
    """

    with open(COMPAT_DIR / "adapters.toml", "rb") as f:
        return tomllib.load(f)


def distribution_name(requirement: str) -> str:
    """
    Return the name of the distribution of a requirement, without extras or
    version specifiers (`'timesfm[torch]>=3'` -> `'timesfm'`).
    """

    return re.match(r"[A-Za-z0-9._-]+", requirement).group(0)


def version_key(version: str) -> tuple[int, ...]:
    """
    Return the numeric components of a version, to compare two releases.
    """

    return tuple(int(part) for part in re.findall(r"\d+", version))


def latest_version_on_pypi(package: str) -> str | None:
    """
    Return the latest release of `package` on PyPI (pre-releases excluded), or
    `None` if PyPI cannot be reached.
    """

    try:
        url = PYPI_URL.format(package=package)
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)["info"]["version"]
    except Exception as exc:
        print(f"Could not get the latest version of {package} from PyPI: {exc}")
        return None


def short_error(text: str) -> str:
    """
    Return the first line of `text` that explains an error, truncated.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if "error" in line.lower() or line.startswith("×"):
            return line[:MAX_ERROR_LENGTH]

    return lines[-1][:MAX_ERROR_LENGTH] if lines else ""


def install(venv_python: Path, config: dict, work_dir: Path) -> tuple[bool, str]:
    """
    Install skforecast, pytest and the backend in one resolution.

    Parameters
    ----------
    venv_python : Path
        Python interpreter of the virtual environment.
    config : dict
        Entry of the adapter in `adapters.toml`.
    work_dir : Path
        Directory where the overrides file is written.

    Returns
    -------
    ok : bool
        Whether the installation succeeded.
    output : str
        Output of uv.

    """

    # Only the core dependencies of skforecast: the `test` extra brings
    # packages the adapters do not need and that can hold the backend back.
    command = [
        "uv", "pip", "install",
        "--python", str(venv_python),
        "--torch-backend", "cpu",
        "-e", str(REPO_ROOT),
        "pytest",
        config["backend_package"],
    ]
    if config.get("overrides"):
        overrides_file = work_dir / "overrides.txt"
        overrides_file.write_text("\n".join(config["overrides"]) + "\n")
        command.extend(["--override", str(overrides_file)])

    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, flush=True)

    return result.returncode == 0, output


def installed_versions(venv_python: Path, packages: list[str]) -> dict:
    """
    Return the version of each package installed in the virtual environment
    (`None` when it is not installed).
    """

    code = (
        "import json, sys\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "versions = {}\n"
        "for package in sys.argv[1:]:\n"
        "    try:\n"
        "        versions[package] = version(package)\n"
        "    except PackageNotFoundError:\n"
        "        versions[package] = None\n"
        "print(json.dumps(versions))\n"
    )
    result = subprocess.run(
        [str(venv_python), "-c", code, *packages],
        capture_output=True, text=True, check=True,
    )

    return json.loads(result.stdout)


def run_tests(
    venv_python: Path,
    adapter: str,
    level: str,
    work_dir: Path,
    env: dict,
) -> dict:
    """
    Run the tests of one level inside the virtual environment.

    Parameters
    ----------
    venv_python : Path
        Python interpreter of the virtual environment.
    adapter : str
        Name of the entry in `adapters.toml`.
    level : str
        `'contract'` or `'smoke'`.
    work_dir : Path
        Directory where the JUnit report is written.
    env : dict
        Environment variables of the pytest process.

    Returns
    -------
    result : dict
        `status` (`'passed'` or `'failed'`), number of `tests`, `errors` (one
        line per failed test) and `duration` in seconds.

    """

    report = work_dir / f"{level}.xml"
    command = [
        str(venv_python), "-m", "pytest",
        str(TESTS_DIR / f"test_{level}.py"),
        "--backend", adapter,
        "--junitxml", str(report),
        "-q", "-p", "no:cacheprovider",
    ]
    print("$", " ".join(command), flush=True)
    start = time.monotonic()
    returncode = subprocess.run(command, cwd=REPO_ROOT, env=env).returncode
    duration = round(time.monotonic() - start, 1)

    n_tests = 0
    errors = []
    if report.exists():
        for testcase in ET.parse(report).getroot().iter("testcase"):
            n_tests += 1
            for outcome in testcase.findall("failure") + testcase.findall("error"):
                message = (outcome.get("message") or outcome.text or "").strip()
                first_line = message.splitlines()[0] if message else "failed"
                errors.append(
                    f"{testcase.get('name')}: {first_line}"[:MAX_ERROR_LENGTH]
                )

    if returncode != 0 and not errors:
        errors.append(f"pytest exited with code {returncode}")

    return {
        "status": "passed" if returncode == 0 else "failed",
        "tests": n_tests,
        "errors": errors,
        "duration": duration,
    }


def main() -> int:
    adapters = load_adapters()

    parser = argparse.ArgumentParser(
        description="Check that a foundation model adapter works with its backend."
    )
    parser.add_argument("--adapter", required=True, choices=sorted(adapters))
    parser.add_argument("--level", default="all", choices=(*LEVELS, "all"))
    parser.add_argument(
        "--results-dir", default="results", help="Directory of the JSON result."
    )
    parser.add_argument(
        "--python", default="3.12", help="Python version of the environment."
    )
    args = parser.parse_args()

    config = adapters[args.adapter]
    package = distribution_name(config["backend_package"])
    levels = LEVELS if args.level == "all" else (args.level,)
    start = time.monotonic()

    result = {
        "adapter": args.adapter,
        "adapter_class": config["adapter"],
        "backend": package,
        "model_id": config["model_id"],
        "overrides": config.get("overrides", []),
        "versions": {},
        "latest_on_pypi": None,
        "held_back": False,
        "install": {"status": "failed", "error": ""},
        "levels": {level: {"status": "not run", "errors": []} for level in LEVELS},
    }

    env = os.environ.copy()
    # Progress bars make the logs of the job unreadable.
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TQDM_DISABLE"] = "1"

    with tempfile.TemporaryDirectory(prefix="foundation_compat_") as tmp:
        work_dir = Path(tmp)
        venv_dir = work_dir / "venv"
        if os.name == "nt":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        subprocess.run(
            ["uv", "venv", str(venv_dir), "--python", args.python], check=True
        )
        installed, output = install(venv_python, config, work_dir)

        if installed:
            result["install"] = {"status": "passed", "error": ""}
            result["versions"] = installed_versions(
                venv_python, [package, *REPORTED_PACKAGES]
            )
            result["latest_on_pypi"] = latest_version_on_pypi(package)
            if result["latest_on_pypi"]:
                result["held_back"] = (
                    version_key(result["versions"][package])
                    < version_key(result["latest_on_pypi"])
                )

            missing_secrets = [
                secret for secret in config.get("secrets", [])
                if not os.environ.get(secret)
            ]
            for level in levels:
                contract_failed = result["levels"]["contract"]["status"] == "failed"
                if level == "smoke" and contract_failed:
                    continue
                elif level == "smoke" and missing_secrets:
                    result["levels"][level] = {
                        "status": "skipped",
                        "errors": [f"no {', '.join(missing_secrets)}"],
                    }
                else:
                    result["levels"][level] = run_tests(
                        venv_python, args.adapter, level, work_dir, env
                    )
        else:
            result["install"]["error"] = short_error(output)

    result["duration"] = round(time.monotonic() - start, 1)

    failed = (
        result["install"]["status"] == "failed"
        or result["held_back"]
        or any(level["status"] == "failed" for level in result["levels"].values())
    )
    result["status"] = "failed" if failed else "passed"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / f"{args.adapter}.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nResult written to {result_file}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
