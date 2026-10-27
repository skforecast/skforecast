"""
Tests of affected.py (scope of the verify skill). Run them with
`python -m pytest .claude/hooks .claude/skills -q -p no:cacheprovider`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).parent
PROJECT_DIR = SKILL_DIR.parent.parent.parent

sys.path.insert(0, str(SKILL_DIR))
try:
    import affected
finally:
    sys.path.remove(str(SKILL_DIR))


@pytest.mark.parametrize(
    "path, category",
    [
        ("dev/scratch.py", "out_of_scope"),
        ("benchmarks/run_benchmarks.py", "out_of_scope"),
        ("CITATION.cff", "out_of_scope"),
        ("skforecast/recursive/_forecaster_recursive.py", "package"),
        ("skforecast/recursive/tests/tests_forecaster_recursive/test_fit.py", "tests"),
        ("tests/test_skforecast_version.py", "tests"),
        ("docs/user_guides/backtesting.ipynb", "notebooks"),
        ("docs/user_guides/index.md", "docs"),
        ("README.md", "docs"),
        ("mkdocs.yml", "docs"),
        ("docs/llms-full.txt", "ai_context"),
        ("tools/ai/llms-base.txt", "ai_context"),
        ("skills/feature-selection/SKILL.md", "ai_context"),
        ("CLAUDE.md", "ai_context"),
        (".claude/hooks/ruff_check.py", "harness"),
        ("tools/docs/hooks/page_meta.yml", "docs_hooks"),
        ("tools/foundation_compat/run.py", "foundation_compat"),
        ("pyproject.toml", "deps"),
        (".github/workflows/unit-tests.yml", "ci"),
        ("tools/check_case_in_file_names.py", "other"),
    ],
)
def test_classify(path, category):
    assert affected.classify(path) == category


@pytest.mark.parametrize(
    "module, expected",
    [
        ("skforecast/recursive/_forecaster_recursive.py",
         "skforecast/recursive/tests/tests_forecaster_recursive"),
        ("skforecast/utils/utils.py", "skforecast/utils/tests/tests_utils"),
        ("skforecast/stats/arima/_auto_arima.py",
         "skforecast/stats/arima/tests/test_auto_arima.py"),
        ("skforecast/experimental/_splitter.py",
         "skforecast/experimental/tests/test_splitter.py"),
        ("skforecast/foundation/_adapters.py",
         "skforecast/foundation/tests/tests_foundation_models"),
        ("skforecast/stats/seasonal/_differencing.py",
         "skforecast/stats/seasonal/tests/test_seasonal.py"),
        ("skforecast/__init__.py", "tests/test_skforecast_version.py"),
    ],
)
def test_tests_for_module(module, expected):
    assert affected.tests_for_module(module, str(PROJECT_DIR)) == (expected, "")


def test_every_package_module_maps_to_existing_tests():
    """Catches new modules whose tests break the naming convention."""
    files = subprocess.run(
        ["git", "ls-files", "skforecast/*.py"],
        cwd=PROJECT_DIR, capture_output=True, text=True, check=True
    ).stdout.split()
    unmapped = []
    for path in files:
        if "/tests/" in path or path.endswith("/__init__.py"):
            continue
        test_path, flag = affected.tests_for_module(path, str(PROJECT_DIR))
        assert test_path is None or (PROJECT_DIR / test_path).exists(), path
        if flag:
            unmapped.append(path)
    # Add the module to EXCEPTIONS (or here, if it truly has no tests).
    assert unmapped == ["skforecast/experimental/_experimental.py"]


def test_build_plan_nothing_to_verify_when_only_out_of_scope_files():
    plan = affected.build_plan(["dev/scratch.py", "benchmarks/x.py"], str(PROJECT_DIR))
    assert plan["nothing_to_verify"]
    assert plan["lint"] == []
    assert affected.format_plan(plan, "files given").endswith("nothing to verify")


def test_build_plan_runs_each_test_once():
    folder = "skforecast/recursive/tests/tests_forecaster_recursive"
    plan = affected.build_plan(
        [
            "skforecast/recursive/_forecaster_recursive.py",
            f"{folder}/test_fit.py",
            f"{folder}/test_predict.py",
            "dev/scratch.py",
        ],
        str(PROJECT_DIR),
    )
    assert plan["pytest"] == [folder]
    assert "dev/scratch.py" not in plan["lint"]


def test_build_plan_checks_by_category():
    plan = affected.build_plan(
        [
            "skforecast/recursive/__init__.py",
            ".claude/hooks/ruff_check.py",
            "docs/user_guides/backtesting.ipynb",
            "pyproject.toml",
        ],
        str(PROJECT_DIR),
    )
    checks = "\n".join(plan["checks"])
    assert "generate_ai_context_files.py --check" in checks
    assert ".claude/hooks .claude/skills" in checks
    assert "page_meta.py" in checks
    # The notebook runner takes paths relative to docs/.
    assert "execute_notebooks.py --check user_guides/backtesting.ipynb" in checks
    assert any("pyproject.toml" in note for note in plan["notes"])
    assert any("release-note" in note for note in plan["notes"])


def test_build_plan_flags_shared_code():
    plan = affected.build_plan(["skforecast/utils/utils.py"], str(PROJECT_DIR))
    assert plan["pytest"] == ["skforecast/utils/tests/tests_utils"]
    assert any("shared code" in note for note in plan["notes"])


def git(args, cwd):
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com"] + args,
        cwd=cwd, check=True, capture_output=True,
    )


def test_changed_files_since_release_branch(tmp_path):
    (tmp_path / "skforecast").mkdir()
    (tmp_path / "skforecast" / "__init__.py").write_text('__version__ = "1.2.0"\n')
    (tmp_path / "a.py").write_text("a = 1\n")
    git(["init", "-q", "-b", "1.2.x"], tmp_path)
    git(["add", "."], tmp_path)
    git(["commit", "-q", "-m", "base"], tmp_path)
    git(["switch", "-q", "-c", "feature"], tmp_path)
    (tmp_path / "committed.py").write_text("b = 1\n")
    git(["add", "committed.py"], tmp_path)
    git(["commit", "-q", "-m", "feature"], tmp_path)
    (tmp_path / "a.py").write_text("a = 2\n")
    (tmp_path / "untracked.py").write_text("c = 1\n")

    base_desc, files = affected.changed_files(str(tmp_path))
    assert base_desc == "merge base with 1.2.x"
    assert files == ["a.py", "committed.py", "untracked.py"]


def test_changed_files_on_release_branch_with_unpushed_commits(tmp_path):
    origin, clone = tmp_path / "origin", tmp_path / "clone"
    origin.mkdir()
    (origin / "skforecast").mkdir()
    (origin / "skforecast" / "__init__.py").write_text('__version__ = "1.2.0"\n')
    git(["init", "-q", "-b", "1.2.x"], origin)
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "base"], origin)
    git(["clone", "-q", str(origin), str(clone)], tmp_path)
    (clone / "unpushed.py").write_text("a = 1\n")
    git(["add", "unpushed.py"], clone)
    git(["commit", "-q", "-m", "not pushed"], clone)

    base_desc, files = affected.changed_files(str(clone))
    assert base_desc == "merge base with origin/1.2.x"
    assert files == ["unpushed.py"]


def make_repo(path):
    (path / "a.py").write_text("a = 1\n")
    git(["init", "-q", "-b", "main"], path)
    git(["add", "."], path)
    git(["commit", "-q", "-m", "base"], path)


def test_stamp_survives_a_commit_of_the_verified_files(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "a.py").write_text("a = 2\n")
    (tmp_path / "new.py").write_text("b = 1\n")
    assert affected.matching_stamp(str(tmp_path)) is None
    assert affected.write_stamp(str(tmp_path)) is not None
    assert affected.matching_stamp(str(tmp_path)) is not None

    git(["add", "."], tmp_path)
    git(["commit", "-q", "-m", "verified change"], tmp_path)
    assert affected.matching_stamp(str(tmp_path)) is not None
    assert "new.py" not in git_status(tmp_path)  # The real index was committed.


def test_stamp_is_invalidated_by_any_change(tmp_path):
    make_repo(tmp_path)
    affected.write_stamp(str(tmp_path))
    (tmp_path / "untracked.py").write_text("b = 1\n")
    assert affected.matching_stamp(str(tmp_path)) is None
    (tmp_path / "untracked.py").unlink()
    assert affected.matching_stamp(str(tmp_path)) is not None
    (tmp_path / "a.py").write_text("a = 3\n")
    assert affected.matching_stamp(str(tmp_path)) is None


def test_stamp_does_not_touch_the_index(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "a.py").write_text("a = 2\n")
    (tmp_path / "untracked.py").write_text("b = 1\n")
    before = git_status(tmp_path)
    affected.write_stamp(str(tmp_path))
    affected.matching_stamp(str(tmp_path))
    assert git_status(tmp_path) == before == " M a.py\n?? untracked.py\n"


def git_status(path):
    return subprocess.run(
        ["git", "status", "--short"], cwd=path, capture_output=True, text=True,
        check=True,
    ).stdout


def test_main_with_files(capsys):
    code = affected.main(["--files", "dev/x.py", "--root", str(PROJECT_DIR)])
    assert code == 0
    assert "nothing to verify" in capsys.readouterr().out


def test_release_branch_of_this_repository():
    branch = affected.release_branch(str(PROJECT_DIR))
    assert branch is not None and branch.endswith(".x")
    assert os.path.exists(PROJECT_DIR / "skforecast" / "__init__.py")
