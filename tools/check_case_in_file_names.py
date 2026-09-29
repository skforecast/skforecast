"""
check_case_in_file_names.py

Detect files that break the naming convention. Looks for files that should be
lowercase but contain non-semantic uppercase letters (valid cases such as _X
and _Y for variables are excluded).

Usage:
    python tools/check_case_in_file_names.py
"""
import sys
from pathlib import Path


def check_incorrect_case_in_filenames():
    """
    Find files with incorrect uppercase letters in the skforecast directory.

    Returns
    -------
    issues : list
        Relative paths of the files with issues.

    """
    # Repository root (parent of the tools/ directory)
    repo_root = Path(__file__).resolve().parent.parent
    skforecast_dir = repo_root / "skforecast"

    if not skforecast_dir.exists():
        print(f"❌ Error: directory {skforecast_dir} not found")
        return None

    # Incorrect (non-semantic) patterns
    incorrect_patterns = [
        'fixtures_',  # fixtures must be all lowercase
        'conftest',   # conftest must be all lowercase
    ]

    # Valid exceptions (uppercase letters with semantic meaning)
    valid_uppercase = ['_X', '_Y']

    issues = []
    for pattern in incorrect_patterns:
        for path in skforecast_dir.rglob(f"*{pattern}*.py"):
            filename = path.name
            if any(c.isupper() for c in filename.replace('.py', '')):
                # Skip files whose uppercase letters have a semantic reason
                if not any(valid in filename for valid in valid_uppercase):
                    # Store the relative path for readability
                    relative_path = path.relative_to(repo_root)
                    issues.append(str(relative_path))

    return issues


def main():
    """Main entry point of the script."""
    print("=== Searching for files with INCORRECT uppercase letters ===\n")

    issues = check_incorrect_case_in_filenames()

    if issues is None:
        sys.exit(1)

    if issues:
        print("⚠️  Files with INCORRECT uppercase letters:\n")
        for issue in issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ No issues found")
        sys.exit(0)


if __name__ == "__main__":
    main()
