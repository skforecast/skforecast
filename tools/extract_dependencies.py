# Extract dependencies from pyproject.toml
# ==============================================================================
import os
import tomli
from itertools import chain


def main():
    """
    Extracts dependencies from pyproject.toml and writes them to requirements.txt.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.abspath(os.path.join(script_dir, "..", "pyproject.toml"))
    requirements_path = os.path.join(script_dir, "requirements.txt")

    with open(pyproject_path, mode='rb') as fp:
        pyproject = tomli.load(fp)

    dependencies_optional = [
        v for k, v in pyproject['project']['optional-dependencies'].items() 
        if k != 'docs'
    ]
    dependencies_optional = chain(*dependencies_optional)
    dependencies_optional = list(set(dependencies_optional))
    dependencies = pyproject['project']['dependencies']
    dependencies_all = dependencies + dependencies_optional
    dependencies_all

    with open(requirements_path, mode='w') as fp:
        for dependency in dependencies_all:
            fp.write(f"{dependency}\n")

    print(f"Generated {len(dependencies_all)} dependencies in {requirements_path}")


if __name__ == "__main__":
    main()
