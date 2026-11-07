#!/usr/bin/env bash
# Create an isolated environment to review one foundation adapter against the
# latest release of its backend.
#
# Usage:
#     bash make_env.sh <key> "<backend requirement>" [python_version] [extra requirements...]
#
# Example:
#     bash make_env.sh chronos "chronos-forecasting" 3.12
#
# The environment lives outside the repository (~/.venvs/fm-review-<key>),
# installs CPU torch, skforecast from this checkout in editable mode, pytest and
# the backend in a single resolution, and writes `pip freeze` next to the logs.
set -euo pipefail

KEY="$1"
BACKEND="$2"
PYTHON_VERSION="${3:-3.12}"
shift 3 || shift $#

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
VENV="${HOME}/.venvs/fm-review-${KEY}"
OUT_DIR="${REPO_ROOT}/dev/foundation_adapters_review/evidence/${KEY}"
mkdir -p "${OUT_DIR}"

uv venv "${VENV}" --python "${PYTHON_VERSION}" --clear
uv pip install --python "${VENV}/bin/python" --torch-backend cpu \
    -e "${REPO_ROOT}" pytest "${BACKEND}" "$@"
uv pip freeze --python "${VENV}/bin/python" | grep -v "^-e " > "${OUT_DIR}/requirements-freeze.txt"
echo "Environment ready: ${VENV}"
