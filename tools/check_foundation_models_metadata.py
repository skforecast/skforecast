"""
Check the foundation model metadata of skforecast against the Hugging Face Hub.

The license, the weights repository and the gating of every foundation model
are declared by hand in `skforecast/foundation` (`_MODEL_LICENSES` and the
adapter class attributes), so they go stale when a provider changes its terms
or a backend switches to other weights. This script compares what
`get_model_info` reports with the model cards published on the Hub:

- `weights_repo_id` exists on the Hub.
- `license` is the license of the model card (its `license_name` when the
  card declares `license: other`).
- `requires_hf_auth` matches the gating of the repository.
- `license_url` is reachable.

It needs network access, so it is not part of the unit tests. It runs on a
schedule in `.github/workflows/foundation-models-metadata.yml`.

Usage:
    python tools/check_foundation_models_metadata.py

Exits with code 1 if any mismatch is found.
"""

from __future__ import annotations
import json
import os
import sys
import urllib.error
import urllib.request

from skforecast.foundation import get_model_info, list_adapters

HUB_API_URL = "https://huggingface.co/api/models/{repo_id}"
TIMEOUT = 30

# Model IDs checked in addition to the default one of each adapter: the other
# checkpoints the adapters can load. Add the new ones when a provider
# publishes them.
EXTRA_MODEL_IDS = (
    "amazon/chronos-2",
    "autogluon/chronos-2",
    "autogluon/chronos-2-synth",
    "theforecastingcompany/t0-beta",
    "Synthefy/Nori-30M",
    "Synthefy/Nori-100M",
)


def _request(url: str, method: str = "GET") -> bytes:
    """
    Return the body of `url`. Uses the `HF_TOKEN` environment variable, when
    set, to authenticate against the Hub.
    """

    headers = {"User-Agent": "skforecast-metadata-check"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def check_model(model_id: str) -> list[str]:
    """
    Compare the metadata skforecast reports for `model_id` with the Hub.

    Parameters
    ----------
    model_id : str
        Model ID to check.

    Returns
    -------
    errors : list
        Description of every mismatch found. Empty when everything matches.

    """

    info = get_model_info(model_id)
    errors = []

    try:
        card = json.loads(_request(HUB_API_URL.format(repo_id=info.weights_repo_id)))
    except urllib.error.HTTPError as exc:
        # The Hub answers 401 both for private and for non-existent repos.
        return [
            f"weights_repo_id '{info.weights_repo_id}' not found on the Hub "
            f"(HTTP {exc.code})."
        ]

    card_data = card.get("cardData") or {}
    hub_license = card_data.get("license")
    if hub_license == "other":
        hub_license = card_data.get("license_name")
    if str(hub_license).lower() != info.license.lower():
        errors.append(
            f"license is '{info.license}' in skforecast and '{hub_license}' "
            f"in the model card of '{info.weights_repo_id}'."
        )

    hub_gated = bool(card.get("gated"))
    if hub_gated != info.requires_hf_auth:
        errors.append(
            f"requires_hf_auth is {info.requires_hf_auth} in skforecast and "
            f"the repository '{info.weights_repo_id}' has gated={hub_gated}."
        )

    try:
        _request(info.license_url, method="HEAD")
    except urllib.error.HTTPError as exc:
        errors.append(
            f"license_url '{info.license_url}' is not reachable "
            f"(HTTP {exc.code})."
        )

    return errors


def main() -> int:
    model_ids = [info.model_id for info in list_adapters()]
    model_ids.extend(EXTRA_MODEL_IDS)

    n_errors = 0
    for model_id in model_ids:
        errors = check_model(model_id)
        n_errors += len(errors)
        print(f"{'FAIL' if errors else 'OK  '} {model_id}")
        for error in errors:
            print(f"       {error}")

    if n_errors:
        print(
            f"\n{n_errors} mismatch(es). Update `_MODEL_LICENSES` in "
            "skforecast/foundation/_utils.py or the adapter class attributes "
            "in skforecast/foundation/_adapters.py."
        )
        return 1

    print("\nFoundation model metadata matches the Hugging Face Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
