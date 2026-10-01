---
paths:
  - "skforecast/foundation/**"
---

# Foundation models

Keep the adapters thin. Generic handling of heterogeneous multi-series input (grouping series by covariate signature, aligning `context` with `context_exog`, reindexing future `exog` to the horizon, per-series quirks) belongs once in `FoundationModel` (`skforecast/foundation/_foundation_model.py`), driven by small adapter class attributes that declare backend constraints (for example whether the backend needs homogeneous covariate keys in a batch).

The adapter contract lives in `skforecast/foundation/_adapter_base.py` (`_AdapterBase`): required capability class attributes, checked when the class is defined, and the `fit`/`predict`/`get_params`/`set_params` methods. New adapters subclass it in `_adapters.py`, declare every attribute in `_REQUIRED_CLASS_ATTRIBUTES` and register in `_ADAPTER_REGISTRY`; change the base only when the contract itself changes.

Adapters only translate already-normalized per-series inputs into the backend call. Do not add per-adapter loops, padding or grouping code; when proposing designs for this module, present the centralized option as the recommendation and treat per-adapter handling as something to remove.
