---
paths:
  - "skforecast/foundation/**"
---

# Foundation models

Keep the adapters thin. Generic handling of heterogeneous multi-series input (grouping series by covariate signature, aligning `context` with `context_exog`, reindexing future `exog` to the horizon, per-series quirks) belongs once in `FoundationModel` (`skforecast/foundation/_foundation_model.py`), driven by small adapter class attributes that declare backend constraints (for example whether the backend needs homogeneous covariate keys in a batch).

Adapters only translate already-normalized per-series inputs into the backend call. Do not add per-adapter loops, padding or grouping code; when proposing designs for this module, present the centralized option as the recommendation and treat per-adapter handling as something to remove.
