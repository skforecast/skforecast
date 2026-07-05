# Evaluation: Grouping prediction-interval arguments in backtesting functions

**Status:** Rejected — not worth it.
**Scope:** `backtesting_forecaster`, `backtesting_forecaster_multiseries` (and their private
helpers) in [`skforecast/model_selection/_validation.py`](../skforecast/model_selection/_validation.py).

---

## Problem definition

The backtesting functions expose several prediction-interval parameters as separate
arguments:

- `interval`
- `interval_method` (`'bootstrapping'` / `'conformal'`)
- `n_boot`
- `use_in_sample_residuals`
- `use_binned_residuals`
- `random_state`

The idea was to reduce the argument count by extracting the "how the interval is
estimated" knobs into a single reusable object, mirroring how `cv` already bundles the
splitting configuration via `TimeSeriesFold`.

Two motivations:

1. **Ergonomics** — fewer top-level arguments, one reusable interval configuration that
   can be passed across several calls.
2. **A latent inconsistency** — the default `interval_method` differs between functions
   (`'bootstrapping'` for single series, `'conformal'` for multiseries).

---

## Possible solutions considered

### Option A — Passive config class (data bag)

A class that only stores the 5 estimation knobs, e.g. `IntervalConfig(method=..., n_boot=...,
use_in_sample_residuals=..., use_binned_residuals=..., random_state=...)`, while keeping
`interval` as a top-level argument.

- Pro: groups parameters that always travel together.
- Con: it is a *passive* object — it holds values but does no work.

### Option B — Behavioral strategy object

A class that owns the estimation logic, e.g. `BootstrapInterval(...)` / `ConformalInterval(...)`,
each exposing a method the backtester calls (analogous to `cv.split()`).

- Pro: this is the only variant that is genuinely scikit-learn idiomatic (see below).
- Con: largest change; touches the prediction code paths, not just signatures.

### Option C — Reusable external dict

Pass the knobs as a plain `dict` that can be unpacked (`**interval_kwargs`).

- Pro: trivial to reuse.
- Con: loses validation, defaults, type hints, and autocomplete. No real scikit-learn
  precedent for grouping a single call's configuration this way.

---

## The scikit-learn philosophy

The decision hinges on what scikit-learn actually prescribes
([developer guide](https://scikit-learn.org/stable/developers/develop.html),
[API design paper, Buitinck et al. 2013](https://arxiv.org/abs/1309.0238)):

1. **Flat keyword arguments are the default.** *"Ideally, the arguments accepted by
   `__init__` should all be keyword arguments with a default value."* Estimators routinely
   carry 15+ flat parameters and do **not** group them into sub-config objects.

2. **Non-proliferation of classes.** A core stated design principle. A class whose only job
   is to hold a few values for one call is exactly what this principle discourages.

3. **Objects-as-arguments only when the object encapsulates behavior.** Every object
   scikit-learn passes as a single argument *does work*:
   - `cv` splitters (`KFold`, `TimeSeriesSplit`) → have `.split()`
   - estimators / `Pipeline` → have `.fit()` / `.transform()`
   - scorers (`make_scorer`) → callable

   The `cv` precedent in skforecast is idiomatic **because `TimeSeriesFold` is behavioral**
   (`.split()`), not because it "groups the split arguments."

4. **Dicts are used for value collections to search/route** (`param_grid`,
   `param_distributions`, metadata routing `**params`) — never to bundle a single function's
   configuration.

**Conclusion from the philosophy:** the dict (C) and the passive config class (A) are the
*least* scikit-learn-like options. Only the behavioral strategy object (B) would be
idiomatic — and that is the most invasive change of all.

---

## Final decision: not worth it

**Rejected.** None of the options clears the bar.

- **It is a breaking API change.** Any of these options alters the public signatures of the
  backtesting functions. Existing user code that passes `interval_method`, `n_boot`,
  `use_in_sample_residuals`, `use_binned_residuals`, or `random_state` as keyword arguments
  would break, requiring a deprecation cycle and migration for every user.

- **There is no clear benefit.** The current flat-argument design is already the
  scikit-learn-idiomatic default. Reducing the argument count is cosmetic, not functional —
  it changes nothing about behavior, correctness, or capability.

- **The two idiomatic alternatives both lose.** The passive config class (A) and the dict (C)
  contradict scikit-learn conventions (non-proliferation of classes; dicts are for
  search/routing). The only convention-aligned variant (B) is a large refactor of the
  prediction paths whose payoff does not justify the churn and breakage.

The cost (breaking change + deprecation cycle + refactor) is real; the benefit (slightly
shorter signatures) is marginal and purely aesthetic.

### The one issue worth addressing separately

The genuine wart — the **`interval_method` default divergence** (`'bootstrapping'` for single
series vs `'conformal'` for multiseries) — does **not** require any of this. It can be
unified (or at least documented) on its own, with no new class and no signature grouping.
