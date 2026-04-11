# conftest.py — test helpers for the new dict-based adapter API
# ==============================================================================
import pandas as pd
from skforecast.foundation._utils import (
    check_preprocess_series_foundation,
    normalize_exog_to_dict,
)


def fit_adapter(adapter, series, exog=None):
    """
    Bridge old-style ``adapter.fit(series=y)`` calls to the new
    dict-based API that adapters now require.

    Mirrors the normalization that ``FoundationModel.fit`` performs.
    """
    series_dict, _ = check_preprocess_series_foundation(series)
    series_names = list(series_dict.keys())
    exog_dict = normalize_exog_to_dict(exog, series_names)
    return adapter.fit(
        series_dict=series_dict,
        exog_dict=exog_dict,
        is_multiple_series=len(series_names) > 1,
    )


def predict_adapter(
    adapter,
    steps,
    exog=None,
    quantiles=None,
    last_window=None,
    last_window_exog=None,
):
    """
    Bridge old-style ``adapter.predict(steps=5, last_window=y)`` calls
    to the new dict-based API.

    Mirrors the normalization that ``FoundationModel.predict`` performs.
    """
    if last_window is not None:
        lw_dict, _ = check_preprocess_series_foundation(last_window)
        series_names = list(lw_dict.keys())
    else:
        lw_dict = None
        series_names = list(adapter._history.keys())

    # Resolve history
    if lw_dict is not None:
        history_dict = {
            name: s.iloc[-adapter.context_length :]
            for name, s in lw_dict.items()
        }
    else:
        history_dict = adapter._history

    # Resolve past exog
    if lw_dict is not None:
        past_exog_dict = normalize_exog_to_dict(last_window_exog, series_names)
        past_exog_dict = {
            name: (
                e.iloc[-adapter.context_length :] if e is not None else None
            )
            for name, e in past_exog_dict.items()
        }
    else:
        history_exog = getattr(adapter, '_history_exog', None)
        if history_exog is not None:
            past_exog_dict = history_exog
        else:
            past_exog_dict = {name: None for name in series_names}

    future_exog_dict = normalize_exog_to_dict(exog, series_names)

    return adapter.predict(
        steps=steps,
        history_dict=history_dict,
        past_exog_dict=past_exog_dict,
        future_exog_dict=future_exog_dict,
        quantiles=quantiles,
        is_multiple_series=len(series_names) > 1,
    )
