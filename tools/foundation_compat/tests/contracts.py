# Backend API used by each foundation model adapter
# ==============================================================================
# For every entry of `adapters.toml`, the symbols of the backend library that
# the adapter in `skforecast/foundation/_adapters.py` imports and the arguments
# it passes to them. `test_contract.py` compares them with the signatures of
# the installed backend, so a rename in a new release of the backend is
# detected without downloading any weights.
#
# Update this file whenever an adapter changes how it calls its backend. To
# list the calls of the adapters:
#
#     grep -n "self._model\.\|self._pipeline\.\|self._forecast_obj\." \
#         skforecast/foundation/_adapters.py
#
# Keys of each call:
#
# - target:       `'module:qualified.name'` of a class, method or attribute.
#                 It must be importable. A class stands for its constructor.
# - kwargs:       arguments the adapter passes by name. Each one must be
#                 declared explicitly in the signature.
# - n_positional: number of arguments the adapter passes by position
#                 (`self` and `cls` excluded).
# - weak_kwargs:  arguments passed by name that the backend takes through
#                 `**kwargs`. Their name cannot be checked in the signature;
#                 only the smoke tests cover them.
#
# A call without `kwargs` and `n_positional` only checks that the target
# exists. That is the case of the `from_pretrained` methods inherited from
# `huggingface_hub`, whose signature is generic.
#
# The contract does not see what the backend returns (attributes, shapes,
# column names): that is checked by the smoke tests.

BACKEND_CONTRACTS = {
    "chronos": [
        {"target": "chronos:BaseChronosPipeline.from_pretrained"},
        # `BaseChronosPipeline.from_pretrained` returns a `Chronos2Pipeline` for
        # the registered checkpoints.
        {
            "target": "chronos:Chronos2Pipeline.predict_quantiles",
            "kwargs": ["inputs", "prediction_length", "quantile_levels"],
            "weak_kwargs": ["cross_learning"],
        },
    ],
    "timesfm25": [
        {"target": "timesfm:TimesFM_2p5_200M_torch.from_pretrained"},
        # Overridden by the adapter to absorb arguments of `huggingface_hub`.
        {"target": "timesfm:TimesFM_2p5_200M_torch._from_pretrained"},
        {
            "target": "timesfm:TimesFM_2p5_200M_torch.forecast",
            "kwargs": ["horizon", "inputs"],
        },
        {"target": "timesfm:TimesFM_2p5_200M_torch.compile", "n_positional": 1},
        {
            "target": "timesfm:ForecastConfig",
            "kwargs": ["max_context", "max_horizon"],
        },
    ],
    "timesfm3": [
        {
            "target": "timesfm:TimesFM3Forecaster.from_pretrained",
            "n_positional": 1,
            "kwargs": ["device"],
        },
        {
            "target": "timesfm:TimesFM3Forecaster.predict_batch",
            "kwargs": [
                "contexts",
                "horizon",
                "past_only_covariates",
                "past_future_covariates",
                "return_quantiles",
                "padding_mode",
            ],
        },
    ],
    "moirai": [
        {"target": "uni2ts.model.moirai2:Moirai2Module.from_pretrained"},
        {
            "target": "uni2ts.model.moirai2:Moirai2Forecast",
            "kwargs": [
                "module",
                "prediction_length",
                "context_length",
                "target_dim",
                "feat_dynamic_real_dim",
                "past_feat_dynamic_real_dim",
            ],
        },
        {
            "target": "uni2ts.model.moirai2:Moirai2Forecast.hparams_context",
            "kwargs": ["prediction_length"],
        },
        {
            "target": "uni2ts.model.moirai2:Moirai2Forecast.predict",
            "n_positional": 1,
        },
    ],
    "tabicl": [
        {
            "target": "tabicl.forecast:TabICLForecaster",
            "kwargs": [
                "max_context_length",
                "temporal_features",
                "point_estimate",
                "tabicl_config",
            ],
        },
        {
            "target": "tabicl.forecast:TabICLForecaster.predict_df",
            "kwargs": ["context_df", "future_df", "quantiles"],
        },
    ],
    "tabpfn": [
        {"target": "tabpfn_time_series:TabPFNMode.LOCAL"},
        {"target": "tabpfn_time_series:TabPFNMode.CLIENT"},
        {
            "target": "tabpfn_time_series:TabPFNTSPipeline",
            "kwargs": [
                "max_context_length",
                "tabpfn_mode",
                "tabpfn_output_selection",
                "tabpfn_model_config",
                "temporal_features",
            ],
        },
        {
            "target": "tabpfn_time_series:TabPFNTSPipeline.predict_df",
            "kwargs": ["context_df", "future_df", "quantiles"],
        },
    ],
    "t0": [
        {"target": "t0:T0Forecaster.from_pretrained"},
        {
            "target": "t0:T0Forecaster.predict",
            "n_positional": 1,
            "kwargs": ["horizon", "quantile_levels", "future_covariates"],
        },
    ],
    "nori": [
        {"target": "synthefy_nori:NoriRegressor", "kwargs": ["model"]},
        {"target": "synthefy_nori:NoriRegressor.fit", "n_positional": 2},
        {
            "target": "synthefy_nori:NoriRegressor.predict",
            "n_positional": 1,
            "kwargs": ["output_type", "quantiles"],
        },
    ],
    "tsicl": [
        {
            "target": "tsicl:TSICL",
            "kwargs": ["checkpoint_version", "allow_auto_download"],
        },
        {
            "target": "tsicl:TSICL.forecast",
            "kwargs": [
                "inputs",
                "prediction_length",
                "quantile_levels",
                "context_length",
                "device",
                "denormalize",
                "squeeze_output",
            ],
        },
    ],
    "timestral": [
        {
            "target": "timestral:TimestralForecaster.from_pretrained",
            "n_positional": 1,
            "kwargs": ["device"],
        },
        {
            "target": "timestral:TimestralForecaster.forecast",
            "kwargs": [
                "context",
                "prediction_length",
                "past_covariates",
                "known_covariates",
                "categorical_past_covariates",
                "categorical_known_covariates",
            ],
        },
        {"target": "timestral:Forecast", "kwargs": ["quantiles", "quantile_levels"]},
    ],
}
