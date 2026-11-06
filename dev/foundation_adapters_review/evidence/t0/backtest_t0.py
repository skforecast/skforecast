"""
Backtesting of the user guide configuration (same data, folds and metric as
`../common/backtest_guide.py`) for T0, through `FoundationModel`, with the
loader of `T0Adapter` patched in this process so that it also runs on
tfc-t0>=0.5 through the diagnostic shim of `t0_shim.py`.

Usage:
    python backtest_t0.py --model-id theforecastingcompany/t0-alpha --exog \
        [--predict-extra '{"trim_leading_missing": true}'] [--dtype bfloat16] \
        [--adapter-dtype bfloat16] [--out file.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_guide import hardware, load_data, versions  # noqa: E402
from t0_shim import make_adapter_model  # noqa: E402

from skforecast.foundation import ForecasterFoundation, FoundationModel  # noqa: E402
from skforecast.foundation._adapters import T0Adapter  # noqa: E402
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-length", type=int, default=500)
    parser.add_argument("--exog", action="store_true")
    parser.add_argument("--predict-extra", default="{}")
    parser.add_argument("--dtype", default=None, help="dtype passed to from_pretrained")
    parser.add_argument("--adapter-dtype", default=None, help="torch_dtype of the adapter (.to(dtype))")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Divide the target by this factor before forecasting; the MAE is reported in original units.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype) if args.dtype else None
    predict_extra = json.loads(args.predict_extra)

    data = load_data()
    end_train = "2014-11-30 23:59:00"
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_train]), refit=False)

    # Backtesting clones the estimator and `model=` is not a constructor
    # parameter returned by `get_params`, so the injected model would be lost:
    # patch the loader of the adapter instead (diagnostic only).
    def _load_model(self) -> None:
        if self._model is not None:
            return
        model = make_adapter_model(self.model_id, dtype=dtype, **predict_extra)
        if args.adapter_dtype:
            model = model.to(getattr(torch, args.adapter_dtype))
        self._model = model

    T0Adapter._load_model = _load_model
    estimator = FoundationModel(model_id=args.model_id, context_length=args.context_length)
    forecaster = ForecasterFoundation(estimator=estimator)
    exog = data[["Temperature", "Holiday"]] if args.exog else None

    start = time.perf_counter()
    metrics, predictions = backtesting_foundation(
        forecaster=forecaster,
        series=data["Demand"] / args.scale,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        suppress_warnings=True,
        show_progress=False,
    )
    elapsed = time.perf_counter() - start
    result = {
        "model_id": args.model_id,
        "context_length": args.context_length,
        "exog": args.exog,
        "predict_extra": predict_extra,
        "dtype": args.dtype,
        "adapter_dtype": args.adapter_dtype,
        "torch_threads": torch.get_num_threads(),
        "scale": args.scale,
        "mae": round(float(metrics["mean_absolute_error"].iloc[0]) * args.scale, 4),
        "elapsed_s": round(elapsed, 2),
        "n_predictions": len(predictions),
        "versions": versions(),
        "hardware": hardware(),
    }
    print(json.dumps(result))
    if args.out:
        predictions.to_csv(args.out)


if __name__ == "__main__":
    main()
