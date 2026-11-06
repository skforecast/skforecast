"""
Print whether `Chronos2Pipeline.predict` declares the arguments the adapter
passes through `predict_quantiles(**predict_kwargs)`. A contract entry on
`Chronos2Pipeline.predict` with `kwargs=["cross_learning"]` would detect a
backend older than 2.2.0, which the current entry (weak kwarg on
`predict_quantiles`) cannot.
"""

import importlib.metadata as md
import inspect

from chronos import BaseChronosPipeline, Chronos2Pipeline

print("chronos-forecasting", md.version("chronos-forecasting"))
for target in [Chronos2Pipeline.predict, Chronos2Pipeline.predict_quantiles,
               BaseChronosPipeline.from_pretrained]:
    print(target.__qualname__, inspect.signature(target))
params = inspect.signature(Chronos2Pipeline.predict).parameters
for name in ["cross_learning", "batch_size", "context_length", "limit_prediction_length"]:
    print(f"  predict declares {name}: {name in params}")
