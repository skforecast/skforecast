"""
Smoke test of the raw backend and checks of the batching assumptions:
left padding with NaN, independence of the series in a batch, CPU time.
"""
import time
import numpy as np
import torch
from timestral import TimestralForecaster

rng = np.random.default_rng(0)
t = np.arange(1200)
long = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(1200).cumsum() * 0.3
short = (20 + 5 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(1200))[-337:]

for model_id in ["mistralai/Timestral-5M", "mistralai/Timestral-28M", "mistralai/Timestral-209M"]:
    t0 = time.perf_counter()
    fc = TimestralForecaster.from_pretrained(model_id)
    load = time.perf_counter() - t0
    n_params = sum(p.numel() for p in fc.model.parameters())

    # Alone
    q_short = fc.forecast(short, prediction_length=48).quantiles[0, 0].numpy()
    q_long = fc.forecast(long, prediction_length=48).quantiles[0, 0].numpy()
    # In one batch, the short series left-padded with NaN
    batch = np.full((2, 1200), np.nan)
    batch[0] = long
    batch[1, -337:] = short
    q_batch = fc.forecast(batch, prediction_length=48).quantiles[:, 0].numpy()
    d_short = np.abs(q_batch[1] - q_short).max()
    d_long = np.abs(q_batch[0] - q_long).max()

    # CPU time of one call: 1 series and 100 series, context 512, horizon 24
    x1 = long[-512:]
    x100 = np.stack([long[-512:] + rng.standard_normal(512) for _ in range(100)])
    fc.forecast(x1, prediction_length=24)
    t0 = time.perf_counter(); fc.forecast(x1, prediction_length=24); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); fc.forecast(x100, prediction_length=24); t100 = time.perf_counter() - t0
    x1_8k = np.tile(long, 7)[:8192]
    t0 = time.perf_counter(); fc.forecast(x1_8k, prediction_length=24); t8k = time.perf_counter() - t0

    print(
        f"{model_id}: params={n_params/1e6:.1f}M load={load:.1f}s "
        f"max|batch-alone| short={d_short:.2e} long={d_long:.2e} "
        f"t(1x512)={t1*1000:.0f}ms t(100x512)={t100:.2f}s t(1x8192)={t8k:.2f}s"
    )
print("torch", torch.__version__, "threads", torch.get_num_threads())
