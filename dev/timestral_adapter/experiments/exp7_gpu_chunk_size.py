"""
PR C, step C3: time and peak GPU memory of one Timestral call versus chunked
calls, to choose `TimestralAdapter.max_series_per_call`.

Backend called directly (as in exp4), on CUDA. Each configuration: 1 target
and 2 known covariates (or none), 24 steps, chunks of `chunk` series (`all` is
one call). Time is the minimum of `REPEATS` runs after one warm-up call; peak
memory is `torch.cuda.max_memory_allocated` over the runs. A configuration
that runs out of GPU memory is reported as OOM.

Run from this folder with the GPU environment (one mode per run):
    python -W ignore exp7_gpu_chunk_size.py gpu > exp7.log
    python -W ignore exp7_gpu_chunk_size.py gpu-long > exp7_long.log
    python -W ignore exp7_gpu_chunk_size.py cpu > exp7_cpu.log
    python -W ignore exp7_gpu_chunk_size.py gpu-extra > exp7_gpu_extra.log
    python -W ignore exp7_gpu_chunk_size.py cpu-extra > exp7_cpu_extra.log

`gpu-long` uses the default `context_length` of the adapter (4096) and skips
the single call with covariates (hundreds of seconds once the memory spills
over the 8 GB of the GPU). `cpu` compares chunks of 50 and 100 on CPU (time
only, peak memory is not measured).
"""
import sys
import time
import platform
import numpy as np
import torch
import timestral
from timestral import TimestralForecaster

MODE = sys.argv[1] if len(sys.argv) > 1 else "gpu"
DEVICE = "cpu" if MODE.startswith("cpu") else "cuda"
REPEATS = 2
STEPS = 24
MODELS = ["mistralai/Timestral-28M", "mistralai/Timestral-209M"]
if MODE == "gpu":
    CONFIGS = [
        # (context length, number of known covariates, number of series, chunks)
        (512, 2, 100, [None, 25, 50]),
        (512, 2, 400, [None, 25, 50, 100, 200]),
        (512, 2, 1000, [None, 25, 50, 100, 200]),
        (512, 0, 1000, [None, 25, 50, 100, 200]),
        (2048, 2, 400, [None, 25, 50, 100, 200]),
        (4096, 2, 100, [None, 25, 50]),
    ]
elif MODE == "gpu-long":
    CONFIGS = [
        (4096, 0, 400, [None, 50, 100, 200]),
        (4096, 2, 400, [50, 100]),
        (4096, 4, 100, [50, 100]),
    ]
elif MODE == "gpu-extra":
    # A chunk of 100 with 100 series is one call (`all`)
    CONFIGS = [(4096, 4, 100, [None])]
elif MODE == "cpu-extra":
    MODELS = ["mistralai/Timestral-28M"]
    CONFIGS = [(2048, 2, 100, [None])]
else:
    MODELS = ["mistralai/Timestral-28M"]
    CONFIGS = [
        (512, 2, 400, [50, 100]),
        (512, 0, 400, [50, 100]),
        (2048, 2, 100, [50, 100]),
    ]

print(
    f"Python {platform.python_version()}, torch {torch.__version__}, "
    f"timestral {getattr(timestral, '__version__', '1.0.0')}, numpy {np.__version__}"
)
if DEVICE == "cuda":
    print(
        f"GPU {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
    )
else:
    print(f"CPU {platform.processor()}, {torch.get_num_threads()} torch threads")
rng = np.random.default_rng(7)


def make_inputs(n_series, context_length, n_known):
    t = np.arange(context_length + STEPS)
    ctx = (
        100 + 20 * np.sin(2 * np.pi * t[:context_length] / 24)
        + rng.standard_normal((n_series, context_length)).cumsum(-1)
    )[:, None].astype(np.float32)
    known = None
    if n_known:
        rows = [
            15 + 8 * np.sin(2 * np.pi * t / (24 * (k // 2 + 1)))
            if k % 2 == 0 else (t % (7 * (k // 2 + 1))).astype(float)
            for k in range(n_known)
        ]
        known = np.repeat(np.stack(rows)[None], n_series, axis=0).astype(np.float32)
        known = known + rng.standard_normal(known.shape).astype(np.float32) * 0.1
    return ctx, known


def run(fc, ctx, known, chunk):
    n_series = ctx.shape[0]
    size = n_series if chunk is None else chunk
    for i in range(0, n_series, size):
        kwargs = {} if known is None else {"known_covariates": known[i:i + size]}
        fc.forecast(ctx[i:i + size], prediction_length=STEPS, **kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()


for model_id in MODELS:
    fc = TimestralForecaster.from_pretrained(model_id, device=DEVICE)
    ctx, known = make_inputs(4, 512, 2)
    run(fc, ctx, known, None)
    print(f"\n{model_id}")
    print(f"{'context':>7} {'cov':>3} {'series':>6} {'chunk':>5} {'time s':>8} {'peak GB':>8}")
    for context_length, n_known, n_series, chunks in CONFIGS:
        ctx, known = make_inputs(n_series, context_length, n_known)
        for chunk in chunks:
            if chunk is not None and chunk >= n_series:
                continue
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                run(fc, ctx, known, chunk)  # warm-up for this shape
                times = []
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    run(fc, ctx, known, chunk)
                    times.append(time.perf_counter() - t0)
                elapsed = f"{min(times):8.2f}"
                peak = (
                    f"{torch.cuda.max_memory_allocated() / 2**30:8.2f}"
                    if DEVICE == "cuda" else f"{'n/a':>8}"
                )
            except torch.OutOfMemoryError:
                elapsed, peak = f"{'OOM':>8}", f"{'OOM':>8}"
            label = "all" if chunk is None else str(chunk)
            print(
                f"{context_length:>7} {n_known:>3} {n_series:>6} {label:>5} {elapsed} {peak}",
                flush=True,
            )
    del fc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

sys.exit(0)
