# ASV Quick Guide (for `skforecast`)

> [!IMPORTANT] 
> Run all commands **from the folder that contains `asv.conf.json`** (e.g., `asv_benchmarks/`).

---

## 1) Install

```bash
pip install asv
```

> [!NOTE]
> You only need `libmambapy` if you later switch ASV to the conda/mamba backend.
With the recommended `environment_type: "virtualenv"` you can ignore the
“Couldn't load asv.plugins._mamba_helpers …” warning.

## 2) Sanity-check the suite

```bash
asv check -v
```

Verifies discovery (`time_`, `mem_`, `peakmem_`, `timeraw_`, `track_`), imports, and structure.

## 3) Fast development run

```bash
asv run --python=same --quick --show-stderr 
```

+ `--python=same` → use the current interpreter (no new env).
+ `--quick` → run each benchmark once (no statistics) and don’t save results.
+ `--show-stderr` → print full tracebacks (great for debugging failures).


## 4) Fast run specific benchmarks

```bash
asv run --python=same --quick --show-stderr -b ForecasterRecursive
asv run --python=same --quick --show-stderr -b TimeForecasterRecursive_Fit
asv run --python=same --quick --show-stderr -b TimeForecasterRecursive_Predict.time_predict
asv run --python=same --quick --show-stderr -b TimeForecasterRecursive_Backtesting.time_backtesting_conformal
```

+ `-b` → filters by substring or regex over the full benchmark name (`Class.method`).

## 5) Normal runs & reports

```bash
# Re-run only new commits (use -e to use existing env)
asv run NEW

# Build static HTML
asv publish

# Open interactive report locally
asv preview
```

```bash
# Compare 2 branches
asv run 0.17.x..feature_asv_benchmark
```

If some tests are heavy, raise the timeout in `asv.conf.json`:

```json
{ "default_benchmark_timeout": 60 }
```

## Machine information (avoid the interactive prompt)

Create ~/.asv-machine.json once so ASV won’t ask:

**Quick way to print the fields you need**

```python
import platform, psutil, sys

print("CPU:", platform.processor())
print("Núcleos lógicos:", psutil.cpu_count(logical=True))
print("RAM GB:", round(psutil.virtual_memory().total / 1e9, 2))
print("SO:", platform.platform())
print("Arquitectura:", platform.machine())
print("Python:", sys.version.split()[0])
```

**Example file (`~/.asv-machine.json`)**
```json
{
  "machine": "my-windows-laptop",
  "arch": "x86_64",
  "cpu": "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel",
  "num_cpu": 16,
  "ram": "34.07GB",
  "os": "Windows-11-10.0.26100-SP0",
  "python": "3.13.2"
}
```
