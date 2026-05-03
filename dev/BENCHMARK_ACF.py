import numpy as np
import timeit
import platform
import scipy
import statsmodels
from statsmodels.tsa.stattools import acf, pacf as sm_pacf
from scipy import signal
from scipy.fft import rfft as scipy_rfft, irfft as scipy_irfft


def fast_acf_scipy(x, nlags):
    n = len(x)
    x_centered = x - np.mean(x)
    autocorr_full = signal.correlate(x_centered, x_centered, mode='full', method='fft')
    autocorr = autocorr_full[len(autocorr_full)//2:]
    var = autocorr[0]
    if var == 0.0:
        return np.ones(nlags + 1)
    return autocorr[:nlags + 1] / var


def fast_acf_numpy(x, nlags):
    n = len(x)
    x_centered = x - np.mean(x)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_x = np.fft.rfft(x_centered, n=n_fft)
    power_spectrum = (fft_x * fft_x.conj()).real
    autocorr = np.fft.irfft(power_spectrum, n=n_fft)
    autocorr = autocorr[:nlags + 1]
    var = autocorr[0]
    if var == 0.0:
        return np.ones(nlags + 1)
    return autocorr / var


def fast_acf_scipy_fft(x, nlags):
    n = len(x)
    x_centered = x - np.mean(x)
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    fft_x = scipy_rfft(x_centered, n=n_fft)
    power_spectrum = (fft_x * fft_x.conj()).real
    autocorr = scipy_irfft(power_spectrum, n=n_fft)
    autocorr = autocorr[:nlags + 1]
    var = autocorr[0]
    if var == 0.0:
        return np.ones(nlags + 1)
    return autocorr / var


def fast_pacf_levinson(x, nlags):
    """PACF via Levinson-Durbin. Uses O(nlags) memory (two rolling vectors)."""
    acf_vals = fast_acf_numpy(x, nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    if nlags < 1:
        return pacf
    phi      = np.zeros(nlags + 1)
    phi_prev = np.zeros(nlags + 1)
    phi_prev[1] = acf_vals[1]
    pacf[1] = acf_vals[1]
    for k in range(2, nlags + 1):
        num = acf_vals[k] - phi_prev[1:k] @ acf_vals[k-1:0:-1]
        den = 1.0 - phi_prev[1:k] @ acf_vals[1:k]
        kk = num / den if den != 0.0 else 0.0
        phi[1:k] = phi_prev[1:k] - kk * phi_prev[k-1:0:-1]
        phi[k] = kk
        pacf[k] = kk
        phi, phi_prev = phi_prev, phi
    return pacf


np.random.seed(42)
N = 10_000
nlags = 500
x = np.cumsum(np.random.randn(N))

print(f"Dataset Size: {N:,} points")
print(f"Lags Calculated: {nlags}")
print(f"Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__} | statsmodels {statsmodels.__version__}\n")

acf_methods = {
    "Statsmodels (fft=False)": lambda: acf(x, nlags=nlags, fft=False),
    "Statsmodels (fft=True) ": lambda: acf(x, nlags=nlags, fft=True),
    "SciPy signal.correlate ": lambda: fast_acf_scipy(x, nlags),
    "NumPy rfft             ": lambda: fast_acf_numpy(x, nlags),
    "SciPy fft.rfft         ": lambda: fast_acf_scipy_fft(x, nlags),
}

pacf_methods = {
    "Statsmodels yw (default)": lambda: sm_pacf(x, nlags=nlags, method="yw"),
    "Statsmodels ld          ": lambda: sm_pacf(x, nlags=nlags, method="ld"),
    "Statsmodels burg        ": lambda: sm_pacf(x, nlags=nlags, method="burg"),
    "NumPy rfft + Levinson   ": lambda: fast_pacf_levinson(x, nlags),
}

runs = 15


def benchmark(methods, label):
    baseline_values = None
    print(f"\n--- {label} ---")
    print(f"{'Method':<26} | {'Avg Time (ms)':<15} | {'Max Diff vs Baseline'}")
    print("-" * 66)
    for name, func in methods.items():
        func()  # warmup
        vals = func()
        if baseline_values is None:
            baseline_values = vals
            max_diff = 0.0
        else:
            n = min(len(vals), len(baseline_values))
            max_diff = np.max(np.abs(vals[:n] - baseline_values[:n]))
        elapsed = timeit.timeit(func, number=runs)
        avg_time_ms = (elapsed / runs) * 1000
        print(f"{name:<26} | {avg_time_ms:>13.2f} ms | {max_diff:>10.2e}")


benchmark(acf_methods, "ACF")
benchmark(pacf_methods, "PACF")
