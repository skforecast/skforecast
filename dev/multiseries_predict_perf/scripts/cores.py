"""Speed of a pure Python loop pinned to each logical processor (ms, min of 5)."""
import ctypes
import time

k = ctypes.windll.kernel32
k.GetCurrentProcess.restype = ctypes.c_void_p
k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
k.SetPriorityClass(k.GetCurrentProcess(), 0x00000080)


def work():
    t0 = time.perf_counter()
    s = 0
    for i in range(300_000):
        s += i * i % 7
    return (time.perf_counter() - t0) * 1e3


out = []
for cpu in range(22):
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), 1 << cpu):
        continue
    time.sleep(0.01)
    runs = [work() for _ in range(5)]
    out.append(f"{cpu}: {min(runs):.1f} (max {max(runs):.1f})")
print(" | ".join(out))
