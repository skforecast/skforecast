"""
Copy semantics of pandas 2 vs 3: DataFrame/Series constructors from ndarray,
to_numpy()/.values writeability, pd.concat(axis=1) blocks and copies, the
copy= keyword deprecation (Pandas4Warning), DataFrame.insert fragmentation and
whether df[list] / iloc slices share memory with the source.

    bash dev/pandas3_migration/run_in_both.sh experiments/exp_cow_basics.py
"""
import warnings; warnings.simplefilter("always")
import numpy as np, pandas as pd
print("pandas", pd.__version__)
arr = np.empty((1000, 5), order="F")
with warnings.catch_warnings(record=True) as w:
    df = pd.DataFrame(arr, columns=list("abcde"), copy=False)
    print("ctor copy=False warns:", [str(x.message)[:80] for x in w])
print("ctor copy=False shares:", np.shares_memory(arr, df.to_numpy()))
with warnings.catch_warnings(record=True) as w:
    df2 = pd.DataFrame(arr, columns=list("abcde"))
    print("ctor default warns:", [str(x.message)[:80] for x in w])
print("ctor default shares:", np.shares_memory(arr, df2.to_numpy()))
v = df.to_numpy()
print("to_numpy writeable:", v.flags.writeable, "F:", v.flags.f_contiguous, "C:", v.flags.c_contiguous)
print("values writeable:", df.values.flags.writeable)
s = pd.Series(np.arange(5.0))
sv = s.to_numpy()
print("series to_numpy writeable:", sv.flags.writeable, "np.asarray writeable:", np.asarray(s).flags.writeable)
# concat axis=1
a = pd.DataFrame(np.ones((1000, 3)), columns=list("abc"))
b = pd.DataFrame(np.ones((1000, 2)), columns=list("de"))
with warnings.catch_warnings(record=True) as w:
    c = pd.concat([a, b], axis=1)
    print("concat warns:", [str(x.message)[:80] for x in w])
try:
    print("concat nblocks:", c._mgr.nblocks)
except Exception as e:
    print("nblocks err", e)
cv = c.to_numpy()
print("concat to_numpy shares with a:", np.shares_memory(cv, a.to_numpy()), "shares with c col a:", np.shares_memory(cv, c["a"].to_numpy()))
with warnings.catch_warnings(record=True) as w:
    c2 = pd.concat([a, b], axis=1, copy=False)
    print("concat copy=False warns:", [type(x.message).__name__ + ": " + str(x.message)[:80] for x in w])
# insert
d = pd.DataFrame(np.ones((1000, 3)), columns=list("abc"))
for i in range(5):
    d.insert(1, f"x{i}", np.ones(1000), allow_duplicates=True)
print("after 5 inserts nblocks:", d._mgr.nblocks)
# df[list] view or copy
e = pd.DataFrame(np.ones((1000, 4)), columns=list("abcd"))
sub = e[["a", "b"]]
print("df[list] shares:", np.shares_memory(sub.to_numpy(), e.to_numpy()))
# iloc slice
sl = e.iloc[10:20]
print("iloc slice shares:", np.shares_memory(sl.to_numpy(), e.to_numpy()))
# astype copy=False
with warnings.catch_warnings(record=True) as w:
    f = e.astype({"a": float}, copy=False)
    print("astype copy=False warns:", [type(x.message).__name__ + ": " + str(x.message)[:80] for x in w])
# Series ctor copy=False
with warnings.catch_warnings(record=True) as w:
    s2 = pd.Series(np.ones(10), copy=False)
    print("Series ctor copy=False warns:", [str(x.message)[:80] for x in w])
# np.asarray of mixed frame
m = pd.DataFrame({"a": np.ones(10), "b": np.arange(10)})
print("mixed to_numpy writeable:", m.to_numpy().flags.writeable)
# to_numpy of a float frame with a single block -> view in pandas 3?
g = pd.DataFrame(np.ones((10, 3)))
gv = g.to_numpy()
print("single block to_numpy shares:", np.shares_memory(gv, g._mgr.blocks[0].values), "writeable:", gv.flags.writeable)
