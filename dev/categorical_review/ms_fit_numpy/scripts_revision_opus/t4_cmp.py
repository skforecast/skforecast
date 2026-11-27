import pickle, numpy as np, collections
a = pickle.load(open("t4_base.pkl", "rb")); b = pickle.load(open("t4_branch.pkl", "rb"))
c = collections.Counter(); diffs = []
for k in a:
    x, y = a[k], b[k]
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        xx = x[:, 1:] if x.dtype == object and x.ndim == 2 else x
        yy = y[:, 1:] if y.dtype == object and y.ndim == 2 else y
        same = xx.shape == yy.shape and np.array_equal(xx.astype(float), yy.astype(float), equal_nan=True)
    else:
        same = x == y
    c[(k[0], same)] += 1
    if not same: diffs.append((k, str(x)[:80], str(y)[:80]))
for k, n in sorted(c.items(), key=str): print(k, n)
for d in diffs: print(d)
