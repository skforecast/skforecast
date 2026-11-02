# Check the datasets registry against a local clone of skforecast-datasets
# ==============================================================================
# Usage: python check_datasets_local.py /path/to/skforecast-datasets/data
import os
import sys
import warnings
import pandas as pd
import skforecast
from skforecast.datasets import datasets

print(skforecast.__file__)
local = sys.argv[1]
for name, info in datasets.items():
    url = info['url']
    if 'raw.githubusercontent.com' not in url:
        print(f"{name}: hosted at {url}, skipped")
        continue
    path = os.path.join(local, url.rsplit('/', 1)[-1])
    df = (pd.read_csv(path, sep=info['sep']) if info['file_type'] == 'csv'
          else pd.read_parquet(path))
    unique = None
    n_before = len(df)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        try:
            if df.index.name != info['index_col']:
                df.index = df.pop(info['index_col'])
            df.index = pd.to_datetime(df.index, format=info['date_format'])
            unique = df.index.is_unique
            if unique:
                df = df.asfreq(info['freq'])
            if not df.index.is_monotonic_increasing:
                df = df.sort_index(kind='stable')
            status = 'PREPROCESS OK'
        except Exception as e:
            status = f'PREPROCESS FAILS: {type(e).__name__}: {e}'
    inserted = len(df) - n_before if status == 'PREPROCESS OK' and unique else 0
    print(
        f"{name}: {status} shape={df.shape} unique={unique} "
        f"freq={getattr(df.index, 'freqstr', None)} asfreq_inserted={inserted} "
        f"warnings={[w.category.__name__ for w in record]}"
    )
