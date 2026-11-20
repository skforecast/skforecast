import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from skforecast.recursive import ForecasterEquivalentDate
idx=pd.date_range("2020",periods=60,freq="D"); y=pd.Series(np.arange(60,dtype=float),index=idx); y.iloc[-7]=np.nan
for off in (7, pd.offsets.Day(7)):
    f=ForecasterEquivalentDate(offset=off,n_offsets=2); f.fit(y)
    print(repr(off), f.predict(1).iloc[0])
