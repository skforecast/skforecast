"""
Direct forecasters with the proposed fix emulated from outside the package:
`estimators_` is emptied before `fit`, so the step tasks sent to the joblib
workers (which receive `forecaster=self`) do not pickle the estimators fitted
in the previous fit.

The classes keep the original names because skforecast selects behavior with
`type(forecaster).__name__` (allowed forecasters, `select_n_jobs_*`). They
live in an importable module so the loky workers can unpickle them.
"""

from __future__ import annotations

from skforecast.direct import ForecasterDirect as _ForecasterDirect
from skforecast.direct import (
    ForecasterDirectMultiVariate as _ForecasterDirectMultiVariate,
)


class ForecasterDirect(_ForecasterDirect):
    def fit(self, *args, **kwargs):
        self.estimators_ = {}
        return super().fit(*args, **kwargs)


class ForecasterDirectMultiVariate(_ForecasterDirectMultiVariate):
    def fit(self, *args, **kwargs):
        self.estimators_ = {}
        return super().fit(*args, **kwargs)
