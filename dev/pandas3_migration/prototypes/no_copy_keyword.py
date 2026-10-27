"""
Prototype: do not pass the deprecated `copy=` keyword to pandas 3
(pd.concat(copy=False), DataFrame.astype(copy=False), Series.astype(copy=False)).

What the library change would be: a helper in a compat module that passes
copy=False only when pandas < 3 (in pandas 3 Copy-on-Write never copies
eagerly in these calls). The prototype therefore edits the source only when
running under pandas 3; under pandas 2 it is a no-op, which is exactly the
behaviour of the proposed helper.
"""

from _source_patch import is_pandas3, patch_source, restore_all

TARGETS = [
    ("skforecast.recursive._forecaster_recursive:ForecasterRecursive.create_train_X_y",
     [("X_train.astype(X_train_dtypes, copy=False)", "X_train.astype(X_train_dtypes)")]),
    ("skforecast.recursive._forecaster_recursive:ForecasterRecursive.create_predict_X",
     [("X_predict.astype(X_predict_dtypes, copy=False)", "X_predict.astype(X_predict_dtypes)")]),
    ("skforecast.recursive._forecaster_recursive_classifier:ForecasterRecursiveClassifier.create_train_X_y",
     [("X_train.astype(X_train_dtypes, copy=False)", "X_train.astype(X_train_dtypes)")]),
    ("skforecast.recursive._forecaster_recursive_classifier:ForecasterRecursiveClassifier.create_predict_X",
     [("X_predict.astype(X_predict_dtypes, copy=False)", "X_predict.astype(X_predict_dtypes)")]),
    ("skforecast.direct._forecaster_direct:ForecasterDirect.create_train_X_y",
     [("X_train.astype(X_train_dtypes, copy=False)", "X_train.astype(X_train_dtypes)")]),
    ("skforecast.direct._forecaster_direct:ForecasterDirect.create_predict_X",
     [("X_predict.astype(X_predict_dtypes, copy=False)", "X_predict.astype(X_predict_dtypes)")]),
    ("skforecast.direct._forecaster_direct_multivariate:ForecasterDirectMultiVariate.create_train_X_y",
     [("X_train.astype(X_train_dtypes, copy=False)", "X_train.astype(X_train_dtypes)")]),
    ("skforecast.direct._forecaster_direct_multivariate:ForecasterDirectMultiVariate.create_predict_X",
     [("X_predict.astype(X_predict_dtypes, copy=False)", "X_predict.astype(X_predict_dtypes)")]),
    ("skforecast.recursive._forecaster_recursive_multiseries:ForecasterRecursiveMultiSeries._assemble_train_X_y",
     [("X_train = pd.concat(X_train, axis=1, copy=False)", "X_train = pd.concat(X_train, axis=1)")]),
    ("skforecast.recursive._forecaster_recursive_multiseries:ForecasterRecursiveMultiSeries._create_train_X_y",
     [("X_train_exog = pd.concat(X_train_exog_buffer, axis=0, copy=False)", "X_train_exog = pd.concat(X_train_exog_buffer, axis=0)")]),
    ("skforecast.recursive._forecaster_recursive_multiseries:ForecasterRecursiveMultiSeries.create_predict_X",
     [("X_predict = X_predict.astype(self.exog_dtypes_out_, copy=False)", "X_predict = X_predict.astype(self.exog_dtypes_out_)")]),
    ("skforecast.preprocessing._preprocessing:reshape_exog_long_to_dict",
     [("v.astype(new_dtypes, copy=False)", "v.astype(new_dtypes)")]),
    ("skforecast.drift_detection._population_drift:PopulationDriftDetector._fit",
     [('ref.astype("float64", copy=False)', 'ref.astype("float64")')]),
]


def apply():
    if not is_pandas3():
        return
    for target, repl in TARGETS:
        patch_source(target, repl)


def restore():
    restore_all()
