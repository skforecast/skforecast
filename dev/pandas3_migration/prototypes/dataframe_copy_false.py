"""
Prototype: build the public training matrix without copying the ndarray.

create_train_X_y of ForecasterRecursive, ForecasterRecursiveClassifier,
ForecasterDirect and ForecasterDirectMultiVariate wrap the ndarray returned by
_create_train_X_y in `pd.DataFrame(data=X_train, ...)`. pandas 3 copies an
ndarray in the constructor unless copy=False (pandas 2 does not copy). The
array is private to the method, so sharing it is safe in both versions.
Independent of the pandas version (the same text change in both).
"""

from _source_patch import patch_source, restore_all

OLD_RECURSIVE = """                      columns = X_train_features_names_out_
                  )"""
NEW_RECURSIVE = """                      columns = X_train_features_names_out_,
                      copy    = False
                  )"""
OLD_DIRECT = """                      columns = X_train_direct_features_names_out_
                  )"""
NEW_DIRECT = """                      columns = X_train_direct_features_names_out_,
                      copy    = False
                  )"""

TARGETS = [
    ("skforecast.recursive._forecaster_recursive:ForecasterRecursive.create_train_X_y", OLD_RECURSIVE, NEW_RECURSIVE),
    ("skforecast.recursive._forecaster_recursive_classifier:ForecasterRecursiveClassifier.create_train_X_y", OLD_RECURSIVE, NEW_RECURSIVE),
    ("skforecast.direct._forecaster_direct:ForecasterDirect.create_train_X_y", OLD_DIRECT, NEW_DIRECT),
    ("skforecast.direct._forecaster_direct_multivariate:ForecasterDirectMultiVariate.create_train_X_y", OLD_DIRECT, NEW_DIRECT),
]


def apply():
    for target, old, new in TARGETS:
        patch_source(target, [(old, new)])


def restore():
    restore_all()
