| scenario | base | branch | minimal |
|---|---|---|---|
| catboost|user=fit_name|cf=None|enc=ordinal_category ** | ERR TypeError | ok [LEVEL,cat] df | ERR TypeError |
| catboost|user=fit_name|cf=None|enc=ordinal ** | ERR TypeError | ok [cat] df | ERR TypeError |
| catboost|user=fit_pos|cf=None|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL,cat] df | ERR CatBoostError |
| catboost|user=fit_pos|cf=None|enc=ordinal | ok [cat] df | ok [cat] df | ok [cat] df |
| catboost|user=init_name|cf=None|enc=ordinal_category ** | ERR CatBoostError | ERR ValueError | ERR CatBoostError |
| catboost|user=init_name|cf=None|enc=ordinal | ok [cat] df | ok [cat] df | ok [cat] df |
| catboost|user=init_pos|cf=None|enc=ordinal_category ** | ERR CatBoostError | ERR ValueError | ERR CatBoostError |
| catboost|user=init_pos|cf=None|enc=ordinal | ok [cat] df | ok [cat] df | ok [cat] df |
| catboost|user=none|cf=None|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL] df | ERR CatBoostError |
| catboost|user=none|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| hgb|user=init_mask|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_mask|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_name|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_name|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_pos|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_pos|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=none|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL] np | ok [LEVEL] df |
| hgb|user=none|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| lgbm|user=fit_name|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| lgbm|user=fit_name|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=init_name|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL,cat] np | ok [LEVEL] df |
| lgbm|user=init_name|cf=None|enc=ordinal ** | ERR LightGBMError | ok [cat] np | ERR LightGBMError |
| lgbm|user=init_pos|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL,cat] np | ok [LEVEL] df |
| lgbm|user=init_pos|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=none|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL] np | ok [LEVEL] df |
| lgbm|user=none|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| xgb|user=init_enable|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL] np | ok [LEVEL] df |
| xgb|user=init_enable|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| xgb|user=init_pos|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| xgb|user=init_pos|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| xgb|user=none|cf=None|enc=ordinal_category ** | ok [LEVEL] df | ok [LEVEL] np | ok [LEVEL] df |
| xgb|user=none|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |

base {'ERR': 7, 'ok': 27} n = 34
branch {'ok': 32, 'ERR': 2} n = 34
minimal {'ERR': 7, 'ok': 27} n = 34

== errors base
  [4] CatBoostError: features data: pandas.DataFrame column '_level_skforecast' has dtype 'category' but is not in  cat_features list
        catboost|user=none|cf=None|enc=ordinal_category
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
        catboost|user=fit_pos|cf=None|enc=ordinal_category
  [2] TypeError: list indices must be integers or slices, not str
        catboost|user=fit_name|cf=None|enc=ordinal
        catboost|user=fit_name|cf=None|enc=ordinal_category
  [1] LightGBMError: categorical_feature is not a number,
if you want to use a column name,
please add the prefix "name:" to the column name
        lgbm|user=init_name|cf=None|enc=ordinal
== errors branch
  [2] ValueError: With `encoding='ordinal_category'` and `categorical_features=None`, the series identifier (`_level_skforecast`) is added to the categorical features
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
== errors minimal
  [4] CatBoostError: features data: pandas.DataFrame column '_level_skforecast' has dtype 'category' but is not in  cat_features list
        catboost|user=none|cf=None|enc=ordinal_category
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
        catboost|user=fit_pos|cf=None|enc=ordinal_category
  [2] TypeError: list indices must be integers or slices, not str
        catboost|user=fit_name|cf=None|enc=ordinal
        catboost|user=fit_name|cf=None|enc=ordinal_category
  [1] LightGBMError: categorical_feature is not a number,
if you want to use a column name,
please add the prefix "name:" to the column name
        lgbm|user=init_name|cf=None|enc=ordinal
