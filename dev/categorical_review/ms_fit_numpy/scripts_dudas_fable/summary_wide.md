| scenario | base | branch | minimal |
|---|---|---|---|
| catboost|user=fit_name|cf=None|enc=None ** | ERR TypeError | ok [cat] df | ok [cat] df |
| catboost|user=fit_name|cf=None|enc=onehot ** | ERR TypeError | ok [cat] df | ok [cat] df |
| catboost|user=fit_name|cf=None|enc=ordinal_category ** | ERR TypeError | ok [LEVEL,cat] df | ERR CatBoostError |
| catboost|user=fit_name|cf=None|enc=ordinal ** | ERR TypeError | ok [cat] df | ok [cat] df |
| catboost|user=fit_name|cf=auto|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=auto|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=auto|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=fit_name|cf=auto|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=list_all|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=list_all|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=list_all|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=fit_name|cf=list_all|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_name|cf=list_partial|enc=None ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=fit_name|cf=list_partial|enc=onehot ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=fit_name|cf=list_partial|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL,cat2] df | ok [LEVEL,cat2] df |
| catboost|user=fit_name|cf=list_partial|enc=ordinal ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=fit_pos|cf=None|enc=None ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=fit_pos|cf=None|enc=onehot ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=fit_pos|cf=None|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL,cat] df | ERR CatBoostError |
| catboost|user=fit_pos|cf=None|enc=ordinal ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=fit_pos|cf=auto|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=auto|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=auto|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=fit_pos|cf=auto|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=list_all|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=list_all|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=list_all|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=fit_pos|cf=list_all|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=fit_pos|cf=list_partial|enc=None ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=fit_pos|cf=list_partial|enc=onehot ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=fit_pos|cf=list_partial|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL,cat2] df | ok [LEVEL,cat2] df |
| catboost|user=fit_pos|cf=list_partial|enc=ordinal ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=init_name|cf=None|enc=None ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_name|cf=None|enc=onehot ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_name|cf=None|enc=ordinal_category ** | ERR CatBoostError | ERR ValueError | ERR CatBoostError |
| catboost|user=init_name|cf=None|enc=ordinal ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_name|cf=auto|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=auto|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=auto|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=auto|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_all|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_all|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_all|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_all|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_partial|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_partial|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_partial|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_name|cf=list_partial|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=None|enc=None ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_pos|cf=None|enc=onehot ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_pos|cf=None|enc=ordinal_category ** | ERR CatBoostError | ERR ValueError | ERR CatBoostError |
| catboost|user=init_pos|cf=None|enc=ordinal ** | ERR CatBoostError | ok [cat] df | ok [cat] df |
| catboost|user=init_pos|cf=auto|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=auto|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=auto|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=auto|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_all|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_all|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_all|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_all|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_partial|enc=None | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_partial|enc=onehot | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_partial|enc=ordinal_category | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=init_pos|cf=list_partial|enc=ordinal | ERR CatBoostError | ERR CatBoostError | ERR CatBoostError |
| catboost|user=none|cf=None|enc=None ** | ERR CatBoostError | ok [-] np | ok [-] df |
| catboost|user=none|cf=None|enc=onehot ** | ERR CatBoostError | ok [-] np | ok [-] df |
| catboost|user=none|cf=None|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL] df | ERR CatBoostError |
| catboost|user=none|cf=None|enc=ordinal ** | ERR CatBoostError | ok [-] np | ok [-] df |
| catboost|user=none|cf=auto|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=auto|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=auto|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=none|cf=auto|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=list_all|enc=None | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=list_all|enc=onehot | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=list_all|enc=ordinal_category | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] df |
| catboost|user=none|cf=list_all|enc=ordinal | ok [cat,cat2] df | ok [cat,cat2] df | ok [cat,cat2] df |
| catboost|user=none|cf=list_partial|enc=None ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=none|cf=list_partial|enc=onehot ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| catboost|user=none|cf=list_partial|enc=ordinal_category ** | ERR CatBoostError | ok [LEVEL,cat2] df | ok [LEVEL,cat2] df |
| catboost|user=none|cf=list_partial|enc=ordinal ** | ERR CatBoostError | ok [cat2] df | ok [cat2] df |
| hgb|user=init_mask|cf=None|enc=None ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_mask|cf=None|enc=onehot ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_mask|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_mask|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_mask|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_mask|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_mask|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_mask|cf=list_partial|enc=None ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_mask|cf=list_partial|enc=onehot ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_mask|cf=list_partial|enc=ordinal_category ** | ok [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| hgb|user=init_mask|cf=list_partial|enc=ordinal ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_name|cf=None|enc=None ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_name|cf=None|enc=onehot ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_name|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_name|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_name|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_name|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_name|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_name|cf=list_partial|enc=None ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_name|cf=list_partial|enc=onehot ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_name|cf=list_partial|enc=ordinal_category ** | ok [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| hgb|user=init_name|cf=list_partial|enc=ordinal ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_pos|cf=None|enc=None ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_pos|cf=None|enc=onehot ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_pos|cf=None|enc=ordinal_category ** | ok [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| hgb|user=init_pos|cf=None|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| hgb|user=init_pos|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_pos|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=init_pos|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=init_pos|cf=list_partial|enc=None ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_pos|cf=list_partial|enc=onehot ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=init_pos|cf=list_partial|enc=ordinal_category ** | ok [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| hgb|user=init_pos|cf=list_partial|enc=ordinal ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=none|cf=None|enc=None ** | ok [cat,cat2] df | ok [-] np | ok [-] df |
| hgb|user=none|cf=None|enc=onehot ** | ok [cat,cat2] df | ok [-] np | ok [-] df |
| hgb|user=none|cf=None|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL] np | ok [LEVEL] df |
| hgb|user=none|cf=None|enc=ordinal ** | ok [cat,cat2] df | ok [-] np | ok [-] df |
| hgb|user=none|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=none|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| hgb|user=none|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| hgb|user=none|cf=list_partial|enc=None ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=none|cf=list_partial|enc=onehot ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| hgb|user=none|cf=list_partial|enc=ordinal_category ** | ok [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| hgb|user=none|cf=list_partial|enc=ordinal ** | ok [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_name|cf=None|enc=None ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_name|cf=None|enc=onehot ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_name|cf=None|enc=ordinal_category ** | INCONSISTENT [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| lgbm|user=fit_name|cf=None|enc=ordinal ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_name|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=fit_name|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=fit_name|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_name|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_name|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_name|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| lgbm|user=fit_name|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_pos|cf=None|enc=None ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=None|enc=onehot ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=None|enc=ordinal_category ** | INCONSISTENT [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=None|enc=ordinal ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| lgbm|user=fit_pos|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=fit_pos|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=fit_pos|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=fit_pos|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_pos|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=fit_pos|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| lgbm|user=fit_pos|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_name|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ERR LightGBMError |
| lgbm|user=init_name|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ERR LightGBMError |
| lgbm|user=init_name|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [LEVEL,cat] np | ok [LEVEL] df |
| lgbm|user=init_name|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ERR LightGBMError |
| lgbm|user=init_name|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=init_name|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=init_name|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_name|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_name|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_name|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| lgbm|user=init_name|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_pos|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| lgbm|user=init_pos|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| lgbm|user=init_pos|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [LEVEL,cat] np | ok [LEVEL] df |
| lgbm|user=init_pos|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| lgbm|user=init_pos|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=init_pos|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=init_pos|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=init_pos|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_pos|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=init_pos|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| lgbm|user=init_pos|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=none|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| lgbm|user=none|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| lgbm|user=none|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [LEVEL] np | ok [LEVEL] df |
| lgbm|user=none|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| lgbm|user=none|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=none|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| lgbm|user=none|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| lgbm|user=none|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=none|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| lgbm|user=none|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| lgbm|user=none|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| pipe_lgbm|user=none|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [-] np | ok [LEVEL] df |
| pipe_lgbm|user=none|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=auto|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=auto|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| pipe_lgbm|user=none|cf=auto|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_all|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_all|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| pipe_lgbm|user=none|cf=list_all|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_partial|enc=None ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_partial|enc=onehot ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| pipe_lgbm|user=none|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat] df | ok [-] np | ok [LEVEL] df |
| pipe_lgbm|user=none|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=None|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=None|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=None|enc=ordinal_category ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=None|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=auto|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=auto|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=auto|enc=ordinal_category ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=auto|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_all|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_all|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_all|enc=ordinal_category ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_all|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_partial|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_partial|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_partial|enc=ordinal_category ** | ok [-] df | ok [-] np | ok [-] df |
| ridge|user=none|cf=list_partial|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=init_pos|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [cat] np | ok [LEVEL] df |
| ttr_lgbm|user=init_pos|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=auto|enc=None ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=auto|enc=onehot ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=auto|enc=ordinal_category ** | ok [LEVEL] df | ok [cat] np | ok [LEVEL] df |
| ttr_lgbm|user=init_pos|cf=auto|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_all|enc=None ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_all|enc=onehot ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_all|enc=ordinal_category ** | ok [LEVEL] df | ok [cat] np | ok [LEVEL] df |
| ttr_lgbm|user=init_pos|cf=list_all|enc=ordinal ** | ok [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_partial|enc=None ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_partial|enc=onehot ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=init_pos|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat] df | ok [cat] np | ok [LEVEL] df |
| ttr_lgbm|user=init_pos|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| ttr_lgbm|user=none|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [-] np | ok [LEVEL] df |
| ttr_lgbm|user=none|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=auto|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=auto|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| ttr_lgbm|user=none|cf=auto|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_all|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_all|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| ttr_lgbm|user=none|cf=list_all|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_partial|enc=None ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_partial|enc=onehot ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ttr_lgbm|user=none|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat] df | ok [-] np | ok [LEVEL] df |
| ttr_lgbm|user=none|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [-] np | ok [LEVEL] df |
| ttr_xgb|user=none|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=auto|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=auto|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| ttr_xgb|user=none|cf=auto|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_all|enc=None ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_all|enc=onehot ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL] df | ok [-] np | ok [LEVEL] df |
| ttr_xgb|user=none|cf=list_all|enc=ordinal ** | ok [-] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_partial|enc=None ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_partial|enc=onehot ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| ttr_xgb|user=none|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat] df | ok [-] np | ok [LEVEL] df |
| ttr_xgb|user=none|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat] df | ok [-] np | ok [-] df |
| xgb|user=init_enable|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=init_enable|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=init_enable|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [LEVEL] np | ok [LEVEL] df |
| xgb|user=init_enable|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=init_enable|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=init_enable|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=init_enable|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_enable|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=init_enable|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=init_enable|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| xgb|user=init_enable|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=init_pos|cf=None|enc=None ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| xgb|user=init_pos|cf=None|enc=onehot ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| xgb|user=init_pos|cf=None|enc=ordinal_category ** | INCONSISTENT [cat] df | ok [LEVEL,cat] np | ok [cat] df |
| xgb|user=init_pos|cf=None|enc=ordinal ** | INCONSISTENT [cat] df | ok [cat] np | ok [cat] df |
| xgb|user=init_pos|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=init_pos|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=init_pos|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=init_pos|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=init_pos|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=init_pos|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| xgb|user=init_pos|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=none|cf=None|enc=None ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=none|cf=None|enc=onehot ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=none|cf=None|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat,cat2] df | ok [LEVEL] np | ok [LEVEL] df |
| xgb|user=none|cf=None|enc=ordinal ** | INCONSISTENT [cat,cat2] df | ok [-] np | ok [-] df |
| xgb|user=none|cf=auto|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=auto|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=auto|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=none|cf=auto|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=list_all|enc=None ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=list_all|enc=onehot ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=list_all|enc=ordinal_category ** | ok [LEVEL,cat,cat2] df | ok [LEVEL,cat,cat2] np | ok [LEVEL,cat,cat2] df |
| xgb|user=none|cf=list_all|enc=ordinal ** | ok [cat,cat2] df | ok [cat,cat2] np | ok [cat,cat2] df |
| xgb|user=none|cf=list_partial|enc=None ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=none|cf=list_partial|enc=onehot ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |
| xgb|user=none|cf=list_partial|enc=ordinal_category ** | INCONSISTENT [LEVEL,cat2] df | ok [LEVEL,cat2] np | ok [LEVEL,cat2] df |
| xgb|user=none|cf=list_partial|enc=ordinal ** | INCONSISTENT [cat2] df | ok [cat2] np | ok [cat2] df |

base {'ERR': 56, 'ok': 200, 'INCONSISTENT': 96} n = 352
branch {'ok': 326, 'ERR': 26} n = 352
minimal {'ok': 320, 'ERR': 32} n = 352

== errors base
  [15] CatBoostError: features data: pandas.DataFrame column 'exog_cat' has dtype 'category' but is not in  cat_features list
        catboost|user=none|cf=list_partial|enc=ordinal
        catboost|user=none|cf=list_partial|enc=ordinal_category
        catboost|user=none|cf=list_partial|enc=onehot
        catboost|user=none|cf=list_partial|enc=None
        catboost|user=none|cf=None|enc=ordinal
        catboost|user=none|cf=None|enc=onehot
        catboost|user=none|cf=None|enc=None
        catboost|user=fit_name|cf=list_partial|enc=ordinal
        catboost|user=fit_name|cf=list_partial|enc=ordinal_category
        catboost|user=fit_name|cf=list_partial|enc=onehot
        catboost|user=fit_name|cf=list_partial|enc=None
        catboost|user=fit_pos|cf=list_partial|enc=ordinal
        catboost|user=fit_pos|cf=list_partial|enc=ordinal_category
        catboost|user=fit_pos|cf=list_partial|enc=onehot
        catboost|user=fit_pos|cf=list_partial|enc=None
  [4] CatBoostError: features data: pandas.DataFrame column '_level_skforecast' has dtype 'category' but is not in  cat_features list
        catboost|user=none|cf=None|enc=ordinal_category
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
        catboost|user=fit_pos|cf=None|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_name|cf=auto|enc=ordinal
        catboost|user=init_name|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_name|cf=auto|enc=ordinal_category
        catboost|user=init_name|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_name|cf=auto|enc=onehot
        catboost|user=init_name|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_name|cf=auto|enc=None
        catboost|user=init_name|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6]
        catboost|user=init_name|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_name|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [8]
        catboost|user=init_name|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5]
        catboost|user=init_name|cf=list_partial|enc=None
  [9] CatBoostError: features data: pandas.DataFrame column 'exog_cat2' has dtype 'category' but is not in  cat_features list
        catboost|user=init_name|cf=None|enc=ordinal
        catboost|user=init_name|cf=None|enc=onehot
        catboost|user=init_name|cf=None|enc=None
        catboost|user=init_pos|cf=None|enc=ordinal
        catboost|user=init_pos|cf=None|enc=onehot
        catboost|user=init_pos|cf=None|enc=None
        catboost|user=fit_pos|cf=None|enc=ordinal
        catboost|user=fit_pos|cf=None|enc=onehot
        catboost|user=fit_pos|cf=None|enc=None
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_pos|cf=auto|enc=ordinal
        catboost|user=init_pos|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_pos|cf=auto|enc=ordinal_category
        catboost|user=init_pos|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_pos|cf=auto|enc=onehot
        catboost|user=init_pos|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_pos|cf=auto|enc=None
        catboost|user=init_pos|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6]
        catboost|user=init_pos|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_pos|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [8]
        catboost|user=init_pos|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [5]
        catboost|user=init_pos|cf=list_partial|enc=None
  [4] TypeError: list indices must be integers or slices, not str
        catboost|user=fit_name|cf=None|enc=ordinal
        catboost|user=fit_name|cf=None|enc=ordinal_category
        catboost|user=fit_name|cf=None|enc=onehot
        catboost|user=fit_name|cf=None|enc=None
== errors branch
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_name|cf=auto|enc=ordinal
        catboost|user=init_name|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_name|cf=auto|enc=ordinal_category
        catboost|user=init_name|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_name|cf=auto|enc=onehot
        catboost|user=init_name|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_name|cf=auto|enc=None
        catboost|user=init_name|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6]
        catboost|user=init_name|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_name|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [8]
        catboost|user=init_name|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5]
        catboost|user=init_name|cf=list_partial|enc=None
  [2] ValueError: With `encoding='ordinal_category'` and `categorical_features=None`, the series identifier (`_level_skforecast`) is added to the categorical features
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_pos|cf=auto|enc=ordinal
        catboost|user=init_pos|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_pos|cf=auto|enc=ordinal_category
        catboost|user=init_pos|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_pos|cf=auto|enc=onehot
        catboost|user=init_pos|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_pos|cf=auto|enc=None
        catboost|user=init_pos|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6]
        catboost|user=init_pos|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_pos|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [8]
        catboost|user=init_pos|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [5]
        catboost|user=init_pos|cf=list_partial|enc=None
== errors minimal
  [5] CatBoostError: features data: pandas.DataFrame column '_level_skforecast' has dtype 'category' but is not in  cat_features list
        catboost|user=none|cf=None|enc=ordinal_category
        catboost|user=init_name|cf=None|enc=ordinal_category
        catboost|user=init_pos|cf=None|enc=ordinal_category
        catboost|user=fit_name|cf=None|enc=ordinal_category
        catboost|user=fit_pos|cf=None|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_name|cf=auto|enc=ordinal
        catboost|user=init_name|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_name|cf=auto|enc=ordinal_category
        catboost|user=init_name|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_name|cf=auto|enc=onehot
        catboost|user=init_name|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_name|cf=auto|enc=None
        catboost|user=init_name|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6]
        catboost|user=init_name|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_name|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [8]
        catboost|user=init_name|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to ('exog_cat',). categorical features passed to fit function are set to [5]
        catboost|user=init_name|cf=list_partial|enc=None
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6]
        catboost|user=init_pos|cf=auto|enc=ordinal
        catboost|user=init_pos|cf=list_all|enc=ordinal
  [2] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [5, 6, 3]
        catboost|user=init_pos|cf=auto|enc=ordinal_category
        catboost|user=init_pos|cf=list_all|enc=ordinal_category
  [2] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [7, 8]
        catboost|user=init_pos|cf=auto|enc=onehot
        catboost|user=init_pos|cf=list_all|enc=onehot
  [2] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [4, 5]
        catboost|user=init_pos|cf=auto|enc=None
        catboost|user=init_pos|cf=list_all|enc=None
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6]
        catboost|user=init_pos|cf=list_partial|enc=ordinal
  [1] CatBoostError: categorical features in the model are set to (5,). categorical features passed to fit function are set to [6, 3]
        catboost|user=init_pos|cf=list_partial|enc=ordinal_category
  [1] CatBoostError: categorical features in the model are set to (7,). categorical features passed to fit function are set to [8]
        catboost|user=init_pos|cf=list_partial|enc=onehot
  [1] CatBoostError: categorical features in the model are set to (4,). categorical features passed to fit function are set to [5]
        catboost|user=init_pos|cf=list_partial|enc=None
  [3] LightGBMError: categorical_feature is not a number,
if you want to use a column name,
please add the prefix "name:" to the column name
        lgbm|user=init_name|cf=None|enc=ordinal
        lgbm|user=init_name|cf=None|enc=onehot
        lgbm|user=init_name|cf=None|enc=None
