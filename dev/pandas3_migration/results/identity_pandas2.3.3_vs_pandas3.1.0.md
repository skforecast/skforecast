# Identity check: pandas 2.3.3 (A) vs pandas 3.1.0 (B)

Totals: {'outputs': 110, 'strict_equal': 37, 'normalized_equal': 69, 'different': 4, 'errors': 0}

| case | output | result | dtype/index differences |
|---|---|---|---|
| recursive_numeric_exog | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | exog_dtypes_in_ | identical |  |
| recursive_numeric_exog | exog_dtypes_out_ | identical |  |
| recursive_numeric_exog | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | predict_interval_bootstrapping | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | predict_interval_binned | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | predict_interval_conformal | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_numeric_exog | in_sample_residuals_ | identical |  |
| recursive_numeric_exog | in_sample_residuals_by_bin_ | identical |  |
| recursive_numeric_exog | create_predict_X | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_calendar_features | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_calendar_features | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_categorical_exog_lgbm | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_categorical_exog_lgbm | exog_dtypes_in_ | identical |  |
| recursive_categorical_exog_lgbm | exog_dtypes_out_ | identical |  |
| recursive_categorical_exog_lgbm | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_text_exog_lgbm | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_text_exog_lgbm | exog_dtypes_in_ | values identical, dtype/unit differ |  |
| recursive_text_exog_lgbm | exog_dtypes_out_ | identical |  |
| recursive_text_exog_lgbm | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_text_exog_linear_no_categorical | error | identical |  |
| direct | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| direct | y_train | values identical, dtype/unit differ |  |
| direct | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| direct | predict_interval_conformal | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| direct | predict_interval_bootstrapping | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | exog_dtypes_in_ | identical |  |
| multiseries_wide_ordinal | exog_dtypes_out_ | identical |  |
| multiseries_wide_ordinal | series_names_in_ | identical |  |
| multiseries_wide_ordinal | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | predict_interval_bootstrapping | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal | in_sample_residuals_ | identical |  |
| multiseries_wide_ordinal | level_row_slices | identical |  |
| multiseries_wide_ordinal_category | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal_category | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal_category | exog_dtypes_in_ | identical |  |
| multiseries_wide_ordinal_category | exog_dtypes_out_ | identical |  |
| multiseries_wide_ordinal_category | series_names_in_ | identical |  |
| multiseries_wide_ordinal_category | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal_category | predict_interval_bootstrapping | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal_category | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_ordinal_category | in_sample_residuals_ | identical |  |
| multiseries_wide_ordinal_category | level_row_slices | identical |  |
| multiseries_wide_onehot | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_onehot | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_onehot | exog_dtypes_in_ | identical |  |
| multiseries_wide_onehot | exog_dtypes_out_ | identical |  |
| multiseries_wide_onehot | series_names_in_ | identical |  |
| multiseries_wide_onehot | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_onehot | predict_interval_bootstrapping | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_onehot | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_wide_onehot | in_sample_residuals_ | identical |  |
| multiseries_wide_onehot | level_row_slices | identical |  |
| multiseries_dict_ordinal | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_dict_ordinal | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_dict_ordinal | exog_dtypes_in_ | identical |  |
| multiseries_dict_ordinal | exog_dtypes_out_ | identical |  |
| multiseries_dict_ordinal | series_names_in_ | identical |  |
| multiseries_dict_ordinal | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_dict_ordinal | predict_interval_bootstrapping | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_dict_ordinal | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_dict_ordinal | in_sample_residuals_ | identical |  |
| multiseries_dict_ordinal | level_row_slices | identical |  |
| multiseries_long_format | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_long_format | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_long_format | exog_dtypes_in_ | identical |  |
| multiseries_long_format | exog_dtypes_out_ | identical |  |
| multiseries_long_format | series_names_in_ | identical |  |
| multiseries_long_format | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_long_format | predict_interval_bootstrapping | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_long_format | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| multiseries_long_format | in_sample_residuals_ | identical |  |
| multiseries_long_format | level_row_slices | identical |  |
| multiseries_long_format | reshaped_series | values identical, dtype/unit differ |  |
| multiseries_long_format | reshaped_exog | values identical, dtype/unit differ |  |
| direct_multivariate | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| direct_multivariate | y_train | values identical, dtype/unit differ |  |
| direct_multivariate | predict | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| direct_multivariate | predict_interval_conformal | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_classifier | X_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_classifier | y_train | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_classifier | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| recursive_classifier | predict_proba | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| stats_arima | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| stats_arima | predict_interval | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| equivalent_date | predict | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| backtesting_recursive | metric | identical |  |
| backtesting_recursive | predictions | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| backtesting_multiseries | metric | values identical, dtype/unit differ | dtypes: {'levels': ('object', 'str')} |
| backtesting_multiseries | predictions | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| backtesting_stats | metric | identical |  |
| backtesting_stats | predictions | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| grid_search_recursive | results | values identical, dtype/unit differ | dtypes: {'lags': ('object', 'str'), 'params': ('object', 'str')} |
| grid_search_multiseries | results | values identical, dtype/unit differ | dtypes: {'lags': ('object', 'str'), 'params': ('object', 'str'), 'levels': ('object', 'str')} |
| save_load_joblib_skops | joblib_recursive_predict_after_load | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| save_load_joblib_skops | joblib_recursive_exog_dtypes_in_after_load | values identical, dtype/unit differ |  |
| save_load_joblib_skops | joblib_recursive_file_size | DIFFERENT VALUES |  |
| save_load_joblib_skops | joblib_multiseries_predict_after_load | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| save_load_joblib_skops | joblib_multiseries_exog_dtypes_in_after_load | identical |  |
| save_load_joblib_skops | joblib_multiseries_file_size | DIFFERENT VALUES |  |
| save_load_joblib_skops | skops_recursive_predict_after_load | values identical, dtype/unit differ | index_dtype: datetime64[ns] -> datetime64[us] |
| save_load_joblib_skops | skops_recursive_exog_dtypes_in_after_load | values identical, dtype/unit differ |  |
| save_load_joblib_skops | skops_recursive_file_size | DIFFERENT VALUES |  |
| save_load_joblib_skops | skops_multiseries_predict_after_load | values identical, dtype/unit differ | dtypes: {'level': ('object', 'str')}; index_dtype: datetime64[ns] -> datetime64[us] |
| save_load_joblib_skops | skops_multiseries_exog_dtypes_in_after_load | identical |  |
| save_load_joblib_skops | skops_multiseries_file_size | DIFFERENT VALUES |  |

## Deprecation warnings present in only one run

| case | run | warning |
|---|---|---|
| recursive_numeric_exog | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive.py:1109) |
| recursive_numeric_exog | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive.py:2042) |
| recursive_calendar_features | only pandas 3.1.0 | Pandas4Warning: DatetimeIndex.dayofweek is deprecated and will be removed in a future version. Use DatetimeIndex.day_of_week instead. (_calendar.py:247) |
| recursive_categorical_exog_lgbm | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive.py:1109) |
| recursive_text_exog_lgbm | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive.py:1109) |
| direct | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_direct.py:1520) |
| multiseries_wide_ordinal | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| multiseries_wide_ordinal_category | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| multiseries_wide_onehot | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| multiseries_dict_ordinal | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| multiseries_long_format | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| direct_multivariate | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_direct_multivariate.py:1637) |
| recursive_classifier | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_classifier.py:1129) |
| equivalent_date | only pandas 3.1.0 | Pandas4Warning: isinstance(obj, DateOffset) is deprecated for offsets that are not DateOffset instances and will return False in a futur (_forecaster_equivalent_date.py:218) |
| equivalent_date | only pandas 3.1.0 | Pandas4Warning: isinstance(obj, DateOffset) is deprecated for offsets that are not DateOffset instances and will return False in a futur (_forecaster_equivalent_date.py:382) |
| equivalent_date | only pandas 3.1.0 | Pandas4Warning: isinstance(obj, DateOffset) is deprecated for offsets that are not DateOffset instances and will return False in a futur (_forecaster_equivalent_date.py:412) |
| equivalent_date | only pandas 3.1.0 | Pandas4Warning: isinstance(obj, DateOffset) is deprecated for offsets that are not DateOffset instances and will return False in a futur (_forecaster_equivalent_date.py:506) |
| equivalent_date | only pandas 3.1.0 | Pandas4Warning: isinstance(obj, DateOffset) is deprecated for offsets that are not DateOffset instances and will return False in a futur (_forecaster_equivalent_date.py:670) |
| backtesting_multiseries | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| grid_search_multiseries | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
| save_load_joblib_skops | only pandas 3.1.0 | Pandas4Warning: The copy keyword is deprecated and will be removed in a future version. Copy-on-Write is active in pandas since 3.0 whic (_forecaster_recursive_multiseries.py:1567) |
