################################################################################
#                             Population Drift Detector                        #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ecdf
from scipy.spatial.distance import jensenshannon
from sklearn.exceptions import NotFittedError
import warnings

from .. import __version__
from ..exceptions import UnknownLevelWarning
from ..utils import get_style_repr_html


def ks_2samp_from_ecdf(
    ecdf1: object, 
    ecdf2: object, 
    alternative: str = "two-sided"
) -> float:
    """
    Calculate the Kolmogorov-Smirnov distance (statistic) using precomputed
    scipy.stats.ecdf objects. This function replicates the behavior of
    scipy.stats.ks_2samp but uses ECDFs instead of raw data. Only the KS statistic
    is computed, not the p-value.

    Parameters
    ----------
    ecdf1 : scipy.stats.ecdf.ECDF
        ECDF object from sample 1.
    ecdf2 : scipy.stats.ecdf.ECDF
        ECDF object from sample 2.
    alternative : 'two-sided', 'less', 'greater', default 'two-sided'
        Defines the alternative hypothesis.

    Returns
    -------
    distance : float
        KS distance (sup of |F1-F2| for two-sided, etc.).
    
    """

    # Common evaluation grid (all jump points from both ECDFs)
    grid = np.union1d(ecdf1.cdf.quantiles, ecdf2.cdf.quantiles)

    cdf1 = ecdf1.cdf.evaluate(grid)
    cdf2 = ecdf2.cdf.evaluate(grid)

    if alternative == "two-sided":
        distance = np.max(np.abs(cdf1 - cdf2))
    elif alternative == "greater":
        distance = np.max(cdf1 - cdf2)
    elif alternative == "less":
        distance = np.max(cdf2 - cdf1)
    else:
        raise ValueError("Invalid alternative")

    return distance


class PopulationDriftDetector:
    """
    A class to detect population drift between reference and new datasets.
    This implementation computes Kolmogorov-Smirnov (KS) test for numeric features,
    Chi-Square test for categorical features, and Jensen-Shannon (JS) distance
    for all features. It calculates empirical distributions of these statistics
    from the reference data and uses quantile thresholds to determine drift in
    new data.
    
    This implementation is inspired by NannyML's DriftDetector. See Notes for
    details.

    For an in-depth explanation of the underlying calculations, see 
    https://skforecast.org/0.18.0/user_guides/drift-detection.html#deep-dive-into-temporal-drift-detection-in-time-series

    Parameters
    ----------
    chunk_size : int, string, pandas DateOffset, default None
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : float, default 0.95
        The quantile threshold (between 0 and 1) for determining drift based on 
        empirical distributions.
    
    Attributes
    ----------
    chunk_size : int, string, pandas DateOffset
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : float
        The quantile threshold (between 0 and 1) for determining drift based on 
        empirical distributions.
    is_fitted_ : bool
        Indicates if the detector has been fitted with reference data.
    ref_features_ : list
        List of features in the reference data.
    empirical_dist_ks_ : dict
        Empirical distributions of KS test statistics for each numeric feature in
        reference data.
    empirical_dist_chi2_ : dict
        Empirical distributions of Chi-Square test statistics for each categorical
        feature in reference data.
    empirical_dist_js_ : dict
        Empirical distributions of Jensen-Shannon distance for each feature in
        reference data (numeric and categorical).
    empirical_threshold_ks_ : dict
        Thresholds for KS statistics based on empirical distributions for each
        numeric feature in reference data.
    empirical_threshold_chi2_ : dict
        Thresholds for Chi-Square statistics based on empirical distributions for
        each categorical feature in reference data.
    empirical_threshold_js_ : dict
        Thresholds for Jensen-Shannon distance based on empirical distributions
        for each feature in reference data (numeric and categorical).
    n_chunks_reference_data_ : int
        Number of chunks in the reference data used during fitting to compute
        empirical distributions.
    ref_ecdf_ : dict
        Precomputed ECDFs for numeric features in the reference data.
    ref_bins_edges_ : dict
        Precomputed bin edges for numeric features in the reference data.
    ref_hist_ : dict
        Precomputed histograms for numeric features in the reference data.
    ref_probs_ : dict
        Precomputed normalized value counts (probabilities) for each category of
        categorical features in the reference data.
    ref_ranges_ : dict
        Min and max values for numeric features in the reference data.
    ref_categories_ : dict
        Unique categories for categorical features in the reference data.
    detectors_ : dict
        Dictionary of PopulationDriftDetector instances for each group when
        fitting/predicting on MultiIndex DataFrames.
    series_names_in_ : list
        List of series IDs present during fitting when using MultiIndex DataFrames.
    
    Notes
    -----
    This implementation is inspired by NannyML's DriftDetector [1]_.

    It is a lightweight version adapted for skforecast's needs:
    - It does not store the raw reference data, only the necessary precomputed
    information to calculate the statistics efficiently during prediction.
    - All empirical thresholds are calculated using the specified quantile from
    the empirical distributions obtained from the reference data chunks.
    - It includes checks for out of range values in numeric features and new
    categories in categorical features.
    - It supports multiple time series by fitting separate detectors for each
    series ID when provided with a MultiIndex DataFrame.

    If user requires more advanced features, such as multivariate drift detection
    or data quality checks, consider using https://nannyml.readthedocs.io/en/stable/
    directly.

    References
    ----------
    .. [1] NannyML API Reference.
           https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html
    
    """

    def __init__(
        self, 
        chunk_size=None, 
        threshold=0.95
    ) -> None:
        
        self.ref_features_             = None
        self.is_fitted_                = False
        self.ref_ecdf_                 = {}
        self.ref_bins_edges_           = {}
        self.ref_hist_                 = {}
        self.ref_probs_                = {}
        self.ref_counts_               = {}
        self.empirical_dist_ks_        = {}
        self.empirical_dist_chi2_      = {}
        self.empirical_dist_js_        = {}
        self.empirical_threshold_ks_   = {}
        self.empirical_threshold_chi2_ = {}
        self.empirical_threshold_js_   = {}
        self.ref_ranges_               = {}
        self.ref_categories_           = {}
        self.n_chunks_reference_data_  = None
        self.detectors_                = {}    # NOTE: Only used for multiseries
        self.series_names_in_          = None  # NOTE: Only used for multiseries

        if not (0 < threshold < 1):
            raise ValueError(f"`threshold` must be between 0 and 1. Got {threshold}.")
        
        self.threshold = threshold

        error_msg = (
            "`chunk_size` must be a positive integer, a string compatible with "
            "pandas DateOffset (e.g., 'D', 'W', 'M'), a pandas DateOffset object, or None."
        )
        if not (isinstance(chunk_size, (int, str, pd.DateOffset, type(None)))):
            raise ValueError(f"{error_msg} Got {type(chunk_size)}.")
        if isinstance(chunk_size, str):
            try:
                chunk_size = pd.tseries.frequencies.to_offset(chunk_size)
            except ValueError:
                raise ValueError(f"{error_msg} Got {type(chunk_size)}.")
        if isinstance(chunk_size, int) and chunk_size <= 0:
            raise ValueError(f"{error_msg} Got {chunk_size}.")
        
        self.chunk_size = chunk_size

    def __repr__(self) -> str:
        """
        Information displayed when a RangeDriftDetector object is printed.
        """
    
        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Fitted features = {self.ref_features_} \n"
            f"Is fitted       = {self.is_fitted_}"
        )

        return info
    
    def _repr_html_(self):
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        style, unique_id = get_style_repr_html(self.is_fitted_)
        content = f"""
        <div class="container-{unique_id}">
            <h2>{type(self).__name__}</h2>
            <details>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Fitted features:</strong> {self.ref_features_}</li>
                    <li><strong>Is fitted:</strong> {self.is_fitted_}</li>
                </ul>
            </details>
            <p>
                <a href="https://skforecast.org/{__version__}/api/drift_detection.html#skforecast.drift_detection._population_drift.PopulationDriftDetector">&#128712 <strong>API Reference</strong></a>
                &nbsp;&nbsp;
                <a href="https://skforecast.org/{__version__}/user_guides/drift-detection.html">&#128462 <strong>User Guide</strong></a>
            </p>
        </div>
        """
        
        return style + content
        
    def _fit(self, X) -> None:
        """
        Fit the drift detector by calculating empirical distributions and thresholds
        from reference data. The empirical distributions are computed by chunking
        the reference data according to the specified `chunk_size` and calculating
        the statistics for each chunk.

        Parameters
        ----------
        X : pandas DataFrame
            Reference data used as the baseline for drift detection.
        
        """

        self.ref_features_             = []
        self.is_fitted_                = False
        self.ref_ecdf_                 = {}
        self.ref_bins_edges_           = {}
        self.ref_hist_                 = {}
        self.ref_probs_                = {}
        self.ref_counts_               = {}
        self.empirical_dist_ks_        = {}
        self.empirical_dist_chi2_      = {}
        self.empirical_dist_js_        = {}
        self.empirical_threshold_ks_   = {}
        self.empirical_threshold_chi2_ = {}
        self.empirical_threshold_js_   = {}
        self.ref_ranges_               = {}
        self.ref_categories_           = {}
        self.n_chunks_reference_data_  = None
        self.detectors_                = {}    # NOTE: Only used for multiseries
        self.series_names_in_          = None  # NOTE: Only used for multiseries

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, pd.offsets.DateOffset) and not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError(
                    "`chunk_size` is a pandas DateOffset but `X` does not have a DatetimeIndex."
                )

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, int):
                chunks_ref = [
                    X.iloc[i : i + self.chunk_size]
                    for i in range(0, len(X), self.chunk_size)
                ]
            elif isinstance(
                self.chunk_size, (str, pd.offsets.DateOffset)
            ) and isinstance(X.index, pd.DatetimeIndex):
                chunks_ref = [group for _, group in X.resample(self.chunk_size)]
        else:
            chunks_ref = [X]

        self.n_chunks_reference_data_ = len(chunks_ref)

        features = X.columns.tolist()
        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(X[feature])
            ref = X[feature].dropna()
            if ref.empty:
                warnings.warn(
                    f"Feature '{feature}' contains only NaN values in the reference dataset. "
                    f"Drift detection skipped.",
                    UnknownLevelWarning
                )
                continue

            self.empirical_dist_ks_[feature] = []
            self.empirical_dist_chi2_[feature] = []
            self.empirical_dist_js_[feature] = []
            self.ref_features_.append(feature)

            if is_numeric:
                # Precompute histogram with bins for Jensen-Shannon distance
                # This may not perfectly align with bins used in predict if new data
                # extends the range, but it provides a reasonable approximation
                # for efficiency.
                min_val = ref.min()
                max_val = ref.max()
                bins_edges = np.histogram_bin_edges(ref.astype("float64"), bins='doane')
                ref_hist = np.histogram(ref, bins=bins_edges)[0] / len(ref)
                self.ref_bins_edges_[feature] = bins_edges
                self.ref_hist_[feature] = ref_hist
                self.ref_ranges_[feature] = (min_val, max_val)

                # Precompute ECDF for Kolmogorov-Smirnov test
                self.ref_ecdf_[feature] = ecdf(ref)
            else:
                counts_raw = ref.value_counts()
                counts_norm = counts_raw / counts_raw.sum()
                self.ref_counts_[feature] = counts_raw
                self.ref_probs_[feature] = counts_norm
                self.ref_categories_[feature] = counts_raw.index.tolist()

            for chunk in chunks_ref:
                new = chunk[feature].dropna()
                if new.empty:
                    continue
                ref = ref[~ref.index.isin(new.index)]
                ks_stat = np.nan
                chi2_stat = np.nan
                js_distance = np.nan

                if is_numeric:
                    new_ecdf = ecdf(new)
                    new_hist = np.histogram(new, bins=self.ref_bins_edges_[feature])[0] / len(new)
                    # Handle out-of-bin data: if new data contains values outside the reference range,
                    # they will not be counted in the histogram, leading to a sum < 1. To ensure
                    # the histograms are comparable, we add an extra bin for "out-of-range" data with
                    # the leftover probability mass in the new histogram and a corresponding zero bin
                    # in the reference histogram.
                    leftover = 1 - np.sum(new_hist)
                    if leftover > 0:
                        new_hist = np.append(new_hist, leftover)
                        ref_hist_appended = np.append(self.ref_hist_[feature], 0)
                        js_distance = jensenshannon(ref_hist_appended, new_hist, base=2)
                    else:
                        js_distance = jensenshannon(self.ref_hist_[feature], new_hist, base=2)

                    ks_stat = ks_2samp_from_ecdf(
                        ecdf1=self.ref_ecdf_.get(feature),
                        ecdf2=new_ecdf,
                        alternative="two-sided"
                    )
                else:
                    new_probs = new.value_counts(normalize=True).sort_index()
                    ref_probs = self.ref_probs_.get(feature)
                    # Align categories and fill missing with 0
                    all_cats = ref_probs.index.union(new_probs.index)
                    ref_probs = ref_probs.reindex(all_cats, fill_value=0)
                    new_probs = new_probs.reindex(all_cats, fill_value=0)
                    js_distance = jensenshannon(ref_probs.to_numpy(), new_probs.to_numpy())

                    # Align categories and fill missing with 0
                    new_counts = new.value_counts().reindex(all_cats, fill_value=0).to_numpy()
                    ref_counts = self.ref_counts_.get(feature).reindex(all_cats, fill_value=0).to_numpy()
                    if new_counts.sum() > 0 and ref_counts.sum() > 0:
                        # Create contingency table with rows = [reference, new], columns = categories
                        contingency_table = np.array([ref_counts, new_counts])
                        chi2_stat = chi2_contingency(contingency_table)[0]

                self.empirical_dist_ks_[feature].append(ks_stat)
                self.empirical_dist_chi2_[feature].append(chi2_stat)
                self.empirical_dist_js_[feature].append(js_distance)

            # Calculate empirical thresholds using the the specified quantile
            # Using pandas Series quantile method to handle NaNs properly and warnings
            self.empirical_threshold_ks_[feature] = pd.Series(
                self.empirical_dist_ks_[feature]
            ).quantile(self.threshold)
            self.empirical_threshold_chi2_[feature] = pd.Series(
                self.empirical_dist_chi2_[feature]
            ).quantile(self.threshold)
            self.empirical_threshold_js_[feature] = pd.Series(
                self.empirical_dist_js_[feature]
            ).quantile(self.threshold)

        self.is_fitted_ = True

    def fit(self, X) -> None:
        """
        Fit the drift detector by calculating empirical distributions and thresholds
        from reference data. The empirical distributions are computed by chunking
        the reference data according to the specified `chunk_size` and calculating
        the statistics for each chunk.

        Parameters
        ----------
        X : pandas DataFrame
            Reference data used as the baseline for drift detection.

        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"`X` must be a pandas DataFrame. Got {type(X)} instead."
            )

        if isinstance(X.index, pd.MultiIndex):
            X = X.groupby(level=0)

            for idx, group in X:
                group = group.droplevel(0)
                self.detectors_[idx] = PopulationDriftDetector(
                                           chunk_size = self.chunk_size,
                                           threshold  = self.threshold
                                       )
                self.detectors_[idx]._fit(group)
        else:
            self._fit(X)

        self.is_fitted_ = True
        self.series_names_in_ = list(self.detectors_.keys()) if self.detectors_ else None
        self._collect_attributes()

    def _predict(self, X) -> pd.DataFrame:
        """
        Predict drift in new data by comparing the estimated statistics to
        reference thresholds.

        Parameters
        ----------
        X : pandas DataFrame
            New data to compare against the reference.

        Returns
        -------
        results : pandas DataFrame
            DataFrame with the drift detection results for each chunk.

        """
        
        if self.chunk_size is not None:
            if isinstance(self.chunk_size, pd.offsets.DateOffset) and not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError(
                    "`chunk_size` is a pandas DateOffset but `X` does not have a DatetimeIndex."
                )

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, int):
                chunks = [
                    X.iloc[i:i + self.chunk_size]
                    for i in range(0, len(X), self.chunk_size)
                ]
            else:
                chunks = [group for _, group in X.resample(self.chunk_size)]
        else:
            chunks = [X]

        results = []
        features = X.columns.tolist()
        for feature in features:
            if feature not in self.ref_features_:
                warnings.warn(
                    f"Feature '{feature}' was not present during fitting. Drift detection skipped."
                    f"for this feature.",
                    UnknownLevelWarning
                )
                continue

            is_numeric = pd.api.types.is_numeric_dtype(X[feature])
            ref_bin_edges = self.ref_bins_edges_.get(feature, None)
            ref_hist = self.ref_hist_.get(feature, None)
            ref_probs = self.ref_probs_.get(feature, None)
            ref_counts = self.ref_counts_.get(feature, None)
            ref_ecdf = self.ref_ecdf_.get(feature, None)
            threshold_ks = self.empirical_threshold_ks_.get(feature, np.nan)
            threshold_chi2 = self.empirical_threshold_chi2_.get(feature, np.nan)
            threshold_js = self.empirical_threshold_js_.get(feature, np.nan)
            ref_range = self.ref_ranges_.get(feature, (np.nan, np.nan))

            for chunk_idx, chunk in enumerate(chunks):

                chunk_label = chunk_idx if self.chunk_size else "full"
                new = chunk[feature].dropna()
                ks_stat = np.nan
                chi2_stat = np.nan
                js_distance = np.nan
                is_out_of_range = np.nan

                if not new.empty:

                    if is_numeric:
                        new_ecdf = ecdf(new)
                        # Compute histogram for new data using reference bin edges and normalize
                        new_hist = np.histogram(new, bins=ref_bin_edges)[0] / len(new)
                        # Handle out-of-bin data: if new data contains values outside the reference range,
                        # they will not be counted in the histogram, leading to a sum < 1. To ensure
                        # the histograms are comparable, we add an extra bin for "out-of-range" data with
                        # the leftover probability mass in the new histogram and a corresponding zero bin
                        # in the reference histogram.
                        leftover = 1 - np.sum(new_hist)
                        if leftover > 0:
                            new_hist = np.append(new_hist, leftover)
                            ref_hist_appended = np.append(ref_hist, 0)
                            js_distance = jensenshannon(ref_hist_appended, new_hist, base=2)
                        else:
                            js_distance = jensenshannon(ref_hist, new_hist, base=2)

                        ks_stat = ks_2samp_from_ecdf(
                            ecdf1=ref_ecdf,
                            ecdf2=new_ecdf,
                            alternative="two-sided"
                        )
                        is_out_of_range = (
                            np.min(new) < ref_range[0] or
                            np.max(new) > ref_range[1]
                        )
                    else:
                        ref_categories = self.ref_categories_[feature]
                        ref_probs = ref_probs.reindex(ref_categories, fill_value=0).to_numpy()
                        # Map new data to reference categories
                        new_counts_dict = new.value_counts().to_dict()
                        new_counts_on_ref = [new_counts_dict.get(cat, 0) for cat in ref_categories]
                        new_probs = (
                            np.array(new_counts_on_ref) / len(new) if len(new) > 0
                            else np.zeros(len(ref_categories))
                        )
                        # Compute leftover (probability of new categories not in reference): if new data
                        # contains categories not seen in reference, they will not be counted in the
                        # histogram, leading to a sum < 1. To ensure the histograms are comparable,
                        # we add an extra bin for "new categories" with the leftover probability mass
                        # in the new histogram and a corresponding zero bin in the reference histogram.
                        leftover = 1 - np.sum(new_probs)
                        if leftover > 0:
                            new_probs = np.append(new_probs, leftover)
                            ref_probs_appended = np.append(ref_probs, 0)
                            js_distance = jensenshannon(ref_probs_appended, new_probs, base=2)
                        else:
                            js_distance = jensenshannon(ref_probs, new_probs, base=2)

                        all_cats = set(self.ref_categories_[feature]).union(set(new_counts_dict.keys()))
                        new_counts = new.value_counts().reindex(all_cats, fill_value=0).to_numpy()
                        ref_counts_aligned = ref_counts.reindex(all_cats, fill_value=0).to_numpy()
                        if new_counts.sum() > 0 and ref_counts_aligned.sum() > 0:
                            # Create contingency table: rows = [reference, new], columns = categories
                            contingency_table = np.array([ref_counts_aligned, new_counts])
                            chi2_stat = chi2_contingency(contingency_table)[0]

                results.append({
                    "chunk": chunk_label,
                    "chunk_start": chunk.index.min(),
                    "chunk_end": chunk.index.max(),
                    "feature": feature,
                    "ks_statistic": ks_stat,
                    "threshold_ks": threshold_ks,
                    "chi2_statistic": chi2_stat,
                    "threshold_chi2": threshold_chi2,
                    "jensen_shannon": js_distance,
                    "threshold_js": threshold_js,
                    "reference_range": ref_range,
                    "is_out_of_range": is_out_of_range,
                })

        results_df = pd.DataFrame(results)
        results_df['drift_ks_statistic'] = results_df['ks_statistic'] > results_df['threshold_ks']
        results_df['drift_chi2_statistic'] = results_df['chi2_statistic'] > results_df['threshold_chi2']
        results_df['drift_js'] = results_df['jensen_shannon'] > results_df['threshold_js']
        results_df['drift_detected'] = (
            results_df['drift_ks_statistic']
            | results_df['drift_chi2_statistic']
            | results_df['drift_js']
            | results_df['is_out_of_range']
        )
        
        return results_df

    def predict(self, X) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict drift in new data by comparing the estimated statistics to
        reference thresholds. Two dataframes are returned, the first one with
        detailed information of each chunk, the second only the total number
        of chunks where drift have been detected.

        Parameters
        ----------
        X : pandas DataFrame
            New data to compare against the reference.

        Returns
        -------
        results : pandas DataFrame
            DataFrame with the drift detection results for each chunk.
        summary : pandas DataFrame
            Summary DataFrame with the total number and percentage of chunks
            with detected drift per feature (or per series_id and feature if
            MultiIndex).
        
        """
        
        if not self.is_fitted_:
            raise NotFittedError(
                "This PopulationDriftDetector instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"`X` must be a pandas DataFrame. Got {type(X)} instead.")

        if isinstance(X.index, pd.MultiIndex):
            results = []
            for idx, group in X.groupby(level=0):
                group = group.droplevel(0)
                if idx not in self.detectors_:
                    warnings.warn(
                        f"Series '{idx}' was not present during fitting. Drift detection skipped.",
                        UnknownLevelWarning
                    )
                    continue

                detector = self.detectors_[idx]
                result = detector._predict(group)
                result.insert(0, 'series_id', idx)
                results.append(result)

            results = pd.concat(results, ignore_index=True)
        else:
            results = self._predict(X)

        if results.columns[0] == 'series_id':
            summary = (
                results.groupby(['series_id', 'feature'])['drift_detected']
                .agg(['sum', 'mean'])
                .reset_index()
                .rename(columns={'sum': 'n_chunks_with_drift', 'mean': 'pct_chunks_with_drift'})
            )
        else:
            summary = (
                results.groupby(['feature'])['drift_detected']
                .agg(['sum', 'mean'])
                .reset_index()
                .rename(columns={'sum': 'n_chunks_with_drift', 'mean': 'pct_chunks_with_drift'})
            )
        summary['pct_chunks_with_drift'] = summary['pct_chunks_with_drift'] * 100
        
        return results, summary

    def _collect_attributes(self) -> None:
        """
        Collect attributes for representation and inspection and update the instance
        dictionary with the collected values. For multi-series (when detectors_ is
        populated), attributes are aggregated into nested dictionaries keyed by
        detector names. For single-series, attributes remain unchanged.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        attr_names = [
            k 
            for k in self.__dict__.keys() 
            if k not in ['is_fitted_', 'detectors_', 'series_names_in_']
        ]
        
        if self.detectors_:
            for attr_name in attr_names:
                collected = {}
                for detector_key, detector in self.detectors_.items():
                    collected[detector_key] = getattr(detector, attr_name, None)
                self.__dict__[attr_name] = deepcopy(collected)
