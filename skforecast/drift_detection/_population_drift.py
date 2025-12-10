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
        raise ValueError(
            "Invalid `alternative`. Must be 'two-sided', 'less', or 'greater'."
        )

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
    https://skforecast.org/latest/user_guides/drift-detection.html#deep-dive-into-temporal-drift-detection-in-time-series

    Parameters
    ----------
    chunk_size : int, str, default None
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : int, float, default 3
        Threshold value for determining drift. Interpretation depends on
        `threshold_method`:
        
        - If `threshold_method='std'`: number of standard deviations above the mean.
        - If `threshold_method='quantile'`: quantile threshold (between 0 and 1).
    threshold_method : str, default 'std'
        Method for calculating thresholds from empirical distributions:
        
        - `'std'`: Uses mean + threshold * std of the empirical distribution.
        This is faster since it does not use leave-one-chunk-out.
        - `'quantile'`: Uses the specified quantile of the empirical distribution.
        Thresholds are computed using leave-one-chunk-out cross-validation to
        avoid self-comparison bias. This is statistically more correct for
        quantile-based thresholds but computationally more expensive.
    
    Attributes
    ----------
    chunk_size : int, str
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : float
        Threshold value for determining drift. Interpretation depends on
        `threshold_method`.
    threshold_method : str
        Method for calculating thresholds ('quantile' or 'std').
    is_fitted : bool
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
        chunk_size: int | str | None = None,
        threshold: int | float = 3,
        threshold_method: str = 'std'
    ) -> None:
        
        self.ref_features_             = []
        self.is_fitted                 = False
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

        error_msg = (
            "`chunk_size` must be a positive integer, a string compatible with "
            "pandas frequencies (e.g., 'D', 'W', 'MS'), or None."
        )
        if not (isinstance(chunk_size, (int, str, pd.DateOffset, type(None)))):
            raise TypeError(f"{error_msg} Got {type(chunk_size)}.")
        
        if isinstance(chunk_size, str):
            try:
                chunk_size = pd.tseries.frequencies.to_offset(chunk_size)
            except ValueError:
                raise ValueError(f"{error_msg} Got {type(chunk_size)}.")
        
        if isinstance(chunk_size, int) and chunk_size <= 0:
            raise ValueError(f"{error_msg} Got {chunk_size}.")
        
        self.chunk_size = chunk_size

        valid_threshold_methods = ['quantile', 'std']
        if threshold_method not in valid_threshold_methods:
            raise ValueError(
                f"`threshold_method` must be one of {valid_threshold_methods}. "
                f"Got '{threshold_method}'."
            )
        self.threshold_method = threshold_method

        if threshold_method == 'quantile':
            if not (0 < threshold < 1):
                raise ValueError(
                    f"When `threshold_method='quantile'`, `threshold` must be between "
                    f"0 and 1. Got {threshold}."
                )
        else:
            if threshold < 0:
                raise ValueError(
                    f"When `threshold_method='std'`, `threshold` must be >= 0. "
                    f"Got {threshold}."
                )
        
        self.threshold = threshold

    def __repr__(self) -> str:
        """
        Information displayed when a PopulationDriftDetector object is printed.
        """
    
        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Threshold        = {self.threshold} \n"
            f"Threshold method = {self.threshold_method} \n"
            f"Chunk size       = {self.chunk_size} \n"
            f"Is fitted        = {self.is_fitted} \n"
            f"Fitted features  = {self.ref_features_}"
        )

        return info
    
    def _repr_html_(self) -> str:
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.
        """

        style, unique_id = get_style_repr_html(self.is_fitted)
        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Threshold:</strong> {self.threshold}</li>
                    <li><strong>Threshold method:</strong> {self.threshold_method}</li>
                    <li><strong>Chunk size:</strong> {self.chunk_size}</li>
                    <li><strong>Is fitted:</strong> {self.is_fitted}</li>
                </ul>
            </details>
            <details>
                <summary>Fitted features</summary>
                <ul>
                    {self.ref_features_}
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

    def _reset_attributes(self) -> None:
        """
        Reset all fitted attributes to their initial state.
        """
        self.ref_features_             = []
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
        self.is_fitted                = False

    def _create_chunks(
        self, 
        X: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """
        Split X into chunks based on chunk_size.
        
        Parameters
        ----------
        X : pandas DataFrame
            Data to be chunked.
            
        Returns
        -------
        chunks : list
            List of DataFrames, each representing a chunk of the data.
        
        """
        
        if self.chunk_size is not None:
            if isinstance(self.chunk_size, pd.offsets.DateOffset) and not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError(
                    "`chunk_size` is a pandas frequency but `X` does not have a DatetimeIndex."
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

        return chunks

    def _compute_js_with_leftover(
        self, 
        ref_probs: np.ndarray, 
        new_probs: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon distance handling out-of-range/unseen values.
        
        If new data contains values outside the reference range (numeric) or
        unseen categories (categorical), they won't be counted in the probability
        distribution, leading to a sum < 1. To ensure distributions are comparable,
        we add an extra bin for "leftover" probability mass in the new distribution
        and a corresponding zero bin in the reference distribution.

        Parameters
        ----------
        ref_probs : numpy ndarray
            Probability distribution from reference data (histogram for numeric,
            normalized counts for categorical).
        new_probs : numpy ndarray
            Probability distribution from new data.

        Returns
        -------
        js_distance : float
            Jensen-Shannon distance between the two distributions.

        """

        leftover = 1 - np.sum(new_probs)
        if leftover > 0:
            new_probs_extended = np.append(new_probs, leftover)
            ref_probs_extended = np.append(ref_probs, 0)
            js_distance = jensenshannon(ref_probs_extended, new_probs_extended, base=2)
        else:
            js_distance = jensenshannon(ref_probs, new_probs, base=2)

        return js_distance
        
    def _fit(self, X: pd.DataFrame) -> None:
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

        self._reset_attributes()

        chunks = self._create_chunks(X)
        self.n_chunks_reference_data_ = len(chunks)

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
                bins_edges = np.histogram_bin_edges(ref.astype("float64"), bins='doane')
                ref_hist = np.histogram(ref, bins=bins_edges)[0] / len(ref)

                self.ref_bins_edges_[feature] = bins_edges
                self.ref_hist_[feature] = ref_hist
                self.ref_ranges_[feature] = (ref.min(), ref.max())

                # Precompute ECDF for Kolmogorov-Smirnov test
                self.ref_ecdf_[feature] = ecdf(ref)
            else:
                counts_raw = ref.value_counts()
                counts_norm = counts_raw / counts_raw.sum()

                self.ref_counts_[feature] = counts_raw
                self.ref_probs_[feature] = counts_norm
                self.ref_categories_[feature] = counts_raw.index.tolist()

            # NOTE: Precompute Leave One Chunk Out (LOO) indices once per feature 
            # (only for 'quantile' method)
            use_loo = self.threshold_method == 'quantile'
            if use_loo:
                chunk_indices = [chunk.index for chunk in chunks]
                # List of indices of the reference DataFrame excluding each time 
                # one of the chunk in order (Leave One Out)
                loo_indices = [
                    X.index.difference(chunk_indices[i])
                    for i in range(len(chunks))
                ]

            for i, chunk in enumerate(chunks):

                new = chunk[feature].dropna()
                if new.empty:
                    continue

                # If 'quantile', use LOO and exclude current chunk from reference data
                if use_loo:
                    ref_data_for_stats = X.loc[loo_indices[i], feature].dropna()
                    if ref_data_for_stats.empty:
                        continue
                else:
                    ref_data_for_stats = ref

                ks_stat = np.nan
                chi2_stat = np.nan
                js_distance = np.nan

                if is_numeric:
                    # Compute ECDF and histogram for reference
                    ref_stats_ecdf = ecdf(ref_data_for_stats)
                    ref_stats_hist = np.histogram(
                        ref_data_for_stats, bins=self.ref_bins_edges_[feature]
                    )[0] / len(ref_data_for_stats)

                    new_ecdf = ecdf(new)
                    new_hist = np.histogram(new, bins=self.ref_bins_edges_[feature])[0] / len(new)
                    
                    # Handle out-of-bin data: if new data contains values outside the reference range,
                    # they will not be counted in the histogram, leading to a sum < 1. To ensure
                    # the histograms are comparable, we add an extra bin for "out-of-range" data with
                    # the leftover probability mass in the new histogram and a corresponding zero bin
                    # in the reference histogram.
                    js_distance = self._compute_js_with_leftover(
                        ref_probs=ref_stats_hist, new_probs=new_hist
                    )

                    ks_stat = ks_2samp_from_ecdf(
                        ecdf1=ref_stats_ecdf,
                        ecdf2=new_ecdf,
                        alternative="two-sided"
                    )
                else:
                    # Compute counts and probs for reference
                    if use_loo:
                        ref_stats_counts = ref_data_for_stats.value_counts()
                        ref_stats_probs = ref_stats_counts / ref_stats_counts.sum()
                    else:
                        ref_stats_counts = self.ref_counts_[feature]
                        ref_stats_probs = self.ref_probs_[feature]

                    new_probs = new.value_counts(normalize=True).sort_index()

                    # Align categories and fill missing with 0
                    all_cats = ref_stats_probs.index.union(new_probs.index)
                    ref_probs_aligned = ref_stats_probs.reindex(all_cats, fill_value=0)
                    new_probs_aligned = new_probs.reindex(all_cats, fill_value=0)
                    js_distance = jensenshannon(
                        ref_probs_aligned.to_numpy(), new_probs_aligned.to_numpy(), base=2
                    )

                    # Align categories and fill missing with 0
                    new_counts = new.value_counts().reindex(all_cats, fill_value=0).to_numpy()
                    ref_counts_aligned = ref_stats_counts.reindex(all_cats, fill_value=0).to_numpy()
                    if new_counts.sum() > 0 and ref_counts_aligned.sum() > 0:
                        # Create contingency table with rows = [reference, new], columns = categories
                        contingency_table = np.array([ref_counts_aligned, new_counts])
                        chi2_stat = chi2_contingency(contingency_table)[0]

                self.empirical_dist_ks_[feature].append(ks_stat)
                self.empirical_dist_chi2_[feature].append(chi2_stat)
                self.empirical_dist_js_[feature].append(js_distance)

            if self.threshold_method == 'quantile':
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
            else:
                # Mean + k*std thresholds
                ks_values = self.empirical_dist_ks_[feature]
                chi2_values = self.empirical_dist_chi2_[feature]
                js_values = self.empirical_dist_js_[feature]
                
                # Suppress RuntimeWarnings when all values are NaN
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice')
                    warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
                    self.empirical_threshold_ks_[feature] = (
                        np.nanmean(ks_values) + self.threshold * np.nanstd(ks_values, ddof=0)
                    )
                    self.empirical_threshold_chi2_[feature] = (
                        np.nanmean(chi2_values) + self.threshold * np.nanstd(chi2_values, ddof=0)
                    )
                    self.empirical_threshold_js_[feature] = (
                        np.nanmean(js_values) + self.threshold * np.nanstd(js_values, ddof=0)
                    )

            # NOTE: Clip thresholds to their theoretical bounds
            # KS statistic is bounded in [0, 1]
            if not np.isnan(self.empirical_threshold_ks_[feature]):
                self.empirical_threshold_ks_[feature] = np.clip(
                    self.empirical_threshold_ks_[feature], 0, 1
                )
            
            # Jensen-Shannon distance is bounded in [0, 1]
            if not np.isnan(self.empirical_threshold_js_[feature]):
                self.empirical_threshold_js_[feature] = np.clip(
                    self.empirical_threshold_js_[feature], 0, 1
                )
            
            # Chi-square statistic is bounded in [0, inf), only clip lower bound
            if not np.isnan(self.empirical_threshold_chi2_[feature]):
                self.empirical_threshold_chi2_[feature] = np.clip(
                    self.empirical_threshold_chi2_[feature], 0, None
                )


        if self.n_chunks_reference_data_ < 10:
            warnings.warn(
                f"Only {self.n_chunks_reference_data_} chunks in reference data. "
                f"Empirical thresholds may not be reliable. Consider using more "
                f"data or smaller chunk_size."
            )

        self.is_fitted = True

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

            - If `X` is a regular DataFrame, a single detector is fitted for all data.
            The index is assumed to be the temporal index and each column a feature.
            - If `X` has a MultiIndex, the first level is assumed to be the series ID
            and the second level the temporal index. A separate detector is fitted for
            each series.        

        """

        self._reset_attributes()
        self.detectors_       = {}    # NOTE: Only used for multiseries
        self.series_names_in_ = None  # NOTE: Only used for multiseries

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"`X` must be a pandas DataFrame. Got {type(X)} instead."
            )

        if isinstance(X.index, pd.MultiIndex):
            X = X.groupby(level=0)

            for idx, group in X:
                group = group.droplevel(0)
                self.detectors_[idx] = PopulationDriftDetector(
                                           chunk_size       = self.chunk_size,
                                           threshold        = self.threshold,
                                           threshold_method = self.threshold_method
                                       )
                self.detectors_[idx]._fit(group)
        else:
            self._fit(X)

        self.is_fitted = True
        self.series_names_in_ = list(self.detectors_.keys()) if self.detectors_ else None
        self._collect_attributes()

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
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

        chunks = self._create_chunks(X)

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
            ks_threshold = self.empirical_threshold_ks_.get(feature, np.nan)
            chi2_threshold = self.empirical_threshold_chi2_.get(feature, np.nan)
            js_threshold = self.empirical_threshold_js_.get(feature, np.nan)
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

                        js_distance = self._compute_js_with_leftover(
                            ref_probs=ref_hist, new_probs=new_hist
                        )

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
                        ref_probs_ = ref_probs.reindex(ref_categories, fill_value=0).to_numpy()

                        # Map new data to reference categories
                        new_counts_dict = new.value_counts().to_dict()
                        new_counts_on_ref = [new_counts_dict.get(cat, 0) for cat in ref_categories]
                        new_probs = (
                            np.array(new_counts_on_ref) / len(new) if len(new) > 0
                            else np.zeros(len(ref_categories))
                        )

                        js_distance = self._compute_js_with_leftover(
                            ref_probs=ref_probs_, new_probs=new_probs
                        )

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
                    "ks_threshold": ks_threshold,
                    "chi2_statistic": chi2_stat,
                    "chi2_threshold": chi2_threshold,
                    "js_statistic": js_distance,
                    "js_threshold": js_threshold,
                    "reference_range": ref_range,
                    "is_out_of_range": is_out_of_range,
                })

        results_df = pd.DataFrame(results)
        results_df['drift_ks_statistic'] = results_df['ks_statistic'] > results_df['ks_threshold']
        results_df['drift_chi2_statistic'] = results_df['chi2_statistic'] > results_df['chi2_threshold']
        results_df['drift_js'] = results_df['js_statistic'] > results_df['js_threshold']
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
            MultiIndex), and the list of chunk IDs where drift was detected.
        
        """
        
        if not self.is_fitted:
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
            groupby_cols = ['series_id', 'feature']
        else:
            groupby_cols = ['feature']

        def _get_drift_chunk_ids(group):
            return group.loc[group['drift_detected'], 'chunk'].tolist()

        summary = (
            results.groupby(groupby_cols)
            .agg(
                n_chunks_with_drift=('drift_detected', 'sum'),
                pct_chunks_with_drift=('drift_detected', 'mean'),
                chunks_with_drift=('drift_detected', lambda x: _get_drift_chunk_ids(
                    results.loc[x.index]
                ))
            )
            .reset_index()
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
        self

        Returns
        -------
        None

        """

        attr_names = [
            k 
            for k in self.__dict__.keys() 
            if k not in ['is_fitted', 'detectors_', 'series_names_in_']
        ]
        
        if self.detectors_:
            for attr_name in attr_names:
                collected = {}
                for detector_key, detector in self.detectors_.items():
                    collected[detector_key] = getattr(detector, attr_name, None)
                
                self.__dict__[attr_name] = deepcopy(collected)

    def get_thresholds(
        self
    ) -> pd.DataFrame:
        """
        Return a DataFrame with all computed thresholds per feature.
        For multi-series, returns thresholds per series_id and feature.

        Parameters
        ----------
        self

        Returns
        -------
        thresholds : pandas DataFrame
            DataFrame with the computed thresholds per feature (and per series_id
            if MultiIndex was used during fitting).
        
        """

        if not self.is_fitted:
            raise NotFittedError(
                "This PopulationDriftDetector instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Multi-series case: ref_features_ is a dict keyed by series_id
        if self.detectors_:
            thresholds = {
                "series_id": [],
                "feature": [],
                "ks_threshold": [],
                "chi2_threshold": [],
                "js_threshold": []
            }
            for series_id, detector in self.detectors_.items():
                for feature in detector.ref_features_:
                    thresholds["series_id"].append(series_id)
                    thresholds["feature"].append(feature)
                    thresholds["ks_threshold"].append(
                        detector.empirical_threshold_ks_.get(feature)
                    )
                    thresholds["chi2_threshold"].append(
                        detector.empirical_threshold_chi2_.get(feature)
                    )
                    thresholds["js_threshold"].append(
                        detector.empirical_threshold_js_.get(feature)
                    )
        else:
            # Single-series case
            thresholds = {
                "feature": [],
                "ks_threshold": [],
                "chi2_threshold": [],
                "js_threshold": []
            }
            for feature in self.ref_features_:
                thresholds["feature"].append(feature)
                thresholds["ks_threshold"].append(
                    self.empirical_threshold_ks_.get(feature)
                )
                thresholds["chi2_threshold"].append(
                    self.empirical_threshold_chi2_.get(feature)
                )
                thresholds["js_threshold"].append(
                    self.empirical_threshold_js_.get(feature)
                )

        return pd.DataFrame(thresholds)
