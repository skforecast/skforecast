import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare, ecdf
from scipy.spatial.distance import jensenshannon


def ks_2samp_from_ecdf(ecdf1, ecdf2, alternative="two-sided"):
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
    alternative : {'two-sided', 'less', 'greater'}
        Defines the alternative hypothesis. Default is 'two-sided'.

    Returns
    -------
    distance : float
        KS distance (sup of |F1-F2| for two-sided, etc.).
    """
    # Common evaluation grid (all jump points from both ECDFs)
    grid = np.union1d(ecdf1.cdf.quantiles, ecdf2.cdf.quantiles)

    # Evaluate both CDFs on the grid
    cdf1 = ecdf1.cdf.evaluate(grid)
    cdf2 = ecdf2.cdf.evaluate(grid)

    # KS distance depending on alternative hypothesis
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
    Chi-Square test for categorical features, and Jensen-Shannon (JS) divergence
    for all features. It calculates empirical distributions of these statistics
    from the reference data and uses quantile thresholds to determine drift in
    new data.
    
    This implementation focuses on computational efficiency by precomputing necessary
    information during fitting without storing the raw reference data.

    Parameters
    ----------
    chunk_size : int, string, pandas DateOffset, None, default None
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : float, default 0.95
        The quantile threshold (between 0 and 1) for determining drift based on 
        empirical distributions.
    
    Attributes
    ----------
    chunk_size : int, string, pandas DateOffset, None, default None
        Size of chunks for sequential drift analysis. If int, number of rows per
        chunk. If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks
        assuming a datetime index. If None, analyzes the full dataset as a single
        chunk.
    threshold : float, default 0.95
        The quantile threshold (between 0 and 1) for determining drift based on 
        empirical distributions.
    empirical_dist_ks_ : dict
        Empirical distributions of KS test statistics for each numeric feature in
        reference data.
    empirical_dist_chi2_ : dict
        Empirical distributions of Chi-Square test statistics for each categorical
        feature in reference data.
    empirical_dist_js_ : dict
        Empirical distributions of Jensen-Shannon divergence for each feature in
        reference data (numeric and categorical).
    empirical_threshold_ks_ : dict
        Thresholds for KS statistics based on empirical distributions for each
        numeric feature in reference data.
    empirical_threshold_chi2_ : dict
        Thresholds for Chi-Square statistics based on empirical distributions for
        each categorical feature in reference data.
    empirical_threshold_js_ : dict
        Thresholds for Jensen-Shannon divergence based on empirical distributions
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
    
    Notes
    -----
    This implementation is inspired by NannyML's DriftDetector
    https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html

    It is a lightweight version adapted for skforecast's needs:
    - It does not store the raw reference data, only the necessary precomputed
    information to calculate the statistics efficiently during prediction.
    - All empirical thresholds are calculated using the specified quantile from
    the empirical distributions obtained from the reference data chunks.
    - It also check out of range values in numeric features and new categories in
    categorical features.
    """

    def __init__(self, chunk_size=None, threshold=0.95):
        self.chunk_size                = chunk_size
        self.threshold                 = threshold
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

    def fit(self, X):
        """
        Fit the drift detector by calculating empirical distributions and thresholds
        from reference data. The empirical distributions are computed by chunking
        the reference data according to the specified `chunk_size` and calculating
        the statistics for each chunk.

        Parameters
        ----------
        X : pandas.DataFrame
            Reference data used as the baseline for drift detection.
        """

        features = X.columns.tolist()

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

        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(X[feature])
            ref = X[feature].dropna()
            self.empirical_dist_ks_[feature] = []
            self.empirical_dist_chi2_[feature] = []
            self.empirical_dist_js_[feature] = []

            if is_numeric:
                # Precompute histogram with bins for Jensen-Shannon divergence
                # This may not perfectly align with bins used in predict if new data
                # extends the range, but it provides a reasonable approximation
                # for efficiency.
                min_val = ref.min()
                max_val = ref.max()
                bins_edges = np.linspace(min_val, max_val, 50)
                ref_hist, _ = np.histogram(
                    ref, bins=bins_edges, range=(min_val, max_val), density=True
                )
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
                self.ref_categories_[feature] = set(counts_raw.index)

            for chunk in chunks_ref:
                new = chunk[feature].dropna()
                ref = ref[~ref.index.isin(new.index)]
                ks_stat = np.nan
                chi2_stat = np.nan
                js_divergence = np.nan

                if is_numeric:
                    new_ecdf = ecdf(new)
                    new_hist, _ = np.histogram(
                        new,
                        bins=self.ref_bins_edges_.get(feature),
                        range=(
                            self.ref_bins_edges_.get(feature)[0],
                            self.ref_bins_edges_.get(feature)[-1]
                        ),
                        density=True
                    )       
                    ks_stat = ks_2samp_from_ecdf(
                        ecdf1=self.ref_ecdf_.get(feature),
                        ecdf2=new_ecdf,
                        alternative="two-sided"
                    )
                    js_divergence = jensenshannon(
                        p = self.ref_hist_.get(feature) + 1e-10,
                        q = new_hist + 1e-10
                    )
                else:
                    new_probs = new.value_counts(normalize=True).sort_index()
                    ref_probs = self.ref_probs_.get(feature)
                    # Align categories and fill missing with 0
                    all_cats = ref_probs.index.union(new_probs.index)
                    ref_probs = ref_probs.reindex(all_cats, fill_value=0)
                    new_probs = new_probs.reindex(all_cats, fill_value=0)
                    js_divergence = jensenshannon(ref_probs.values, new_probs.values)

                    # Align categories and fill missing with 0
                    new_counts = new.value_counts().reindex(all_cats, fill_value=0).values
                    ref_counts = self.ref_counts_.get(feature).reindex(all_cats, fill_value=0).values
                    if new_counts.sum() > 0 and ref_counts.sum() > 0:
                        expected = ref_counts / ref_counts.sum() * new_counts.sum()
                        chi2_stat, _ = chisquare(f_obs=new_counts, f_exp=expected)

                self.empirical_dist_ks_[feature].append(ks_stat)
                self.empirical_dist_chi2_[feature].append(chi2_stat)
                self.empirical_dist_js_[feature].append(js_divergence)

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

    def predict(self, X):
        """
        Predict drift in new data by comparing the estimated statistics to
        reference thresholds.

        Parameters
        ----------
        X : pandas.DataFrame
            New data to compare against the reference.

        Returns
        -------
        results : pandas.DataFrame
            DataFrame with the drift detection results for each chunk.

        """

        features = X.columns.tolist()
        results = []

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, int):
                chunks = [
                    X.iloc[i:i+self.chunk_size]
                    for i in range(0, len(X), self.chunk_size)
                ]
            else:
                chunks = [group for _, group in X.resample(self.chunk_size)]
        else:
            chunks = [X]

        for feature in features:
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
                js_divergence = np.nan
                is_out_of_range = False

                if is_numeric:
                    new_ecdf = ecdf(new)
                    new_hist, _ = np.histogram(
                        new,
                        bins=ref_bin_edges,
                        range=(ref_bin_edges[0], ref_bin_edges[-1]) if ref_bin_edges is not None else None,
                        density=True
                    )
                    ks_stat = ks_2samp_from_ecdf(
                        ecdf1=ref_ecdf,
                        ecdf2=new_ecdf,
                        alternative="two-sided"
                    )
                    js_divergence = jensenshannon(
                        p = ref_hist + 1e-10,
                        q = new_hist + 1e-10
                    )
                    is_out_of_range = (
                        np.min(new) < ref_range[0] or
                        np.max(new) > ref_range[1]
                    )
                else:
                    new_probs = new.value_counts(normalize=True).sort_index()
                    all_cats = ref_probs.index.union(new_probs.index)
                    ref_probs_aligned = ref_probs.reindex(all_cats, fill_value=0)
                    new_probs_aligned = new_probs.reindex(all_cats, fill_value=0)
                    js_divergence = jensenshannon(ref_probs_aligned.values, new_probs_aligned.values)

                    new_counts = new.value_counts().reindex(all_cats, fill_value=0).values
                    ref_counts_aligned = ref_counts.reindex(all_cats, fill_value=0).values
                    if new_counts.sum() > 0 and ref_counts_aligned.sum() > 0:
                        expected = ref_counts_aligned / ref_counts_aligned.sum() * new_counts.sum()
                        chi2_stat, _ = chisquare(f_obs=new_counts, f_exp=expected)
                    else:
                        chi2_stat = np.nan
                        js_divergence = np.nan

                results.append({
                    "chunk": chunk_label,
                    "chunk_start": chunk.index.min(),
                    "chunk_end": chunk.index.max(),
                    "feature": feature,
                    "ks_statistic": ks_stat,
                    "chi2_statistic": chi2_stat,
                    "jensen_shannon": js_divergence,
                    "threshold_ks": threshold_ks if is_numeric else np.nan,
                    "threshold_chi2": threshold_chi2 if not is_numeric else np.nan,
                    "threshold_js": threshold_js,
                    "reference_range": ref_range,
                    "is_out_of_range": is_out_of_range,
                })

        results_df = pd.DataFrame(results)
        results_df['drift_ks_statistic'] = results_df['ks_statistic'] > results_df['threshold_ks']
        results_df['drift_chi2_statistic'] = results_df['chi2_statistic'] > results_df['threshold_chi2']
        results_df['drift_js'] = results_df['jensen_shannon'] > results_df['threshold_js']
        results_df['drift_detected'] = (
            results_df['drift_ks_statistic'] | results_df['drift_chi2_statistic'] | results_df['drift_js']
        )

        return results_df
