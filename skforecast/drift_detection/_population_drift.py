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


class PopulationDriftDetectorFast:
    """
    A class to detect population drift between reference and new datasets.

    Parameters
    ----------
    chunk_size : int or str, optional
        Size of chunks for sequential drift analysis. If int, number of rows per chunk.
        If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks assuming a datetime index.
        If None, analyzes the full dataset as a single chunk. Default is None.
    threshold : float, optional
        The quantile threshold (between 0 and 1) for determining drift based on empirical distributions.
        Default is 0.95 (95th percentile).
    
    Attributes
    ----------
    chunk_size : int or str, optional
        Size of chunks for sequential drift analysis. If int, number of rows per chunk.
        If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks assuming a datetime index.
        If None, analyzes the full dataset as a single chunk. Default is None.
    threshold : float, optional
        The quantile threshold (between 0 and 1) for determining drift based on empirical distributions.
        Default is 0.95 (95th percentile).
    empirical_dist_ks_ : dict
        Empirical distributions of KS test statistics for each numeric feature in reference data.
    empirical_dist_chi2_ : dict
        Empirical distributions of Chi-Square test statistics for each categorical feature in reference data.
    empirical_dist_js_ : dict
        Empirical distributions of Jensen-Shannon divergence for each feature in reference data (numeric and categorical).
    empirical_threshold_ks_ : dict
        Thresholds for KS statistics based on empirical distributions for each numeric feature in reference data.
    empirical_threshold_chi2_ : dict
        Thresholds for Chi-Square statistics based on empirical distributions for each categorical feature in reference data.
    empirical_threshold_js_ : dict
        Thresholds for Jensen-Shannon divergence based on empirical distributions for each feature in reference data (numeric and categorical).
    X_ : pandas.DataFrame
        The reference DataFrame used for fitting.
    n_chunks_reference_data_ : int
        Number of chunks in the reference data.
    ref_hist_ : dict
        Precomputed histograms for numeric features in the reference data.
    ref_bins_ : dict
        Precomputed bin edges for numeric features in the reference data.
    ref_probs_ : dict
        Precomputed value counts (probabilities) for categorical features in the reference data.
    
    """

    # TODO: add RangeDriftDetector

    def __init__(self, chunk_size=None, threshold=0.95):
        self.chunk_size  = chunk_size
        self.threshold   = threshold
        self.ref_ecdf_   = {}
        self.ref_bins_edges_= {}
        self.ref_hist_   = {}
        self.ref_probs_  = {}
        self.ref_counts_ = {}
        self.empirical_dist_ks_    = {}
        self.empirical_dist_chi2_  = {}
        self.empirical_dist_js_    = {}
        self.empirical_threshold_ks_  = {}
        self.empirical_threshold_chi2_ = {}
        self.empirical_threshold_js_   = {}
        self.n_chunks_reference_data_ = None

    def fit(self, X):
        """
        Fit the drift detector by calculating empirical distributions and thresholds
        from reference data.

        Parameters
        ----------
        X : pandas.DataFrame
            Reference DataFrame (e.g., training data) used as the baseline for
            drift detection.
        """

        features = X.columns.tolist()
        self.X_ = X.copy()

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

                # Precompute ECDF for Kolmogorov-Smirnov test
                self.ref_ecdf_[feature] = ecdf(ref)
            else:
                counts_raw = ref.value_counts()
                counts_norm = counts_raw / counts_raw.sum()
                self.ref_counts_[feature] = counts_raw
                self.ref_probs_[feature] = counts_norm

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
            self.empirical_threshold_ks_[feature] = pd.Series(self.empirical_dist_ks_[feature]).quantile(self.threshold)
            self.empirical_threshold_chi2_[feature] = pd.Series(self.empirical_dist_chi2_[feature]).quantile(self.threshold)
            self.empirical_threshold_js_[feature] = pd.Series(self.empirical_dist_js_[feature]).quantile(self.threshold)

    def predict(self, X):
        """
        Predict drift in new data by comparing feature distributions to reference thresholds.

        Parameters
        ----------
        X : pandas.DataFrame
            New DataFrame (e.g., production or test data) to compare against the reference.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: 'chunk', 'feature', 'test_stat', 'p_value', 'js_divergence', 'drift_detected'.
            - 'chunk': Label for the chunk (e.g., 'chunk_0' or 'full').
            - 'feature': Feature name.
            - 'test_stat': Test statistic (KS for numeric, Chi-Square for categorical).
            - 'p_value': P-value from the statistical test.
            - 'js_divergence': Jensen-Shannon divergence (0 to 1, symmetric measure).
            - 'drift_detected': Boolean indicating drift (True if p_value < significance_level or js_divergence > js_threshold).
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

            for chunk_idx, chunk in enumerate(chunks):
                chunk_label = chunk_idx if self.chunk_size else "full"
                new = chunk[feature].dropna()
                
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
                    "ks_statistic": ks_stat if is_numeric else np.nan,
                    "chi2_statistic": chi2_stat if not is_numeric else np.nan,
                    "jensen_shannon": js_divergence,
                    "threshold_ks": threshold_ks if is_numeric else np.nan,
                    "threshold_chi2": threshold_chi2 if not is_numeric else np.nan,
                    "threshold_js": threshold_js,
                })
        results_df = pd.DataFrame(results)
        results_df['drift_ks_statistic'] = results_df['ks_statistic'] > results_df['threshold_ks']
        results_df['drift_chi2_statistic'] = results_df['chi2_statistic'] > results_df['threshold_chi2']
        results_df['drift_js'] = results_df['jensen_shannon'] > results_df['threshold_js']
        results_df['drift_detected'] = (
            results_df['drift_ks_statistic'] | results_df['drift_chi2_statistic'] | results_df['drift_js']
        )

        return results_df
    


# ---Deprecated ---
def _calculate_drift_metrics(ref, new):
    """
    Calculate drift metrics between reference and new data for a single feature.
    Always computes Kolmogorov-Smirnov (KS) test and Jensen-Shannon (JS) divergence for numeric features,
    and Chi-Square test and JS divergence for categorical features.

    Parameters
    ----------
    ref : pandas.Series
        Reference data for the feature.
    new : pandas.Series
        New data for the feature.

    Returns
    -------
    tuple
        A tuple containing the test statistic, p-value, and JS divergence.
    """

    if len(ref) == 0 or len(new) == 0:
        stat, p_value, js_div = np.nan, np.nan, np.nan
    else:
        # Always compute JS divergence
        if pd.api.types.is_numeric_dtype(ref):
            # JS divergence for numeric features using histograms
            bins = np.linspace(min(ref.min(), ref.min()), max(ref.max(), ref.max()), 50)
            ref_hist, _ = np.histogram(ref, bins=bins, density=True)
            new_hist, _ = np.histogram(new, bins=bins, density=True)
            js_div = jensenshannon(ref_hist + 1e-10, new_hist + 1e-10)
            # Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(ref, new)
        else:
            # JS divergence for categorical features using value counts
            ref_probs = ref.value_counts(normalize=True).sort_index()
            new_probs = new.value_counts(normalize=True).sort_index()
            all_cats = ref_probs.index.union(new_probs.index)
            ref_probs = ref_probs.reindex(all_cats, fill_value=0)
            new_probs = new_probs.reindex(all_cats, fill_value=0)
            js_div = jensenshannon(ref_probs.values, new_probs.values)
            # Chi-Square goodness-of-fit test
            ref_counts = ref.value_counts().reindex(all_cats, fill_value=0).values
            new_counts = new.value_counts().reindex(all_cats, fill_value=0).values
            if new_counts.sum() == 0 or ref_counts.sum() == 0:
                stat, p_value = np.nan, np.nan
            else:
                # Scale expected frequencies to match new sample size
                expected = ref_counts / ref_counts.sum() * new_counts.sum()
                stat, p_value = chisquare(f_obs=new_counts, f_exp=expected)

    return stat, p_value, js_div

class PopulationDriftDetector:
    """
    A class to detect population drift between reference and new datasets.

    Parameters
    ----------
    chunk_size : int or str, optional
        Size of chunks for sequential drift analysis. If int, number of rows per chunk.
        If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks assuming a datetime index.
        If None, analyzes the full dataset as a single chunk. Default is None.
    threshold : float, optional
        The quantile threshold (between 0 and 1) for determining drift based on empirical distributions.
        Default is 0.95 (95th percentile).
    
    Attributes
    ----------
    chunk_size : int or str, optional
        Size of chunks for sequential drift analysis. If int, number of rows per chunk.
        If str (e.g., 'D' for daily, 'W' for weekly), time-based chunks assuming a datetime index.
        If None, analyzes the full dataset as a single chunk. Default is None.
    threshold : float, optional
        The quantile threshold (between 0 and 1) for determining drift based on empirical distributions.
        Default is 0.95 (95th percentile).
    empirical_distributions_ : dict
        Empirical distributions of test statistics for each feature in reference data.
    empirical_thresholds_ : dict
        Thresholds for statistics based on empirical distributions.
    X_ : pandas.DataFrame
        The reference DataFrame used for fitting.
    n_chunks_reference_data_ : int
        Number of chunks in the reference data.
    """

    def __init__(self, chunk_size=None, threshold=0.95):
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.X_ = None
        self.empirical_distributions_ = {}
        self.empirical_thresholds_ = {}
        self.n_chunks_reference_data_ = None


    def fit(self, X):
        """
        Fit the drift detector by calculating empirical distributions and thresholds
        from reference data.

        Parameters
        ----------
        X : pandas.DataFrame
            Reference DataFrame (e.g., training data) used as the baseline for
            drift detection.
        """

        # TODO: almacenar la info necesaria para calcular las distancias sin tener que almacenar el raw data.


        features = X.columns.tolist()
        self.X_ = X.copy()

        if self.chunk_size is not None:
            if isinstance(self.chunk_size, int):
                # Row-based chunks
                chunks_ref = [
                    X.iloc[i:i+self.chunk_size]
                    for i in range(0, len(X), self.chunk_size)
                ]
            else:
                # Time-based chunks (assume datetime index)
                chunks_ref = [group for _, group in X.resample(self.chunk_size)]
        else:
            # Single chunk (whole dataset)
            chunks_ref = [X]

        self.n_chunks_reference_data_ = len(chunks_ref)
        self.empirical_distributions = {feature: [] for feature in features}
        self.upper_thresholds = {feature: None for feature in features}
        self.lower_thresholds = {feature: None for feature in features}

        for feature in features:
            ref = X[feature].dropna()
            for chunk in chunks_ref:
                new = chunk[feature].dropna()
                #TODO: Remove overlapping data points. Is this needed?
                stat, _, jensen_shannon = _calculate_drift_metrics(ref[~ref.index.isin(new.index)], new)
                self.empirical_distributions[feature].append([stat, jensen_shannon])

            self.empirical_distributions[feature] = pd.DataFrame(
                                                self.empirical_distributions[feature],
                                                columns=['stat', 'jensen_shannon']
                                               )
            self.empirical_thresholds_[feature] = (
                self.empirical_distributions[feature]
                [['stat', 'jensen_shannon']]
                .quantile(self.threshold)
            )

    def predict(self, X):
        """
        Predict drift in new data by comparing feature distributions to reference thresholds.

        Parameters
        ----------
        X : pandas.DataFrame
            New DataFrame (e.g., production or test data) to compare against the reference.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: 'chunk', 'feature', 'test_stat', 'p_value', 'js_divergence', 'drift_detected'.
            - 'chunk': Label for the chunk (e.g., 'chunk_0' or 'full').
            - 'feature': Feature name.
            - 'test_stat': Test statistic (KS for numeric, Chi-Square for categorical).
            - 'p_value': P-value from the statistical test.
            - 'js_divergence': Jensen-Shannon divergence (0 to 1, symmetric measure).
            - 'drift_detected': Boolean indicating drift (True if p_value < significance_level or js_divergence > js_threshold).
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
            ref = self.X_[feature].dropna()
            for chunk_idx, chunk in enumerate(chunks):
                chunk_label = chunk_idx if self.chunk_size else "full"
                new = chunk[feature].dropna()

                stat, _, jensen_shannon = _calculate_drift_metrics(ref, new)

                results.append({
                    "chunk": chunk_label,
                    "chunk_start": chunk.index.min(),
                    "chunk_end": chunk.index.max(),
                    "feature": feature,
                    "statistic": stat,
                    "threshold_stat": self.empirical_thresholds_[feature]['stat'],
                    "jensen_shannon": jensen_shannon,
                    "threshold_jensen_shannon": self.empirical_thresholds_[feature]['jensen_shannon'],
                })

        results_df = pd.DataFrame(results)
        results_df['drift_statistic'] = results_df['statistic'] > results_df['threshold_stat']
        results_df['drift_js'] = results_df['jensen_shannon'] > results_df['threshold_jensen_shannon']
        results_df['drift_detected'] = results_df['drift_statistic'] | results_df['drift_js']

        return results_df