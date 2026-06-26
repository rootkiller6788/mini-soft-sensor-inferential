/**
 * @file dr_measurement.h
 * @brief Measurement data structures and statistical operations.
 *
 * Handles measurement value representation, uncertainty propagation,
 * outlier detection, and covariance matrix construction.
 *
 * Key concepts from metrology (GUM / JCGM 100:2008):
 *   - Type A uncertainty: evaluated by statistical analysis of observations
 *   - Type B uncertainty: evaluated by means other than statistical analysis
 *   - Combined standard uncertainty: sqrt(sum of squared components)
 *   - Expanded uncertainty: k * u_c (typically k=2 for ~95% confidence)
 *
 * Outlier detection methods:
 *   - IQR (Interquartile Range) method (Tukey, 1977)
 *   - Grubbs test (Grubbs, 1950; ISO 5725-2)
 *   - Mahalanobis distance for multivariate outliers
 *
 * References:
 *   [1] JCGM 100:2008. "Evaluation of measurement data - Guide to the
 *       expression of uncertainty in measurement (GUM)."
 *   [2] ISO 5725-2:2019. "Accuracy (trueness and precision) of measurement
 *       methods and results."
 *   [3] Grubbs, F.E. (1950). "Sample Criteria for Testing Outlying
 *       Observations." Annals of Math. Statistics, 21(1), 27-58.
 *   [4] Tukey, J.W. (1977). "Exploratory Data Analysis." Addison-Wesley.
 *   [5] Box, G.E.P. (1954). "Some Theorems on Quadratic Forms Applied in
 *       the Study of Analysis of Variance Problems." Annals of Math.
 *       Statistics, 25(2), 290-302.
 */

#ifndef DR_MEASUREMENT_H
#define DR_MEASUREMENT_H

#include "dr_core.h"

/* ============================================================================
 * L1: Measurement Statistics Structures
 * ============================================================================ */

/**
 * @brief Summary statistics for a set of measurements.
 *
 * Computed from a sample of repeated measurements of the same quantity.
 */
typedef struct {
    int     n;          /**< Number of observations */
    double  mean;       /**< Sample mean x_bar = (1/n) * sum x_i */
    double  stddev;     /**< Sample standard deviation s = sqrt(sum(x_i-x_bar)^2/(n-1)) */
    double  min_val;    /**< Minimum observed value */
    double  max_val;    /**< Maximum observed value */
    double  q1;         /**< First quartile (25th percentile) */
    double  median;     /**< Median (50th percentile) */
    double  q3;         /**< Third quartile (75th percentile) */
    double  skewness;   /**< Sample skewness: measure of asymmetry */
    double  kurtosis;   /**< Excess kurtosis: measure of tail weight */
} dr_meas_stats_t;

/**
 * @brief Correlation structure between pairs of measurements.
 *
 * For data reconciliation with correlated measurements,
 * the full covariance matrix must be used instead of diagonal.
 */
typedef struct {
    int      nvar;         /**< Number of variables */
    double  *covariance;   /**< Covariance matrix (row-major, nvar x nvar) */
    double  *correlation;  /**< Correlation matrix (row-major, nvar x nvar) */
    int      is_diagonal;  /**< 1 if covariance is strictly diagonal */
} dr_correlation_t;

/* ============================================================================
 * L2: Measurement Construction and Statistics
 * ============================================================================ */

/**
 * @brief Compute summary statistics from an array of repeated measurements.
 *
 * Implements the standard one-pass algorithm for mean, two-pass for variance
 * (more numerically stable than one-pass), and a sorting-based approach for
 * quartiles.
 *
 * @param values   Array of n observation values.
 * @param n        Number of observations (n >= 2).
 * @param stats    Output: summary statistics structure.
 * @return         DR_OK or error code.
 *
 * Complexity: O(n log n) due to sorting for quartiles.
 */
int dr_meas_compute_statistics(const double *values, int n, dr_meas_stats_t *stats);

/**
 * @brief Detect outliers using the IQR method (Tukey's fences).
 *
 * An observation x is flagged as an outlier if:
 *   x < Q1 - k * IQR   or   x > Q3 + k * IQR
 *
 * where IQR = Q3 - Q1 and typically k = 1.5 (mild) or k = 3.0 (extreme).
 *
 * @param values    Array of n observations.
 * @param n         Number of observations.
 * @param k         Multiplier (1.5 for standard Tukey fences).
 * @param outlier   Output array of n ints: 1 if outlier, 0 otherwise.
 * @param n_out     Output: number of outliers detected.
 * @return          DR_OK or error code.
 *
 * Complexity: O(n log n).
 *
 * Reference: Tukey, J.W. (1977). "Exploratory Data Analysis." Addison-Wesley.
 */
int dr_meas_detect_outliers_iqr(const double *values, int n, double k,
                                int *outlier, int *n_out);

/**
 * @brief Grubbs test for a single outlier in univariate data.
 *
 * Test statistic: G = max_i |x_i - x_bar| / s
 *
 * Critical value at significance level α:
 *   G_crit = (n-1)/sqrt(n) * sqrt(t^2_{α/(2n), n-2} / (n-2 + t^2_{α/(2n), n-2}))
 *
 * where t_{α/(2n), n-2} is the upper critical value of the t-distribution
 * with n-2 degrees of freedom at significance level α/(2n).
 *
 * H0: No outliers. H1: The most extreme value is an outlier.
 *
 * @param values     Array of n observations.
 * @param n          Number of observations (n >= 3).
 * @param alpha      Significance level (e.g., 0.05).
 * @param idx_out    Output: index of suspected outlier, or -1 if none.
 * @param g_stat     Output: computed Grubbs statistic G.
 * @param g_crit     Output: critical value at level alpha.
 * @return           DR_OK or error code.
 *
 * Complexity: O(n).
 *
 * Reference: Grubbs, F.E. (1950). "Sample Criteria for Testing Outlying
 * Observations." Annals of Math. Statistics, 21(1), 27-58.
 */
int dr_meas_grubbs_test(const double *values, int n, double alpha,
                        int *idx_out, double *g_stat, double *g_crit);

/* ============================================================================
 * L3: Covariance and Correlation Operations
 * ============================================================================ */

/**
 * @brief Build a full covariance matrix from measurement standard deviations
 * and pairwise correlation coefficients.
 *
 * Cov(i,j) = rho(i,j) * sigma_i * sigma_j
 *
 * where rho(i,j) is the correlation coefficient (-1 <= rho <= 1),
 * and Cov(i,i) = sigma_i^2.
 *
 * @param stddev  Array of n standard deviations (sigma_i > 0).
 * @param rho     Correlation matrix (row-major, n x n). Pass NULL for
 *                independent measurements (diagonal covariance).
 * @param n       Number of variables.
 * @param cov_out Output covariance matrix (row-major, n x n).
 * @return        DR_OK or error code.
 *
 * Complexity: O(n^2).
 */
int dr_meas_build_covariance(const double *stddev, const double *rho,
                             int n, double *cov_out);

/**
 * @brief Convert a covariance matrix to a correlation matrix.
 *
 * Corr(i,j) = Cov(i,j) / sqrt(Cov(i,i) * Cov(j,j))
 *
 * Diagonal entries are always 1.0 by definition.
 *
 * @param cov    Covariance matrix (row-major, n x n).
 * @param n      Dimension.
 * @param corr   Output correlation matrix (row-major, n x n).
 * @return       DR_OK or error code.
 *
 * Complexity: O(n^2).
 */
int dr_meas_covariance_to_correlation(const double *cov, int n, double *corr);

/**
 * @brief Standardize measurements using mean and standard deviation.
 *
 * z_i = (x_i - mu) / sigma
 *
 * Results in zero mean and unit variance for the standardized values.
 * This is a preprocessing step for PCA-based methods and for numerical
 * conditioning of the constraint matrix.
 *
 * @param values  Input: raw values. Output: standardized values.
 * @param mean    Mean value mu.
 * @param stddev  Standard deviation sigma.
 * @param n       Number of values.
 * @return        DR_OK or error code (DR_ERR_NULL_POINTER if stddev==0).
 *
 * Complexity: O(n).
 *
 * Reference: Jackson, J.E. (1991). "A User's Guide to Principal Components."
 * Wiley.
 */
int dr_meas_standardize(double *values, double mean, double stddev, int n);

/* ============================================================================
 * L4: Distribution Critical Values
 * ============================================================================ */

/**
 * @brief Compute the critical value of the chi-squared distribution
 * using the Wilson-Hilferty approximation.
 *
 * Approximation (Wilson & Hilferty, 1931):
 *   chi2_{nu}(alpha) ≈ nu * (1 - 2/(9*nu) + z_alpha * sqrt(2/(9*nu)))^3
 *
 * where z_alpha is the standard normal quantile for upper tail probability alpha.
 *
 * Used for the global test in data reconciliation.
 *
 * @param nu     Degrees of freedom (nu > 0).
 * @param alpha  Significance level (e.g., 0.05 for 95% confidence).
 * @return       Approximate critical value.
 *
 * Complexity: O(1).
 *
 * Reference: Wilson, E.B., Hilferty, M.M. (1931). "The Distribution of
 * Chi-Square." Proc. Nat. Acad. Sci., 17(12), 684-688.
 */
double dr_meas_chi2_critical(int nu, double alpha);

/**
 * @brief Compute the critical value of the standard normal distribution
 * using the Abramowitz and Stegun rational approximation.
 *
 * Approximation (Abramowitz & Stegun §26.2.23):
 * Uses a rational Chebyshev approximation with maximum absolute error
 * less than 4.5e-4 for z in [-inf, inf].
 *
 * @param alpha  Upper tail probability (e.g., 0.025 for two-tailed 95%).
 * @return       z-score for the given tail probability.
 *
 * Complexity: O(1).
 *
 * Reference: Abramowitz, M., Stegun, I.A. (1964). "Handbook of Mathematical
 * Functions." Dover, §26.2.23.
 */
double dr_meas_normal_critical(double alpha);

/**
 * @brief Compute the critical value of the Student t-distribution
 * using the Hill (1970) approximation.
 *
 * Approximation accurate to within 0.001 for nu >= 1.
 * Used for the measurement test critical values.
 *
 * @param nu     Degrees of freedom.
 * @param alpha  Upper tail probability.
 * @return       Approximate critical t-value.
 *
 * Complexity: O(1).
 *
 * Reference: Hill, G.W. (1970). "Algorithm 395: Student's t-Distribution."
 * Comm. ACM, 13(10), 617-619.
 */
double dr_meas_t_critical(int nu, double alpha);

/**
 * @brief Compute measurement uncertainty following GUM guidelines.
 *
 * Combines Type A uncertainty (from repeated measurements) and Type B
 * uncertainty (from instrument specifications, calibration certificates,
 * etc.) using the root-sum-of-squares formula:
 *
 *   u_c = sqrt(u_A^2 + u_B1^2 + u_B2^2 + ...)
 *
 * @param u_type_a      Type A standard uncertainty (standard deviation of mean).
 * @param u_type_b      Array of Type B standard uncertainty components.
 * @param n_type_b      Number of Type B components.
 * @return              Combined standard uncertainty u_c.
 *
 * Complexity: O(n_type_b).
 *
 * Reference: JCGM 100:2008, Clause 5.
 */
double dr_meas_combined_uncertainty(double u_type_a, const double *u_type_b,
                                    int n_type_b);

/**
 * @brief Scale measurement uncertainty for different confidence levels.
 *
 * Expanded uncertainty: U = k * u_c
 *
 * Common coverage factors k:
 *   k = 1: approx. 68% confidence
 *   k = 2: approx. 95% confidence (most common in industry)
 *   k = 3: approx. 99.7% confidence
 *
 * @param u_combined  Combined standard uncertainty.
 * @param k           Coverage factor.
 * @return            Expanded uncertainty U.
 *
 * Complexity: O(1).
 */
double dr_meas_expanded_uncertainty(double u_combined, double k);

/**
 * @brief Compute the variance inflation factor (VIF) and condition number
 * of the covariance matrix.
 *
 * VIF_i = diagonal element of the inverse of the correlation matrix.
 * Measures how much the variance of a coefficient is inflated due to
 * correlation with other variables. VIF > 10 indicates serious
 * multicollinearity.
 *
 * Condition number: kappa = lambda_max / lambda_min
 * (ratio of largest to smallest eigenvalue). Large kappa indicates
 * ill-conditioning.
 *
 * @param cov    Covariance matrix (row-major, n x n).
 * @param n      Dimension.
 * @param vif    Output: VIF values (length n).
 * @param cond   Output: condition number estimate.
 * @return       DR_OK or error code.
 *
 * Complexity: O(n^3) for matrix inversion.
 *
 * Reference: Belsley, D.A., Kuh, E., Welsch, R.E. (1980). "Regression
 * Diagnostics." Wiley.
 */
int dr_meas_vif_condition(const double *cov, int n, double *vif, double *cond);

#endif /* DR_MEASUREMENT_H */
