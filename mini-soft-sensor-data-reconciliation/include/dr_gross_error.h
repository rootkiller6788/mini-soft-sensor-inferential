/**
 * @file dr_gross_error.h
 * @brief Gross error detection and identification for data reconciliation.
 *
 * Gross errors are non-random measurement errors caused by sensor faults,
 * process leaks, biases, or other systematic effects. They invalidate the
 * standard Gaussian error assumption and must be detected and identified
 * before reconciliation results can be trusted.
 *
 * Detection strategies:
 *   1. Global Test (GT): Tests if ANY gross error exists in the system
 *   2. Nodal Test (NT): Tests each constraint node individually
 *   3. Measurement Test (MT): Tests each measurement individually
 *
 * Identification strategies:
 *   1. Serial elimination: Remove most suspect measurement, re-reconcile
 *   2. Simultaneous estimation: Estimate gross error magnitudes alongside
 *      reconciled values
 *   3. Robust M-estimation: Use robust loss functions to down-weight
 *      outliers automatically
 *
 * Statistical framework:
 *   H0: y = x + epsilon, epsilon ~ N(0, Sigma)  (no gross errors)
 *   H1: y = x + delta + epsilon                   (gross error delta present)
 *
 * Test statistics and their null distributions:
 *   - Global Test:  z_GT = r' * V^{-1} * r ~ chi^2(nu), nu = m
 *   - Nodal Test:   z_NT(j) = r_j / sqrt(V_jj) ~ N(0,1)
 *   - Measurement Test: z_MT(i) = |d_i| / sqrt(W_ii) ~ N(0,1)
 *
 * where:
 *   r = A*y - b  (constraint residuals)
 *   V = A*Sigma*A' (residual covariance)
 *   d = Sigma * A' * V^{-1} * r (measurement adjustments)
 *   W = Sigma * A' * V^{-1} * A * Sigma (adjustment covariance)
 *
 * References:
 *   [1] Mah, R.S.H., Tamhane, A.C. (1982). "Detection of Gross Errors in
 *       Process Data." AIChE Journal, 28(5), 828-835.
 *   [2] Narasimhan, S., Mah, R.S.H. (1987). "Generalized Likelihood Ratio
 *       Method for Gross Error Identification." AIChE Journal, 33(9),
 *       1514-1521.
 *   [3] Tong, H., Crowe, C.M. (1995). "Detection of Gross Errors in Data
 *       Reconciliation by Principal Component Analysis." AIChE Journal,
 *       41(7), 1712-1722.
 *   [4] Ozyurt, D.B., Pike, R.W. (2004). "Theory and Practice of
 *       Simultaneous Data Reconciliation and Gross Error Detection."
 *       Comp. Chem. Eng., 28(3), 381-402.
 *   [5] Huber, P.J., Ronchetti, E.M. (2009). "Robust Statistics," 2nd ed.
 *       Wiley.
 *   [6] Beaton, A.E., Tukey, J.W. (1974). "The Fitting of Power Series,
 *       Meaning Polynomials, Illustrated on Band-Spectroscopic Data."
 *       Technometrics, 16(2), 147-185.
 *   [7] Fair, R.C. (1974). "On the Robust Estimation of Econometric
 *       Models." Annals Econ. Social Measurement, 3(4), 667-677.
 */

#ifndef DR_GROSS_ERROR_H
#define DR_GROSS_ERROR_H

#include "dr_core.h"

/* ============================================================================
 * L1: Gross Error Test Structures
 * ============================================================================ */

/**
 * @brief Result of a statistical test for gross errors.
 */
typedef struct {
    int     test_type;      /**< 0=Global, 1=Nodal, 2=Measurement, 3=Collective */
    double  statistic;      /**< Computed test statistic value */
    double  critical_value; /**< Critical value at significance level alpha */
    double  p_value;        /**< Approximate p-value */
    int     reject_H0;      /**< 1 if H0 rejected (gross error suspected), 0 otherwise */
    int     suspect_index;  /**< Index of suspect node/measurement (-1 = none/global) */
} dr_ge_test_result_t;

/**
 * @brief Parameters for robust M-estimation based data reconciliation.
 */
typedef struct {
    int     max_iterations;    /**< Maximum IRLS iterations (default 50) */
    double  convergence_tol;   /**< Convergence tolerance (default 1e-6) */
    double  tuning_constant;   /**< Tuning constant for the robust function (default 1.345) */
    int     robust_function;   /**< 0=Huber, 1=Biweight, 2=Cauchy, 3=Fair */
    int     use_median_init;   /**< 1 = use median for initial scale estimate */
} dr_ge_robust_params_t;

/**
 * @brief Result of gross error identification via serial elimination.
 */
typedef struct {
    int     n_errors_found;    /**< Number of gross errors identified */
    int    *error_indices;     /**< Indices of measurements with gross errors */
    double *error_magnitudes;  /**< Estimated magnitudes of gross errors */
    double *error_stddevs;     /**< Standard deviations of magnitude estimates */
    double  final_objective;   /**< WLS objective after error removal */
    int     iterations;        /**< Number of elimination rounds */
} dr_ge_identification_t;

/* ============================================================================
 * L2: Statistical Tests for Gross Error Detection
 * ============================================================================ */

/**
 * @brief Global Test (GT): chi-squared test for any gross error.
 *
 * H0: No gross errors present (all delta_i = 0).
 * H1: At least one gross error present.
 *
 * Test statistic: z = r^T * V^{-1} * r
 * Under H0: z ~ chi^2(nu) where nu = rank(A * Sigma * A^T).
 *
 * If z > chi^2_{nu, alpha}, reject H0 → at least one gross error.
 *
 * @param prob     Problem description with constraints and covariances.
 * @param result   Output test result.
 * @param alpha    Significance level (e.g., 0.05).
 * @return         DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2) for V^{-1} computation.
 *
 * Theorem (Mah & Tamhane, 1982): Under H0 and Gaussian measurement errors,
 * the test statistic follows exactly a chi-squared distribution with
 * nu = m degrees of freedom.
 */
int dr_ge_global_test(const dr_problem_t *prob, dr_ge_test_result_t *result,
                      double alpha);

/**
 * @brief Nodal Test (NT): test each constraint node for gross error.
 *
 * Tests each constraint residual individually:
 *   z_NT(j) = r_j / sqrt(V_jj) ~ N(0,1) under H0.
 *
 * Bonferroni correction: test each at alpha/m to control family-wise
 * error rate.
 *
 * @param prob        Problem description.
 * @param results     Output array of test results (length ncon).
 * @param alpha       Family-wise significance level.
 * @return            DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2) for V decomposition, then O(m^2) per node.
 *
 * Reference: Mah, R.S.H., Stanley, G.M., Downing, D.M. (1976).
 * "Reconciliation and Rectification of Process Flow and Inventory Data."
 * Ind. Eng. Chem. Proc. Des. Dev., 15(1), 175-183.
 */
int dr_ge_nodal_test(const dr_problem_t *prob, dr_ge_test_result_t *results,
                     double alpha);

/**
 * @brief Measurement Test (MT): test each measurement for gross error.
 *
 * Tests each individual measurement:
 *   z_MT(i) = |d_i| / sqrt(W_ii) ~ N(0,1) under H0.
 *
 * where d = Sigma * A^T * V^{-1} * r is the adjustment vector,
 * and W = Sigma * A^T * V^{-1} * A * Sigma is its covariance.
 *
 * Under H0 for measurement i, no gross error.
 *
 * @param prob        Problem description.
 * @param results     Output array of test results (length nvar).
 * @param alpha       Significance level per measurement.
 * @return            DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2 + n^2 * m).
 *
 * Reference: Tamhane, A.C., Mah, R.S.H. (1985). "Data Reconciliation
 * and Gross Error Detection in Chemical Process Networks."
 * Technometrics, 27(4), 409-422.
 */
int dr_ge_measurement_test(const dr_problem_t *prob,
                           dr_ge_test_result_t *results, double alpha);

/* ============================================================================
 * L3: Gross Error Identification
 * ============================================================================ */

/**
 * @brief Serial elimination strategy for gross error identification.
 *
 * Algorithm:
 *   1. Perform global test
 *   2. If significant, compute measurement test statistics
 *   3. Remove the most suspect measurement (largest z_MT)
 *   4. Re-reconcile without that measurement
 *   5. Repeat until global test is not significant or max iterations reached
 *
 * @param prob     Original problem with all measurements.
 * @param ident    Output: identification results.
 * @param alpha    Significance level for global test.
 * @return         DR_OK or error code.
 *
 * Complexity: O(k * (m^3 + n * m^2)) where k is number of errors found.
 *
 * Reference: Serth, R.W., Heenan, W.A. (1986). "Gross Error Detection
 * and Data Reconciliation in Steam-Metering Systems." AIChE Journal,
 * 32(5), 733-742.
 */
int dr_ge_serial_elimination(const dr_problem_t *prob,
                             dr_ge_identification_t *ident, double alpha);

/**
 * @brief Simultaneous gross error estimation using the Collective Test
 * approach (Keller et al., 1994).
 *
 * Estimates gross error magnitudes simultaneously with reconciliation
 * by solving an augmented linear system.
 *
 * @param prob       Problem description.
 * @param ident      Output: identification results with error magnitudes.
 * @param suspects   Array of suspected measurement indices.
 * @param n_suspects Number of suspected measurements.
 * @return           DR_OK or error code.
 *
 * Complexity: O((m + n_s)^3).
 *
 * Reference: Keller, J.Y., et al. (1994). "A General Approach to Data
 * Reconciliation and Gross Error Detection." Comp. Chem. Eng., 18(7),
 * 653-666.
 */
int dr_ge_simultaneous_estimation(const dr_problem_t *prob,
                                  dr_ge_identification_t *ident,
                                  const int *suspects, int n_suspects);

/**
 * @brief Allocate and initialize a gross error identification structure.
 *
 * @param nvar  Number of variables.
 * @return      Pointer to allocated structure, or NULL.
 */
dr_ge_identification_t *dr_ge_identification_create(int nvar);

/**
 * @brief Free a gross error identification structure.
 *
 * @param ident  Structure to free. Safe to pass NULL.
 */
void dr_ge_identification_free(dr_ge_identification_t *ident);

/* ============================================================================
 * L4: Robust M-Estimation for Data Reconciliation
 * ============================================================================ */

/**
 * @brief Robust data reconciliation using the Huber M-estimator.
 *
 * The Huber function limits the influence of large residuals:
 *   rho(r) = { 0.5 * r^2                    if |r| <= c
 *            { c * |r| - 0.5 * c^2          if |r| >  c
 *   psi(r) = rho'(r) = { r                  if |r| <= c
 *                       { c * sign(r)        if |r| >  c
 *   w(r)   = psi(r)/r  = { 1                if |r| <= c
 *                         { c / |r|          if |r| >  c
 *
 * where c is the tuning constant (typically 1.345 for 95% efficiency
 * at the Gaussian).
 *
 * Solved via Iteratively Reweighted Least Squares (IRLS).
 *
 * @param prob       Problem description.
 * @param x_out      Output: robust reconciled values (length nvar).
 * @param params     Robust estimation parameters.
 * @return           DR_OK or error code.
 *
 * Complexity: O(k * (m^3 + n * m^2)) where k is IRLS iterations.
 *
 * Reference: Huber, P.J. (1964). "Robust Estimation of a Location
 * Parameter." Annals of Math. Statistics, 35(1), 73-101.
 */
int dr_ge_robust_huber(const dr_problem_t *prob, double *x_out,
                       const dr_ge_robust_params_t *params);

/**
 * @brief Robust data reconciliation using Tukey's biweight (bisquare) function.
 *
 * The biweight function completely rejects observations beyond c:
 *   rho(r) = { (c^2/6) * [1 - (1 - (r/c)^2)^3]   if |r| <= c
 *            { c^2/6                                if |r| >  c
 *   w(r)   = { [1 - (r/c)^2]^2                      if |r| <= c
 *            { 0                                    if |r| >  c
 *
 * Tuning constant c = 4.685 for 95% efficiency at Gaussian.
 *
 * @param prob    Problem description.
 * @param x_out   Output: robust reconciled values (length nvar).
 * @param c       Tuning constant.
 * @param max_iter Maximum IRLS iterations.
 * @param tol     Convergence tolerance.
 * @return        DR_OK or error code.
 *
 * Complexity: O(k * (m^3 + n * m^2)).
 *
 * Reference: Beaton, A.E., Tukey, J.W. (1974). "The Fitting of Power
 * Series." Technometrics, 16(2), 147-185.
 */
int dr_ge_robust_biweight(const dr_problem_t *prob, double *x_out,
                          double c, int max_iter, double tol);

/**
 * @brief Robust data reconciliation using the Cauchy M-estimator.
 *
 * The Cauchy (Lorentzian) function:
 *   rho(r) = (c^2/2) * log(1 + (r/c)^2)
 *   w(r)   = 1 / (1 + (r/c)^2)
 *
 * This function has a heavy tail and never completely rejects any
 * observation, but down-weights extreme values smoothly.
 *
 * @param prob    Problem description.
 * @param x_out   Output: robust reconciled values (length nvar).
 * @param c       Tuning constant (typically 2.385).
 * @param max_iter Maximum IRLS iterations.
 * @param tol     Convergence tolerance.
 * @return        DR_OK or error code.
 *
 * Complexity: O(k * (m^3 + n * m^2)).
 */
int dr_ge_robust_cauchy(const dr_problem_t *prob, double *x_out,
                        double c, int max_iter, double tol);

/**
 * @brief Robust data reconciliation using the Fair function.
 *
 * The Fair function:
 *   rho(r) = c^2 * [|r|/c - log(1 + |r|/c)]
 *   w(r)   = 1 / (1 + |r|/c)
 *
 * This function has a linear asymptotic behavior for large residuals,
 * similar to the Huber function but smoother.
 *
 * @param prob    Problem description.
 * @param x_out   Output: robust reconciled values (length nvar).
 * @param c       Tuning constant.
 * @param max_iter Maximum IRLS iterations.
 * @param tol     Convergence tolerance.
 * @return        DR_OK or error code.
 *
 * Complexity: O(k * (m^3 + n * m^2)).
 *
 * Reference: Fair, R.C. (1974). "Robust Estimation of Econometric Models."
 * Annals Econ. Social Measurement, 3(4), 667-677.
 */
int dr_ge_robust_fair(const dr_problem_t *prob, double *x_out,
                      double c, int max_iter, double tol);

/**
 * @brief Initialize robust estimation parameters with default values.
 *
 * Sets: max_iterations=50, convergence_tol=1e-6, tuning_constant=1.345,
 *       robust_function=0 (Huber), use_median_init=1.
 *
 * @param params  Output: initialized parameters.
 */
void dr_ge_robust_params_init(dr_ge_robust_params_t *params);

/* ============================================================================
 * L5: Advanced Detection Methods
 * ============================================================================ */

/**
 * @brief Compute the power of the global test against a specified
 * gross error magnitude.
 *
 * Power = P(reject H0 | H1 is true)
 *       = P(chi^2_{nu}(lambda) > chi^2_{nu, alpha})
 *
 * where lambda = delta^T * A^T * V^{-1} * A * delta is the non-centrality
 * parameter.
 *
 * @param prob      Problem description.
 * @param delta     Hypothetical gross error vector.
 * @param alpha     Significance level.
 * @param power_out Output: test power (0-1).
 * @return          DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2).
 *
 * Reference: Rollins, D.K., Davis, J.F. (1992). "Unbiased Estimation of
 * Gross Errors in Process Measurements." AIChE Journal, 38(4), 563-572.
 */
int dr_ge_test_power(const dr_problem_t *prob, const double *delta,
                     double alpha, double *power_out);

/**
 * @brief Expected Value Test (EVT) for gross error detection.
 *
 * Tests whether the expected value of the reconciliation adjustments
 * is significantly different from zero, which would indicate a
 * systematic bias rather than random error.
 *
 * @param prob     Problem description.
 * @param x_hat    Reconciled values (from dr_solve).
 * @param results  Output: per-measurement EVT results.
 * @param alpha    Significance level.
 * @return         DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2).
 */
int dr_ge_expected_value_test(const dr_problem_t *prob, const double *x_hat,
                              dr_ge_test_result_t *results, double alpha);

/**
 * @brief Principal Component Analysis-based gross error detection
 * using the Tong and Crowe (1995) method.
 *
 * Uses PCA on the constraint residuals to identify patterns of
 * gross errors that may not be detected by individual tests.
 *
 * @param prob        Problem description.
 * @param n_components Output: number of significant principal components.
 * @param scores      Output: PC scores matrix (m x min(m,n)).
 *                    Must be pre-allocated.
 * @param alpha       Significance level.
 * @return            DR_OK or error code.
 *
 * Complexity: O(m^3 + n * m^2).
 *
 * Reference: Tong, H., Crowe, C.M. (1995). "Detection of Gross Errors
 * in Data Reconciliation by PCA." AIChE Journal, 41(7), 1712-1722.
 */
int dr_ge_pca_detection(const dr_problem_t *prob, int *n_components,
                        double *scores, double alpha);

#endif /* DR_GROSS_ERROR_H */
