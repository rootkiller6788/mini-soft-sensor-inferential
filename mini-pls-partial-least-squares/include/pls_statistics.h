#ifndef PLS_STATISTICS_H
#define PLS_STATISTICS_H
#include "pls_model.h"
#include "matrix_ops.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_statistics.h — PLS Model Diagnostics and Statistics
 *
 * Implements statistical diagnostics for PLS model quality evaluation
 * and process monitoring / fault detection.
 *
 * Knowledge Coverage:
 *   L1 - Definitions: R2X, R2Y, Q2, Hotelling T2, SPE, VIP, DModX, PRESS
 *   L2 - Core Concepts: model diagnostics decompose deviation into
 *        in-model (T2) and out-of-model (SPE) components
 *   L5 - Algorithms: VIP computation, T2/SPE control limits
 *   L7 - Applications: Process monitoring and fault detection
 *
 * References:
 *   Eriksson et al. "Multi- and Megavariate Data Analysis", Umetrics, 2006.
 *   Nomikos & MacGregor, AIChE J., 40(8):1361-1375, 1994.
 *   Jackson, J.E. "A User's Guide to Principal Components", Wiley, 1991.
 *   Wise & Gallagher, J. Process Control, 6(6):329-348, 1996.
 * ================================================================= */

/* ---- R2 and Q2: Goodness of Fit and Predictive Ability (L1) ---- */

/*
 * R2X: proportion of X-variance explained by A latent variables.
 * R2X = 1 - SS(E) / SS(X_centered)
 * Returns value in [0, 1], or -1 on error.
 */
double compute_R2X(const Matrix *X_orig, const Matrix *X_resid);

/*
 * R2Y: proportion of Y-variance explained by A latent variables.
 * R2Y = 1 - SS(F) / SS(Y_centered)
 */
double compute_R2Y(const Matrix *Y_orig, const Matrix *Y_resid);

/*
 * Q2: cross-validated predictive ability.
 * Q2 = 1 - PRESS / SS(Y_total)
 * Q2 > 0.5 is good, Q2 > 0.9 is excellent.
 */
double compute_Q2(double PRESS, double SSY_total);

/*
 * Compute total sum of squares of a matrix.
 * SS(A) = sum_i sum_j (A_ij)^2
 */
double matrix_sum_of_squares(const Matrix *A);

/* ---- Hotelling T2: In-Model Distance (L1) ---- */

/*
 * T2 for a single observation.
 * T2_i = sum_{a=1..A} (t_{ia}^2 / var(t_a))
 * Measures distance from center within the model plane.
 *
 * Reference: Hotelling, H. "The generalization of Student's ratio",
 * Ann. Math. Statist., 2(3):360-378, 1931.
 */
double compute_T2_single(const Vector *t_scores,
                         const Vector *score_variances);

/*
 * T2 for a batch of observations. Returns new Vector (m x 1). Caller frees.
 */
Vector* compute_T2_batch(const Matrix *T_scores,
                         const Vector *score_variances);

/*
 * T2 confidence limit at significance level alpha.
 * Uses Wilson-Hilferty F-distribution approximation.
 *
 * T2_alpha = A * (n^2 - 1) / (n * (n - A)) * F_{A, n-A, alpha}
 *
 * Reference: Jackson, "A User's Guide to Principal Components", 1991.
 */
double compute_T2_limit(size_t n_samples, size_t n_lvs, double alpha);

/*
 * Compute score variances from training score matrix.
 * Returns new Vector (a_lvs x 1). Caller frees.
 */
Vector* compute_score_variances(const Matrix *T_scores);

/* ---- SPE (Q-statistic): Residual Distance (L1) ---- */

/*
 * SPE for a single observation.
 * SPE_i = sum_j (x_{ij} - x_hat_{ij})^2
 * where x_hat = t_i * P^T is the projection onto the model plane.
 * Measures how well the model represents the observation.
 */
double compute_SPE_single(const Vector *x_residual);

/*
 * SPE for a batch of observations. Returns new Vector (m x 1). Caller frees.
 */
Vector* compute_SPE_batch(const Matrix *X_residuals);

/*
 * SPE confidence limit (Jackson-Mudholkar approximation).
 *
 * SPE_alpha = theta1 * [1 + c_alpha * sqrt(2*theta2*h0^2)/theta1
 *             + theta2*h0*(h0-1)/theta1^2]^{1/h0}
 *
 * where h0 = 1 - 2*theta1*theta3 / (3*theta2^2),
 * theta_k = sum_{j=A+1..p} lambda_j^k (residual eigenvalues).
 *
 * Reference: Jackson & Mudholkar, Technometrics, 21(3):341-349, 1979.
 */
double compute_SPE_limit(double theta1, double theta2, double theta3,
                         double alpha);

/*
 * Estimate theta parameters for SPE limit from residuals.
 */
void compute_SPE_theta(const Matrix *X_residuals,
                       double *theta1, double *theta2, double *theta3);

/* ---- VIP: Variable Importance in Projection (L1, L5) ---- */

/*
 * Compute VIP scores for all X-variables.
 *
 * VIP_j = sqrt( p * sum_a [w_{ja}^2 * SSY_a] / sum_a SSY_a )
 * where SSY_a = b_a^2 * (t_a^T * t_a) is Y-variance explained by LV a.
 *
 * VIP > 1.0: important variable
 * VIP < 0.8: unimportant variable
 *
 * Complexity: O(p * A). Returns new Vector (p x 1). Caller frees.
 */
Vector* compute_VIP(const PLSModel *model);

/*
 * Compute the Y-variance explained by each latent variable.
 * SSY_a = b_a^2 * t_a^T * t_a
 * Returns new Vector (a_lvs x 1). Caller frees.
 */
Vector* compute_SSY_per_LV(const PLSModel *model);

/* ---- DModX: Distance to Model in X-space (L1) ---- */

/*
 * DModX for a single observation.
 * DModX_i = sqrt( SPE_i / (p - A) )
 * Normalized residual standard deviation.
 */
double compute_DModX_single(const Vector *x_residual,
                            size_t p_vars, size_t a_lvs);

/*
 * DModX for a batch. Returns new Vector (m x 1). Caller frees.
 */
Vector* compute_DModX_batch(const Matrix *X_residuals, size_t a_lvs);

/*
 * DModX critical value (Dcrit).
 * Dcrit = sqrt( chi2_{p-A, alpha} / (p - A) )
 * Uses Wilson-Hilferty F-approximation.
 */
double compute_DModX_critical(const Matrix *X_residuals,
                              size_t a_lvs, double alpha);

/* ---- Prediction Error Metrics (L1) ---- */

/*
 * PRESS: Predictive Residual Error Sum of Squares.
 * PRESS = sum_i sum_j (y_{ij} - y_hat_{ij})^2
 * Lower PRESS = better predictive performance.
 */
double compute_PRESS(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * RMSEC: Root Mean Square Error of Calibration.
 * Training set error. Complexity: O(n * q).
 */
double compute_RMSEC(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * RMSEP: Root Mean Square Error of Prediction.
 * Independent test set error.
 */
double compute_RMSEP(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * Bias: mean(Y_pred - Y_true) for each response variable.
 * Returns new Vector (q x 1). Caller frees.
 */
Vector* compute_bias(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * SEP: Standard Error of Prediction for each response.
 * SEP_j = stddev of (y_pred_{ij} - y_true_{ij})
 * Returns new Vector (q x 1). Caller frees.
 */
Vector* compute_SEP(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * RPD: Ratio of Performance to Deviation.
 * RPD_j = stddev(Y_true_j) / SEP_j
 * RPD > 2.0 = good model, RPD > 3.0 = excellent.
 * Returns new Vector (q x 1). Caller frees.
 */
Vector* compute_RPD(const Matrix *Y_true, const Matrix *Y_pred);

/*
 * F-distribution critical value approximation (Wilson-Hilferty, 1931).
 *
 * F_{d1, d2, alpha} is approximated via the chi-square transformation:
 *   z = (F^{1/3} * (1 - 2/(9*d2)) - (1 - 2/(9*d1))) / sqrt(2/(9*d1) + 2/(9*d2) * F^{2/3})
 * using Newton iteration.
 *
 * @param d1    numerator degrees of freedom
 * @param d2    denominator degrees of freedom
 * @param alpha significance level (e.g., 0.05 for 95% limit)
 * @return      approximate F critical value
 */
double f_distribution_critical(double d1, double d2, double alpha);

/*
 * Normal distribution quantile (inverse CDF) approximation.
 * Uses the rational approximation from Abramowitz & Stegun (1964),
 * formula 26.2.23, with relative error < 4.5e-4.
 *
 * @param p  probability (0 < p < 1)
 * @return   z-score such that Phi(z) = p
 */
double normal_quantile(double p);

#ifdef __cplusplus
}
#endif
#endif /* PLS_STATISTICS_H */
