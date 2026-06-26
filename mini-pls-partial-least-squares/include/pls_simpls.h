#ifndef PLS_SIMPLS_H
#define PLS_SIMPLS_H
#include "pls_model.h"
#include "matrix_ops.h"
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_simpls.h — SIMPLS Algorithm for PLS Model Fitting
 *
 * SIMPLS (Statistically Inspired Modification of PLS) computes the
 * PLS model by directly solving for weight vectors that maximize
 * covariance while maintaining orthogonality constraints.
 *
 * Key advantage over NIPALS: avoids repeated deflation of X and Y,
 * computing all results from the original data matrices.
 *
 * Knowledge Coverage:
 *   L5 — Algorithms: SIMPLS weight computation, covariance maximization
 *        with orthogonality constraints, direct coefficient calculation
 *   L4 — Engineering Laws: SIMPLS solves the same optimization problem
 *        as NIPALS but via a different numerical path (de Jong, 1993)
 *
 * Algorithm (de Jong, 1993):
 *   S = X^T * Y                         (cross-covariance matrix)
 *   For a = 1..A:
 *     If a == 1: compute dominant eigenvector of S * S^T -> w
 *     Else:      compute dominant eigenvector of (I - P_prev * pinv(P_prev)) * S * S^T -> w
 *     w = w / ||w||
 *     t = X * w
 *     p = X^T * t / (t^T * t)
 *     Store w, p
 *
 * Theorem (de Jong): SIMPLS and NIPALS produce identical weight vectors
 * and regression coefficients when Y is univariate (q = 1). For q > 1,
 * they produce different but equivalent models.
 *
 * References:
 *   de Jong, S. "SIMPLS: an alternative approach to partial least squares
 *   regression", Chemometrics Intell. Lab. Syst., 18:251-263, 1993.
 *   ter Braak, C.J.F., de Jong, S. "The objective function of partial
 *   least squares regression", J. Chemometrics, 12:41-54, 1998.
 * ================================================================= */

/*
 * SIMPLS algorithm configuration.
 */
typedef struct {
    double  tolerance;       /* Convergence tolerance for power iteration */
    int     max_iterations;  /* Max iterations for dominant eigenvector   */
    int     verbose;         /* Print diagnostics if non-zero             */
} SIMPLSConfig;

SIMPLSConfig simpls_config_default(void);

/*
 * simpls_fit: Fit a PLS model using the SIMPLS algorithm.
 *
 * Unlike NIPALS, SIMPLS does NOT modify the input data matrices.
 * All computations are performed on the original X and Y.
 *
 * @param X       training predictor matrix (n x p), NOT modified
 * @param Y       training response matrix (n x q), NOT modified
 * @param a_lvs   number of latent variables to extract
 * @param config  SIMPLS configuration (NULL for defaults)
 * @param model   pre-allocated PLS model to fill
 * @return        0 on success, -1 on error
 *
 * Complexity: O(A * (n*p*q + p^3)) per LV
 */
int simpls_fit(const Matrix *X, const Matrix *Y, size_t a_lvs,
               const SIMPLSConfig *config, PLSModel *model);

/*
 * simpls_compute_weights: Compute SIMPLS weight vectors directly.
 *
 * Low-level function that extracts weight vectors from cross-covariance
 * matrix S = X^T * Y, respecting orthogonality constraints.
 *
 * @param S       cross-covariance matrix (p x q), NOT modified
 * @param a_lvs   number of weight vectors to compute
 * @param W_out   output weight matrix (p x a_lvs), caller-allocated
 * @param config  SIMPLS configuration
 * @return        0 on success, -1 on error
 */
int simpls_compute_weights(const Matrix *S, size_t a_lvs,
                           Matrix *W_out, const SIMPLSConfig *config);

/*
 * simpls_predict: Predict Y from new X using a SIMPLS-fitted model.
 *
 * @param model   fitted PLS model
 * @param X_new   new predictor data (m x p)
 * @param Y_pred  output predictions (m x q), caller-allocated
 * @return        0 on success
 */
int simpls_predict(const PLSModel *model, const Matrix *X_new,
                   Matrix *Y_pred);

/*
 * simpls_compute_beta_direct: Compute PLS regression coefficients
 * directly from weights without storing intermediate matrices.
 *
 * Beta = W * inv(P^T * W) * Q^T
 *
 * This is the same formula as NIPALS but computed from the SIMPLS
 * decomposition, which may differ in the orthogonality of scores.
 *
 * @param model  fitted PLS model (must have W, P, Q set)
 * @return       0 on success, -1 on singular P^T * W
 */
int simpls_compute_beta_direct(PLSModel *model);

#ifdef __cplusplus
}
#endif
#endif /* PLS_SIMPLS_H */
