#ifndef PLS_NIPALS_H
#define PLS_NIPALS_H
#include "pls_model.h"
#include "matrix_ops.h"
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_nipals.h — NIPALS Algorithm for PLS Model Fitting
 *
 * Implements the Nonlinear Iterative Partial Least Squares (NIPALS)
 * algorithm, the original and most widely-used PLS algorithm.
 *
 * Knowledge Coverage:
 *   L5 — Algorithms: NIPALS single-component extraction, multi-LV iteration,
 *        convergence monitoring, deflation
 *   L4 — Engineering Laws: NIPALS convergence property (monotonic decrease
 *        of residual), equivalence to dominant eigenvector of X^T*Y*Y^T*X
 *
 * Algorithm (Wold, 1975):
 *   For each latent variable a = 1..A:
 *     1. Initialize u = column of Y
 *     2. Repeat until convergence:
 *        w = X^T * u / (u^T * u)       (X-weights)
 *        w = w / ||w||                  (normalize)
 *        t = X * w                      (X-scores)
 *        c = Y^T * t / (t^T * t)        (Y-weights)
 *        u = Y * c                      (Y-scores, update)
 *     3. p = X^T * t / (t^T * t)       (X-loadings)
 *     4. b_inner = u^T * t / (t^T * t)  (inner relation coefficient)
 *     5. Deflate: X := X - t * p^T, Y := Y - b_inner * t * c^T
 *
 * Theorem: The NIPALS weight vector w converges to the dominant
 * eigenvector of X^T * Y * Y^T * X (Hoskuldsson, 1988).
 *
 * References:
 *   Wold, H. "Estimation of principal components and related models
 *   by iterative least squares", in Krishnaiah (ed.), Multivariate
 *   Analysis, Academic Press, pp. 391-420, 1966.
 *   Hoskuldsson, A. "PLS regression methods", J. Chemometrics,
 *   2:211-228, 1988.
 *   Geladi, P., Kowalski, B. "Partial Least-Squares Regression:
 *   A Tutorial", Analytica Chimica Acta, 185:1-17, 1986.
 * ================================================================= */

/*
 * NIPALS algorithm configuration.
 * Controls convergence behaviour and numerical safeguards.
 */
typedef struct {
    double  tolerance;       /* Convergence tolerance on u change (default 1e-10) */
    int     max_iterations;  /* Maximum iterations per component (default 200)   */
    int     verbose;         /* Non-zero to print convergence diagnostics         */
} NIPALSConfig;

/*
 * Initialize default NIPALS configuration.
 * tolerance = 1e-10, max_iterations = 200, verbose = 0.
 */
NIPALSConfig nipals_config_default(void);

/*
 * nipals_fit: Fit a PLS model using the NIPALS algorithm.
 *
 * This is the primary model-fitting entry point. It iteratively
 * extracts A latent variables from the training data.
 *
 * @param X      training predictor matrix (n x p), will be deflated in-place
 * @param Y      training response matrix (n x q), will be deflated in-place
 * @param a_lvs  number of latent variables to extract
 * @param config NIPALS configuration (NULL for defaults)
 * @param model  pre-allocated PLS model to fill with decomposition
 * @return       0 on success, -1 on error (null input, dimension mismatch,
 *               non-convergence)
 *
 * Side effects: X and Y are deflated (modified in-place).
 * Copy them before calling if original data is needed later.
 *
 * Complexity: O(A * n * p * q * max_iter) per LV extraction
 */
int nipals_fit(Matrix *X, Matrix *Y, size_t a_lvs,
               const NIPALSConfig *config, PLSModel *model);

/*
 * nipals_extract_one_component: Extract a single latent variable.
 *
 * Low-level function that implements one iteration of the outer NIPALS loop.
 * Useful for understanding the algorithm and for custom deflation strategies.
 *
 * @param X       current deflated X (n x p), modified in-place
 * @param Y       current deflated Y (n x q), modified in-place
 * @param w_out   output X-weight vector (p x 1), caller-allocated
 * @param t_out   output X-score vector (n x 1), caller-allocated
 * @param p_out   output X-loading vector (p x 1), caller-allocated
 * @param c_out   output Y-weight vector (q x 1), caller-allocated
 * @param u_out   output Y-score vector (n x 1), caller-allocated
 * @param b_out   pointer to receive inner relation coefficient
 * @param config  NIPALS configuration
 * @param iters_out pointer to receive actual iterations used, may be NULL
 * @return        0 on convergence, -1 on non-convergence after max_iters
 */
int nipals_extract_one_component(Matrix *X, Matrix *Y,
                Vector *w_out, Vector *t_out, Vector *p_out,
                Vector *c_out, Vector *u_out, double *b_out,
                const NIPALSConfig *config, int *iters_out);

/*
 * nipals_deflate: Deflate X and Y after extracting one component.
 *
 * X := X - t * p^T
 * Y := Y - b * t * c^T
 *
 * @param X  (n x p), modified in-place
 * @param Y  (n x q), modified in-place
 * @param t  X-score vector (n x 1)
 * @param p  X-loading vector (p x 1)
 * @param c  Y-weight vector (q x 1)
 * @param b  inner relation coefficient
 */
void nipals_deflate(Matrix *X, Matrix *Y,
                    const Vector *t, const Vector *p,
                    const Vector *c, double b);

/*
 * nipals_predict: Predict Y from new X using a NIPALS-fitted model.
 *
 * Projects X_new through the sequence of weights and loadings to
 * compute predicted Y. This is the standard PLS prediction path.
 *
 * @param model  fitted PLS model (must have W, P, Q, B_inner, Beta set)
 * @param X_new  new predictor data (m x p), in original (uncentered) units
 * @param Y_pred output predictions (m x q), caller-allocated
 * @return       0 on success, -1 on error
 */
int nipals_predict(const PLSModel *model, const Matrix *X_new,
                   Matrix *Y_pred);

#ifdef __cplusplus
}
#endif
#endif /* PLS_NIPALS_H */
