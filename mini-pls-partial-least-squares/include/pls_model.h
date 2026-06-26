#ifndef PLS_MODEL_H
#define PLS_MODEL_H
#include "matrix_ops.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_model.h — Partial Least Squares Model Data Structures
 *
 * Defines the core data types for PLS regression models.
 * Covers the complete PLS decomposition:
 *
 *   X = T * P^T + E     (L1: outer relation, predictor block)
 *   Y = U * Q^T + F     (L1: outer relation, response block)
 *   U = T * B + H       (L1: inner relation)
 *
 * Beta = W * inv(P^T * W) * Q^T   (L2: regression coefficients)
 * Y_hat = X * Beta + 1 * b0^T     (L2: prediction equation)
 *
 * Knowledge Coverage:
 *   L1 — Definitions: PLSModel, PLSComponentStats struct types
 *   L2 — Core Concepts: latent variable decomposition, inner/outer relations
 *   L3 — Engineering Structures: model lifecycle, preprocessing config,
 *         batch/single prediction pipelines
 *   L4 — Engineering Laws: orthogonality of scores (T^T * T is diagonal)
 *
 * References:
 *   Wold, H. "Soft Modeling by Latent Variables", in Perspectives in
 *   Probability and Statistics, Academic Press, 1975.
 *   Wold, S., Sjostrom, M., Eriksson, L. "PLS-regression: a basic tool
 *   of chemometrics", Chemometrics Intell. Lab. Syst., 58:109-130, 2001.
 *   de Jong, S. "SIMPLS: an alternative approach to partial least squares
 *   regression", Chemometrics Intell. Lab. Syst., 18:251-263, 1993.
 * ================================================================= */

/* -----------------------------------------------------------------
 * L1: Definition — PLS Model Struct
 * -----------------------------------------------------------------
 * Complete mathematical representation of a fitted PLS model.
 * All decomposition matrices are stored in the order of extracted
 * latent variables (column k = k-th LV).
 *
 *   W  (p x a): X-weights, direction of maximum (X,Y) covariance
 *   P  (p x a): X-loadings, projection of X onto the score space
 *   T  (n x a): X-scores, latent variable representation of X
 *   C  (q x a): Y-weights (SIMPLS variant)
 *   U  (n x a): Y-scores
 *   Q  (q x a): Y-loadings
 *   B_inner (a x a): diagonal inner relation matrix
 *   Beta (p x q): final PLS regression coefficients
 *   b0  (q x 1): intercept
 */
typedef struct {
    Matrix *W;
    Matrix *P;
    Matrix *T;
    Matrix *C;
    Matrix *U;
    Matrix *Q;
    Matrix *B_inner;
    Matrix *Beta;
    Vector *b0;

    /* Preprocessing parameters */
    Vector *X_mean;
    Vector *X_std;
    Vector *Y_mean;
    Vector *Y_std;

    /* Dimensions */
    size_t  n_samples;
    size_t  p_vars;
    size_t  q_vars;
    size_t  a_lvs;

    /* Preprocessing flags */
    int     center_x;
    int     scale_x;
    int     center_y;
    int     scale_y;

    /* Cumulative fit statistics */
    double  R2X_cum;
    double  R2Y_cum;
    double  Q2_cum;
} PLSModel;

/* -----------------------------------------------------------------
 * L1: Definition — Component-wise Statistics
 * -----------------------------------------------------------------
 * Tracks variance explained by each successive latent variable.
 * Used for scree plots and model dimension selection.
 *
 * Theorem: sum_{k=1..a} R2X_k = R2X_cum (additivity of variance explained)
 */
typedef struct {
    size_t   n_lvs;
    double  *R2X;
    double  *R2Y;
    double  *Q2;
    double  *PRESS;
} PLSComponentStats;

/* -----------------------------------------------------------------
 * L2: Core Concepts — Model Lifecycle
 * ----------------------------------------------------------------- */

PLSModel* pls_model_alloc(size_t n_samples, size_t p_vars,
                          size_t q_vars, size_t a_lvs);
PLSModel* pls_model_copy(const PLSModel *model);
void      pls_model_free(PLSModel *model);

/*
 * Compute regression coefficients Beta from the fitted decomposition.
 * Beta = W * inv(P^T * W) * Q^T
 * This is the PLS regression coefficient matrix mapping X to Y.
 * Complexity: O(p * a^2 + a^3). Returns 0 on success.
 */
int       pls_model_compute_beta(PLSModel *model);

/*
 * Set preprocessing parameters to enable correct prediction on new data.
 */
void      pls_model_set_preprocessing(PLSModel *model,
                    const Vector *x_mean, const Vector *x_std,
                    int center_x, int scale_x,
                    const Vector *y_mean, const Vector *y_std,
                    int center_y, int scale_y);

void      pls_model_set_stats(PLSModel *model, double R2X_cum,
                              double R2Y_cum, double Q2_cum);

/*
 * Predict Y for a single new sample x_new (in original units).
 * Applies the same centering/scaling as training.
 * y_pred must be pre-allocated with length q_vars.
 * Returns 0 on success.
 */
int       pls_model_predict_single(const PLSModel *model,
                                   const Vector *x_new, Vector *y_pred);

/*
 * Predict Y for a batch of new samples X_new (m x p_vars).
 * Y_pred must be pre-allocated (m x q_vars).
 * Returns 0 on success.
 */
int       pls_model_predict_batch(const PLSModel *model,
                                  const Matrix *X_new, Matrix *Y_pred);

/*
 * Compute X-scores for new data: T_new = X_new_processed * W.
 * For diagnostic score plots (T1 vs T2, Hotelling T2).
 * Returns (m x a_lvs) matrix, caller must free.
 */
Matrix*   pls_model_compute_scores(const PLSModel *model,
                                   const Matrix *X_new);

/*
 * Compute X-residuals: E = X_new - T_new * P^T.
 * For SPE (Q-statistic) diagnostics.
 * Returns (m x p_vars) matrix, caller must free.
 */
Matrix*   pls_model_compute_residuals(const PLSModel *model,
                                      const Matrix *X_new);

/*
 * Compute the PLS inner relation: for given X-score t (a_lvs x 1),
 * compute the predicted Y-score u_hat = B_inner * t.
 * Returns new Vector (a_lvs x 1), caller must free.
 */
Vector*   pls_model_inner_predict(const PLSModel *model,
                                  const Vector *t_score);

/* Component statistics lifecycle */
PLSComponentStats* pls_component_stats_alloc(size_t max_lvs);
void               pls_component_stats_free(PLSComponentStats *stats);

#ifdef __cplusplus
}
#endif
#endif /* PLS_MODEL_H */
