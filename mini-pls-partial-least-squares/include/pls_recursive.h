#ifndef PLS_RECURSIVE_H
#define PLS_RECURSIVE_H
#include "pls_model.h"
#include "matrix_ops.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_recursive.h — Recursive PLS for Adaptive Soft Sensing
 *
 * Implements recursive/adaptive PLS algorithms that update the model
 * as new data arrives, enabling the soft sensor to track time-varying
 * process characteristics without full re-training.
 *
 * Knowledge Coverage:
 *   L8 — Advanced Topics: Recursive PLS (RPLS) with forgetting factor,
 *        exponential weighting, moving window PLS, just-in-time PLS
 *   L5 — Algorithms: Recursive update formulas for mean, covariance,
 *        and PLS decomposition
 *   L7 — Applications: Adaptive soft sensing in continuous processes
 *
 * Key Algorithms:
 *   1. RPLS with forgetting factor (Qin, 1998):
 *      Updates X^T*X and X^T*Y recursively with exponential forgetting,
 *      then re-decomposes via NIPALS.
 *   2. Moving window PLS:
 *      Maintains a sliding window of the most recent N samples.
 *   3. Recursive mean/variance update (Welford's algorithm):
 *      Updates mean and variance in O(1) per new sample.
 *
 * References:
 *   Qin, S.J. "Recursive PLS algorithms for adaptive data modeling",
 *   Computers & Chemical Engineering, 22(4-5):503-514, 1998.
 *   Dayal, B.S., MacGregor, J.F. "Recursive exponentially weighted
 *   PLS and its applications to adaptive control and prediction",
 *   J. Process Control, 7(3):169-179, 1997.
 *   Welford, B.P. "Note on a method for calculating corrected sums
 *   of squares and products", Technometrics, 4(3):419-420, 1962.
 *   Mu, S. et al. "Online dual updating with recursive PLS model
 *   and its application in predicting crystal size of purified
 *   terephthalic acid process", J. Process Control, 16(6):557-566, 2006.
 * ================================================================= */

/*
 * Recursive PLS configuration.
 */
typedef struct {
    double   forgetting_factor;  /* lambda in (0, 1], 1 = no forgetting           */
    size_t   window_size;        /* Moving window size (0 = no window, uses lambda) */
    size_t   update_interval;    /* Number of samples between model updates         */
    int      use_exponential;    /* 1 = exponential forgetting, 0 = moving window   */
    double   outlier_threshold;  /* T2 threshold to reject outliers during update   */
    int      verbose;
} RecursivePLSConfig;

RecursivePLSConfig recursive_pls_config_default(void);

/*
 * Recursive PLS model state.
 * Maintains the accumulated statistics needed for model updating.
 */
typedef struct {
    /* Current PLS model */
    PLSModel  *model;

    /* Accumulated statistics for recursive updates */
    Matrix    *XtX;           /* Accumulated X^T * X (p x p)                      */
    Matrix    *XtY;           /* Accumulated X^T * Y (p x q)                      */
    Vector    *sum_x;         /* Running sum of X columns (p x 1)                  */
    Vector    *sum_y;         /* Running sum of Y columns (q x 1)                  */
    double     sum_sq_y;      /* Running sum of squares of Y                      */
    size_t     n_effective;   /* Effective number of samples (with forgetting)     */
    size_t     n_total;       /* Total samples processed                          */

    /* Moving window data buffer (ring buffer) */
    Matrix    *window_X;      /* (window_size x p) ring buffer                     */
    Matrix    *window_Y;      /* (window_size x q) ring buffer                     */
    size_t     window_pos;    /* Current insertion position                       */
    size_t     window_count;  /* Number of samples currently in window            */

    /* Configuration */
    RecursivePLSConfig config;
    size_t    samples_since_update; /* Counter for periodic updates               */
} RecursivePLS;

/* ---- Recursive PLS Lifecycle ---- */

/*
 * Allocate and initialize a recursive PLS model.
 *
 * @param p_vars   number of X variables
 * @param q_vars   number of Y variables
 * @param a_lvs    number of latent variables
 * @param config   recursive PLS configuration
 * @return         initialized RecursivePLS, or NULL on failure
 */
RecursivePLS* recursive_pls_alloc(size_t p_vars, size_t q_vars,
                                   size_t a_lvs,
                                   const RecursivePLSConfig *config);

void recursive_pls_free(RecursivePLS *rpls);

/*
 * recursive_pls_initialize: Initialize recursive PLS with a batch of data.
 *
 * Computes initial X^T*X, X^T*Y, and fits the initial PLS model.
 * This should be called before any online updates.
 *
 * @param rpls  recursive PLS state
 * @param X     initial training data (n x p)
 * @param Y     initial training data (n x q)
 * @return      0 on success
 */
int recursive_pls_initialize(RecursivePLS *rpls,
                             const Matrix *X, const Matrix *Y);

/*
 * recursive_pls_update: Update the model with a new sample.
 *
 * Updates the accumulated statistics with exponential forgetting
 * or adds to the moving window. If update_interval samples have
 * accumulated, re-fits the PLS model.
 *
 * @param rpls   recursive PLS state
 * @param x_new  new X sample (p x 1)
 * @param y_new  new Y sample (q x 1), can be NULL if Y is unknown
 *               (model update is deferred until Y becomes available)
 * @return       0 on success, 1 if model was updated, -1 on error
 */
int recursive_pls_update(RecursivePLS *rpls,
                         const Vector *x_new, const Vector *y_new);

/*
 * recursive_pls_predict: Predict Y for a new sample using the
 * current recursively-updated model.
 *
 * @param rpls    recursive PLS state
 * @param x_new   new X sample (p x 1)
 * @param y_pred  output prediction (q x 1), caller-allocated
 * @return        0 on success
 */
int recursive_pls_predict(const RecursivePLS *rpls,
                          const Vector *x_new, Vector *y_pred);

/*
 * recursive_pls_refit: Force a model re-fit using current statistics.
 *
 * Decomposes the current X^T*X and X^T*Y to produce updated
 * PLS weights, loadings, and regression coefficients.
 *
 * @param rpls  recursive PLS state
 * @return      0 on success
 */
int recursive_pls_refit(RecursivePLS *rpls);

/*
 * welford_update_mean_var: Update running mean and variance using
 * Welford's online algorithm (single-pass, numerically stable).
 *
 * For a new value x:
 *   n_new = n + 1
 *   delta = x - mean
 *   mean_new = mean + delta / n_new
 *   M2_new = M2 + delta * (x - mean_new)
 *   var_new = M2_new / n_new
 *
 * This is O(1) per sample and avoids catastrophic cancellation.
 *
 * Reference: Welford, B.P., Technometrics, 4(3):419-420, 1962.
 *
 * @param n       pointer to sample count (updated in-place)
 * @param mean    pointer to running mean (updated in-place)
 * @param M2      pointer to sum of squared differences (updated in-place)
 * @param x       new observation
 */
void welford_update(size_t *n, double *mean, double *M2, double x);

/*
 * welford_finalize: Extract current mean and variance from Welford accumulators.
 *
 * variance = M2 / n  (population variance)
 * stddev = sqrt(M2 / n)
 */
void welford_finalize(size_t n, double mean, double M2,
                      double *variance, double *stddev);

/*
 * Compute exponentially weighted mean and variance update.
 *
 * mean_new = lambda * mean + (1 - lambda) * x
 * M2_new = lambda * M2 + (1 - lambda) * (x - mean) * (x - mean_new)
 *
 * where lambda is the forgetting factor.
 *
 * @param lambda  forgetting factor in (0, 1]
 */
void ewma_update(double lambda, double *mean, double *M2, double x);

#ifdef __cplusplus
}
#endif
#endif /* PLS_RECURSIVE_H */
