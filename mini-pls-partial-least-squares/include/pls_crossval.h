#ifndef PLS_CROSSVAL_H
#define PLS_CROSSVAL_H
#include "pls_model.h"
#include "pls_nipals.h"
#include "pls_statistics.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_crossval.h — Cross-Validation for PLS Model Selection
 *
 * Implements cross-validation methods for determining the optimal
 * number of latent variables in a PLS model.
 *
 * Knowledge Coverage:
 *   L5 — Algorithms: K-fold cross-validation, leave-one-out CV,
 *        Monte Carlo cross-validation, PRESS-based LV selection
 *   L4 — Engineering Laws: bias-variance tradeoff in PLS model selection,
 *        one-standard-error rule for parsimony
 *
 * Methods:
 *   1. K-fold CV:     Split data into K folds, train on K-1, test on 1
 *   2. Leave-One-Out: K = n (each sample held out once)
 *   3. Monte Carlo:   Random train/test splits with specified ratio
 *
 * Model Selection Rules:
 *   1. Minimum PRESS: select A with lowest PRESS
 *   2. One-standard-error rule: smallest A within 1 SE of min PRESS
 *   3. Q2 threshold: smallest A with Q2 > threshold
 *
 * References:
 *   Wold, S. "Cross-validatory estimation of the number of components
 *   in factor and principal components models", Technometrics,
 *   20(4):397-405, 1978.
 *   Hastie, T., Tibshirani, R., Friedman, J. "The Elements of
 *   Statistical Learning", 2nd ed., Springer, 2009, Section 7.10.
 * ================================================================= */

/*
 * Cross-validation configuration.
 */
typedef struct {
    int     n_folds;          /* Number of folds (2 = split-half, n = LOO)    */
    int     n_repeats;        /* Number of MC repetitions (1 for standard CV) */
    int     random_seed;      /* Seed for reproducible splits (0 = time-based) */
    double  test_ratio;       /* Test fraction for Monte Carlo CV (e.g., 0.2) */
    int     verbose;          /* Print progress if non-zero                   */
} CrossValConfig;

CrossValConfig crossval_config_default(void);

/*
 * Cross-validation result for a range of latent variables.
 */
typedef struct {
    size_t   max_lvs;         /* Maximum number of LVs evaluated          */
    double  *PRESS;            /* PRESS for each LV count (length max_lvs) */
    double  *PRESS_std;        /* Standard deviation of PRESS (length max_lvs) */
    double  *Q2;               /* Q2 for each LV count                    */
    double  *R2Y;              /* R2Y for each LV count                   */
    size_t   opt_lvs_min_press;  /* Optimal LVs by minimum PRESS rule    */
    size_t   opt_lvs_onese;      /* Optimal LVs by one-standard-error rule */
    size_t   opt_lvs_q2;          /* Optimal LVs by Q2 > 0.5 rule         */
} CrossValResult;

/*
 * crossval_kfold: Perform K-fold cross-validation for PLS.
 *
 * Repeatedly fits PLS models on K-1 folds and evaluates prediction
 * error on the held-out fold. Computes PRESS and Q2 for each
 * candidate number of latent variables.
 *
 * @param X       training data (n x p), NOT modified
 * @param Y       training data (n x q), NOT modified
 * @param max_lvs maximum number of LVs to evaluate
 * @param config  cross-validation configuration (NULL for defaults)
 * @param result  output cross-validation results, caller-allocated
 * @return        0 on success, -1 on error
 *
 * Complexity: O(K * max_lvs * n * p * q * max_iters)
 */
int crossval_kfold(const Matrix *X, const Matrix *Y,
                   size_t max_lvs, const CrossValConfig *config,
                   CrossValResult *result);

/*
 * crossval_leave_one_out: Leave-One-Out cross-validation.
 *
 * Special case of K-fold with K = n. Each sample is held out once.
 * Most computationally expensive but provides nearly unbiased
 * estimate of prediction error.
 *
 * Complexity: O(n * max_lvs * n * p * q * max_iters) ~ O(n^2 * p * q)
 */
int crossval_leave_one_out(const Matrix *X, const Matrix *Y,
                           size_t max_lvs, CrossValResult *result);

/*
 * crossval_monte_carlo: Monte Carlo cross-validation.
 *
 * Repeatedly splits data into random train/test sets, fits PLS on
 * the training set, and evaluates on the test set. Averaged over
 * multiple repetitions for stability.
 *
 * @param X         training data (n x p)
 * @param Y         training data (n x q)
 * @param max_lvs   maximum LVs to evaluate
 * @param n_repeats number of random splits
 * @param test_ratio fraction for test set (0 < ratio < 1)
 * @param seed      random seed (0 for time-based)
 * @param result    output results
 * @return          0 on success
 */
int crossval_monte_carlo(const Matrix *X, const Matrix *Y,
                         size_t max_lvs, int n_repeats,
                         double test_ratio, int seed,
                         CrossValResult *result);

/*
 * crossval_select_optimal: Apply model selection rules to CV results.
 *
 * Populates the opt_lvs_* fields in the result struct using:
 *   1. Minimum PRESS
 *   2. One-standard-error rule (parsimonious model)
 *   3. Q2 > 0.5 rule
 */
void crossval_select_optimal(CrossValResult *result);

/*
 * CrossValResult lifecycle.
 */
CrossValResult* crossval_result_alloc(size_t max_lvs);
void            crossval_result_free(CrossValResult *result);

/*
 * crossval_print_results: Pretty-print cross-validation results.
 * Prints a table of LVs vs PRESS, Q2, R2Y with optimal selections marked.
 */
void crossval_print_results(const CrossValResult *result);

#ifdef __cplusplus
}
#endif
#endif /* PLS_CROSSVAL_H */
