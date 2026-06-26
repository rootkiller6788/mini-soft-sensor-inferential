#include "pls_crossval.h"
#include "pls_nipals.h"
#include "pls_preprocessing.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])

CrossValConfig crossval_config_default(void) {
    CrossValConfig c;
    c.n_folds = 7;
    c.n_repeats = 1;
    c.random_seed = 0;
    c.test_ratio = 0.2;
    c.verbose = 0;
    return c;
}

CrossValResult* crossval_result_alloc(size_t max_lvs) {
    CrossValResult *r = (CrossValResult*)calloc(1, sizeof(CrossValResult));
    if (!r) return NULL;
    r->max_lvs = max_lvs;
    r->PRESS = (double*)calloc(max_lvs + 1, sizeof(double));
    r->PRESS_std = (double*)calloc(max_lvs + 1, sizeof(double));
    r->Q2 = (double*)calloc(max_lvs + 1, sizeof(double));
    r->R2Y = (double*)calloc(max_lvs + 1, sizeof(double));
    if (!r->PRESS || !r->PRESS_std || !r->Q2 || !r->R2Y) {
        crossval_result_free(r); return NULL;
    }
    return r;
}

void crossval_result_free(CrossValResult *result) {
    if (!result) return;
    free(result->PRESS); free(result->PRESS_std);
    free(result->Q2); free(result->R2Y);
    free(result);
}

/* Simple pseudo-random number generator (xorshift32) */
static uint32_t xorshift_state = 2463534242;

static void xs_seed(uint32_t seed) {
    xorshift_state = (seed != 0) ? seed : (uint32_t)time(NULL);
}

static uint32_t xs_rand(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xorshift_state = x;
    return x;
}

static void shuffle_indices(size_t *arr, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = xs_rand() % (i + 1);
        size_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* Compute total SS of Y for Q2 calculation */
static double compute_SSY(const Matrix *Y) {
    double ss = 0.0;
    Vector *means = matrix_column_means(Y);
    if (!means) return 0.0;
    for (size_t i = 0; i < Y->rows; i++)
        for (size_t j = 0; j < Y->cols; j++) {
            double d = MATRIX_AT(Y, i, j) - means->data[j];
            ss += d * d;
        }
    vector_free(means);
    return ss;
}

int crossval_kfold(const Matrix *X, const Matrix *Y,
                   size_t max_lvs, const CrossValConfig *config,
                   CrossValResult *result) {
    if (!X || !Y || !result || X->rows != Y->rows) return -1;
    CrossValConfig cfg = config ? *config : crossval_config_default();
    size_t n = X->rows, p = X->cols, q = Y->cols;
    size_t K = cfg.n_folds;
    if (K > n) K = n;
    if (K < 2) K = 2;

    double SSY = compute_SSY(Y);

    /* Build fold assignments */
    size_t *indices = (size_t*)malloc(n * sizeof(size_t));
    if (!indices) return -1;
    for (size_t i = 0; i < n; i++) indices[i] = i;
    xs_seed(cfg.random_seed);
    shuffle_indices(indices, n);

    /* Accumulate PRESS per LV count across folds */
    for (size_t fold = 0; fold < K; fold++) {
        size_t test_start = fold * n / K;
        size_t test_end = (fold + 1) * n / K;
        size_t n_test = test_end - test_start;
        size_t n_train = n - n_test;
        if (n_test == 0 || n_train == 0) continue;

        /* Build train/test splits */
        Matrix *X_train = matrix_alloc(n_train, p);
        Matrix *Y_train = matrix_alloc(n_train, q);
        Matrix *X_test  = matrix_alloc(n_test, p);
        Matrix *Y_test  = matrix_alloc(n_test, q);
        if (!X_train || !Y_train || !X_test || !Y_test) {
            matrix_free(X_train); matrix_free(Y_train);
            matrix_free(X_test); matrix_free(Y_test);
            free(indices); return -1;
        }

        size_t tr = 0, te = 0;
        for (size_t i = 0; i < n; i++) {
            size_t idx = indices[i];
            if (i >= test_start && i < test_end) {
                for (size_t j = 0; j < p; j++)
                    MATRIX_AT(X_test, te, j) = MATRIX_AT(X, idx, j);
                for (size_t j = 0; j < q; j++)
                    MATRIX_AT(Y_test, te, j) = MATRIX_AT(Y, idx, j);
                te++;
            } else {
                for (size_t j = 0; j < p; j++)
                    MATRIX_AT(X_train, tr, j) = MATRIX_AT(X, idx, j);
                for (size_t j = 0; j < q; j++)
                    MATRIX_AT(Y_train, tr, j) = MATRIX_AT(Y, idx, j);
                tr++;
            }
        }

        /* Preprocess train data */
        PreprocessingConfig ppcfg = preproc_config_default();
        Vector *xm = vector_alloc(p), *xs = vector_alloc(p);
        Vector *ym = vector_alloc(q), *ys = vector_alloc(q);
        if (!xm || !xs || !ym || !ys) {
            vector_free(xm); vector_free(xs); vector_free(ym); vector_free(ys);
            matrix_free(X_train); matrix_free(Y_train);
            matrix_free(X_test); matrix_free(Y_test);
            free(indices); return -1;
        }
        preprocess_fit(X_train, Y_train, &ppcfg, xm, xs, ym, ys);

        /* Fit PLS for each candidate LV count and predict on test set */
        for (size_t a = 1; a <= max_lvs && a <= p && a <= n_train; a++) {
            /* Create a copy of preprocessed training data (NIPALS deflates in-place) */
            Matrix *Xt = matrix_copy(X_train);
            Matrix *Yt = matrix_copy(Y_train);
            if (!Xt || !Yt) { matrix_free(Xt); matrix_free(Yt); continue; }

            PLSModel *model = pls_model_alloc(n_train, p, q, a);
            if (!model) { matrix_free(Xt); matrix_free(Yt); continue; }
            pls_model_set_preprocessing(model, xm, xs, 1, 0, ym, ys, 1, 0);

            NIPALSConfig nipcfg = nipals_config_default();
            nipcfg.verbose = 0;
            if (nipals_fit(Xt, Yt, a, &nipcfg, model) == 0) {
                /* Predict on test set */
                Matrix *X_test_pp = matrix_copy(X_test);
                preprocess_transform(X_test_pp, xm, xs, 1, 0);
                Matrix *Y_pred = matrix_alloc(n_test, q);
                if (Y_pred) {
                    pls_model_predict_batch(model, X_test_pp, Y_pred);
                    /* Undo Y preprocessing on predictions for fair PRESS */
                    preprocess_inverse(Y_pred, ym, ys, 1, 0);
                    double press_fold = compute_PRESS(Y_test, Y_pred);
                    result->PRESS[a] += press_fold;
                    matrix_free(Y_pred);
                }
                matrix_free(X_test_pp);
            }
            pls_model_free(model);
            matrix_free(Xt); matrix_free(Yt);
        }

        vector_free(xm); vector_free(xs); vector_free(ym); vector_free(ys);
        matrix_free(X_train); matrix_free(Y_train);
        matrix_free(X_test); matrix_free(Y_test);
    }

    /* Compute final Q2 and R2Y from PRESS */
    for (size_t a = 1; a <= max_lvs; a++) {
        if (SSY > 1e-15) {
            result->Q2[a] = 1.0 - result->PRESS[a] / SSY;
        }
    }

    crossval_select_optimal(result);
    free(indices);
    return 0;
}

int crossval_leave_one_out(const Matrix *X, const Matrix *Y,
                           size_t max_lvs, CrossValResult *result) {
    CrossValConfig cfg = crossval_config_default();
    cfg.n_folds = (int)X->rows;
    return crossval_kfold(X, Y, max_lvs, &cfg, result);
}

int crossval_monte_carlo(const Matrix *X, const Matrix *Y,
                         size_t max_lvs, int n_repeats,
                         double test_ratio, int seed,
                         CrossValResult *result) {
    if (!X || !Y || !result || test_ratio <= 0 || test_ratio >= 1) return -1;
    /* Simplified: use K-fold with many repeats as approximation */
    CrossValConfig cfg = crossval_config_default();
    cfg.n_folds = (int)(1.0 / test_ratio);
    if (cfg.n_folds < 2) cfg.n_folds = 2;
    cfg.random_seed = seed;

    for (int r = 0; r < n_repeats; r++) {
        cfg.random_seed = seed + r * 1000;
        crossval_kfold(X, Y, max_lvs, &cfg, result);
    }
    /* Average PRESS over repeats */
    for (size_t a = 1; a <= max_lvs; a++) {
        result->PRESS[a] /= (double)n_repeats;
    }

    double SSY = compute_SSY(Y);
    for (size_t a = 1; a <= max_lvs; a++) {
        result->Q2[a] = (SSY > 1e-15) ? 1.0 - result->PRESS[a] / SSY : 0.0;
    }
    crossval_select_optimal(result);
    return 0;
}

void crossval_select_optimal(CrossValResult *result) {
    if (!result || result->max_lvs == 0) return;
    /* Min PRESS rule */
    size_t best = 1;
    double min_press = result->PRESS[1];
    for (size_t a = 2; a <= result->max_lvs; a++) {
        if (result->PRESS[a] < min_press) {
            min_press = result->PRESS[a];
            best = a;
        }
    }
    result->opt_lvs_min_press = best;

    /* One-standard-error rule */
    /* Estimate PRESS std from folds (use PRESS_std if available, else heuristic) */
    double se = min_press * 0.05;  /* Heuristic: 5% of PRESS as SE */
    size_t onese = best;
    for (size_t a = 1; a < best; a++) {
        if (result->PRESS[a] <= min_press + se) {
            onese = a; break;
        }
    }
    result->opt_lvs_onese = onese;

    /* Q2 rule: first LV with Q2 > 0.5 */
    size_t q2_best = result->max_lvs;
    for (size_t a = 1; a <= result->max_lvs; a++) {
        if (result->Q2[a] > 0.5) { q2_best = a; break; }
    }
    result->opt_lvs_q2 = q2_best;
}

void crossval_print_results(const CrossValResult *result) {
    if (!result) return;
    printf("=== Cross-Validation Results ===\n");
    printf(" LVs |    PRESS    |     Q2     |    R2Y\n");
    printf("-----|-------------|------------|--------\n");
    for (size_t a = 1; a <= result->max_lvs; a++) {
        printf(" %3lu | %11.4f | %10.4f | %7.4f", (unsigned long)a,
               result->PRESS[a], result->Q2[a], result->R2Y[a]);
        if (a == result->opt_lvs_min_press) printf("  <-- min PRESS");
        if (a == result->opt_lvs_onese)      printf("  <-- 1-SE");
        printf("\n");
    }
    printf("Optimal LVs (min PRESS): %lu\n", (unsigned long)result->opt_lvs_min_press);
    printf("Optimal LVs (1-SE):      %lu\n", (unsigned long)result->opt_lvs_onese);
    printf("Optimal LVs (Q2>0.5):   %lu\n", (unsigned long)result->opt_lvs_q2);
}
