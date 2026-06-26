#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#include "pls_model.h"
#include "pls_nipals.h"
#include "pls_preprocessing.h"
#include "pls_statistics.h"
#include "pls_crossval.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* =================================================================
 * Example 3: Model Selection using Cross-Validation
 *
 * Demonstrates how to select the optimal number of latent variables
 * using K-fold cross-validation with PRESS, Q2, and the
 * one-standard-error rule.
 *
 * Knowledge: L5 — Algorithm (cross-validation, model selection)
 *            L6 — Canonical Problem (model complexity selection)
 * ================================================================= */

#define N_SAMPLES 60
#define N_XVARS   8
#define N_YVARS   1
#define MAX_LVS   6

int main(void) {
    printf("=== Example 3: Cross-Validation for LV Selection ===\n\n");

    /* ---- Generate data with known latent structure ---- */
    printf("Generating data with %d samples, %d X-variables...\n", N_SAMPLES, N_XVARS);
    printf("(True latent structure: 2 LVs, 4 noise variables)\n\n");

    Matrix *X = matrix_alloc(N_SAMPLES, N_XVARS);
    Matrix *Y = matrix_alloc(N_SAMPLES, N_YVARS);

    srand(789);
    for (size_t i = 0; i < N_SAMPLES; i++) {
        /* Two latent factors */
        double t1 = 5.0 * ((double)rand() / RAND_MAX - 0.5);
        double t2 = 3.0 * ((double)rand() / RAND_MAX - 0.5);

        /* X variables: first 4 related to t1, t2; last 4 are noise */
        MATRIX_AT(X, i, 0) = 2.0 * t1 + 0.5 * t2 + 0.2 * ((double)rand()/RAND_MAX - 0.5);
        MATRIX_AT(X, i, 1) = 1.5 * t1 - 1.0 * t2 + 0.2 * ((double)rand()/RAND_MAX - 0.5);
        MATRIX_AT(X, i, 2) = 0.5 * t1 + 2.0 * t2 + 0.2 * ((double)rand()/RAND_MAX - 0.5);
        MATRIX_AT(X, i, 3) = -1.0* t1 + 1.5 * t2 + 0.2 * ((double)rand()/RAND_MAX - 0.5);
        MATRIX_AT(X, i, 4) = 5.0 * ((double)rand() / RAND_MAX - 0.5);  /* noise */
        MATRIX_AT(X, i, 5) = 3.0 * ((double)rand() / RAND_MAX - 0.5);  /* noise */
        MATRIX_AT(X, i, 6) = 4.0 * ((double)rand() / RAND_MAX - 0.5);  /* noise */
        MATRIX_AT(X, i, 7) = 2.0 * ((double)rand() / RAND_MAX - 0.5);  /* noise */

        /* Y depends only on t1 and t2 */
        MATRIX_AT(Y, i, 0) = 3.0 * t1 + 2.0 * t2 + 0.3 * ((double)rand()/RAND_MAX - 0.5);
    }

    /* ---- Preprocess ---- */
    printf("Preprocessing data...\n");
    Matrix *X_pp = matrix_copy(X);
    Matrix *Y_pp = matrix_copy(Y);

    Vector *x_mean = vector_alloc(N_XVARS);
    Vector *x_std  = vector_alloc(N_XVARS);
    Vector *y_mean = vector_alloc(N_YVARS);
    Vector *y_std  = vector_alloc(N_YVARS);

    PreprocessingConfig ppcfg;
    ppcfg.x_method = PREPROC_AUTOSCALE;
    ppcfg.y_method = PREPROC_MEAN_CENTER;
    preprocess_fit(X_pp, Y_pp, &ppcfg, x_mean, x_std, y_mean, y_std);

    /* ---- Cross-validation ---- */
    printf("Running 5-fold cross-validation (1-%d LVs)...\n\n", MAX_LVS);

    CrossValConfig cv_cfg = crossval_config_default();
    cv_cfg.n_folds = 5;
    cv_cfg.verbose = 0;

    CrossValResult *cv_result = crossval_result_alloc(MAX_LVS);

    int cv_ok = crossval_kfold(X_pp, Y_pp, MAX_LVS, &cv_cfg, cv_result);

    if (cv_ok == 0) {
        printf("Cross-validation results:\n\n");
        crossval_print_results(cv_result);

        /* ---- Fit final model with optimal number of LVs ---- */
        size_t opt_lvs = cv_result->opt_lvs_onese;  /* Use parsimonious choice */
        printf("\n--- Fitting final model with %lu LVs (1-SE rule) ---\n", (unsigned long)opt_lvs);

        PLSModel *model = pls_model_alloc(N_SAMPLES, N_XVARS, N_YVARS, (unsigned long)opt_lvs);
        pls_model_set_preprocessing(model, x_mean, x_std, 1, 1, y_mean, y_std, 1, 0);

        Matrix *X_fit = matrix_copy(X_pp);
        Matrix *Y_fit = matrix_copy(Y_pp);

        NIPALSConfig nipcfg = nipals_config_default();
        nipcfg.verbose = 0;
        nipals_fit(X_fit, Y_fit, opt_lvs, &nipcfg, model);

        /* Evaluate final model */
        Matrix *Y_pred = matrix_alloc(N_SAMPLES, N_YVARS);
        pls_model_predict_batch(model, X, Y_pred);

        double RMSEC = compute_RMSEC(Y, Y_pred);
        double R2Y = compute_R2Y(Y_pp, Y_fit);

        printf("  Final model R2Y:   %.4f\n", R2Y);
        printf("  Final model RMSEC: %.4f\n", RMSEC);

        /* Variable importance */
        Vector *vip = compute_VIP(model);
        printf("\n  Variable Importance (VIP):\n");
        for (size_t j = 0; j < N_XVARS; j++) {
            printf("    X%-3zu: VIP = %6.3f %s\n", j+1, vip->data[j],
                   vip->data[j] > 1.0 ? "***" : "");
        }
        printf("  (Variables X5-X8 are noise, should have low VIP)\n");

        vector_free(vip);
        matrix_free(Y_pred);
        matrix_free(X_fit);
        matrix_free(Y_fit);
        pls_model_free(model);
    }

    /* ---- Cleanup ---- */
    crossval_result_free(cv_result);
    matrix_free(X); matrix_free(Y);
    matrix_free(X_pp); matrix_free(Y_pp);
    vector_free(x_mean); vector_free(x_std);
    vector_free(y_mean); vector_free(y_std);

    printf("\nExample 3 completed.\n");
    return 0;
}
