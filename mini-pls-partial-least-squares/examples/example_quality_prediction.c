#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#include "pls_model.h"
#include "pls_nipals.h"
#include "pls_preprocessing.h"
#include "pls_statistics.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* =================================================================
 * Example 1: Quality Prediction using PLS (Soft Sensor)
 *
 * Scenario: A chemical reactor produces a product whose quality (purity)
 * is measured only once per day in the lab. However, 5 process variables
 * (temperature, pressure, flow rate, pH, residence time) are measured
 * online every minute. PLS builds a soft sensor to predict product
 * quality in real-time from the process measurements.
 *
 * Knowledge: L6 — Canonical Problem (soft sensor / inferential measurement)
 *            L7 — Industrial Application (quality estimation)
 * ================================================================= */

#define N_SAMPLES 50
#define N_XVARS   5
#define N_YVARS   1
#define N_LVS     2

int main(void) {
    printf("=== Example 1: Quality Prediction Soft Sensor ===\n\n");

    /* ---- Build synthetic training data ---- */
    printf("Generating synthetic process data (%d samples, %d variables)...\n",
           N_SAMPLES, N_XVARS);

    Matrix *X = matrix_alloc(N_SAMPLES, N_XVARS);
    Matrix *Y = matrix_alloc(N_SAMPLES, N_YVARS);

    /* Generate correlated data:
     * Y = 2*X1 + 1.5*X2 + 0.5*X3 + noise
     * X4 and X5 are noise variables
     */
    srand(42);
    for (size_t i = 0; i < N_SAMPLES; i++) {
        double x1 = 10.0 + 2.0 * ((double)rand() / RAND_MAX - 0.5);
        double x2 = 20.0 + 3.0 * ((double)rand() / RAND_MAX - 0.5);
        double x3 = 30.0 + 1.0 * ((double)rand() / RAND_MAX - 0.5);
        double x4 = 15.0 + 4.0 * ((double)rand() / RAND_MAX - 0.5);
        double x5 = 25.0 + 2.0 * ((double)rand() / RAND_MAX - 0.5);

        double purity = 2.0 * x1 + 1.5 * x2 + 0.5 * x3
                       + 0.1 * ((double)rand() / RAND_MAX - 0.5);

        MATRIX_AT(X, i, 0) = x1;
        MATRIX_AT(X, i, 1) = x2;
        MATRIX_AT(X, i, 2) = x3;
        MATRIX_AT(X, i, 3) = x4;
        MATRIX_AT(X, i, 4) = x5;
        MATRIX_AT(Y, i, 0) = purity;
    }

    /* ---- Preprocess data ---- */
    printf("Preprocessing data (mean-center + auto-scale)...\n");
    PreprocessingConfig ppcfg;
    ppcfg.x_method = PREPROC_AUTOSCALE;
    ppcfg.y_method = PREPROC_MEAN_CENTER;

    Vector *x_mean = vector_alloc(N_XVARS);
    Vector *x_std  = vector_alloc(N_XVARS);
    Vector *y_mean = vector_alloc(N_YVARS);
    Vector *y_std  = vector_alloc(N_YVARS);

    Matrix *X_pp = matrix_copy(X);
    Matrix *Y_pp = matrix_copy(Y);
    preprocess_fit(X_pp, Y_pp, &ppcfg, x_mean, x_std, y_mean, y_std);

    /* ---- Fit PLS model using NIPALS ---- */
    printf("Fitting PLS model with %d latent variables...\n", N_LVS);

    PLSModel *model = pls_model_alloc(N_SAMPLES, N_XVARS, N_YVARS, N_LVS);
    pls_model_set_preprocessing(model, x_mean, x_std, 1, 1, y_mean, y_std, 1, 0);

    Matrix *X_fit = matrix_copy(X_pp);
    Matrix *Y_fit = matrix_copy(Y_pp);

    NIPALSConfig nipcfg = nipals_config_default();
    int fit_ok = nipals_fit(X_fit, Y_fit, N_LVS, &nipcfg, model);

    if (fit_ok == 0) {
        printf("PLS model fitted successfully!\n");

        /* ---- Model diagnostics ---- */
        Matrix *Y_pred = matrix_alloc(N_SAMPLES, N_YVARS);
        pls_model_predict_batch(model, X, Y_pred);

        double RMSEC = compute_RMSEC(Y, Y_pred);
        double PRESS = compute_PRESS(Y, Y_pred);
        double SSY = matrix_sum_of_squares(Y_pp);
        double Q2 = compute_Q2(PRESS, SSY);

        printf("\n--- Model Performance ---\n");
        printf("  RMSEC: %.4f\n", RMSEC);
        printf("  PRESS: %.4f\n", PRESS);
        printf("  Q2:    %.4f\n", Q2);
        printf("  R2Y:   %.4f\n", model->R2Y_cum);

        /* ---- Variable importance ---- */
        Vector *vip = compute_VIP(model);
        printf("\n--- Variable Importance (VIP) ---\n");
        const char *var_names[] = {"Temp", "Pressure", "Flow", "pH", "Res.Time"};
        for (size_t j = 0; j < N_XVARS; j++) {
            printf("  %-10s: VIP = %.3f %s\n", var_names[j], vip->data[j],
                   vip->data[j] > 1.0 ? "(important)" : "");
        }

        /* ---- Real-time prediction demo ---- */
        printf("\n--- Real-time Prediction Demo ---\n");
        double new_sample[] = {10.5, 21.0, 30.2, 14.8, 24.5};
        Vector *x_new = vector_from_array(N_XVARS, new_sample);
        Vector *y_hat = vector_alloc(N_YVARS);

        pls_model_predict_single(model, x_new, y_hat);
        printf("  New sample: [");
        for (size_t j = 0; j < N_XVARS; j++)
            printf("%.1f%s", new_sample[j], j < N_XVARS - 1 ? ", " : "");
        printf("]\n");
        printf("  Predicted purity: %.4f\n", y_hat->data[0]);
        printf("  Expected purity:  %.4f (2*10.5 + 1.5*21 + 0.5*30.2 = %.1f)\n",
               2.0*10.5 + 1.5*21.0 + 0.5*30.2);
        printf("  Prediction error: %.4f\n",
               y_hat->data[0] - (2.0*10.5 + 1.5*21.0 + 0.5*30.2));

        vector_free(vip);
        matrix_free(Y_pred);
        vector_free(x_new);
        vector_free(y_hat);
    } else {
        printf("PLS fitting failed!\n");
    }

    /* ---- Cleanup ---- */
    pls_model_free(model);
    matrix_free(X); matrix_free(Y);
    matrix_free(X_pp); matrix_free(Y_pp);
    matrix_free(X_fit); matrix_free(Y_fit);
    vector_free(x_mean); vector_free(x_std);
    vector_free(y_mean); vector_free(y_std);

    printf("\nExample 1 completed.\n");
    return 0;
}
