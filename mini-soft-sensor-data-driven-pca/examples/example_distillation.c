/**
 * example_distillation.c — PCA Soft Sensor for Distillation Column
 *
 * L6 Canonical Problem: Estimate product composition from tray temperatures.
 *
 * In a distillation column, composition analyzers are expensive and slow
 * (5-15 min sample intervals). Tray temperatures are cheap and fast (1 sec).
 * A PCA-based soft sensor uses tray temperatures to infer composition
 * in real-time, enabling tighter control.
 *
 * This example demonstrates:
 *   1. PCA model training on historical temperature-composition data
 *   2. Online composition prediction from temperature measurements
 *   3. Process monitoring with T2 and SPE for sensor fault detection
 *   4. Variance analysis to select number of PCs

 * Reference: Kresta, MacGregor & Marlin (1991)
 *            Qin (2003) "Statistical Process Monitoring"
 */
#include "pca_core.h"
#include "pca_decomposition.h"
#include "pca_inferential.h"
#include "pca_monitoring.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_TRAIN 50
#define N_VARS  6
#define N_TEST  5

int main(void)
{
    size_t i, j;
    pca_matrix *X_train, *Y_train;
    pca_model *model;
    pca_soft_sensor *ss;

    printf("=== Distillation Column Soft Sensor ===\n\n");

    /* Step 1: Generate synthetic training data.
     * 6 tray temperatures (secondary variables) predict
     * product purity (primary variable). */
    printf("Step 1: Generating synthetic distillation data...\n");
    X_train = pca_matrix_alloc(N_TRAIN, N_VARS);
    Y_train = pca_matrix_alloc(N_TRAIN, 1);

    for (i = 0; i < N_TRAIN; i++) {
        double base_temp = 150.0 + 20.0 * sin((double)i * 0.2);
        /* Tray temperatures with correlated structure */
        for (j = 0; j < N_VARS; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            X_train->data[i * N_VARS + j] = base_temp + (double)j * 5.0 + noise;
        }
        /* Product purity = f(temperatures) + noise */
        double purity = 0.95 - 0.002 * base_temp
                        + 0.001 * X_train->data[i * N_VARS + 5]
                        + ((double)rand() / RAND_MAX - 0.5) * 0.01;
        Y_train->data[i] = purity;
    }
    printf("  Training data: %zu observations x %zu variables\n", (size_t)N_TRAIN, (size_t)N_VARS);

    /* Step 2: PCA model fitting */
    printf("\nStep 2: Fitting PCA model...\n");
    model = pca_model_alloc(N_VARS);
    if (pca_fit_full(X_train, model, 50, 1e-8) != 0) {
        printf("  PCA fitting failed!\n"); return 1;
    }
    printf("  Eigenvalues: ");
    for (j = 0; j < N_VARS && j < 5; j++) printf("%.3f ", model->eigenvalues[j]);
    printf("\n");

    printf("  Variance explained:\n");
    for (j = 0; j < N_VARS; j++)
        printf("    PC%zu: %.1f%% (cum: %.1f%%)\n",
               j+1, model->var_expl[j]*100, model->cum_var[j]*100);

    /* PC selection */
    size_t n_pcs = pca_cumvar_rule(model->cum_var, N_VARS, 0.95);
    printf("  Selected PCs: %zu (95%% variance)\n", n_pcs);

    /* Step 3: Soft sensor training */
    printf("\nStep 3: Training PCR soft sensor...\n");
    ss = pca_soft_sensor_alloc(N_VARS, 1, n_pcs);
    /* Since X_train was modified in-place by pca_fit_full, we need fresh data */
    pca_matrix *X_fresh = pca_matrix_alloc(N_TRAIN, N_VARS);
    pca_matrix *Y_fresh = pca_matrix_alloc(N_TRAIN, 1);
    for (i = 0; i < N_TRAIN; i++) {
        double base_temp = 150.0 + 20.0 * sin((double)i * 0.2);
        for (j = 0; j < N_VARS; j++) {
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            X_fresh->data[i * N_VARS + j] = base_temp + (double)j * 5.0 + noise;
        }
        Y_fresh->data[i] = 0.95 - 0.002 * base_temp
                           + 0.001 * X_fresh->data[i * N_VARS + 5]
                           + ((double)rand() / RAND_MAX - 0.5) * 0.01;
    }
    if (pca_pcr_train(X_fresh, Y_fresh, n_pcs, ss) != 0) {
        printf("  Soft sensor training failed!\n"); return 1;
    }
    printf("  Soft sensor trained: %zu PCs, %zu->%d variables\n",
           n_pcs, (size_t)N_VARS, 1);

    /* Step 4: Online prediction test */
    printf("\nStep 4: Online composition prediction...\n");
    double x_test[N_VARS] = {152.0, 157.5, 162.0, 167.5, 172.0, 177.5};
    double y_pred;
    pca_pcr_predict(ss, x_test, &y_pred);
    printf("  Tray temperatures: ");
    for (j = 0; j < N_VARS; j++) printf("%.1f ", x_test[j]);
    printf("\n  Predicted purity: %.4f\n", y_pred);

    /* Step 5: Process monitoring */
    printf("\nStep 5: Process monitoring with PCA...\n");
    /* Scale test observation */
    double x_scaled[N_VARS];
    for (j = 0; j < N_VARS; j++) {
        double s = ss->pca->std_vec[j];
        if (s < 1e-12) s = 1.0;
        x_scaled[j] = (x_test[j] - ss->pca->mean_vec[j]) / s;
    }
    pca_fault_result *fr = pca_monitor_observation(
        x_scaled, model->loadings, model->eigenvalues,
        n_pcs, N_VARS, N_TRAIN, 0.05);
    pca_fault_result_print(fr);

    /* Step 6: Summary */
    printf("\n=== Summary ===\n");
    printf("  PCA model: %zu variables, %zu PCs (%.1f%% variance)\n",
           (size_t)N_VARS, n_pcs, model->cum_var[n_pcs-1]*100);
    printf("  Soft sensor: PCR with %zu latent variables\n", n_pcs);
    printf("  Monitoring: T2 limit=%.2f, SPE limit=%.4f\n",
           fr->t2_threshold, fr->spe_threshold);

    /* Cleanup */
    pca_fault_result_free(fr);
    pca_soft_sensor_free(ss);
    pca_model_free(model);
    pca_matrix_free(X_fresh); pca_matrix_free(Y_fresh);
    pca_matrix_free(X_train); pca_matrix_free(Y_train);

    printf("\nDone.\n");
    return 0;
}
