#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#include "pls_model.h"
#include "pls_nipals.h"
#include "pls_preprocessing.h"
#include "pls_statistics.h"
#include "pls_online.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* =================================================================
 * Example 2: Multivariate Process Monitoring with PLS
 *
 * Scenario: A distillation column is monitored using PLS-based MSPC
 * (Multivariate Statistical Process Control). T2 and SPE statistics
 * detect process faults in real-time.
 *
 * Knowledge: L6 — Canonical Problem (process monitoring)
 *            L7 — Industrial Application (fault detection in refining)
 * ================================================================= */

#define N_TRAIN   100
#define N_MONITOR  30
#define N_VARS      6
#define N_LVS       3

int main(void) {
    printf("=== Example 2: Process Monitoring with PLS ===\n\n");

    /* ---- Generate normal operating data ---- */
    printf("Generating normal operating data (%d samples)...\n", N_TRAIN);
    Matrix *X_train = matrix_alloc(N_TRAIN, N_VARS);
    Matrix *Y_train = matrix_alloc(N_TRAIN, 1);

    srand(12345);
    for (size_t i = 0; i < N_TRAIN; i++) {
        /* Normal operation: variables have typical means and correlations */
        double t = (double)i / N_TRAIN;
        double x0 = 100.0 + 5.0 * sin(2.0 * 3.14159 * t) + 1.0 * ((double)rand()/RAND_MAX - 0.5);
        double x1 = 50.0  + 3.0 * cos(2.0 * 3.14159 * t) + 0.5 * ((double)rand()/RAND_MAX - 0.5);
        double x2 = 75.0  + 2.0 * x0 / 100.0 + 0.3 * ((double)rand()/RAND_MAX - 0.5);
        double x3 = 30.0  + 1.5 * x1 / 50.0  + 0.2 * ((double)rand()/RAND_MAX - 0.5);
        double x4 = 60.0  + 0.5 * x0 / 100.0 + 0.8 * ((double)rand()/RAND_MAX - 0.5);
        double x5 = 45.0  + 1.0 * sin(3.14159 * t) + 0.4 * ((double)rand()/RAND_MAX - 0.5);

        MATRIX_AT(X_train, i, 0) = x0;
        MATRIX_AT(X_train, i, 1) = x1;
        MATRIX_AT(X_train, i, 2) = x2;
        MATRIX_AT(X_train, i, 3) = x3;
        MATRIX_AT(X_train, i, 4) = x4;
        MATRIX_AT(X_train, i, 5) = x5;

        /* Y = quality indicator (related to first 3 variables) */
        MATRIX_AT(Y_train, i, 0) = 0.4*x0 + 0.3*x1 + 0.3*x2
                                    + 0.5*((double)rand()/RAND_MAX - 0.5);
    }

    /* ---- Preprocess and fit PLS ---- */
    printf("Fitting PLS model...\n");
    Matrix *X_pp = matrix_copy(X_train);
    Matrix *Y_pp = matrix_copy(Y_train);

    Vector *x_mean = vector_alloc(N_VARS);
    Vector *x_std  = vector_alloc(N_VARS);
    Vector *y_mean = vector_alloc(1);
    Vector *y_std  = vector_alloc(1);

    PreprocessingConfig ppcfg;
    ppcfg.x_method = PREPROC_AUTOSCALE;
    ppcfg.y_method = PREPROC_MEAN_CENTER;
    preprocess_fit(X_pp, Y_pp, &ppcfg, x_mean, x_std, y_mean, y_std);

    PLSModel *model = pls_model_alloc(N_TRAIN, N_VARS, 1, N_LVS);
    pls_model_set_preprocessing(model, x_mean, x_std, 1, 1, y_mean, y_std, 1, 0);

    NIPALSConfig nipcfg = nipals_config_default();
    nipals_fit(X_pp, Y_pp, N_LVS, &nipcfg, model);

    /* ---- Initialize online monitor ---- */
    printf("Initializing online monitor...\n");
    OnlineMonitor monitor;
    online_monitor_init(model, &monitor, 0.05, 0.01);

    printf("\nControl Limits:\n");
    printf("  T2  95%%: %.2f, 99%%: %.2f\n", monitor.T2_limit_95, monitor.T2_limit_99);
    printf("  SPE 95%%: %.4f, 99%%: %.4f\n", monitor.SPE_limit_95, monitor.SPE_limit_99);

    /* ---- Simulate monitoring (first 20 normal, last 10 with fault) ---- */
    printf("\n--- Online Monitoring Simulation ---\n");
    printf("(First 20 samples: normal, last 10: simulated fault)\n\n");

    for (size_t i = 0; i < N_MONITOR; i++) {
        double x[6];
        double t = (double)i / N_MONITOR;

        if (i < 20) {
            /* Normal operation */
            x[0] = 100.0 + 5.0 * sin(2.0*3.14159*(t+1.0)) + 0.5*((double)rand()/RAND_MAX-0.5);
            x[1] = 50.0  + 3.0 * cos(2.0*3.14159*(t+1.0)) + 0.3*((double)rand()/RAND_MAX-0.5);
            x[2] = 75.0  + 2.0 * x[0]/100.0 + 0.2*((double)rand()/RAND_MAX-0.5);
            x[3] = 30.0  + 1.5 * x[1]/50.0  + 0.1*((double)rand()/RAND_MAX-0.5);
            x[4] = 60.0  + 0.5 * x[0]/100.0 + 0.6*((double)rand()/RAND_MAX-0.5);
            x[5] = 45.0  + 1.0 * sin(3.14159*(t+1.0)) + 0.3*((double)rand()/RAND_MAX-0.5);
        } else {
            /* Fault: variable 2 drifts up (sensor fault or process upset) */
            double drift = 1.0 + 0.5 * (double)(i - 20);
            x[0] = 100.0 + 5.0 * sin(2.0*3.14159*(t+1.0)) + 0.5*((double)rand()/RAND_MAX-0.5);
            x[1] = 50.0  + 3.0 * cos(2.0*3.14159*(t+1.0)) + 0.3*((double)rand()/RAND_MAX-0.5);
            x[2] = 75.0  + 2.0 * x[0]/100.0 * drift + 0.2*((double)rand()/RAND_MAX-0.5);
            x[3] = 30.0  + 1.5 * x[1]/50.0  + 0.1*((double)rand()/RAND_MAX-0.5);
            x[4] = 60.0  + 0.5 * x[0]/100.0 + 0.6*((double)rand()/RAND_MAX-0.5);
            x[5] = 45.0  + 1.0 * sin(3.14159*(t+1.0)) + 0.3*((double)rand()/RAND_MAX-0.5);
        }

        Vector *x_new = vector_from_array(N_VARS, x);
        OnlinePrediction *pred = online_prediction_alloc(1, N_LVS);
        online_predict_sample(&monitor, x_new, pred);

        printf("Sample %2zu | T2=%6.2f | SPE=%7.4f | Alarm: %s\n",
               i+1, pred->T2, pred->SPE, alarm_state_string(pred->alarm));

        online_prediction_free(pred);
        vector_free(x_new);
    }

    /* ---- Summary ---- */
    size_t total, alarms;
    double rate;
    online_monitor_summary(&monitor, &total, &alarms, &rate);
    printf("\n--- Monitoring Summary ---\n");
    printf("  Total samples: %zu\n", total);
    printf("  Alarms:        %zu\n", alarms);
    printf("  Alarm rate:    %.1f%%\n", rate * 100.0);

    /* ---- Cleanup ---- */
    online_monitor_free(&monitor);
    pls_model_free(model);
    matrix_free(X_train); matrix_free(Y_train);
    matrix_free(X_pp); matrix_free(Y_pp);
    vector_free(x_mean); vector_free(x_std);
    vector_free(y_mean); vector_free(y_std);

    printf("\nExample 2 completed.\n");
    return 0;
}
