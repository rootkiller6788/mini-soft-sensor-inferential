/**
 * example_tep.c — Tennessee Eastman Process Monitoring with PCA
 *
 * L6 Canonical Problem: Fault detection in the TE benchmark process.
 *
 * The Tennessee Eastman process (Downs & Vogel, 1993) is the standard
 * benchmark for process monitoring. It has 52 variables and 21 pre-programmed
 * faults. PCA-based MSPC with T2 and SPE is the most widely used method.
 *
 * This simplified example uses a subset of TE-like variables to demonstrate
 * multivariate fault detection with PCA monitoring.
 *
 * Reference: Downs & Vogel (1993), Chiang, Russell & Braatz (2001)
 */
#include "pca_core.h"
#include "pca_decomposition.h"
#include "pca_monitoring.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_TRAIN 100
#define N_VARS  8
#define N_TEST  20

int main(void)
{
    size_t i, j;
    pca_matrix *X;
    pca_model *model;
    srand((unsigned)time(NULL));

    printf("=== Tennessee Eastman Process Monitoring (Simplified) ===\n\n");

    /* Generate "normal" TE-like data with correlations */
    X = pca_matrix_alloc(N_TRAIN, N_VARS);
    for (i = 0; i < N_TRAIN; i++) {
        double base1 = 50.0 + ((double)rand()/RAND_MAX-0.5)*2.0;
        double base2 = 120.0 + ((double)rand()/RAND_MAX-0.5)*3.0;
        X->data[i*N_VARS+0] = base1;                          /* A feed */
        X->data[i*N_VARS+1] = base1 * 2.0 + ((double)rand()/RAND_MAX-0.5); /* D feed */
        X->data[i*N_VARS+2] = base2;                          /* Reactor temp */
        X->data[i*N_VARS+3] = base2 * 0.8 + ((double)rand()/RAND_MAX-0.5); /* Reactor press */
        X->data[i*N_VARS+4] = 30.0 + 0.1*base2 + ((double)rand()/RAND_MAX-0.5); /* Sep temp */
        X->data[i*N_VARS+5] = 2700.0 + ((double)rand()/RAND_MAX-0.5)*5.0;  /* Sep press */
        X->data[i*N_VARS+6] = 60.0 + 0.05*base1 + ((double)rand()/RAND_MAX-0.5); /* Stripper temp */
        X->data[i*N_VARS+7] = 80.0 + 0.02*base1 + ((double)rand()/RAND_MAX-0.5); /* Flow */
    }

    /* Fit PCA */
    model = pca_model_alloc(N_VARS);
    pca_fit_full(X, model, 50, 1e-10);

    size_t n_pcs = pca_cumvar_rule(model->cum_var, N_VARS, 0.90);
    printf("PCA model: %zu PCs retain %.1f%% variance\n\n",
           n_pcs, model->cum_var[n_pcs-1]*100);

    /* Monitoring: test normal operation */
    printf("Testing normal operation (%d samples):\n", N_TEST);
    {
        int alarms = 0;
        for (i = 0; i < N_TEST; i++) {
            double x[N_VARS];
            double base1 = 50.0 + ((double)rand()/RAND_MAX-0.5)*2.0;
            double base2 = 120.0 + ((double)rand()/RAND_MAX-0.5)*3.0;
            x[0] = base1; x[1] = base1*2.0; x[2] = base2;
            x[3] = base2*0.8; x[4] = 30.0+0.1*base2;
            x[5] = 2700.0; x[6] = 60.0+0.05*base1; x[7] = 80.0+0.02*base1;
            /* Scale */
            double xs[N_VARS];
            for (j = 0; j < N_VARS; j++)
                xs[j] = (x[j]-model->mean_vec[j])/model->std_vec[j];
            pca_fault_result *fr = pca_monitor_observation(
                xs, model->loadings, model->eigenvalues,
                n_pcs, N_VARS, N_TRAIN, 0.01);
            if (fr->t2_alarm || fr->spe_alarm) alarms++;
            pca_fault_result_free(fr);
        }
        printf("  Normal: %d/%d alarms (expect ~1%% false alarm rate)\n",
               alarms, N_TEST);
    }

    /* Monitoring: test with a fault (sensor bias on variable 1) */
    printf("\nTesting with sensor bias fault on variable 1:\n");
    {
        int alarms_t2 = 0, alarms_spe = 0;
        for (i = 0; i < N_TEST; i++) {
            double x[N_VARS];
            double base1 = 50.0 + ((double)rand()/RAND_MAX-0.5)*2.0;
            double base2 = 120.0 + ((double)rand()/RAND_MAX-0.5)*3.0;
            x[0] = base1;
            x[1] = base1*2.0 + 3.0;  /* 3-unit bias fault! */
            x[2] = base2; x[3] = base2*0.8; x[4] = 30.0+0.1*base2;
            x[5] = 2700.0; x[6] = 60.0+0.05*base1; x[7] = 80.0+0.02*base1;
            double xs[N_VARS];
            for (j = 0; j < N_VARS; j++)
                xs[j] = (x[j]-model->mean_vec[j])/model->std_vec[j];
            pca_fault_result *fr = pca_monitor_observation(
                xs, model->loadings, model->eigenvalues,
                n_pcs, N_VARS, N_TRAIN, 0.01);
            if (fr->t2_alarm) alarms_t2++;
            if (fr->spe_alarm) alarms_spe++;
            pca_fault_result_free(fr);
        }
        printf("  Fault: T2 alarms=%d/%d, SPE alarms=%d/%d\n",
               alarms_t2, N_TEST, alarms_spe, N_TEST);
        printf("  (SPE should detect correlation-breaking sensor faults)\n");
    }

    /* Monitoring: test with process upset (shift in reactor temp) */
    printf("\nTesting with process upset (reactor temp shift):\n");
    {
        int alarms_t2 = 0;
        for (i = 0; i < N_TEST; i++) {
            double x[N_VARS];
            double base1 = 50.0 + ((double)rand()/RAND_MAX-0.5)*2.0;
            double base2 = 125.0 + ((double)rand()/RAND_MAX-0.5)*3.0; /* 5C shift! */
            x[0] = base1; x[1] = base1*2.0; x[2] = base2;
            x[3] = base2*0.8; x[4] = 30.0+0.1*base2;
            x[5] = 2700.0; x[6] = 60.0+0.05*base1; x[7] = 80.0+0.02*base1;
            double xs[N_VARS];
            for (j = 0; j < N_VARS; j++)
                xs[j] = (x[j]-model->mean_vec[j])/model->std_vec[j];
            pca_fault_result *fr = pca_monitor_observation(
                xs, model->loadings, model->eigenvalues,
                n_pcs, N_VARS, N_TRAIN, 0.01);
            if (fr->t2_alarm) alarms_t2++;
            pca_fault_result_free(fr);
        }
        printf("  Upset: T2 alarms=%d/%d\n", alarms_t2, N_TEST);
        printf("  (T2 should detect shifts in the PC subspace)\n");
    }

    pca_model_free(model);
    pca_matrix_free(X);

    printf("\n=== TEP Monitoring Complete ===\n");
    return 0;
}
