/**
 * example_reactor.c — PCA Soft Sensor for Chemical Reactor
 *
 * L6 Canonical Problem: Infer reactant concentration from
 * temperature, pressure, and flow measurements.
 *
 * In exothermic reactors, concentration measurement requires
 * sampling and lab analysis (30+ min delay). Temperature and
 * pressure are measured in real-time. A PCA soft sensor uses
 * these fast measurements to infer concentration for advanced
 * control (e.g., MPC with concentration feedback).
 */
#include "pca_core.h"
#include "pca_decomposition.h"
#include "pca_inferential.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(void)
{
    size_t i;
    const size_t N = 60, M = 4;
    pca_matrix *X, *Y;
    pca_soft_sensor *ss;
    double rmsecv[4], r2cv[4];

    srand((unsigned)time(NULL));

    printf("=== Chemical Reactor Soft Sensor ===\n\n");
    printf("Goal: Infer reactant concentration from:\n");
    printf("  - Reactor temperature (C)\n");
    printf("  - Jacket temperature (C)\n");
    printf("  - Feed flow rate (L/min)\n");
    printf("  - Reactor pressure (bar)\n\n");

    /* Generate correlated reactor data */
    X = pca_matrix_alloc(N, M);
    Y = pca_matrix_alloc(N, 1);

    for (i = 0; i < N; i++) {
        double T_r = 80.0 + 15.0 * sin((double)i * 0.15);  /* reactor temp */
        double T_j = T_r - 10.0 + ((double)rand()/RAND_MAX-0.5)*5.0; /* jacket */
        double F   = 10.0 + ((double)rand()/RAND_MAX-0.5)*2.0;       /* flow */
        double P   = 2.5 + 0.01 * T_r + ((double)rand()/RAND_MAX-0.5)*0.1; /* press */
        X->data[i * M + 0] = T_r;
        X->data[i * M + 1] = T_j;
        X->data[i * M + 2] = F;
        X->data[i * M + 3] = P;
        /* Concentration decreases with temperature (reaction consumption) */
        Y->data[i] = 1.0 - 0.005 * T_r + 0.002 * T_j
                     + ((double)rand()/RAND_MAX-0.5)*0.02;
    }

    printf("Data: %zu samples x %zu variables\n", N, M);

    /* Cross-validation for PC selection */
    printf("\nCross-validation for PC selection:\n");
    pca_pcr_cross_validate(X, Y, 3, rmsecv, r2cv);
    for (i = 0; i < 3; i++) {
        printf("  %zu PC(s): RMSECV=%.4f, R2CV=%.4f\n", i+1, rmsecv[i], r2cv[i]);
    }

    /* Find best number of PCs */
    size_t best_pc = 1;
    for (i = 1; i < 3; i++) {
        if (rmsecv[i] < rmsecv[best_pc-1]) best_pc = i + 1;
    }
    printf("  Best: %zu PCs (RMSECV=%.4f)\n", best_pc, rmsecv[best_pc-1]);

    /* Train final model */
    ss = pca_soft_sensor_alloc(M, 1, best_pc);
    pca_pcr_train(X, Y, best_pc, ss);

    /* Test predictions */
    printf("\nOnline predictions:\n");
    double test_cases[][4] = {
        {85.0, 75.0, 10.0, 2.8},
        {90.0, 80.0, 11.0, 3.0},
        {75.0, 65.0, 9.5, 2.6}
    };
    for (i = 0; i < 3; i++) {
        double pred;
        pca_pcr_predict(ss, test_cases[i], &pred);
        printf("  T=%.0f J=%.0f F=%.1f P=%.1f -> Conc=%.4f\n",
               test_cases[i][0], test_cases[i][1],
               test_cases[i][2], test_cases[i][3], pred);
    }

    printf("\nModel quality:\n");
    printf("  PCA variance: %.1f%% (%zu PCs)\n",
           ss->pca->cum_var[best_pc-1]*100, best_pc);

    pca_soft_sensor_free(ss);
    pca_matrix_free(X); pca_matrix_free(Y);
    printf("\nDone.\n");
    return 0;
}
