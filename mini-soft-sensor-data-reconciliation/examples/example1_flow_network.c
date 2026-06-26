/**
 * Example 1: Flow Network Data Reconciliation
 *
 * A simple pipeline network with 5 flow streams and 3 mass balance nodes.
 *
 * Network topology:
 *   Node 1: F1 + F2 = F3  (inlet mixing)
 *   Node 2: F3 = F4 + F5  (flow split)
 *   Node 3: F1 + F2 = F4 + F5  (overall balance, redundant)
 *
 * Measurements (with uncertainties):
 *   F1 = 10.2 +/- 0.5 t/h
 *   F2 = 5.1  +/- 0.3 t/h
 *   F3 = 15.0 +/- 0.4 t/h
 *   F4 = 8.3  +/- 0.3 t/h
 *   F5 = 6.5  +/- 0.3 t/h
 *
 * Note: direct measurements show F1+F2=15.3 but F3=15.0 (imbalance of 0.3)
 * and F3=15.0 but F4+F5=14.8 (imbalance of 0.2).
 * Data reconciliation adjusts all flows to satisfy the mass balances
 * while staying close to the measurements.
 */

#include "dr_core.h"
#include "dr_gross_error.h"
#include "dr_redundancy.h"
#include "dr_measurement.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    int i;
    int nvar = 5, ncon = 2;

    printf("============================================================\n");
    printf("  Example 1: Flow Network Data Reconciliation\n");
    printf("============================================================\n\n");

    /* Create problem */
    dr_problem_t *prob = dr_problem_create(nvar, ncon);
    if (!prob) { printf("Failed to create problem\n"); return 1; }

    /* Set measurements: F1..F5 with uncertainties */
    dr_set_measurement(prob, 0, 10.2, 0.5, 101);
    dr_set_measurement(prob, 1,  5.1, 0.3, 102);
    dr_set_measurement(prob, 2, 15.0, 0.4, 103);
    dr_set_measurement(prob, 3,  8.3, 0.3, 104);
    dr_set_measurement(prob, 4,  6.5, 0.3, 105);

    printf("Measurement data:\n");
    printf("  F1 = 10.2 +/- 0.5 t/h\n");
    printf("  F2 =  5.1 +/- 0.3 t/h\n");
    printf("  F3 = 15.0 +/- 0.4 t/h\n");
    printf("  F4 =  8.3 +/- 0.3 t/h\n");
    printf("  F5 =  6.5 +/- 0.3 t/h\n\n");

    /* Mass balance constraints (independent set only) */
    double coeffs1[5] = { 1.0,  1.0, -1.0,  0.0,  0.0};
    double coeffs2[5] = { 0.0,  0.0,  1.0, -1.0, -1.0};

    dr_set_constraint(prob, 0, coeffs1, 0.0, DR_CONSTRAINT_MASS);
    dr_set_constraint(prob, 1, coeffs2, 0.0, DR_CONSTRAINT_MASS);

    printf("Constraints (independent):\n");
    printf("  Node 1: F1 + F2 - F3 = 0\n");
    printf("  Node 2: F3 - F4 - F5 = 0\n");
    printf("  (Overall balance F1+F2=F4+F5 implied by above)\n\n");

    /* Solve */
    dr_result_t *result = dr_result_create(nvar, ncon);
    if (!result) { dr_problem_free(prob); return 1; }

    int status = dr_solve(prob, result, DR_SOLVER_QR_ORTHOG);
    if (status != DR_OK) {
        printf("Solver failed with status %d\n", status);
        dr_result_free(result); dr_problem_free(prob);
        return 1;
    }

    /* Print results */
    printf("Reconciliation results (Lagrange method):\n");
    printf("  Stream  Measured   Reconciled  Adjustment  |Adjustment|/sigma\n");
    const char *names[] = {"F1", "F2", "F3", "F4", "F5"};
    for (i = 0; i < nvar; i++) {
        double measured = prob->measurements[i].value;
        double reconciled = result->x_reconciled[i];
        double adj = result->x_adjustments[i];
        double sigma = prob->measurements[i].stddev;
        printf("  %s     %7.3f    %7.3f     %+7.3f      %.2f\n",
               names[i], measured, reconciled, adj, fabs(adj)/sigma);
    }

    printf("\nConstraint residuals (should be ~0):\n");
    for (i = 0; i < ncon; i++) {
        printf("  Node %d residual: %.3e\n", i+1, result->constraint_resid[i]);
    }

    printf("\nWLS Objective: %.6f\n", result->objective);
    printf("Chi2 threshold (95%%): %.4f\n", result->chi2_threshold);

    /* Check if objective exceeds threshold (gross error test) */
    if (result->objective > result->chi2_threshold) {
        printf("\n*** WARNING: Objective exceeds chi2 threshold! ***\n");
        printf("    Possible gross error in measurements.\n");
    } else {
        printf("\nObjective within chi2 threshold: no gross error suspected.\n");
    }

    /* Run global test */
    double gt_stat;
    int gt_df;
    dr_global_test(prob, result->x_reconciled, &gt_stat, &gt_df);
    printf("\nGlobal Test: z = %.4f (df = %d)\n", gt_stat, gt_df);
    printf("Critical value (alpha=0.05): %.4f\n",
           dr_meas_chi2_critical(gt_df, 0.05));

    /* Redundancy analysis */
    int redundancy[5];
    dr_compute_redundancy(prob, redundancy);
    printf("\nRedundancy degrees:\n");
    for (i = 0; i < nvar; i++) {
        printf("  %s: %d\n", names[i], redundancy[i]);
    }

    dr_result_free(result);
    dr_problem_free(prob);
    printf("\nExample 1 completed successfully.\n");
    return 0;
}
