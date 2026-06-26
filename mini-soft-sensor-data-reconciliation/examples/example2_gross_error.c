/**
 * Example 2: Gross Error Detection in Steam Metering
 *
 * A boiler produces steam measured at 100.0 t/h. The steam passes through
 * a turbine (measured at 60.0 t/h) and a bypass (measured at 35.0 t/h).
 * Mass balance: Boiler = Turbine + Bypass
 *
 * Scenario A: Normal operation (no gross errors)
 *   Boiler=100.0, Turbine=60.0, Bypass=40.0 (exactly balanced)
 *
 * Scenario B: Gross error (faulty turbine meter)
 *   Boiler=100.0, Turbine=50.0 (+10 bias), Bypass=40.0
 *
 * Demonstrates: Global Test, Nodal Test, Measurement Test,
 *               Serial Elimination identification.
 */

#include "dr_core.h"
#include "dr_gross_error.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static void run_scenario(const char *label, double f_boiler, double f_turbine,
                         double f_bypass, double sigma_boiler, double sigma_turbine,
                         double sigma_bypass, double alpha) {
    int nvar = 3, ncon = 1;
    printf("\n--- %s ---\n", label);
    printf("  Measurements: Boiler=%.1f, Turbine=%.1f, Bypass=%.1f\n",
           f_boiler, f_turbine, f_bypass);

    dr_problem_t *prob = dr_problem_create(nvar, ncon);
    if (!prob) return;

    dr_set_measurement(prob, 0, f_boiler, sigma_boiler, 1);
    dr_set_measurement(prob, 1, f_turbine, sigma_turbine, 2);
    dr_set_measurement(prob, 2, f_bypass, sigma_bypass, 3);

    double coeffs[3] = {1.0, -1.0, -1.0};
    dr_set_constraint(prob, 0, coeffs, 0.0, DR_CONSTRAINT_MASS);

    /* Solve */
    dr_result_t *result = dr_result_create(nvar, ncon);
    if (!result) { dr_problem_free(prob); return; }
    if (dr_solve(prob, result, DR_SOLVER_LAGRANGE) != DR_OK) {
        printf("  Solver failed.\n");
        dr_result_free(result); dr_problem_free(prob);
        return;
    }

    printf("  Reconciled: Boiler=%.3f, Turbine=%.3f, Bypass=%.3f\n",
           result->x_reconciled[0], result->x_reconciled[1],
           result->x_reconciled[2]);
    printf("  Adjustments: %+.3f, %+.3f, %+.3f\n",
           result->x_adjustments[0], result->x_adjustments[1],
           result->x_adjustments[2]);
    printf("  Objective: %.4f\n", result->objective);

    /* Global Test */
    dr_ge_test_result_t gt;
    dr_ge_global_test(prob, &gt, alpha);
    printf("\n  Global Test: z=%.4f, critical=%.4f, reject H0=%s\n",
           gt.statistic, gt.critical_value, gt.reject_H0 ? "YES" : "no");

    /* Nodal Test */
    dr_ge_test_result_t nt[1];
    dr_ge_nodal_test(prob, nt, alpha);
    printf("  Nodal Test:  z=%.4f, critical=%.4f, reject H0=%s\n",
           nt[0].statistic, nt[0].critical_value, nt[0].reject_H0 ? "YES" : "no");

    /* Measurement Test */
    dr_ge_test_result_t mt[3];
    dr_ge_measurement_test(prob, mt, alpha);
    const char *vnames[] = {"Boiler", "Turbine", "Bypass"};
    printf("  Measurement Tests:\n");
    for (int i = 0; i < 3; i++) {
        printf("    %s: z=%.4f, critical=%.4f, suspect=%s\n",
               vnames[i], mt[i].statistic, mt[i].critical_value,
               mt[i].reject_H0 ? "YES" : "no");
    }

    /* Serial elimination if gross error detected */
    if (gt.reject_H0) {
        printf("\n  Running serial elimination...\n");
        dr_ge_identification_t *ident = dr_ge_identification_create(nvar);
        if (ident) {
            dr_ge_serial_elimination(prob, ident, alpha);
            printf("  Errors found: %d\n", ident->n_errors_found);
            for (int i = 0; i < ident->n_errors_found; i++) {
                printf("    Variable %d: magnitude ~%.3f\n",
                       ident->error_indices[i],
                       ident->error_magnitudes[i]);
            }
            dr_ge_identification_free(ident);
        }
    }

    dr_result_free(result);
    dr_problem_free(prob);
}

int main(void) {
    printf("================================================================\n");
    printf("  Example 2: Gross Error Detection in Steam Metering\n");
    printf("================================================================\n");

    /* Scenario A: No gross errors */
    run_scenario("Scenario A: Normal Operation",
                 100.0, 60.0, 40.0,  /* measurements */
                 2.0, 1.5, 1.5,       /* uncertainties */
                 0.05);

    /* Scenario B: Gross error on turbine meter (+10 bias) */
    run_scenario("Scenario B: Turbine Meter Bias (+10)",
                 100.0, 50.0, 40.0,  /* turbine reads 50 instead of 60 */
                 2.0, 1.5, 1.5,
                 0.05);

    /* Scenario C: Gross error on bypass meter (-5 bias) */
    run_scenario("Scenario C: Bypass Meter Bias (-5)",
                 100.0, 60.0, 35.0,  /* bypass reads 35 instead of 40 */
                 2.0, 1.5, 1.5,
                 0.05);

    printf("\nExample 2 completed successfully.\n");
    return 0;
}
