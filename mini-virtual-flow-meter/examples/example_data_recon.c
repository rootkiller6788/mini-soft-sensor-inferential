#include <stdio.h>
#include <math.h>
#include "vfm_state_estimation.h"

/* Forward declarations for local data reconciliation functions */
extern int dr_node_mass_balance(const double *measured,
                                 const double *uncertainties,
                                 const int *is_inlet, int n,
                                 double *reconciled);
extern int dr_global_test(const double *x_meas, const double *x_reconciled,
                           const double *variances, int n_meas,
                           int n_cons, double alpha);

int main(void) {
    printf("==========================================\n");
    printf("  Example 4: Data Reconciliation for VFM\n");
    printf("==========================================\n\n");

    /* Three wells feeding into a common manifold.
     * Measurements: well1=10.0, well2=8.5, well3=12.0 m3/h
     * Manifold outlet: 30.0 m3/h
     * Uncertainties: 0.5, 0.4, 0.6, 0.8 m3/h respectively
     */

    double measured[]  = {10.0, 8.5, 12.0, 30.0};
    double uncert[]    = {0.5, 0.4, 0.6, 0.8};
    int is_inlet[]     = {1, 1, 1, -1};  /* +inlet, -outlet */
    double reconciled[4];

    (void)dr_node_mass_balance(measured, uncert, is_inlet, 4, reconciled);

    printf("Oil production allocation - Virtual Flow Meter reconciliation:\n\n");
    printf("  Stream      | Measured | Uncert | Reconciled | Adjustment\n");
    printf("  ------------+----------+--------+------------+-----------\n");

    int i;
    for (i = 0; i < 4; i++) {
        double adj = reconciled[i] - measured[i];
        printf("  %-11s | %8.2f | %6.2f | %10.2f | %+10.2f\n",
               i < 3 ? "Well" : "Manifold",
               measured[i], uncert[i], reconciled[i], adj);
    }

    /* Verify mass balance */
    double sum_in = reconciled[0] + reconciled[1] + reconciled[2];
    printf("\n  Sum of wells (reconciled): %.2f m3/h\n", sum_in);
    printf("  Manifold outlet:          %.2f m3/h\n", reconciled[3]);
    printf("  Mass balance error:       %.6f m3/h\n\n", sum_in - reconciled[3]);

    /* Global test for gross errors */
    double variances[4];
    for (i = 0; i < 4; i++) variances[i] = uncert[i] * uncert[i];
    int ge = dr_global_test(measured, reconciled, variances, 4, 1, 0.05);
    printf("  Gross error test: %s\n", ge ? "SUSPECT (p<0.05)" : "CLEAN");

    printf("\nData reconciliation ensures VFM estimates from multiple\n");
    printf("wells are consistent with the common export meter measurement.\n");
    return 0;
}