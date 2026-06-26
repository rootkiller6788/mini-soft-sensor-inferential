#include <stdio.h>
#include <math.h>
#include "virtual_flow_meter.h"
#include "pipeline_geometry.h"
#include "flow_models.h"
#include "fluid_properties.h"

int main(void) {
    printf("========================================\n");
    printf("  Example 1: Orifice Virtual Flow Meter\n");
    printf("========================================\n\n");

    orifice_params_t params;
    orifice_params_init(&params, 0.1, 0.05);  /* D=0.1m, d=0.05m, beta=0.5 */

    double Cd = orifice_discharge_coeff_iso5167(0.5, 100000.0);
    params.discharge_coeff = Cd;

    double dp[] = {1000, 5000, 10000, 20000, 50000};
    double rho = 998.2;  /* water at 20C */

    printf("Orifice: D=%.2f m, d=%.2f m, beta=%.2f, Cd=%.4f\n",
           params.pipe_diameter, params.bore_diameter,
           params.beta_ratio, Cd);
    printf("Fluid: water at 20 C, density = %.1f kg/m3\n\n", rho);
    printf("  dP [Pa]    |  Q [m3/s]    |  Q [L/s]    |  Q [m3/h]\n");
    printf("  -----------+---------------+---------------+------------\n");

    int i;
    for (i = 0; i < 5; i++) {
        double Q = orifice_vol_flow(&params, dp[i], rho, 1.0);
        printf("  %10.0f | %13.6f | %11.3f | %10.3f\n",
               dp[i], Q, Q*1000.0, Q*3600.0);
    }

    /* Compute Re at max flow */
    double v_max = orifice_vol_flow(&params, dp[4], rho, 1.0)
                 / pipe_cross_section_area(0.1);
    double Re = flow_reynolds_number(v_max, 0.1, 1.004e-6);
    printf("\nMax velocity: %.3f m/s, Re = %.0f (%s)\n",
           v_max, Re, flow_regime_classify(Re)==VFM_REGIME_TURBULENT
           ? "turbulent" : "laminar");

    printf("\nNote: VFM replaces a physical flow meter with this computation.\n");
    printf("The orifice equation maps dP -> Q, using only pressure sensors.\n");
    return 0;
}