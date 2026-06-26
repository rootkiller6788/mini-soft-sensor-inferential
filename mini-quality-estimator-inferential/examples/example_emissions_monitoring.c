/**
 * @file example_emissions_monitoring.c
 * @brief L6-L7 Example: Emissions NOx inference for environmental compliance.
 *
 * Continuous Emissions Monitoring Systems (CEMS) are expensive ($50k-$200k).
 * Inferential sensing using combustion process variables provides a
 * cost-effective alternative for NOx estimation.
 *
 * Reference: Shakil et al. (2012) "Soft sensor for NOx and O2... "— ISA Trans.
 */

#include "quality_estimator_types.h"
#include "kalman_quality.h"
#include "bias_correction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void)
{
    printf("=== Example: NOx Emissions Inferential Estimation ===\n\n");

    /* NOx model using combustion parameters:
     * NOx = 150 + 0.5*(T_flame - 1200) + 0.3*(O2 - 3) - 0.1*(AirFlow - 100)
     *      - 0.2*(NH3_flow - 5)
     *
     * where: T_flame = flame temperature (°C), O2 = excess oxygen (%),
     *        AirFlow = combustion air (t/h), NH3_flow = ammonia injection (kg/h, SCR)
     */

    /* Set up a standalone Kalman filter for NOx estimation */
    /* State: [NOx, bias]^T */
    double A[] = {1.0, 0.0, 0.0, 1.0};  /* Bias is random walk */
    double C[] = {1.0, 1.0};            /* Measurement = NOx + bias */
    double Q[] = {0.5, 0.0, 0.0, 0.01}; /* NOx noise > bias noise */
    double R[] = {4.0};                 /* Measurement noise */
    double x0[] = {150.0, 0.0};
    double P0[] = {100.0, 0.0, 0.0, 10.0};

    kalman_filter_t kf;
    kf_alloc(&kf, 2, 4, 1);
    kf_set_matrices(&kf, A, NULL, C);
    kf_set_noise(&kf, Q, R);
    kf_set_initial(&kf, x0, P0);

    printf("Estimator: NOx Emissions (Kalman filter)\n");
    printf("States: [NOx concentration, bias]\n");
    printf("Inputs: T_flame, O2, AirFlow, NH3_flow\n\n");

    /* Simulate 8 hours with quarterly CEMS validation */
    printf("Hour | T_flame | O2%% | Air | NH3 | KF_est | CEMS | +/-3sigma\n");
    printf("-----|---------|-----|-----|-----|--------|------|----------\n");

    double T_flame[]  = {1200,1210,1220,1215,1205,1195,1185,1190};
    double O2_data[]  = {3.0, 3.2, 3.5, 3.3, 2.8, 2.5, 2.3, 2.8};
    double Air_data[] = {100, 102, 105, 103, 98, 95, 93, 98};
    double NH3_data[] = {5.0, 5.5, 6.0, 5.8, 5.2, 4.8, 4.5, 5.0};

    /* CEMS validation readings every 2 hours */
    double cems_readings[] = {155, 162, 158, 152};
    int cems_idx = 0;

    for (int h = 0; h < 8; h++) {
        /* Compute model-based NOx prediction */
        double T = T_flame[h], O2 = O2_data[h];
        double Air = Air_data[h], NH3 = NH3_data[h];
        double NOx_model = 150.0 + 0.5*(T - 1200.0) + 0.3*(O2 - 3.0)
                         - 0.1*(Air - 100.0) - 0.2*(NH3 - 5.0);

        double u[] = {T, O2, Air, NH3};

        double y_cems = -1;
        /* CEMS reading every 2 hours */
        if (h == 1 || h == 3 || h == 5 || h == 7) {
            y_cems = cems_readings[cems_idx++];
        }

        double y_meas = (y_cems > 0) ? y_cems : NOx_model;
        double y_arr[] = {y_meas};
        kf_step(&kf, u, y_arr, NULL);

        double kval = kf_get_quality(&kf);
        double kvar = kf_get_quality_variance(&kf);
        double sigma = sqrt(kvar);

        printf("  %2d  |  %5.0f  | %3.1f | %3.0f | %3.1f | %6.1f | %4.0f | [%5.1f, %5.1f]\n",
               h, T, O2, Air, NH3, kval, y_cems > 0 ? y_cems : 0.0,
               kval - 3.0*sigma, kval + 3.0*sigma);
    }

    kf_free(&kf);
    printf("\n=== Example complete ===\n");
    return 0;
}
