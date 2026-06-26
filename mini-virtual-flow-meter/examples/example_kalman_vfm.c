#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "virtual_flow_meter.h"
#include "vfm_state_estimation.h"

int main(void) {
    printf("=============================================\n");
    printf("  Example 3: Kalman Filter Virtual Flow Meter\n");
    printf("=============================================\n\n");

    vfm_kalman_t kf;
    vfm_kalman_init(&kf, 0.001, 0.01);

    printf("Kalman filter state: [flow_rate, bias, drift_rate]\n");
    printf("Process noise sigma: 0.001, Measurement noise sigma: 0.01\n\n");

    /* Simulate: true flow = 0.05 m3/s with step change at t=50 */
    printf("Tracking step change from 0.05 -> 0.08 m3/s at t=50:\n");
    printf("  Step | Measurement | Estimate  | Uncertainty | Innovation\n");
    printf("  -----+-------------+-----------+-------------+-----------\n");

    int t;
    for (t = 0; t < 100; t++) {
        double true_flow = (t < 50) ? 0.05 : 0.08;
        double noise = 0.005 * ((double)rand() / RAND_MAX - 0.5);
        double meas = true_flow + noise;

        vfm_kalman_predict(&kf);
        vfm_kalman_update(&kf, meas);

        double flow, uncert;
        vfm_kalman_get_estimate(&kf, &flow, &uncert);

        if (t % 10 == 0) {
            printf("  %4d | %11.6f | %9.6f | %11.6f | %9.6f\n",
                   t, meas, flow, uncert, kf.innovation);
        }
    }

    double flow, uncert;
    vfm_kalman_get_estimate(&kf, &flow, &uncert);
    printf("\nFinal estimate: %.6f +/- %.6f m3/s (true=0.08, last meas=%.6f)\n",
           flow, uncert, 0.08);

    /* Adaptive noise tuning demo */
    printf("\nAdaptive noise tuning:\n");
    double R0 = kf.R;
    double Ra = vfm_kalman_adapt_noise(&kf, 0.1);
    printf("  R before: %.8f, R after: %.8f\n", R0, Ra);

    printf("\nKalman filter provides optimal sensor fusion for VFM,\n");
    printf("combining multiple noisy measurements with a dynamic model.\n");
    return 0;
}