/**
 * @example example_quality_estimator.c
 * @brief Industrial Quality Soft Sensor using Adaptive Kalman Filter
 * L7: Industrial application — infer product quality with confidence bounds
 */
#include "kalman_applications.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== Industrial Quality Soft Sensor ===\n");
    printf("Adaptive Kalman Filter for inferring product quality.\n");
    printf("Application: polymer melt index estimation.\n\n");

    IndustrialQualityEstimator est;
    double beta[4] = {0.8, -0.3, 0.15, 0.05};
    quality_estimator_init(&est, beta, 1.5, 4, 0.0001, 0.01, 1.0);

    printf("Model: quality = 0.8*T1 - 0.3*P1 + 0.15*F1 + 0.05*T2 + 1.5\n\n");

    printf("%6s %10s %10s %12s %12s\n",
           "Time", "Quality", "StdDev", "CI_Lower", "CI_Upper");
    printf("----------------------------------------------------------\n");

    double T1 = 180.0, P1 = 50.0, F1 = 120.0, T2 = 200.0;

    for (int t = 0; t <= 100; t++) {
        /* Process drifts slowly */
        T1 += 0.02;
        P1 += 0.01 * sin((double)t * 0.1);
        F1 += 0.005;
        T2 -= 0.01;

        double x_meas[4] = {T1, P1, F1, T2};
        quality_estimator_update(&est, x_meas);

        /* Every 20 steps, simulate a lab sample */
        if (t > 0 && t % 20 == 0) {
            double true_quality = 0.8*T1 - 0.3*P1 + 0.15*F1 + 0.05*T2 + 1.5;
            double lab_value = true_quality + 0.1 * ((double)(t%5) - 2.0);
            quality_estimator_calibrate(&est, lab_value);
        }

        if (t % 10 == 0) {
            double q, lo, hi;
            quality_estimator_get_prediction(&est, &q, &lo, &hi);
            printf("%6d %10.3f %10.4f %12.4f %12.4f\n",
                   t, q, est.quality_std, lo, hi);
        }
    }

    printf("\nFinal Quality Estimate:\n");
    double q, lo, hi;
    quality_estimator_get_prediction(&est, &q, &lo, &hi);
    printf("  Quality:     %.3f\n", q);
    printf("  95%% CI:     [%.4f, %.4f]\n", lo, hi);
    printf("  EWMA Error:  %.6f\n", est.ewma_error);
    printf("  Updates:     %u\n", est.update_count);

    uint8_t faults = quality_estimator_check_sensors(&est);
    printf("  Sensor Status: %s\n", faults ? "FAULT DETECTED" : "ALL OK");

    return 0;
}
