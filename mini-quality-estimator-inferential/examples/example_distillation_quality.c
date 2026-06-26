/**
 * @file example_distillation_quality.c
 * @brief L6 Example: Distillation column composition inference.
 *
 * A binary distillation column has overhead composition measured by lab
 * analysis every 4 hours. Process temperatures, reflux ratio, and pressure
 * are available every 10 seconds. This example demonstrates a complete
 * inferential quality estimation workflow.
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
    printf("=== Example: Distillation Column Composition Inference ===\n\n");

    /*---------------------------------------------------------------------
     * L1: Define the quality estimator configuration
     *-------------------------------------------------------------------*/
    qest_config_t cfg;
    qest_config_init(&cfg, "DistillateComposition", "QI_TOP_COMP", "mol%", 3);
    cfg.model_type = QMODEL_DATA_DRIVEN;
    cfg.bias_strategy = BIAS_EWMA_FILTERED;
    cfg.fast_sample_period = 60.0;    /* Process variables every 1 minute */
    cfg.lab_sample_period  = 14400.0; /* Lab sample every 4 hours */
    cfg.bias_filter_gain   = 0.15;

    printf("Estimator: %s\n", cfg.estimator_name);
    printf("Quality tag: %s [%s]\n", cfg.quality_tag, cfg.units);
    printf("Input variables: Top tray T (C), Reflux flow (t/h), Column dP (kPa)\n\n");

    /*---------------------------------------------------------------------
     * L2: Linear model: composition = f(T_top, R, dP)
     *
     * From process knowledge: overhead composition is mainly determined
     * by top tray temperature (higher T = lower purity, lighter components).
     * Reflux ratio and pressure drop provide secondary indicators.
     *
     * Model: y = 98.5 - 0.12*T_top + 0.05*R + 0.03*dP
     *-------------------------------------------------------------------*/
    quality_estimator_t *qest = qest_alloc();
    qest_configure(qest, &cfg);

    double coeff[] = {-0.12, 0.05, 0.03};
    qest_set_linear_model(qest, 98.5, coeff, 3, 0.95, 0.25);

    /*---------------------------------------------------------------------
     * L5: Simulate 24 hours of operation with 4 lab samples
     *-------------------------------------------------------------------*/
    printf("Step | Time(h) | T_top | Reflux | dP  | y_model | y_lab | y_corrected | Bias\n");
    printf("-----|---------|-------|--------|-----|---------|-------|-------------|-----\n");

    /* Operating data over 24 hours (hourly snapshots for display) */
    double T_top[]   = {82.0, 82.5, 83.0, 82.8, 82.2, 81.5, 81.0, 80.5,
                        80.0, 79.8, 80.2, 80.5, 81.0, 81.5, 82.0, 82.5,
                        83.0, 83.2, 82.8, 82.0, 81.5, 81.0, 80.5, 80.0};
    double Reflux[]  = {45.0, 45.0, 46.0, 46.0, 44.0, 43.0, 42.0, 41.0,
                        40.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0,
                        46.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0, 40.0};
    double DP[]      = {12.0, 12.0, 12.5, 12.5, 12.0, 11.5, 11.0, 10.5,
                        10.0, 9.8,  10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
                        13.0, 13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0};

    /* Lab sample times (hours 4, 10, 16, 22) and values */
    int lab_hours[] = {4, 10, 16, 22};
    double lab_values[] = {90.2, 91.5, 92.0, 93.1};  /* Ground truth */

    int lab_idx = 0;

    for (int hour = 0; hour < 24; hour++) {
        /* Set process variable inputs via public API */
        double inputs[] = {T_top[hour], Reflux[hour], DP[hour]};
        qest_set_inputs(qest, inputs, 3);

        /* Run estimation */
        const quality_estimate_t *est = qest_step(qest);

        /* Check if lab sample arrives this hour */
        double y_lab = -1.0;
        if (lab_idx < 4 && hour == lab_hours[lab_idx]) {
            y_lab = lab_values[lab_idx];
            lab_sample_t lab;
            memset(&lab, 0, sizeof(lab));
            lab.measured_value = y_lab;
            lab.lab_stddev = 0.15;
            lab.quality_flag = LAB_QUALITY_GOOD;
            qest_timestamp_set(&lab.sample_time, 2026, 6, 23, hour, 0, 0.0);

            qest_process_lab(qest, &lab);
            lab_idx++;
        }

        /* Display results every 2 hours */
        if (hour % 2 == 0) {
            printf(" %3d |  %5.1f  | %5.1f |  %5.1f | %4.1f | %7.2f | %5.2f |  %8.2f   | %+5.2f\n",
                   hour, (double)hour, T_top[hour], Reflux[hour], DP[hour],
                   est->predicted_value,
                   y_lab > 0 ? y_lab : est->bias_corrected_value,
                   est->bias_corrected_value,
                   est->bias_current);
        }
    }

    /*---------------------------------------------------------------------
     * L4: Performance summary
     *-------------------------------------------------------------------*/
    printf("\n=== Performance Summary ===\n");
    qest_performance_t perf;
    qest_get_performance(qest, &perf);
    printf("Predictions made:    %lld\n", (long long)perf.n_predictions);
    printf("Bias updates:        %lld\n", (long long)perf.n_bias_updates);
    printf("RMSE (model vs lab): %.4f %s\n", perf.rmse, cfg.units);
    printf("MAE:                 %.4f %s\n", perf.mae, cfg.units);
    printf("Health status:       %d (0=OK)\n", (int)perf.health);

    qest_free(qest);
    printf("\n=== Example complete ===\n");
    return 0;
}
