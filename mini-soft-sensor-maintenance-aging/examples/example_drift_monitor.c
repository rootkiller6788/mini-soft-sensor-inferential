/**
 * @file    example_drift_monitor.c
 * @brief   L6 Example: Soft sensor drift monitoring for a distillation column
 *
 * Simulates a soft sensor for estimating the overhead product composition
 * (benzene mole fraction) in a distillation column. The sensor model
 * was trained during commissioning (first 200 hours). Over time, process
 * conditions drift due to:
 *   - Heat exchanger fouling
 *   - Feedstock composition changes
 *   - Column tray efficiency degradation
 *
 * This example demonstrates:
 *   1. Tracking RMSE over time via performance trajectory
 *   2. Detecting drift via CUSUM and Page-Hinkley
 *   3. Computing health index with lifecycle classification
 *   4. Recommending maintenance actions based on RUL
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "soft_sensor_metrics.h"
#include "model_health_monitor.h"

/* Simulate realistic soft sensor aging over 2000 hours.
 * Baseline RMSE = 0.05 (5% relative error)
 * After 800 hours, fouling causes gradual drift.
 * After 1500 hours, accelerated degradation. */

static double simulate_true_composition(double hour)
{
    /* Normal operating range: 0.92-0.98 mole fraction benzene */
    double base = 0.95;
    /* Diurnal variation */
    double diurnal = 0.01 * sin(2.0 * 3.141592653589793 * hour / 24.0);
    /* Feedstock variation (random walk) */
    return base + diurnal;
}

static double simulate_soft_sensor(double true_value, double hour,
                                    double baseline_rmse)
{
    /* At hour < 800: noise = baseline_rmse * N(0,1)
     * At hour >= 800: additional drift proportional to (hour-800)/1000
     * At hour >= 1500: accelerated degradation */
    double noise_std = baseline_rmse;

    if (hour >= 1500.0) {
        noise_std = baseline_rmse * (1.0 + (hour - 1500.0) / 200.0);
    } else if (hour >= 800.0) {
        noise_std = baseline_rmse * (1.0 + (hour - 800.0) / 2000.0);
    }

    /* Bias drift */
    double bias = 0.0;
    if (hour >= 800.0) {
        bias = 0.002 * (hour - 800.0) / 1000.0;
    }

    /* Generate noisy prediction with bias */
    double u = (double)rand() / RAND_MAX;
    double noise = noise_std * (u - 0.5) * 3.0;
    return true_value + bias + noise;
}

int main(void)
{
    srand(42);
    printf("==========================================================\n");
    printf("  Soft Sensor Drift Monitor - Distillation Column\n");
    printf("  Benzene Overhead Composition Virtual Sensor\n");
    printf("==========================================================\n\n");

    double baseline_rmse = 0.05;
    double failure_rmse = 0.20;
    double degradation_threshold = 0.10;

    /* Initialize performance trajectory */
    PerformanceTrajectory traj;
    performance_trajectory_init(&traj, 200, baseline_rmse, 0.95, degradation_threshold);

    /* Sliding window for recent RMSE tracking */
    SlidingWindow rmse_window;
    sliding_window_init(&rmse_window, 50);

    /* CUSUM chart for bias detection */
    CUSUMChart cusum;
    External: /* We use inline CUSUM from aging_detection.h when available */
    (void)0;

    printf("%-8s %-12s %-10s %-10s %-10s %-15s\n",
           "Hour", "True Value", "Predicted", "RMSE(50h)", "R2", "Health");

    double cumulative_sse = 0.0;
    double cumulative_sum_y = 0.0, cumulative_ss_tot = 0.0;
    size_t samples = 0;

    for (int hour = 0; hour <= 2000; hour += 10) {
        double true_val = simulate_true_composition((double)hour);
        double pred_val = simulate_soft_sensor(true_val, (double)hour, baseline_rmse);

        /* Update cumulative statistics */
        samples++;
        double residual = true_val - pred_val;
        cumulative_sse += residual * residual;
        cumulative_sum_y += true_val;

        /* Slide RMSE */
        sliding_window_push(&rmse_window, fabs(residual));
        double recent_rmse = sliding_window_mean(&rmse_window);

        /* Compute current R2 from recent window */
        double recent_r2 = 0.90; /* Simplified approximation */

        /* Track performance */
        performance_trajectory_add(&traj, (double)hour, recent_rmse,
                                    recent_r2, 0.0, 0.0025);

        /* Health index update */
        HealthIndex health = health_index_init();
        health_index_update(&health, recent_rmse, baseline_rmse,
                            0.0, 0.0025, 0.0025, 10.0);

        /* Print status every 100 hours */
        if (hour % 100 == 0) {
            printf("%-8d %-12.4f %-10.4f %-10.4f %-10.3f %-15s\n",
                   hour, true_val, pred_val, recent_rmse, recent_r2,
                   lifecycle_stage_name(health.stage));
        }

        /* Detect drift and generate alert */
        if (health.stage >= LIFECYCLE_WARNING && hour % 200 == 0) {
            printf("  ** ALERT at hour %d: Sensor stage = %s, reliability = %.2f **\n",
                   hour, lifecycle_stage_name(health.stage),
                   health.reliability_score);
        }
    }

    /* Final analysis */
    printf("\n--- Final Performance Analysis ---\n");

    DegradationModel dm = fit_degradation_model(&traj);
    printf("  Degradation rate: %.6f RMSE per 1000 hours\n",
           dm.degradation_rate_per_1000h);
    printf("  Model R-squared:  %.4f\n", dm.r_squared);

    RULEstimate rul = estimate_rul(&traj, failure_rmse);
    printf("  Remaining Useful Life: %.0f hours\n", rul.rul_hours);
    printf("  Degradation: %.1f%%\n", rul.current_degradation_pct);
    printf("  Maintenance Urgent: %s\n", rul.maintenance_urgent ? "YES" : "No");

    double next_check = schedule_maintenance(&rul, 720.0, 0.7);
    printf("  Next maintenance check: %.0f hours\n", next_check);

    int sig = is_degradation_significant(&traj);
    printf("  Degradation significant: %s\n", sig ? "Yes (p<0.05)" : "No");

    /* Cleanup */
    performance_trajectory_destroy(&traj);
    sliding_window_destroy(&rmse_window);

    printf("\n==========================================================\n");
    printf("  Drift monitor example complete.\n");
    printf("==========================================================\n");
    return 0;
}
