/**
 * @file    example_adaptive_retraining.c
 * @brief   L6 Example: Adaptive retraining strategy for an aging Bioreactor soft sensor
 *
 * Demonstrates:
 *   1. Model performance monitoring during batch fermentation
 *   2. Trigger detection: when to retrain based on RMSE threshold
 *   3. Multi-model ensemble with dynamic weighting
 *   4. Just-in-Time (JIT) prediction as fallback
 *   5. Maintenance decision logging
 *
 * Scenario: Biomass concentration soft sensor in a pharmaceutical
 * bioreactor. The sensor uses off-gas analysis (CO2, O2) and
 * substrate feed rate to estimate biomass concentration.
 *
 * Over 50 batches, the sensor gradually ages due to:
 *   - Strain evolution
 *   - Sensor probe fouling
 *   - Media lot variability
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "soft_sensor_metrics.h"
#include "adaptive_model_updater.h"

/* Simulate biomass estimation (g/L) from 3 process variables:
 * x[0] = CO2 evolution rate (mmol/L/h)
 * x[1] = Dissolved oxygen (%)
 * x[2] = Substrate feed rate (g/L/h) */

#define N_VARS 3
#define N_SAMPLES 50

static void generate_training_data(double *x, double *y, size_t n,
                                    double noise_level)
{
    for (size_t i = 0; i < n; i++) {
        double phase = (double)i / (double)n; /* 0=start, 1=end of batch */

        /* CO2: increases during exponential phase */
        x[i * N_VARS + 0] = 5.0 + 20.0 * exp(-5.0 * (phase - 0.4) * (phase - 0.4));
        /* DO: decreases then increases */
        x[i * N_VARS + 1] = 80.0 - 60.0 * exp(-3.0 * (phase - 0.3) * (phase - 0.3));
        /* Feed: step function */
        x[i * N_VARS + 2] = (phase > 0.1) ? 2.0 : 0.5;

        /* True biomass: logistic growth */
        double biomass = 15.0 / (1.0 + exp(-8.0 * (phase - 0.3)));
        double noise = noise_level * ((double)rand() / RAND_MAX - 0.5) * 4.0;
        y[i] = biomass + noise;
    }
}

int main(void)
{
    srand(789);
    printf("===============================================================\n");
    printf("  Adaptive Retraining - Bioreactor Biomass Soft Sensor\n");
    printf("  Scenario: 50 batches, 3 process variables\n");
    printf("===============================================================\n\n");

    /* Phase 1: Initial training (batches 1-10, low noise) */
    printf("--- Phase 1: Commissioning (Batches 1-10) ---\n");

    double x_train[N_SAMPLES * N_VARS];
    double y_train[N_SAMPLES];
    generate_training_data(x_train, y_train, N_SAMPLES, 0.5);

    /* Initialize adaptive models */
    RPLSState rpls;
    rpls_init(&rpls, N_VARS, 2, 0.95);
    ForgettingFactor ff;
    forgetting_factor_init(&ff, FORGET_ADAPTIVE, 0.95);
    UpdateTriggerConfig trigger;
    update_trigger_init(&trigger, TRIGGER_PERFORMANCE, 20, 0.8);

    JITPLSState jit;
    jitpls_init(&jit, N_VARS, 200, 10, 1.0);

    /* Train RPLS on initial data */
    rpls_bulk_update(&rpls, x_train, y_train, N_SAMPLES);

    /* Evaluate on training data */
    double train_rmse = 0.0;
    for (size_t i = 0; i < N_SAMPLES; i++) {
        double pred = rpls_predict(&rpls, &x_train[i * N_VARS]);
        /* Also add to JIT database */
        jitpls_add_to_database(&jit, &x_train[i * N_VARS], y_train[i]);
        double e = pred - y_train[i];
        train_rmse += e * e;
    }
    train_rmse = sqrt(train_rmse / N_SAMPLES);
    printf("  Initial training RMSE: %.3f g/L\n", train_rmse);

    /* Phase 2: Operation with aging (batches 11-50) */
    printf("\n--- Phase 2: Operation with gradual aging (Batches 11-50) ---\n");

    double cum_rmse = train_rmse;
    size_t total_preds = 0;
    int retrain_count = 0;

    for (int batch = 11; batch <= 50; batch++) {
        double noise = 0.5 + 0.05 * (batch - 10); /* Increasing noise */
        double x[N_SAMPLES * N_VARS];
        double y[N_SAMPLES];
        generate_training_data(x, y, N_SAMPLES, noise);

        double batch_sse = 0.0;
        for (size_t i = 0; i < N_SAMPLES; i++) {
            double pred_rpls = rpls_predict(&rpls, &x[i * N_VARS]);
            double pred_jit = jitpls_predict(&jit, &x[i * N_VARS], 2);

            /* Ensemble: weighted average of RPLS and JIT */
            double ensemble_pred = 0.7 * pred_rpls + 0.3 * pred_jit;
            double e = ensemble_pred - y[i];
            batch_sse += e * e;
            total_preds++;
        }

        double batch_rmse = sqrt(batch_sse / N_SAMPLES);
        cum_rmse = 0.9 * cum_rmse + 0.1 * batch_rmse;

        if (batch % 5 == 1) {
            printf("  Batch %2d: RMSE=%.3f, CumRMSE=%.3f",
                   batch, batch_rmse, cum_rmse);

            /* Check if update needed */
            int do_update = update_trigger_check(&trigger, batch_rmse,
                                                   0.85, 0.05);
            if (do_update) {
                printf(" [RETRAIN triggered]");
                /* Retrain RPLS on this batch */
                rpls_bulk_update(&rpls, x, y, N_SAMPLES);
                /* Update forgetting factor */
                double new_lambda = forgetting_factor_update(&ff, 0.02);
                rpls.lambda = new_lambda;
                /* Add to JIT database */
                for (size_t i = 0; i < N_SAMPLES; i++)
                    jitpls_add_to_database(&jit, &x[i * N_VARS], y[i]);
                retrain_count++;
            }
            printf("\n");
        }
    }

    /* Final results */
    printf("\n--- Maintenance Summary ---\n");
    printf("  Total predictions: %zu\n", total_preds);
    printf("  Retrain events: %d\n", retrain_count);
    printf("  Final cumulative RMSE: %.3f g/L\n", cum_rmse);
    printf("  Forgetting factor: %.4f\n", ff.current_lambda);
    printf("  JIT database size: %zu\n", jit.db_count);

    /* Prune old JIT database entries */
    jitpls_prune_database(&jit, 150, 100);
    printf("  JIT database after pruning: %zu\n", jit.db_count);

    /* Cleanup */
    rpls_destroy(&rpls);
    jitpls_destroy(&jit);

    printf("\n===============================================================\n");
    printf("  Adaptive retraining example complete.\n");
    printf("===============================================================\n");
    return 0;
}
