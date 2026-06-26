/**
 * @file example_fermentation_soft_sensor.c
 * @brief L6 Example: Neural Network Soft Sensor for Fed-Batch Fermentation
 *
 * Demonstrates a soft sensor for estimating biomass, product, and substrate
 * concentrations from easily measured online variables.
 *
 * Reference: Bailey & Ollis, "Biochemical Engineering Fundamentals" (1986)
 *            James et al., "Comparative study of black-box and hybrid..."
 *            Bioprocess and Biosystems Engineering (2002)
 *
 * Classic Problem (L6): In industrial fermentation, biomass concentration
 * can only be measured offline via dry cell weight (DCW) assays taking hours.
 * Online sensors (pH, DO, temperature, off-gas) provide continuous data.
 * A neural network soft sensor bridges this gap for real-time monitoring
 * and control of feed rate.
 */

#include "nn_soft_sensor.h"
#include "nn_sensor_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Neural Network Soft Sensor for Fermentation Process        ║\n");
    printf("║  L6: Biomass & Product Estimation (Monod Kinetics)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int n_samples = 150;
    int n_features = 5;
    int n_outputs = 3;

    printf("Step 1: Generating fermentation process data (%d samples)...\n", n_samples);
    double *X = (double *)malloc(n_samples * n_features * sizeof(double));
    double *Y = (double *)malloc(n_samples * n_outputs * sizeof(double));
    nn_generate_fermentation_data(X, Y, n_samples, 0.05);
    printf("  Model: Fed-batch with Monod kinetics + Luedeking-Piret product formation\n");
    printf("  Inputs:  Temperature, pH, DO, Agitation, Feed rate\n");
    printf("  Outputs: Biomass (g/L), Product (g/L), Substrate (g/L)\n");

    nn_dataset_t *ds = nn_dataset_create(X, Y, n_samples, n_features, n_outputs, 0.7, 0.15);
    nn_dataset_normalize(ds, NN_NORM_ZSCORE, NN_NORM_ZSCORE);
    printf("  Normalized: Z-score (zero mean, unit variance)\n");

    printf("\nStep 2: Building network [5 → 16 → 8 → 3]...\n");
    int sizes[] = {5, 16, 8, 3};
    nn_architecture_t arch = {4, sizes, NN_ACTIVATION_TANH, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    printf("  Hidden activation: tanh (symmetric, handles Monod saturation)\n");
    printf("  Parameters: %d\n", nn_count_parameters(net));

    /* Configure with Adam + L2 regularization */
    nn_training_config_t config;
    nn_training_config_default(&config);
    config.optimizer = NN_OPTIMIZER_ADAM;
    config.learning_rate = 0.0005;
    config.l2_lambda = 0.001;
    config.max_epochs = 100;
    nn_network_configure_training(net, &config);

    printf("\nStep 3: Training (early stopping with patience=15)...\n");
    double best_val_rmse = 1e100;
    int best_epoch = 0;
    int patience = 15;
    int no_improve = 0;

    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        double train_loss = nn_network_train_epoch(net, ds, epoch);

        nn_regression_metrics_t vm;
        nn_network_evaluate(net, ds, ds->val_start, ds->val_end, &vm);

        if (vm.rmse < best_val_rmse) {
            best_val_rmse = vm.rmse;
            best_epoch = epoch;
            no_improve = 0;
        } else {
            no_improve++;
        }

        if (epoch % 20 == 0 || epoch == config.max_epochs - 1 || no_improve >= patience) {
            printf("  Epoch %3d | Loss: %.6f | Val RMSE: %.6f | R²: %.4f",
                   epoch + 1, train_loss, vm.rmse, vm.r_squared);
            if (no_improve >= patience) printf(" [EARLY STOP]");
            printf("\n");
        }

        if (no_improve >= patience) break;
    }
    net->is_trained = 1;

    printf("\nStep 4: Full evaluation...\n");
    nn_regression_metrics_t tm;
    nn_network_evaluate(net, ds, ds->test_start, ds->test_end, &tm);
    printf("  Test RMSE:   %.6f\n", tm.rmse);
    printf("  Test MAE:    %.6f\n", tm.mae);
    printf("  Test R²:     %.6f\n", tm.r_squared);
    printf("  Best epoch:  %d\n", best_epoch + 1);

    /* Soft sensor deployment simulation */
    soft_sensor_t *sensor = soft_sensor_create(
        "FermentationNN", "BioReactor-F101", "Biomass Concentration",
        "g/L", SENSOR_TYPE_STATIC, net);
    sensor->dataset = ds;
    sensor->val_rmse = tm.rmse;
    sensor->r_squared = tm.r_squared;

    printf("\nStep 5: Simulating online soft sensor deployment...\n");
    printf("  Monitoring performance with lab sample calibration:\n");
    printf("  %-4s %-10s %-10s %-10s %-10s\n",
           "Day", "Lab(g/L)", "NN(g/L)", "Error", "Status");
    printf("  -----------------------------------------------\n");

    for (int day = 1; day <= 14; day++) {
        /* Simulate: every day a lab sample is taken */
        int idx = ds->test_start + day;
        if (idx >= ds->test_end) break;
        double actual = ds->Y.data[idx * n_outputs];
        double pred[3];
        soft_sensor_predict(sensor, ds->X.data + idx * n_features, pred);

        double error = actual - pred[0];
        const char *status = (fabs(error) < 2.0 * tm.rmse) ? "OK" : "WARN";

        printf("  %-4d %-10.2f %-10.2f %-10.4f %-10s\n",
               day, actual, pred[0], error, status);

        /* Update monitoring */
        soft_sensor_monitor_performance(sensor, actual, pred[0]);
    }

    if (sensor->needs_retraining) {
        printf("\n  ⚠ WARNING: Soft sensor performance degraded. Retraining recommended.\n");
    } else {
        printf("\n  ✓ Soft sensor performing within acceptable limits.\n");
    }

    /* Cleanup */
    soft_sensor_free(sensor);
    nn_dataset_free(ds);
    free(X); free(Y);

    printf("\nDone.\n\n");
    return 0;
}
