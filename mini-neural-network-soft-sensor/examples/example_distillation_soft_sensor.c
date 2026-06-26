/**
 * @file example_distillation_soft_sensor.c
 * @brief L6 Example: Neural Network Soft Sensor for Distillation Column Composition
 *
 * This example demonstrates:
 * 1. Generating synthetic distillation column data
 * 2. Training a neural network soft sensor
 * 3. Evaluating model performance
 * 4. Predicting top and bottom product compositions from measured variables
 *
 * Reference: Skogestad, "Dynamics and control of distillation columns" (1997)
 *            Qin, "Neural networks for intelligent sensors..." CEP (1997)
 *
 * Classic Problem: In a binary distillation column, product compositions
 * (top and bottom) are measured infrequently by laboratory analyzers
 * (every 2-4 hours), but tray temperatures, pressures, and flow rates
 * are measured continuously (every second). A soft sensor uses the
 * continuous measurements to estimate compositions in real-time.
 */

#include "nn_soft_sensor.h"
#include "nn_sensor_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Neural Network Soft Sensor for Distillation Column         ║\n");
    printf("║  L6: Composition Estimation from Process Measurements      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Parameters */
    int n_samples = 200;
    int n_features = 5;   /* RR, Feed rate, Feed comp, Reboiler duty, Feed temp */
    int n_outputs = 3;    /* Top composition, Bottom composition, Pressure drop */

    printf("Step 1: Generating synthetic distillation column data (%d samples)...\n", n_samples);
    double *X = (double *)malloc(n_samples * n_features * sizeof(double));
    double *Y = (double *)malloc(n_samples * n_outputs * sizeof(double));
    if (!X || !Y) {
        printf("ERROR: Memory allocation failed\n");
        free(X); free(Y);
        return 1;
    }
    nn_generate_distillation_data(X, Y, n_samples, 0.01);
    printf("  Done. Data ranges:\n");
    printf("  X[0] (Reflux ratio): %6.2f - %6.2f\n", X[0], X[n_features]);
    printf("  Y[0] (Top comp):     %6.4f - %6.4f\n", Y[0], Y[n_outputs]);

    printf("\nStep 2: Creating dataset with train/val/test split (70/15/15)...\n");
    nn_dataset_t *ds = nn_dataset_create(X, Y, n_samples, n_features, n_outputs,
                                          0.70, 0.15);
    if (!ds) {
        printf("ERROR: Dataset creation failed\n");
        free(X); free(Y);
        return 1;
    }
    printf("  Training samples:   %d\n", ds->train_end - ds->train_start);
    printf("  Validation samples: %d\n", ds->val_end - ds->val_start);
    printf("  Test samples:       %d\n", ds->test_end - ds->test_start);

    printf("\nStep 3: Normalizing data (MinMax to [0,1])...\n");
    nn_dataset_normalize(ds, NN_NORM_MINMAX, NN_NORM_MINMAX);
    printf("  Done.\n");

    printf("\nStep 4: Building neural network [5 → 12 → 8 → 3]...\n");
    int layer_sizes[] = {5, 12, 8, 3};
    nn_architecture_t arch = {4, layer_sizes, NN_ACTIVATION_RELU,
                              NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    if (!net) {
        printf("ERROR: Network creation failed\n");
        nn_dataset_free(ds);
        free(X); free(Y);
        return 1;
    }

    /* Configure training */
    nn_training_config_t config;
    nn_training_config_default(&config);
    config.optimizer = NN_OPTIMIZER_ADAM;
    config.learning_rate = 0.001;
    config.l2_lambda = 0.0001;     /* Light L2 regularization */
    nn_network_configure_training(net, &config);

    int n_params = nn_count_parameters(net);
    printf("  Trainable parameters: %d\n", n_params);

    printf("\nStep 5: Training the soft sensor...\n");
    int max_epochs = 50;
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        double train_loss = nn_network_train_epoch(net, ds, epoch);

        if (epoch % 10 == 0 || epoch == max_epochs - 1) {
            nn_regression_metrics_t val_metrics;
            nn_network_evaluate(net, ds, ds->val_start, ds->val_end, &val_metrics);
            printf("  Epoch %3d/%d | Train Loss: %8.6f | Val RMSE: %8.6f | Val R²: %6.4f\n",
                   epoch + 1, max_epochs, train_loss, val_metrics.rmse, val_metrics.r_squared);
        }
    }
    net->is_trained = 1;

    printf("\nStep 6: Evaluating on test set...\n");
    nn_regression_metrics_t test_metrics;
    nn_network_evaluate(net, ds, ds->test_start, ds->test_end, &test_metrics);
    printf("  Test RMSE:      %8.6f\n", test_metrics.rmse);
    printf("  Test MAE:       %8.6f\n", test_metrics.mae);
    printf("  Test R²:        %8.6f\n", test_metrics.r_squared);
    printf("  Adj R²:         %8.6f\n", test_metrics.adj_r_squared);
    printf("  Max Error:      %8.6f\n", test_metrics.max_error);
    printf("  AIC:            %10.2f\n", test_metrics.aic);
    printf("  BIC:            %10.2f\n", test_metrics.bic);

    printf("\nStep 7: Creating soft sensor and making predictions...\n");
    soft_sensor_t *sensor = soft_sensor_create(
        "DistillationNN", "C2 Splitter", "Top Composition",
        "mole fraction", SENSOR_TYPE_STATIC, net);
    sensor->dataset = ds;
    sensor->val_rmse = test_metrics.rmse;
    sensor->val_mae = test_metrics.mae;
    sensor->r_squared = test_metrics.r_squared;

    /* Sample predictions */
    printf("\n  Sample Predictions (Test Set):\n");
    printf("  %-10s %-12s %-10s %-12s %-12s %-12s\n",
           "Sample", "Top True", "Top Pred", "Bot True", "Bot Pred", "dP Pred");
    printf("  ---------------------------------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        int idx = ds->test_start + i * 5;
        if (idx >= ds->test_end) break;
        const double *x_row = ds->X.data + idx * n_features;
        const double *y_row = ds->Y.data + idx * n_outputs;
        double pred[3];
        soft_sensor_predict(sensor, x_row, pred);
        printf("  %-10d %-12.4f %-10.4f %-12.4f %-12.4f %-12.4f\n",
               idx, y_row[0], pred[0], y_row[1], pred[1], pred[2]);
    }

    printf("\nStep 8: Residual analysis...\n");
    int n_test = ds->test_end - ds->test_start;
    double *residuals = (double *)malloc(n_test * sizeof(double));
    double bias, std_dev, skewness, kurtosis;
    double *y_pred_arr = (double *)malloc(n_test * sizeof(double));

    for (int i = 0; i < n_test; i++) {
        int idx = ds->test_start + i;
        double pred[3];
        nn_network_forward(net, ds->X.data + idx * n_features, pred);
        y_pred_arr[i] = pred[0];
    }

    nn_residual_analysis(ds->Y.data + ds->test_start * n_outputs,
                          y_pred_arr, n_test, residuals,
                          &bias, &std_dev, &skewness, &kurtosis);

    printf("  Residual bias:     %+.6f (zero = unbiased)\n", bias);
    printf("  Residual std:      %.6f\n", std_dev);
    printf("  Residual skewness: %+.4f (0 = symmetric)\n", skewness);
    printf("  Residual kurtosis: %+.4f (0 = normal tail)\n", kurtosis);
    printf("  Durbin-Watson:     %.4f (2 = no autocorrelation)\n",
           nn_durbin_watson(residuals, n_test));

    printf("\n  ╔════════════════════════════════╗\n");
    printf("  ║  Soft Sensor Status: TRAINED  ║\n");
    printf("  ╚════════════════════════════════╝\n");

    /* Cleanup */
    free(residuals);
    free(y_pred_arr);
    soft_sensor_free(sensor);
    nn_dataset_free(ds);
    free(X);
    free(Y);

    printf("\nDone.\n\n");
    return 0;
}
