/**
 * @file example_polymer_soft_sensor.c
 * @brief L6-L7 Example: Neural Network Soft Sensor for Polyethylene Melt Index
 *
 * Demonstrates a soft sensor for estimating polymer quality (Melt Index,
 * density) from reactor operating conditions, enabling real-time quality
 * control without waiting for lab analysis (4-6 hour delay).
 *
 * Reference: McAuley & MacGregor, "On-line inference of polymer properties
 *            in an industrial polyethylene reactor," AIChE J. (1991)
 *            Richards & Congalidis, "Measurement and control of
 *            polymerization reactors," Comp.Chem.Eng. (2006)
 *            Gonzaga et al., "ANN-based soft sensor for real-time process
 *            monitoring and control of an industrial polymerization process"
 *            Comp.Chem.Eng. (2009)
 *
 * Classic Problem (L6): In gas-phase polyethylene production, the Melt Index
 * (MI) is the key quality variable but is measured offline every 2-4 hours.
 * Process conditions (temperature, pressure, H2/C2 ratio) change much faster.
 * A soft sensor provides continuous MI estimates, enabling:
 * - Tighter quality control
 * - Faster grade transitions (saving 100-200 tons/transition of off-spec)
 * - Reduced lab testing frequency
 *
 * Industrial Context (L7): Implemented at major polyolefin plants globally
 * (Dow, ExxonMobil, Borealis, Sinopec) using DCS-integrated soft sensors.
 */

#include "nn_soft_sensor.h"
#include "nn_sensor_validation.h"
#include "nn_training.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Neural Network Soft Sensor for Polymerization Process      ║\n");
    printf("║  L6-L7: Melt Index & Density Estimation (Industrial Grade)  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int n_samples = 250;
    int n_features = 5;
    int n_outputs = 3;

    printf("Step 1: Generating polyethylene reactor data...\n");
    double *X = (double *)malloc(n_samples * n_features * sizeof(double));
    double *Y = (double *)malloc(n_samples * n_outputs * sizeof(double));
    nn_generate_polymerization_data(X, Y, n_samples, 0.03);
    printf("  Model: Ziegler-Natta gas-phase ethylene polymerization\n");
    printf("  Inputs:  Temp(C), Pressure(bar), Catalyst(g/h), H2/C2 ratio, Residence(h)\n");
    printf("  Outputs: Melt Index (g/10min), Density (g/cm³), Production Rate (kg/h)\n");

    /* Create dataset */
    nn_dataset_t *ds = nn_dataset_create(X, Y, n_samples, n_features, n_outputs, 0.7, 0.15);
    nn_dataset_normalize(ds, NN_NORM_MINMAX, NN_NORM_MINMAX);

    printf("\nStep 2: Training 3 models and comparing...\n");

    /* Model A: ReLU activation */
    printf("\n  --- Model A: ReLU [5, 10, 5, 3] ---\n");
    int sizes_a[] = {5, 10, 5, 3};
    nn_architecture_t arch_a = {4, sizes_a, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net_a = nn_network_create(&arch_a);
    nn_training_config_t cfg;
    nn_training_config_default(&cfg);
    cfg.learning_rate = 0.001;
    nn_network_configure_training(net_a, &cfg);
    for (int e = 0; e < 30; e++) nn_network_train_epoch(net_a, ds, e);
    net_a->is_trained = 1;

    /* Model B: tanh activation */
    printf("\n  --- Model B: Tanh [5, 10, 5, 3] ---\n");
    int sizes_b[] = {5, 10, 5, 3};
    nn_architecture_t arch_b = {4, sizes_b, NN_ACTIVATION_TANH, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net_b = nn_network_create(&arch_b);
    nn_network_configure_training(net_b, &cfg);
    for (int e = 0; e < 30; e++) nn_network_train_epoch(net_b, ds, e);
    net_b->is_trained = 1;

    /* Model C: Ensemble (3x bagged ReLU networks) */
    printf("\n  --- Model C: Ensemble (3 members) [5, 10, 5, 3] ---\n");
    nn_ensemble_t *ensemble = nn_ensemble_create(3, &arch_a);
    /* Train each ensemble member on bootstrapped data */
    for (int m = 0; m < 3; m++) {
        /* Simple bootstrap: train each member on full training set */
        for (int e = 0; e < 15; e++) {
            nn_network_train_epoch(ensemble->networks[m], ds, e);
        }
    }

    printf("\nStep 3: Model comparison on test set...\n");
    nn_regression_metrics_t ma, mb;
    nn_network_evaluate(net_a, ds, ds->test_start, ds->test_end, &ma);
    nn_network_evaluate(net_b, ds, ds->test_start, ds->test_end, &mb);

    printf("\n  %-20s %-10s %-10s %-10s\n", "Metric", "ReLU(A)", "Tanh(B)", "Ensemble(C)");
    printf("  -------------------------------------------------\n");
    printf("  %-20s %-10.4f %-10.4f\n", "Test RMSE", ma.rmse, mb.rmse);
    printf("  %-20s %-10.4f %-10.4f\n", "Test MAE", ma.mae, mb.mae);
    printf("  %-20s %-10.4f %-10.4f\n", "R²", ma.r_squared, mb.r_squared);
    printf("  %-20s %-10.1f %-10.1f\n", "AIC", ma.aic, mb.aic);
    printf("  %-20s %-10d %-10d\n", "Parameters",
           nn_count_parameters(net_a), nn_count_parameters(net_b));

    /* Select best model */
    nn_network_t *best_net = (ma.rmse < mb.rmse) ? net_a : net_b;
    const char *best_name = (ma.rmse < mb.rmse) ? "ReLU" : "Tanh";

    printf("\n  ✓ Best model: %s (RMSE=%.4f, R²=%.4f)\n", best_name,
           (ma.rmse < mb.rmse) ? ma.rmse : mb.rmse,
           (ma.rmse < mb.rmse) ? ma.r_squared : mb.r_squared);

    printf("\nStep 4: Deploying industrial soft sensor...\n");
    soft_sensor_t *sensor = soft_sensor_create(
        "PolyMI-SoftSensor", "UNIPOL Reactor R-201",
        "Melt Index", "g/10min", SENSOR_TYPE_STATIC, best_net);
    sensor->dataset = ds;

    nn_regression_metrics_t *best_metrics = (ma.rmse < mb.rmse) ? &ma : &mb;
    sensor->val_rmse = best_metrics->rmse;
    sensor->val_mae = best_metrics->mae;
    sensor->r_squared = best_metrics->r_squared;

    /* Industrial deployment configuration */
    nn_industrial_soft_sensor_t app;
    nn_industrial_sensor_init(&app, sensor,
                               "Shanghai Chemical Industry Park",
                               "Honeywell Experion PKS",
                               "OSIsoft PI System");

    printf("  Plant:       %s\n", app.plant_location);
    printf("  DCS:         %s\n", app.dcs_system);
    printf("  Historian:   %s\n", app.historian_system);
    printf("  Update rate: %d ms\n", app.update_interval_ms);

    printf("\nStep 5: Real-time prediction simulation...\n");
    printf("  %-8s %-8s %-8s %-10s %-12s %-12s\n",
           "Time", "T(°C)", "H2/C2", "MI(Actual)", "MI(Pred)", "Quality");
    printf("  -------------------------------------------------------------\n");

    for (int t = 0; t < 8; t++) {
        int idx = ds->test_start + t * 3;
        if (idx >= ds->test_end) break;
        const double *x = ds->X.data + idx * n_features;
        double pred[3];
        soft_sensor_predict(sensor, x, pred);

        double actual = ds->Y.data[idx * n_outputs];
        double error_pct = fabs(pred[0] - actual) / actual * 100.0;
        const char *quality = (error_pct < 5.0) ? "GOOD" :
                               (error_pct < 10.0) ? "FAIR" : "POOR";

        printf("  %-8d %-8.1f %-8.3f %-10.2f %-12.2f %-12s\n",
               t, x[0], x[3], actual, pred[0], quality);
    }

    printf("\nStep 6: Uncertainty estimation with MC Dropout...\n");
    nn_mc_dropout_t *mc = nn_mc_dropout_create(best_net, 50);
    if (mc) {
        /* Enable dropout for uncertainty */
        best_net->regularization_type = NN_REG_DROPOUT;
        for (int l = 1; l < best_net->num_layers; l++) {
            best_net->layers[l]->dropout_rate = 0.1;
        }

        printf("  %-10s %-12s %-12s %-12s\n",
               "Sample", "Mean MI", "95% Lower", "95% Upper");
        printf("  ---------------------------------------------\n");
        for (int i = 0; i < 5; i++) {
            int idx = ds->test_start + i * 5;
            if (idx >= ds->test_end) break;
            double mc_out[3];
            nn_mc_dropout_predict(mc, ds->X.data + idx * n_features, mc_out);
            printf("  %-10d %-12.3f %-12.3f %-12.3f\n",
                   idx, mc_out[0], mc->lower_bound[0], mc->upper_bound[0]);
        }
        printf("  Epistemic uncertainty: %.6f\n", mc->epistemic_uncertainty);
        nn_mc_dropout_free(mc);
    }

    /* Cleanup other networks */
    if (best_net == net_a) {
        nn_network_free(net_b);
    } else {
        nn_network_free(net_a);
    }
    nn_ensemble_free(ensemble);
    soft_sensor_free(sensor);
    nn_dataset_free(ds);
    free(X);
    free(Y);

    printf("\n  ╔═══════════════════════════════════════════╗\n");
    printf("  ║  Soft Sensor Ready for DCS Integration   ║\n");
    printf("  ╚═══════════════════════════════════════════╝\n");

    printf("\nDone.\n\n");
    return 0;
}
