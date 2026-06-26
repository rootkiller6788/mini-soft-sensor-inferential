/**
 * @file test_nn_soft_sensor.c
 * @brief Tests for soft sensor validation, monitoring, and industrial applications (L4-L8).
 *
 * Tests: model evaluation, k-fold CV, performance monitoring,
 * concept drift detection, industrial data generators, ensemble prediction.
 */

#include "nn_soft_sensor.h"
#include "nn_sensor_validation.h"
#include "nn_training.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)
#define CHECK_FLOAT(a, b, tol, msg) do { \
    if (fabs((a)-(b)) < (tol)) PASS(); else { printf("FAIL: %s (%.6f vs %.6f)\n", msg, a, b); tests_failed++; } \
} while(0)

/*===========================================================================
 * L4-L6: Model Evaluation
 *===========================================================================*/

static void test_model_evaluation(void) {
    TEST("model evaluation computes metrics");
    int sizes[] = {1, 2, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_LINEAR, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    /* Set all weights and biases for identity: y = x */
    /* Input layer: pass-through */
    net->layers[0]->weights.data[0] = 1.0;
    net->layers[0]->biases.data[0] = 0.0;
    /* Hidden layer: both neurons carry the signal */
    net->layers[1]->weights.data[0] = 1.0;
    net->layers[1]->weights.data[1] = 0.0;
    net->layers[1]->biases.data[0] = 0.0;
    net->layers[1]->biases.data[1] = 0.0;
    /* Output layer: sum both contributions (linear activation) */
    net->layers[2]->weights.data[0] = 1.0;
    net->layers[2]->weights.data[1] = 0.0;
    net->layers[2]->biases.data[0] = 0.0;

    double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double Y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    nn_dataset_t *ds = nn_dataset_create(X, Y, 5, 1, 1, 1.0, 0.0);

    nn_regression_metrics_t metrics;
    nn_network_evaluate(net, ds, 0, 5, &metrics);
    CHECK(metrics.rmse >= 0.0, "RMSE non-negative");
    CHECK(metrics.rmse <= 5.0, "RMSE reasonable");
    CHECK(metrics.max_error >= 0.0, "max_error non-negative");
    CHECK(metrics.r_squared <= 1.1, "R² not absurd");

    nn_dataset_free(ds);
    nn_network_free(net);
}

/*===========================================================================
 * L4-L6: K-Fold Cross Validation
 *===========================================================================*/

static void test_kfold_cv(void) {
    TEST("k-fold cross-validation runs");
    int sizes[] = {1, 2, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    double X[20], Y[20];
    for (int i = 0; i < 20; i++) {
        X[i] = (double)i / 20.0;
        Y[i] = X[i] * 2.0;
    }

    double mean_rmse, std_rmse;
    int ret = nn_kfold_cross_validate(net, X, Y, 20, 1, 1, 4, 5,
                                       &mean_rmse, &std_rmse);
    CHECK(ret == 0, "k-fold CV returned success");
    CHECK(mean_rmse >= 0.0, "mean RMSE non-negative");
    CHECK(std_rmse >= 0.0, "std RMSE non-negative");

    nn_network_free(net);
}

/*===========================================================================
 * L6: Residual Analysis
 *===========================================================================*/

static void test_residual_stats(void) {
    TEST("residual analysis with zero-mean symmetric errors");
    double y_true[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_pred[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double bias, std_dev, skew, kurt;
    nn_residual_analysis(y_true, y_pred, 5, NULL, &bias, &std_dev, &skew, &kurt);
    CHECK_FLOAT(bias, 0.0, 1e-10, "bias = 0 for perfect fit");
    CHECK_FLOAT(std_dev, 0.0, 1e-10, "std_dev = 0 for perfect fit");
}

/*===========================================================================
 * L7: Performance Monitoring
 *===========================================================================*/

static void test_performance_monitoring(void) {
    TEST("soft sensor performance tracking");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    soft_sensor_t *sensor = soft_sensor_create("Test", "Unit", "Var", "unit",
                                                SENSOR_TYPE_STATIC, net);
    sensor->val_rmse = 0.1;

    /* Check that predictions counter tracks calls */
    sensor->total_predictions = 0;
    soft_sensor_predict(sensor, (double[]){0.5, 0.3}, (double[1]){});
    CHECK(sensor->total_predictions == 1, "prediction counter incremented");

    /* Update with small errors - should not trigger retraining */
    int ret = soft_sensor_monitor_performance(sensor, 1.0, 1.05);
    CHECK(ret == 0, "small error does not trigger retraining");

    /* Test drift detection */
    sensor->drift_indicator = 0.5;
    ret = soft_sensor_detect_drift(sensor, 5.0);
    CHECK(ret == 1, "drift detected with high indicator");

    soft_sensor_free(sensor);
    nn_network_free(net);
}

/*===========================================================================
 * L7: Industrial Deployment
 *===========================================================================*/

static void test_industrial_init(void) {
    TEST("industrial sensor initialization");
    nn_industrial_soft_sensor_t app;
    nn_industrial_sensor_init(&app, NULL, "Shanghai Plant",
                               "Honeywell Experion PKS", "OSIsoft PI");
    CHECK(strcmp(app.plant_location, "Shanghai Plant") == 0, "plant stored");
    CHECK(strcmp(app.dcs_system, "Honeywell Experion PKS") == 0, "DCS stored");
    CHECK(app.update_interval_ms == 1000, "default update interval");
}

/*===========================================================================
 * L6-L7: Process Data Generators
 *===========================================================================*/

static void test_distillation_data(void) {
    TEST("distillation data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_distillation_data(X, Y, n, 0.01);

    /* Check ranges */
    int valid_x = 1, valid_y = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(X[i * 5]) || X[i * 5] < 0.0) valid_x = 0;
        if (!isfinite(Y[i * 3]) || Y[i * 3] < 0.0 || Y[i * 3] > 1.0) valid_y = 0;
    }
    CHECK(valid_x, "distillation X values valid");
    CHECK(valid_y, "distillation Y values in [0,1]");

    free(X); free(Y);
}

static void test_fermentation_data(void) {
    TEST("fermentation data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_fermentation_data(X, Y, n, 0.01);

    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(X[i * 5]) || X[i * 5] < 20.0 || X[i * 5] > 45.0) valid = 0;
        if (!isfinite(Y[i * 3]) || Y[i * 3] < 0.0) valid = 0;
    }
    CHECK(valid, "fermentation data valid");

    free(X); free(Y);
}

static void test_polymerization_data(void) {
    TEST("polymerization data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_polymerization_data(X, Y, n, 0.01);

    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(Y[i * 3 + 1]) || Y[i * 3 + 1] < 0.9 || Y[i * 3 + 1] > 1.0) valid = 0;
    }
    CHECK(valid, "polymerization data valid");

    free(X); free(Y);
}

static void test_cement_kiln_data(void) {
    TEST("cement kiln data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_cement_kiln_data(X, Y, n, 0.01);

    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(X[i * 5]) || X[i * 5] < 1300.0 || X[i * 5] > 1600.0) valid = 0;
    }
    CHECK(valid, "cement kiln temperature in range");

    free(X); free(Y);
}

static void test_fcc_data(void) {
    TEST("FCC unit data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_fcc_data(X, Y, n, 0.01);

    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(Y[i * 3 + 2]) || Y[i * 3 + 2] < 50.0 || Y[i * 3 + 2] > 95.0) valid = 0;
    }
    CHECK(valid, "FCC conversion in realistic range");

    free(X); free(Y);
}

static void test_blast_furnace_data(void) {
    TEST("blast furnace data generator");
    int n = 30;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_blast_furnace_data(X, Y, n, 0.01);

    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(Y[i * 3]) || Y[i * 3] < 1400.0 || Y[i * 3] > 1600.0) valid = 0;
    }
    CHECK(valid, "blast furnace hot metal temp in range");

    free(X); free(Y);
}

/*===========================================================================
 * L8: Ensemble Prediction
 *===========================================================================*/

static void test_ensemble(void) {
    TEST("ensemble creation and prediction");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_ensemble_t *ens = nn_ensemble_create(3, &arch);
    CHECK(ens != NULL, "ensemble created");
    CHECK(ens->num_members == 3, "3 members");

    double input[] = {0.5, 0.3};
    double output[1] = {0.0};
    nn_ensemble_predict(ens, input, output);
    CHECK(isfinite(output[0]), "ensemble prediction finite");

    nn_ensemble_free(ens);
}

/*===========================================================================
 * L8: MC Dropout (Bayesian)
 *===========================================================================*/

static void test_mc_dropout(void) {
    TEST("MC dropout uncertainty estimation");
    int sizes[] = {2, 5, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.2};
    nn_network_t *net = nn_network_create(&arch);

    nn_mc_dropout_t *mc = nn_mc_dropout_create(net, 10);
    CHECK(mc != NULL, "MC dropout created");

    double input[] = {0.5, 0.3};
    double output[1];
    nn_mc_dropout_predict(mc, input, output);
    CHECK(isfinite(output[0]), "MC prediction finite");
    CHECK(mc->variance_predictions[0] >= 0.0, "variance non-negative");
    CHECK(mc->lower_bound[0] <= mc->mean_predictions[0], "lower bound valid");
    CHECK(mc->upper_bound[0] >= mc->mean_predictions[0], "upper bound valid");
    CHECK(mc->epistemic_uncertainty >= 0.0, "epistemic uncertainty non-negative");

    nn_mc_dropout_free(mc);
    nn_network_free(net);
}

/*===========================================================================
 * L6: End-to-End Soft Sensor Training with Generated Data
 *===========================================================================*/

static void test_end_to_end_distillation(void) {
    TEST("end-to-end distillation soft sensor");
    int n = 100;
    double *X = (double *)malloc(n * 5 * sizeof(double));
    double *Y = (double *)malloc(n * 3 * sizeof(double));
    nn_generate_distillation_data(X, Y, n, 0.02);

    nn_dataset_t *ds = nn_dataset_create(X, Y, n, 5, 3, 0.7, 0.15);
    nn_dataset_normalize(ds, NN_NORM_MINMAX, NN_NORM_MINMAX);

    int sizes[] = {5, 10, 5, 3};
    nn_architecture_t arch = {4, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    /* Train briefly */
    for (int e = 0; e < 5; e++) {
        nn_network_train_epoch(net, ds, e);
    }
    net->is_trained = 1;

    /* Evaluate */
    nn_regression_metrics_t metrics;
    nn_network_evaluate(net, ds, ds->test_start, ds->test_end, &metrics);
    CHECK(metrics.rmse >= 0.0, "test RMSE non-negative");
    CHECK(metrics.mae >= 0.0, "test MAE non-negative");

    /* Predict on single sample */
    double test_in[] = {1.5, 100.0, 0.5, 900.0, 100.0};
    double test_out[3];
    nn_network_forward(net, test_in, test_out);
    for (int i = 0; i < 3; i++) CHECK(isfinite(test_out[i]), "prediction finite");

    nn_dataset_free(ds);
    nn_network_free(net);
    free(X); free(Y);
}

/*===========================================================================
 * L7: Model Comparison
 *===========================================================================*/

static void test_compare_models(void) {
    TEST("model comparison via paired test");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net_a = nn_network_create(&arch);
    nn_network_t *net_b = nn_network_create(&arch);

    double X_test[] = {0.5, 1.0, 0.8, 2.0, 0.3, 3.0, 1.0, 4.0};
    double Y_test[] = {1.0, 1.6, 0.9, 2.0};

    soft_sensor_t *sa = soft_sensor_create("A", "", "", "", SENSOR_TYPE_STATIC, net_a);
    soft_sensor_t *sb = soft_sensor_create("B", "", "", "", SENSOR_TYPE_STATIC, net_b);

    double p_value;
    soft_sensor_compare_models(sa, sb, X_test, Y_test, 4, &p_value);
    CHECK(p_value >= 0.0 && p_value <= 1.0, "p-value in [0,1]");

    soft_sensor_free(sa);
    soft_sensor_free(sb);
    nn_network_free(net_a);
    nn_network_free(net_b);
}

/*===========================================================================
 * Test runner
 *===========================================================================*/

int main(void) {
    printf("\n=== Neural Network Soft Sensor: Validation & Applications ===\n\n");

    printf("--- L4-L6: Model Evaluation ---\n");
    test_model_evaluation();
    test_kfold_cv();
    test_residual_stats();

    printf("\n--- L7: Performance Monitoring ---\n");
    test_performance_monitoring();
    test_industrial_init();

    printf("\n--- L6-L7: Industrial Data Generators ---\n");
    test_distillation_data();
    test_fermentation_data();
    test_polymerization_data();
    test_cement_kiln_data();
    test_fcc_data();
    test_blast_furnace_data();

    printf("\n--- L8: Advanced Methods ---\n");
    test_ensemble();
    test_mc_dropout();
    test_end_to_end_distillation();
    test_compare_models();

    printf("\n========================================\n");
    printf("Results: %d PASS, %d FAIL\n", tests_passed, tests_failed);
    printf("========================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
