/**
 * @file test_nn_core.c
 * @brief Tests for neural network core operations (L1-L3).
 *
 * Tests: network creation, forward pass, activation functions,
 * soft sensor lifecycle, dataset operations, normalization.
 */

#include "nn_soft_sensor.h"
#include "nn_sensor_validation.h"
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
 * L1 Tests: Network Creation
 *===========================================================================*/

static void test_network_create_valid(void) {
    TEST("create valid network");
    int sizes[] = {5, 10, 8, 1};
    nn_architecture_t arch = {4, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    CHECK(net != NULL, "network should be created");
    CHECK(net->num_layers == 4, "num_layers should be 4");
    CHECK(net->layers[0]->output_size == 5, "input layer size");
    CHECK(net->layers[1]->output_size == 10, "hidden layer 1 size");
    CHECK(net->layers[3]->output_size == 1, "output layer size");
    nn_network_free(net);
}

static void test_network_create_invalid(void) {
    TEST("create invalid network (NULL arch)");
    nn_network_t *net = nn_network_create(NULL);
    CHECK(net == NULL, "NULL arch should return NULL");

    TEST("create invalid network (too few layers)");
    int sizes[] = {5};
    nn_architecture_t arch = {1, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    net = nn_network_create(&arch);
    CHECK(net == NULL, "1-layer network should return NULL");
}

static void test_weight_initialization(void) {
    TEST("He initialization produces non-zero weights");
    double data[100] = {0};
    nn_he_initialize(data, 10, 10);
    double sum = 0.0;
    for (int i = 0; i < 100; i++) sum += fabs(data[i]);
    CHECK(sum > 0.01, "weights should be non-zero after He init");

    TEST("Xavier initialization produces symmetric range weights");
    double xdata[100] = {0};
    nn_xavier_initialize(xdata, 10, 10);
    double min_val = xdata[0], max_val = xdata[0];
    for (int i = 0; i < 100; i++) {
        if (xdata[i] < min_val) min_val = xdata[i];
        if (xdata[i] > max_val) max_val = xdata[i];
    }
    CHECK(fabs(min_val + max_val) < 0.5, "Xavier weights should be roughly symmetric");
}

/*===========================================================================
 * L2 Tests: Forward Pass and Activations
 *===========================================================================*/

static void test_forward_pass_small(void) {
    TEST("forward pass on 2-layer network");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    CHECK(net != NULL, "network created");

    double input[] = {1.0, 2.0};
    double output[1];
    nn_network_forward(net, input, output);
    /* Output should be some finite value */
    CHECK(isfinite(output[0]), "output should be finite");
    nn_network_free(net);
}

static void test_all_activations(void) {
    TEST("all activation functions produce valid outputs");
    double z[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    double a[5];
    double *da = (double *)malloc(5 * sizeof(double));

    /* Just verify they don't crash and produce finite values */
    nn_activation_type_t acts[] = {
        NN_ACTIVATION_SIGMOID, NN_ACTIVATION_TANH, NN_ACTIVATION_RELU,
        NN_ACTIVATION_LEAKY_RELU, NN_ACTIVATION_ELU, NN_ACTIVATION_SWISH,
        NN_ACTIVATION_LINEAR, NN_ACTIVATION_SOFTMAX, NN_ACTIVATION_GELU
    };
    int all_ok = 1;
    for (int t = 0; t < 9; t++) {
        /* Forward pass is done via nn_layer_forward which uses internal activation */
        int sizes[] = {5, 5};
        nn_architecture_t arch = {2, sizes, acts[t], acts[t], 0.0};
        nn_network_t *net = nn_network_create(&arch);
        if (!net) { all_ok = 0; break; }
        nn_network_forward(net, z, a);
        for (int i = 0; i < 5; i++) {
            if (!isfinite(a[i])) { all_ok = 0; break; }
        }
        nn_network_free(net);
        if (!all_ok) break;
    }
    CHECK(all_ok, "all activations produce finite outputs");
    free(da);
}

static void test_sigmoid_range(void) {
    TEST("sigmoid output in (0,1)");
    int sizes[] = {1, 1};
    nn_architecture_t arch = {2, sizes, NN_ACTIVATION_SIGMOID, NN_ACTIVATION_SIGMOID, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    double input[] = {100.0};  /* Very large positive */
    double output[1];
    nn_network_forward(net, input, output);
    CHECK(output[0] > 0.99 && output[0] <= 1.001, "sigmoid of large positive ~= 1");

    /* Set weight to produce very negative input */
    net->layers[1]->weights.data[0] = -1.0;
    net->layers[1]->biases.data[0] = -100.0;
    nn_network_forward(net, input, output);
    CHECK(output[0] < 0.01 && output[0] >= -0.001, "sigmoid of large negative ~= 0");
    nn_network_free(net);
}

static void test_relu_properties(void) {
    TEST("ReLU: f(0)=0, f(-1)=0, f(1)=1");
    /* Need 3 layers so middle layer uses hidden_activation (ReLU) */
    int sizes[] = {1, 1, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    /* Set all layers to identity */
    for (int l = 0; l < 3; l++) {
        net->layers[l]->weights.data[0] = 1.0;
        net->layers[l]->biases.data[0] = 0.0;
    }

    double input[1], output[1];

    input[0] = -1.0;
    nn_network_forward(net, input, output);
    CHECK_FLOAT(output[0], 0.0, 1e-10, "ReLU(-1)=0");

    input[0] = 0.0;
    nn_network_forward(net, input, output);
    CHECK_FLOAT(output[0], 0.0, 1e-10, "ReLU(0)=0");

    input[0] = 1.0;
    nn_network_forward(net, input, output);
    CHECK_FLOAT(output[0], 1.0, 1e-10, "ReLU(1)=1");

    nn_network_free(net);
}

/*===========================================================================
 * L3 Tests: Dataset Operations
 *===========================================================================*/

static void test_dataset_create(void) {
    TEST("dataset creation with splits");
    double X[100], Y[50];
    for (int i = 0; i < 100; i++) X[i] = (double)i;
    for (int i = 0; i < 50; i++) Y[i] = (double)(i * 2);

    nn_dataset_t *ds = nn_dataset_create(X, Y, 10, 3, 1, 0.6, 0.2);
    CHECK(ds != NULL, "dataset created");
    CHECK(ds->num_samples == 10, "num_samples");
    CHECK(ds->num_features == 3, "num_features");
    CHECK(ds->num_outputs == 1, "num_outputs");
    CHECK(ds->train_end == 6, "train_end (60% of 10 = 6)");
    CHECK(ds->val_end == 8, "val_end (60%+20% = 80% of 10 = 8)");
    CHECK(ds->test_end == 10, "test_end = 10");

    /* Check data copy */
    CHECK_FLOAT(ds->X.data[0], 0.0, 1e-10, "first X value");
    CHECK_FLOAT(ds->Y.data[0], 0.0, 1e-10, "first Y value");

    nn_dataset_free(ds);
}

static void test_dataset_normalize(void) {
    TEST("Z-score normalization");
    double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double Y[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    nn_dataset_t *ds = nn_dataset_create(X, Y, 5, 1, 1, 0.6, 0.2);
    nn_dataset_normalize(ds, NN_NORM_ZSCORE, NN_NORM_ZSCORE);

    CHECK(ds->is_normalized == 1, "should be normalized");
    CHECK(ds->X_norm != NULL, "X_norm params created");
    CHECK(ds->Y_norm != NULL, "Y_norm params created");
    CHECK_FLOAT(ds->X_norm->mean[0], 3.0, 0.01, "X mean = 3");

    nn_dataset_free(ds);
}

/*===========================================================================
 * L2-L3 Tests: Soft Sensor
 *===========================================================================*/

static void test_soft_sensor_create(void) {
    TEST("soft sensor creation");
    int sizes[] = {3, 5, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    soft_sensor_t *sensor = soft_sensor_create("TestSensor", "Reactor1",
                                                "Conversion", "%",
                                                SENSOR_TYPE_STATIC, net);
    CHECK(sensor != NULL, "sensor created");
    CHECK(sensor->network == net, "network attached");
    CHECK(strcmp(sensor->name, "TestSensor") == 0, "name stored");
    CHECK(strcmp(sensor->quality_variable, "Conversion") == 0, "quality var stored");

    soft_sensor_predict(sensor, (double[]){0.5, 0.3, 0.8}, (double[1]){});
    CHECK(sensor->total_predictions == 1, "prediction counter incremented");

    soft_sensor_free(sensor);
    nn_network_free(net);
}

/*===========================================================================
 * L4 Tests: R-squared Computation
 *===========================================================================*/

static void test_r_squared(void) {
    TEST("R² perfect prediction = 1.0");
    double y_true[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_pred[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double r2 = nn_compute_r_squared(y_true, y_pred, 5);
    CHECK_FLOAT(r2, 1.0, 1e-10, "R² = 1.0 for perfect prediction");

    TEST("R² for mean prediction = 0.0");
    double y_pred_mean[] = {3.0, 3.0, 3.0, 3.0, 3.0};
    r2 = nn_compute_r_squared(y_true, y_pred_mean, 5);
    CHECK_FLOAT(r2, 0.0, 1e-10, "R² ~= 0 for mean prediction");
}

/*===========================================================================
 * L4 Tests: Parameter Counting
 *===========================================================================*/

static void test_count_parameters(void) {
    TEST("parameter count 2-3-1 network");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    /* Layer 0: 2x? (input layer has weights=2x2=4 + bias=2, but it's the
     * input layer treated as special). Actually our implementation:
     * Layer 0 (input): 0x2=0 weights, 0 biases (input layer has identity)
     * Wait, in our implementation all layers have weights including input.
     * Let's check the count function directly. */
    int n = nn_count_parameters(net);
    /* All layers: (2*2+2) + (3*2+3) + (1*3+1) = 6 + 9 + 4 = 19
     * Actually input layer has input_size=2, output_size=2, so 2*2+2=6
     * hidden: 2*3+3=9, output: 3*1+1=4, total=19 */
    CHECK(n > 0, "parameter count positive");
    CHECK(n == 19, "2-3-1 network has 19 parameters");
    nn_network_free(net);
}

/*===========================================================================
 * L4 Tests: AIC/BIC
 *===========================================================================*/

static void test_aic_bic(void) {
    TEST("AIC and BIC computation");
    double aic = nn_aic_criterion(100, 1.0, 10);
    double bic = nn_bic_criterion(100, 1.0, 10);
    CHECK(isfinite(aic), "AIC finite");
    CHECK(isfinite(bic), "BIC finite");
    CHECK(bic > aic, "BIC > AIC (stricter penalty)");
}

/*===========================================================================
 * L4 Tests: Residual Analysis
 *===========================================================================*/

static void test_residual_analysis(void) {
    TEST("residual analysis for symmetric errors");
    double y_true[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_pred[] = {1.1, 1.9, 3.1, 3.9, 5.1};
    double residuals[5], bias, std_dev, skewness, kurtosis;
    nn_residual_analysis(y_true, y_pred, 5, residuals, &bias, &std_dev, &skewness, &kurtosis);
    CHECK_FLOAT(bias, -0.02, 0.1, "bias near zero");
    CHECK(std_dev > 0.0, "std_dev positive");
}

static void test_durbin_watson(void) {
    TEST("Durbin-Watson computation returns valid value");
    double residuals[] = {0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.05};
    double dw = nn_durbin_watson(residuals, 8);
    CHECK(dw > 0.0 && dw < 4.0, "DW statistic in valid range [0,4]");
}

/*===========================================================================
 * L4 Tests: Overfitting Detection
 *===========================================================================*/

static void test_overfitting_detection(void) {
    TEST("overfitting detection with diverging curves");
    double train_loss[] = {0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08};
    double val_loss[]   = {0.95, 0.8, 0.6, 0.45, 0.5, 0.55, 0.62, 0.7};
    int overfit = nn_detect_overfitting(train_loss, val_loss, 8, 3);
    CHECK(overfit == 1, "should detect overfitting when val increases, train decreases");
}

/*===========================================================================
 * L4 Tests: Early Stopping
 *===========================================================================*/

static void test_early_stopping(void) {
    TEST("early stopping epoch selection");
    double val_loss[] = {0.8, 0.6, 0.5, 0.45, 0.48, 0.52, 0.55, 0.6};
    int best = nn_early_stopping_epoch(val_loss, 8);
    CHECK(best == 3, "best epoch is 3 (minimum at index 3)");
}

/*===========================================================================
 * L5 Tests: Model Persistence
 *===========================================================================*/

static void test_save_load(void) {
    TEST("network save and load roundtrip");
    int sizes[] = {2, 3, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);
    net->is_trained = 1;

    /* Set known weight */
    net->layers[1]->weights.data[0] = 0.123456;

    int ret = nn_network_save(net, "build/test_model.nn");
    CHECK(ret == 0, "save succeeded");

    nn_network_t *loaded = nn_network_load("build/test_model.nn");
    CHECK(loaded != NULL, "load succeeded");
    if (loaded) {
        CHECK_FLOAT(loaded->layers[1]->weights.data[0], 0.123456, 1e-10,
                     "loaded weight matches");
        CHECK(loaded->is_trained == 1, "trained flag preserved");
        nn_network_free(loaded);
    }

    remove("build/test_model.nn");
    nn_network_free(net);
}

/*===========================================================================
 * Test runner
 *===========================================================================*/

int main(void) {
    printf("\n=== Neural Network Soft Sensor: Core Tests ===\n\n");

    printf("--- L1: Network Creation ---\n");
    test_network_create_valid();
    test_network_create_invalid();
    test_weight_initialization();

    printf("\n--- L2: Forward Pass & Activations ---\n");
    test_forward_pass_small();
    test_all_activations();
    test_sigmoid_range();
    test_relu_properties();

    printf("\n--- L3: Dataset Operations ---\n");
    test_dataset_create();
    test_dataset_normalize();

    printf("\n--- L2-L3: Soft Sensor Lifecycle ---\n");
    test_soft_sensor_create();

    printf("\n--- L4: Validation Metrics ---\n");
    test_r_squared();
    test_count_parameters();
    test_aic_bic();
    test_residual_analysis();
    test_durbin_watson();
    test_overfitting_detection();
    test_early_stopping();

    printf("\n--- L5: Persistence ---\n");
    test_save_load();

    printf("\n========================================\n");
    printf("Results: %d PASS, %d FAIL\n", tests_passed, tests_failed);
    printf("========================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
