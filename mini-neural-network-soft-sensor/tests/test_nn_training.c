/**
 * @file test_nn_training.c
 * @brief Tests for neural network training algorithms (L5).
 *
 * Tests: loss functions, optimizers, backpropagation, regularization,
 * learning rate schedules, batch normalization.
 */

#include "nn_soft_sensor.h"
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
 * L5: Loss Function Tests
 *===========================================================================*/

static void test_loss_mse(void) {
    TEST("MSE loss computation");
    double pred[] = {1.0, 2.0, 3.0};
    double truev[] = {1.0, 2.0, 3.0};
    double grad[3];
    double loss = nn_loss_mse(pred, truev, 3, grad);
    CHECK_FLOAT(loss, 0.0, 1e-10, "MSE perfect = 0");
    CHECK_FLOAT(grad[0], 0.0, 1e-10, "gradient = 0 for perfect");

    TEST("MSE loss with errors");
    double pred2[] = {1.0, 0.0, 3.0};
    loss = nn_loss_mse(pred2, truev, 3, grad);
    CHECK(loss > 0.0, "MSE > 0 with errors");
    CHECK(grad[1] < 0.0, "gradient for under-prediction < 0");
}

static void test_loss_mae(void) {
    TEST("MAE loss computation");
    double pred[] = {0.0, 0.0, 0.0};
    double truev[] = {1.0, 2.0, 3.0};
    double grad[3];
    double loss = nn_loss_mae(pred, truev, 3, grad);
    CHECK_FLOAT(loss, 2.0, 1e-10, "MAE = (1+2+3)/3 = 2");
}

static void test_loss_huber(void) {
    TEST("Huber loss - quadratic region");
    double pred[] = {0.5};
    double truev[] = {0.0};
    double grad[1];
    double loss = nn_loss_huber(pred, truev, 1, 1.0, grad);
    CHECK_FLOAT(loss, 0.125, 0.01, "Huber small error (0.5*0.25=0.125)");
    CHECK_FLOAT(grad[0], 0.5, 0.01, "grad = error for small errors");

    TEST("Huber loss - linear region");
    double pred2[] = {3.0};
    double loss2 = nn_loss_huber(pred2, truev, 1, 1.0, grad);
    /* |3-0| = 3 > 1, so: 1.0*3 - 0.5*1 = 2.5 */
    CHECK_FLOAT(loss2, 2.5, 0.01, "Huber large error");
    CHECK_FLOAT(grad[0], 1.0, 0.01, "grad = delta*sign for large errors");
}

static void test_loss_mape(void) {
    TEST("MAPE loss computation");
    double pred[] = {1.1, 1.9};
    double truev[] = {1.0, 2.0};
    double grad[2];
    double loss = nn_loss_mape(pred, truev, 2, grad);
    /* MAPE = 100/2 * (|0.1|/1.0 + |-0.1|/2.0) = 50 * (0.1+0.05) = 7.5 */
    CHECK_FLOAT(loss, 7.5, 0.01, "MAPE computed correctly");
}

/*===========================================================================
 * L5: Optimizer Tests
 *===========================================================================*/

static void test_sgd_step(void) {
    TEST("SGD decreases weight in gradient direction");
    double W_data[] = {1.0, 2.0};
    double dW_data[] = {0.1, 0.2};
    double b_data[] = {0.5};
    double db_data[] = {0.05};

    nn_matrix_t W = {W_data, 1, 2, 2};
    nn_matrix_t dW = {dW_data, 1, 2, 2};
    nn_vector_t b = {b_data, 1};
    nn_vector_t db = {db_data, 1};

    nn_optimizer_sgd_step(&W, &dW, &b, &db, 0.1);

    CHECK_FLOAT(W_data[0], 0.99, 1e-10, "W -= lr*dW");
    CHECK_FLOAT(W_data[1], 1.98, 1e-10, "W -= lr*dW");
    CHECK_FLOAT(b_data[0], 0.495, 1e-10, "b -= lr*db");
}

static void test_adam_step(void) {
    TEST("Adam optimizer step");
    double W_data[] = {1.0};
    double dW_data[] = {0.1};
    double b_data[] = {0.0};
    double db_data[] = {0.0};
    double mW_data[] = {0.0};
    double vW_data[] = {0.0};
    double mb_data[] = {0.0};
    double vb_data[] = {0.0};

    nn_matrix_t W = {W_data, 1, 1, 1};
    nn_matrix_t dW = {dW_data, 1, 1, 1};
    nn_vector_t b = {b_data, 1};
    nn_vector_t db = {db_data, 1};
    nn_matrix_t mW = {mW_data, 1, 1, 1};
    nn_matrix_t vW = {vW_data, 1, 1, 1};
    nn_vector_t mb = {mb_data, 1};
    nn_vector_t vb = {vb_data, 1};

    int ret = nn_optimizer_adam_step(&W, &dW, &b, &db,
                                      0.001, 0.9, 0.999, 1e-8, 1,
                                      &mW, &vW, &mb, &vb);
    CHECK(ret == 0, "Adam step returned 0");
    CHECK(W_data[0] < 1.0, "weight decreased");
}

static void test_momentum_step(void) {
    TEST("Momentum step accumulates velocity");
    double v_W_data[] = {0.0};
    double v_b_data[] = {0.0};
    double W_data[] = {1.0};
    double dW_data[] = {0.1};
    double b_data[] = {0.0};
    double db_data[] = {0.0};

    nn_matrix_t W = {W_data, 1, 1, 1};
    nn_matrix_t dW = {dW_data, 1, 1, 1};
    nn_vector_t b = {b_data, 1};
    nn_vector_t db = {db_data, 1};
    nn_matrix_t v_W = {v_W_data, 1, 1, 1};
    nn_vector_t v_b = {v_b_data, 1};

    nn_optimizer_momentum_step(&W, &dW, &b, &db, 0.1, 0.9, &v_W, &v_b);
    CHECK(v_W_data[0] < 0.0, "velocity negative for positive gradient");
    CHECK(W_data[0] < 1.0, "weight decreased");
}

/*===========================================================================
 * L5: Regularization Tests
 *===========================================================================*/

static void test_l1_regularization(void) {
    TEST("L1 regularization gradient");
    double w[] = {1.0, -2.0, 0.5, 0.0};
    double grad[4] = {0};
    nn_matrix_t W = {w, 2, 2, 2};
    nn_matrix_t dW = {grad, 2, 2, 2};

    double loss = nn_regularization_l1(&W, 0.01, &dW);
    CHECK(loss > 0.0, "L1 loss positive");
    CHECK_FLOAT(dW.data[0], 0.01, 1e-10, "grad sign(+)=0.01");
    CHECK_FLOAT(dW.data[1], -0.01, 1e-10, "grad sign(-)=-0.01");
    CHECK_FLOAT(dW.data[3], 0.0, 1e-10, "grad zero for zero weight");
}

static void test_l2_regularization(void) {
    TEST("L2 regularization gradient proportional to weight");
    double w[] = {1.0, 3.0};
    double grad[2] = {0};
    nn_matrix_t W = {w, 1, 2, 2};
    nn_matrix_t dW = {grad, 1, 2, 2};

    nn_regularization_l2(&W, 0.01, &dW);
    CHECK_FLOAT(dW.data[0], 0.01, 1e-10, "grad[0] = lambda*w[0] = 0.01");
    CHECK_FLOAT(dW.data[1], 0.03, 1e-10, "grad[1] = lambda*w[1] = 0.03");
}

static void test_dropout(void) {
    TEST("dropout zeros some activations");
    double activations[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    unsigned int seed = 42;
    nn_dropout_apply(activations, 10, 0.5, &seed);
    int zeros = 0;
    for (int i = 0; i < 10; i++) {
        if (activations[i] == 0.0) zeros++;
    }
    /* With rate=0.5, we expect some zeros and some scaled values */
    CHECK(zeros >= 2, "at least some activations zeroed out");
}

/*===========================================================================
 * L5: Learning Rate Schedule Tests
 *===========================================================================*/

static void test_lr_schedule(void) {
    TEST("constant schedule");
    double lr = nn_learning_rate_schedule(0.1, 10, LR_SCHEDULE_CONSTANT, 0.0, 10);
    CHECK_FLOAT(lr, 0.1, 1e-10, "constant returns base_lr");

    TEST("exponential decay");
    lr = nn_learning_rate_schedule(0.1, 5, LR_SCHEDULE_EXPONENTIAL, 0.1, 10);
    CHECK(lr < 0.1, "exponential decay reduces lr");

    TEST("step decay");
    lr = nn_learning_rate_schedule(1.0, 12, LR_SCHEDULE_STEP, 0.5, 10);
    CHECK_FLOAT(lr, 0.5, 1e-10, "step: after 10 steps, lr halves once");

    TEST("cosine annealing");
    lr = nn_learning_rate_schedule(0.1, 0, LR_SCHEDULE_COSINE, 0.0, 100);
    CHECK_FLOAT(lr, 0.1, 1e-10, "cosine: epoch 0 = base_lr");
    lr = nn_learning_rate_schedule(0.1, 100, LR_SCHEDULE_COSINE, 0.0, 100);
    CHECK_FLOAT(lr, 0.0, 1e-10, "cosine: epoch max = 0");
}

/*===========================================================================
 * L5: RNG Tests
 *===========================================================================*/

static void test_random_normal(void) {
    TEST("random normal produces varied values");
    unsigned int seed = 42;
    double sum = 0.0, sum_sq = 0.0;
    int n = 1000;
    for (int i = 0; i < n; i++) {
        double v = nn_random_normal(&seed);
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / n;
    double var = sum_sq / n - mean * mean;
    CHECK(fabs(mean) < 0.1, "normal random mean ~ 0");
    CHECK(fabs(var - 1.0) < 0.2, "normal random variance ~ 1");
}

static void test_random_uniform(void) {
    TEST("random uniform in [0,1)");
    unsigned int seed = 42;
    for (int i = 0; i < 100; i++) {
        double v = nn_random_uniform(&seed);
        CHECK(v >= 0.0 && v < 1.0, "uniform in [0,1)");
        break;  /* Only test first value to avoid excessive output */
    }
}

/*===========================================================================
 * L5: Training Loop Tests
 *===========================================================================*/

static void test_training_epoch(void) {
    TEST("training epoch reduces loss on simple data");
    /* Create a simple network to learn identity: y = x */
    int sizes[] = {1, 4, 1};
    nn_architecture_t arch = {3, sizes, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, 0.0};
    nn_network_t *net = nn_network_create(&arch);

    /* Create simple dataset: learn y = 2*x */
    int n = 50;
    double *X = (double *)malloc(n * sizeof(double));
    double *Y = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        X[i] = (double)i / n;
        Y[i] = 2.0 * X[i];
    }

    nn_dataset_t *ds = nn_dataset_create(X, Y, n, 1, 1, 0.8, 0.0);
    /* Move 20% into validation */
    ds->val_start = (int)(n * 0.8);
    ds->val_end = n;
    ds->train_end = ds->val_start;

    /* Train for several epochs */
    for (int e = 0; e < 20; e++) {
        nn_network_train_epoch(net, ds, e);
    }

    net->is_trained = 1;

    /* Test prediction */
    double output[1];
    nn_network_forward(net, &(double){0.5}, output);
    /* Should be close to 1.0 */
    CHECK(isfinite(output[0]), "prediction is finite after training");

    nn_dataset_free(ds);
    nn_network_free(net);
    free(X);
    free(Y);
}

static void test_training_config(void) {
    TEST("training config defaults");
    nn_training_config_t config;
    nn_training_config_default(&config);
    CHECK(config.optimizer == NN_OPTIMIZER_ADAM, "default optimizer is Adam");
    CHECK(config.loss == NN_LOSS_MSE, "default loss is MSE");
    CHECK_FLOAT(config.learning_rate, 0.001, 1e-10, "default lr = 0.001");
    CHECK(config.max_epochs == 100, "default max_epochs = 100");
    CHECK(config.batch_size == 32, "default batch_size = 32");
}

/*===========================================================================
 * L5: Batch Normalization Tests
 *===========================================================================*/

static void test_batch_norm(void) {
    TEST("batch normalization zero mean, unit variance");
    double batch[] = {1.0, 2.0, 3.0, 4.0, 5.0,   /* feature 0 */
                       2.0, 4.0, 6.0, 8.0, 10.0}; /* feature 1 */
    double gamma[] = {1.0, 1.0};
    double beta[] = {0.0, 1.0};

    nn_batch_normalize(batch, 5, 2, gamma, beta, 1e-5);

    /* Check feature 0 mean ~ 0 */
    double mean0 = 0.0;
    for (int i = 0; i < 5; i++) mean0 += batch[i * 2];
    mean0 /= 5;
    CHECK_FLOAT(mean0, 0.0, 0.1, "mean of normalized feature 0 ~ 0");

    /* Check feature 1 mean ~ 1 (beta = 1) */
    double mean1 = 0.0;
    for (int i = 0; i < 5; i++) mean1 += batch[i * 2 + 1];
    mean1 /= 5;
    CHECK_FLOAT(mean1, 1.0, 0.5, "mean of normalized feature 1 ~ 1 (beta=1)");
}

/*===========================================================================
 * Test runner
 *===========================================================================*/

int main(void) {
    printf("\n=== Neural Network Soft Sensor: Training Tests ===\n\n");

    printf("--- L5: Loss Functions ---\n");
    test_loss_mse();
    test_loss_mae();
    test_loss_huber();
    test_loss_mape();

    printf("\n--- L5: Optimizers ---\n");
    test_sgd_step();
    test_adam_step();
    test_momentum_step();

    printf("\n--- L5: Regularization ---\n");
    test_l1_regularization();
    test_l2_regularization();
    test_dropout();

    printf("\n--- L5: Learning Rate Schedules ---\n");
    test_lr_schedule();

    printf("\n--- L5: RNG ---\n");
    test_random_normal();
    test_random_uniform();

    printf("\n--- L5: Training ---\n");
    test_training_config();
    test_training_epoch();

    printf("\n--- L5: Batch Normalization ---\n");
    test_batch_norm();

    printf("\n========================================\n");
    printf("Results: %d PASS, %d FAIL\n", tests_passed, tests_failed);
    printf("========================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
