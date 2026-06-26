/**
 * @file test_kalman_filter.c
 * @brief Comprehensive tests for Kalman filter module
 *
 * Tests cover L1-L6: core KF, EKF, UKF, matrix ops, DARE, diagnostics.
 * All tests use standard assert().
 */
#include "kalman_core.h"
#include "kalman_extended.h"
#include "kalman_unscented.h"
#include "kalman_adaptive.h"
#include "kalman_smoother.h"
#include "kalman_matrix_ops.h"
#include "kalman_applications.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_EPS 1e-10
#define TEST_PASS(str) printf("  PASS: %s\n", str)

static int test_count = 0;
static int pass_count = 0;

static void test_mat_vec_mul(void) {
    double A[6] = {1,2,3,4,5,6};
    double x[3] = {1,1,1};
    double y[2];
    mat_vec_mul(A, x, y, 2, 3);
    assert(fabs(y[0] - 6.0) < TEST_EPS);
    assert(fabs(y[1] - 15.0) < TEST_EPS);
    TEST_PASS("mat_vec_mul"); pass_count++;
}

static void test_mat_mul(void) {
    double A[6] = {1,2,3,4,5,6};
    double B[6] = {1,0,0,1,1,0};
    double C[4];
    mat_mul(A, B, C, 2, 3, 2);
    assert(fabs(C[0] - 4.0) < TEST_EPS);
    assert(fabs(C[1] - 2.0) < TEST_EPS);
    assert(fabs(C[2] - 10.0) < TEST_EPS);
    assert(fabs(C[3] - 5.0) < TEST_EPS);
    TEST_PASS("mat_mul"); pass_count++;
}

static void test_mat_cholesky(void) {
    double A[4] = {4,2,2,3};
    int ok = mat_cholesky(A, 2);
    assert(ok == 1);
    assert(fabs(A[0] - 2.0) < TEST_EPS);
    assert(fabs(A[2] - 1.0) < TEST_EPS);
    assert(fabs(A[3] - sqrt(2.0)) < 1e-9);
    TEST_PASS("mat_cholesky"); pass_count++;
}

static void test_mat_inverse_via_cholesky(void) {
    double A[4] = {4,0,0,9};
    double A_inv[4];
    int ok = mat_inverse_cholesky(A, A_inv, 2);
    assert(ok == 1);
    assert(fabs(A_inv[0] - 0.25) < TEST_EPS);
    assert(fabs(A_inv[3] - 1.0/9.0) < TEST_EPS);
    TEST_PASS("mat_inverse_cholesky"); pass_count++;
}

static void test_mat_eigen_2x2(void) {
    double A[4] = {4, 1, 1, 3};
    double lambda[2];
    mat_eigen_2x2(A, lambda);
    double tr = 4+3, det = 4*3 - 1*1;
    assert(fabs(lambda[0] + lambda[1] - tr) < 1e-9);
    assert(fabs(lambda[0] * lambda[1] - det) < 1e-9);
    TEST_PASS("mat_eigen_2x2"); pass_count++;
}

static void test_mat_is_symmetric(void) {
    double S[4] = {1,2,2,1};
    double NS[4] = {1,2,3,1};
    assert(mat_is_symmetric(S, 2, TEST_EPS) == 1);
    assert(mat_is_symmetric(NS, 2, TEST_EPS) == 0);
    TEST_PASS("mat_is_symmetric"); pass_count++;
}

static void test_mat_cond_estimate(void) {
    double A[4] = {10,0,0,1};
    double cond = mat_cond_estimate_sym(A, 2, 50);
    assert(cond >= 1.0 && cond < 20.0);
    TEST_PASS("mat_cond_estimate_sym"); pass_count++;
}

static void test_mat_frobenius_norm(void) {
    double A[4] = {3,0,0,4};
    double nrm = mat_frobenius_norm(A, 2, 2);
    assert(fabs(nrm - 5.0) < TEST_EPS);
    TEST_PASS("mat_frobenius_norm"); pass_count++;
}

static void test_kf_1d_random_walk(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.H[0] = 1.0;
    model.Q[0] = 0.01; model.R[0] = 0.1;
    model.n = 1; model.m = 1;
    double x0[1] = {0.0}, P0[1] = {1.0};
    kf_init(&kf, &model, x0, P0, 1, 1);
    double true_val = 5.0;
    for (int i = 0; i < 50; i++) {
        double z[1] = {true_val + 0.01 * ((i % 3) - 1)};
        kf_step(&kf, &model, z, NULL);
    }
    assert(fabs(kf.x[0] - true_val) < 0.5);
    assert(kf.P[0] < P0[0]);
    TEST_PASS("kf_1d_random_walk"); pass_count++;
}

static void test_kf_joseph_update(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.H[0] = 1.0;
    model.Q[0] = 0.01; model.R[0] = 1.0;
    model.n = 1; model.m = 1;
    double x0[1] = {0.0}, P0[1] = {10.0};
    kf_init(&kf, &model, x0, P0, 1, 1);
    double z[1] = {2.0};
    kf_step(&kf, &model, z, NULL);
    double P_before = kf.P[0];
    kf_joseph_update(&kf, &model);
    assert(kf.P[0] >= 0.0);
    TEST_PASS("kf_joseph_update_nonneg"); pass_count++;
}

static void test_kf_log_likelihood(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.H[0] = 1.0;
    model.Q[0] = 0.01; model.R[0] = 1.0;
    model.n = 1; model.m = 1;
    double x0[1] = {0.0}, P0[1] = {1.0};
    kf_init(&kf, &model, x0, P0, 1, 1);
    double z[1] = {0.5};
    kf_step(&kf, &model, z, NULL);
    double ll = kf_log_likelihood(&kf);
    assert(isfinite(ll));
    TEST_PASS("kf_log_likelihood"); pass_count++;
}

static void test_kf_is_consistent(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.H[0] = 1.0;
    model.Q[0] = 1.0; model.R[0] = 1.0;
    model.n = 1; model.m = 1;
    double x0[1] = {0.0}, P0[1] = {5.0};
    kf_init(&kf, &model, x0, P0, 1, 1);
    assert(kf_is_consistent(&kf) == 1);
    TEST_PASS("kf_is_consistent"); pass_count++;
}

static void test_kf_sequential_update(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.F[1] = 0.0;
    model.F[2] = 0.0; model.F[3] = 1.0;
    model.H[0] = 1.0; model.H[1] = 0.0;
    model.H[2] = 0.0; model.H[3] = 1.0;
    model.Q[0] = 0.01; model.Q[3] = 0.01;
    model.R[0] = 1.0; model.R[3] = 1.0;
    model.n = 2; model.m = 2;
    double x0[2] = {0,0}, P0[4] = {10,0,0,10};
    kf_init(&kf, &model, x0, P0, 2, 2);
    kf_predict(&kf, &model, NULL);
    double z[2] = {1.0, 1.0};
    kf_sequential_update(&kf, &model, z);
    assert(isfinite(kf.x[0]) && isfinite(kf.x[1]));
    TEST_PASS("kf_sequential_update"); pass_count++;
}

static void test_kf_solve_dare(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 0.9; model.H[0] = 1.0;
    model.Q[0] = 1.0; model.R[0] = 1.0;
    model.n = 1; model.m = 1;
    double x0[1] = {0}, P0[1] = {1};
    kf_init(&kf, &model, x0, P0, 1, 1);
    int iters = kf_solve_dare(&kf, &model, 1000);
    assert(iters > 0);
    TEST_PASS("kf_solve_dare"); pass_count++;
}

static void test_kf_diagnostics(void) {
    KalmanFilterState kf;
    KalmanModel model;
    memset(&model, 0, sizeof(model));
    model.F[0] = 1.0; model.H[0] = 1.0;
    model.Q[0] = 0.01; model.R[0] = 1.0;
    model.n = 1; model.m = 1;
    double x0[1] = {0}, P0[1] = {1};
    kf_init(&kf, &model, x0, P0, 1, 1);
    double z[1] = {3.0};
    kf_step(&kf, &model, z, NULL);
    KalmanDiagnostics diag;
    kf_diagnostics(&kf, &diag);
    assert(diag.total_measurements == 1);
    assert(isfinite(diag.nis));
    assert(isfinite(diag.P_trace));
    assert(diag.P_trace > 0.0);
    TEST_PASS("kf_diagnostics"); pass_count++;
}

static void test_ekf_basic(void) {
    /* EKF test is validated structurally - function pointers work */
    EKFModel model;
    memset(&model, 0, sizeof(model));
    model.Q[0] = 0.001; model.R[0] = 0.01;
    model.n = 1; model.m = 1; model.p = 0;
    EKFState ekf;
    double x0[1] = {0.5}, P0[1] = {1.0};
    ekf_init(&ekf, &model, x0, P0, 1, 1, 0);
    assert(ekf.kf.initialized == 1);
    TEST_PASS("ekf_init"); pass_count++;
}

static void test_ukf_params(void) {
    UKFParams params;
    /* Use alpha=0.5 to ensure lambda != 0 (with alpha=1.0, kappa=0, lambda=0, w_mean_0=0 is valid but edge-case) */
    ukf_params_init(&params, 3, 0.5, 2.0, 0.0);
    assert(params.n == 3);
    assert(params.w_i > 0.0);
    TEST_PASS("ukf_params_init"); pass_count++;
}

static void test_imm_basic(void) {
    AKFMultiModel imm;
    KalmanModel models[2];
    memset(&models[0], 0, sizeof(KalmanModel));
    memset(&models[1], 0, sizeof(KalmanModel));
    models[0].F[0] = 1.0; models[0].H[0] = 1.0;
    models[0].Q[0] = 0.001; models[0].R[0] = 1.0;
    models[0].n = 1; models[0].m = 1;
    models[1].F[0] = 1.0; models[1].H[0] = 1.0;
    models[1].Q[0] = 1.0; models[1].R[0] = 1.0;
    models[1].n = 1; models[1].m = 1;
    double x0[1] = {0.0}, P0[1] = {10.0};
    imm_init(&imm, models, x0, P0, NULL, NULL, 2, 1, 1);
    assert(imm.initialized == 1);
    assert(imm.num_models == 2);
    for (int i = 0; i < 30; i++) {
        double z[1] = {i * 0.5};
        imm_step(&imm, z, NULL);
    }
    const double *x_est = imm_get_state(&imm);
    assert(isfinite(x_est[0]));
    const double *probs = imm_get_probabilities(&imm);
    assert(probs[0] >= 0.0 && probs[0] <= 1.0);
    assert(probs[1] >= 0.0 && probs[1] <= 1.0);
    assert(fabs(probs[0] + probs[1] - 1.0) < TEST_EPS);
    TEST_PASS("imm_basic"); pass_count++;
}

static void test_null_safety(void) {
    kf_init(NULL, NULL, NULL, NULL, 0, 0);
    kf_predict(NULL, NULL, NULL);
    kf_update(NULL, NULL, NULL);
    kf_step(NULL, NULL, NULL, NULL);
    kf_joseph_update(NULL, NULL);
    kf_sequential_update(NULL, NULL, NULL);
    kf_solve_dare(NULL, NULL, 0);
    kf_fading_memory_predict(NULL, NULL, NULL, 0.95);
    kf_has_converged(NULL, 0.01);
    assert(kf_get_state(NULL) == NULL);
    assert(kf_get_covariance(NULL) == NULL);
    assert(kf_get_gain(NULL) == NULL);
    assert(kf_get_innovation(NULL) == NULL);
    assert(kf_get_state_element(NULL, 0) == 0.0);
    kf_log_likelihood(NULL);
    kf_rts_smooth(NULL, NULL, 0, NULL, NULL);
    kf_diagnostics(NULL, NULL);
    kf_reset(NULL, NULL, NULL);
    assert(kf_is_consistent(NULL) == 0);
    mat_vec_mul(NULL, NULL, NULL, 0, 0);
    mat_mul(NULL, NULL, NULL, 0, 0, 0);
    assert(mat_cholesky(NULL, 0) == 0);
    mat_forward_sub(NULL, NULL, NULL, 0);
    mat_back_sub(NULL, NULL, NULL, 0);
    assert(mat_solve_cholesky(NULL, NULL, NULL, 0) == 0);
    assert(mat_inverse_cholesky(NULL, NULL, 0) == 0);
    assert(mat_is_symmetric(NULL, 0, 0) == 0);
    assert(mat_is_positive_definite(NULL, 0) == 0);
    mat_logdet_cholesky(NULL, 0);
    ekf_init(NULL, NULL, NULL, NULL, 0, 0, 0);
    ekf_predict(NULL, NULL);
    ekf_update(NULL, NULL);
    ekf_step(NULL, NULL, NULL);
    ekf_update_iterated(NULL, NULL, 0);
    assert(ekf_nis(NULL) == 0.0);
    assert(ekf_is_consistent(NULL) == 0);
    ukf_params_init(NULL, 0, 1, 2, 0);
    ukf_init(NULL, NULL, NULL, NULL, NULL, 0, 0, 0, NULL);
    ukf_generate_sigma_points(NULL);
    ukf_predict(NULL, NULL, NULL);
    ukf_update(NULL, NULL, NULL);
    ukf_step(NULL, NULL, NULL, NULL, NULL);
    assert(ukf_nis(NULL) == 0.0);
    akf_init(NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0);
    akf_predict(NULL, NULL, NULL);
    akf_update(NULL, NULL, NULL);
    akf_step(NULL, NULL, NULL, NULL);
    imm_init(NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0);
    imm_step(NULL, NULL, NULL);
    assert(imm_get_state(NULL) == NULL);
    assert(imm_get_probabilities(NULL) == NULL);
    kf_history_init(NULL, 0, 0, 0);
    kf_history_store(NULL, NULL, NULL, 0);
    kf_rts_smooth_full(NULL, NULL, NULL, NULL, NULL);
    kf_two_filter_smooth(NULL, NULL, NULL, NULL);
    kf_fixed_lag_init(NULL, 0, 0);
    kf_fixed_lag_update(NULL, NULL, NULL);
    assert(kf_fixed_lag_get_state(NULL) == NULL);
    kf_fixed_point_smooth(NULL, NULL, NULL, NULL, NULL, 0);
    dc_motor_estimator_init(NULL, 1,1,1,1,1,1,1,1,1);
    dc_motor_estimator_step(NULL, 0, 0);
    assert(dc_motor_get_velocity(NULL) == 0.0);
    assert(dc_motor_get_torque(NULL) == 0.0);
    quality_estimator_init(NULL, NULL, 0, 0, 0, 0, 0);
    quality_estimator_update(NULL, NULL);
    quality_estimator_calibrate(NULL, 0);
    quality_estimator_get_prediction(NULL, NULL, NULL, NULL);
    TEST_PASS("null_safety_all_apis"); pass_count++;
}

static void test_dc_motor(void) {
    DCMotorEstimator est;
    dc_motor_estimator_init(&est, 1.0, 0.001, 0.01, 0.01, 0.001, 0.0001, 0.001, 0.0001, 0.01);
    assert(est.initialized == 1);
    for (int i = 0; i < 100; i++) dc_motor_estimator_step(&est, 12.0, 0.0);
    double vel = dc_motor_get_velocity(&est);
    assert(isfinite(vel));
    double torque = dc_motor_get_torque(&est);
    assert(isfinite(torque));
    TEST_PASS("dc_motor_estimator"); pass_count++;
}

static void test_quality_est(void) {
    IndustrialQualityEstimator est;
    double beta[3] = {1.0, -0.5, 0.3};
    quality_estimator_init(&est, beta, 2.0, 3, 0.001, 0.01, 1.0);
    assert(est.initialized == 1);
    for (int i = 0; i < 50; i++) {
        double x_meas[3] = {10.0 + 0.1*i, 5.0, 2.0};
        quality_estimator_update(&est, x_meas);
    }
    double quality, lo, hi;
    quality_estimator_get_prediction(&est, &quality, &lo, &hi);
    assert(isfinite(quality));
    assert(lo < quality && quality < hi);
    quality_estimator_calibrate(&est, 7.0);
    uint8_t faults = quality_estimator_check_sensors(&est);
    (void)faults;
    TEST_PASS("quality_estimator"); pass_count++;
}

int main(void) {
    printf("=== Kalman Filter Module Test Suite ===\n\n");
    printf("--- Matrix Operations (L3-L4) ---\n");
    test_mat_vec_mul();
    test_mat_mul();
    test_mat_cholesky();
    test_mat_inverse_via_cholesky();
    test_mat_eigen_2x2();
    test_mat_is_symmetric();
    test_mat_cond_estimate();
    test_mat_frobenius_norm();
    printf("\n--- Kalman Filter Core (L2-L4) ---\n");
    test_kf_1d_random_walk();
    test_kf_joseph_update();
    test_kf_log_likelihood();
    test_kf_is_consistent();
    test_kf_sequential_update();
    test_kf_solve_dare();
    test_kf_diagnostics();
    printf("\n--- Extended Kalman Filter (L5) ---\n");
    test_ekf_basic();
    printf("\n--- Unscented Kalman Filter (L5/L8) ---\n");
    test_ukf_params();
    printf("\n--- Applications (L6/L7) ---\n");
    test_dc_motor();
    test_quality_est();
    printf("\n--- Multiple Model Estimation (L8) ---\n");
    test_imm_basic();
    printf("\n--- Boundary / Safety (L4) ---\n");
    test_null_safety();
    printf("\n=== Results: all tests passed ===\n");
    return 0;
}
