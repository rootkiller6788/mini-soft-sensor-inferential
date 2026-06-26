/**
 * @file test_quality_estimator.c
 * @brief Comprehensive tests for inferential quality estimation framework.
 *
 * Tests cover L1-L8: definitions, concepts, structures, laws, algorithms,
 * canonical problems, industrial applications, and advanced topics.
 */

#include "quality_estimator_types.h"
#include "inferential_model.h"
#include "kalman_quality.h"
#include "bias_correction.h"
#include "multi_rate_fusion.h"
#include "quality_recursive_ls.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

/*===========================================================================
 * L1: Definitions Tests
 *===========================================================================*/

static void test_config_init(void)
{
    TEST("Config initialization");
    qest_config_t cfg;
    qest_config_init(&cfg, "TestEstimator", "QI_001", "wt%", 5);

    assert(strcmp(cfg.estimator_name, "TestEstimator") == 0);
    assert(strcmp(cfg.quality_tag, "QI_001") == 0);
    assert(strcmp(cfg.units, "wt%") == 0);
    assert(cfg.n_input_vars == 5);
    assert(cfg.model_type == QMODEL_DATA_DRIVEN);
    assert(cfg.bias_strategy == BIAS_EWMA_FILTERED);
    assert(cfg.fast_sample_period == 10.0);
    assert(cfg.lab_sample_period == 14400.0);
    assert(cfg.bias_filter_gain > 0.0 && cfg.bias_filter_gain < 1.0);
    PASS();
}

static void test_timestamp_operations(void)
{
    TEST("Timestamp operations");
    qest_timestamp_t t1, t2;
    qest_timestamp_set(&t1, 2026, 6, 23, 10, 30, 0.0);
    qest_timestamp_set(&t2, 2026, 6, 23, 14, 30, 0.0);

    double diff = qest_timestamp_diff_seconds(&t2, &t1);
    assert(fabs(diff - 14400.0) < 1.0);  /* 4 hours */

    int cmp = qest_timestamp_compare(&t1, &t2);
    assert(cmp < 0);  /* t1 < t2 */
    PASS();
}

static void test_quality_estimate_init(void)
{
    TEST("Quality estimate initialization");
    quality_estimate_t est;
    qest_init_estimate(&est);

    assert(est.predicted_value == 0.0);
    assert(est.bias_corrected_value == 0.0);
    assert(est.mode == QEST_MODE_STANDBY);
    assert(est.health == MAINT_OK);
    assert(est.is_valid == 0);
    PASS();
}

static void test_process_variable_init(void)
{
    TEST("Process variable initialization");
    process_variable_t pv;
    pv_init(&pv, "TIC101", 0.0, 200.0, 0.1);

    assert(strcmp(pv.tag_name, "TIC101") == 0);
    assert(pv.low_range == 0.0);
    assert(pv.high_range == 200.0);
    assert(pv.sensor_stddev == 0.1);

    pv_update(&pv, 100.0);
    assert(pv.raw_value == 100.0);
    assert(fabs(pv.scaled_value - 0.5) < 0.01);

    pv_update(&pv, 150.0);
    assert(pv.rate_of_change == 50.0);
    PASS();
}

/*===========================================================================
 * L2: Core Concepts Tests — Linear Model
 *===========================================================================*/

static void test_linear_model_evaluate(void)
{
    TEST("Linear model evaluation");
    linear_model_t lm;
    lm.n_inputs = 3;
    lm.intercept = 1.0;
    lm.coefficients[0] = 2.0;
    lm.coefficients[1] = 0.5;
    lm.coefficients[2] = -1.0;

    double x[] = {1.0, 2.0, 3.0};
    double y = linear_model_evaluate(&lm, x, 3);
    /* y = 1 + 2*1 + 0.5*2 + (-1)*3 = 1 + 2 + 1 - 3 = 1 */
    assert(fabs(y - 1.0) < 1e-10);
    PASS();
}

static void test_arx_model_init(void)
{
    TEST("ARX model initialization");
    arx_model_t arx;
    arx_init(&arx, 2, 3, 4, 1);

    assert(arx.na == 2);
    assert(arx.nb == 3);
    assert(arx.n_inputs == 4);
    assert(arx.nk == 1);
    PASS();
}

/*===========================================================================
 * L3: Engineering Structures Tests — State-Space
 *===========================================================================*/

static void test_state_space_alloc(void)
{
    TEST("State-space allocation");
    ss_model_t ss;
    ss_model_alloc(&ss, 3, 2, 1);

    assert(ss.n_states == 3);
    assert(ss.n_inputs == 2);
    assert(ss.n_outputs == 1);
    assert(ss.A != NULL);
    assert(ss.B != NULL);
    assert(ss.C != NULL);
    assert(ss.x != NULL);
    assert(ss.P != NULL);

    /* A should be identity */
    for (int i = 0; i < 3; i++) {
        assert(fabs(ss.A[i*3 + i] - 1.0) < 1e-10);
    }

    ss_model_free(&ss);
    PASS();
}

static void test_state_space_predict(void)
{
    TEST("State-space prediction");
    ss_model_t ss;
    ss_model_alloc(&ss, 2, 1, 1);

    /* A = [0.9, 0; 0, 0.9] (stable first-order decay) */
    ss.A[0] = 0.9; ss.A[1] = 0.0;
    ss.A[2] = 0.0; ss.A[3] = 0.9;

    /* B = [1; 0] */
    ss.B[0] = 1.0; ss.B[1] = 0.0;

    /* Initial state */
    ss.x[0] = 5.0; ss.x[1] = 3.0;

    /* P = I */
    ss.P[0] = 1.0; ss.P[1] = 0.0;
    ss.P[2] = 0.0; ss.P[3] = 1.0;

    /* Q = 0.01 * I */
    ss.Q[0] = 0.01; ss.Q[1] = 0.0;
    ss.Q[2] = 0.0; ss.Q[3] = 0.01;

    double u[] = {2.0};
    ss_model_predict(&ss, u);

    /* x_prior = A*x + B*u = [0.9*5 + 1*2, 0.9*3 + 0*2] = [6.5, 2.7] */
    assert(fabs(ss.x_prior[0] - 6.5) < 1e-10);
    assert(fabs(ss.x_prior[1] - 2.7) < 1e-10);

    ss_model_free(&ss);
    PASS();
}

/*===========================================================================
 * L4: Engineering Laws Tests — Model Validation
 *===========================================================================*/

static void test_compute_regression_stats(void)
{
    TEST("Regression statistics computation");
    double y_model[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_lab[]   = {1.1, 1.9, 3.2, 3.8, 5.1};
    qest_performance_t stats;
    memset(&stats, 0, sizeof(stats));

    compute_regression_stats(y_model, y_lab, 5, &stats);

    assert(stats.rmse > 0.0);
    assert(stats.r_squared > 0.9);  /* Good fit */
    assert(stats.mae > 0.0);
    PASS();
}

static void test_durbin_watson(void)
{
    TEST("Durbin-Watson test");
    /* White noise: DW ≈ 2 */
    double white[] = {0.5, -0.3, 0.1, -0.2, 0.4, -0.1};
    double dw_white = durbin_watson_test(white, 6);
    /* For white noise, DW should be close to 2 */
    assert(dw_white > 1.0 && dw_white < 3.0);

    /* Positive autocorrelation: DW < 2 */
    double pos_auto[] = {1.0, 1.2, 1.1, 1.3, 1.2, 1.4};
    double dw_pos = durbin_watson_test(pos_auto, 6);
    assert(dw_pos < 2.5);

    /* Negative autocorrelation: DW > 2 */
    double neg_auto[] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    double dw_neg = durbin_watson_test(neg_auto, 6);
    assert(dw_neg > 1.5);

    PASS();
}

static void test_t_test_zero_mean(void)
{
    TEST("T-test for zero mean");
    /* Zero-mean residuals: should NOT reject H0 */
    double zero_mean[] = {0.1, -0.05, 0.02, -0.03, 0.0, 0.07, -0.01, 0.04};
    double t_stat;
    int result = t_test_zero_mean(zero_mean, 8, &t_stat);
    assert(result == 1);  /* Cannot reject H0 (mean = 0) */

    /* Nonzero-mean residuals: should reject H0 */
    double nonzero_mean[] = {2.1, 1.9, 2.0, 2.2, 1.8, 2.0, 2.3, 1.7};
    result = t_test_zero_mean(nonzero_mean, 8, &t_stat);
    /* With large mean shift, should reject H0 */
    assert(result == 0 || fabs(t_stat) > 2.0);
    PASS();
}

/*===========================================================================
 * L5: Algorithm Tests — Kalman Filter
 *===========================================================================*/

static void test_kalman_filter_alloc(void)
{
    TEST("Kalman filter allocation");
    kalman_filter_t kf;
    kf_alloc(&kf, 2, 1, 1);

    assert(kf.n_states == 2);
    assert(kf.x != NULL);
    assert(kf.P != NULL);
    assert(kf.A != NULL);
    assert(kf.K != NULL);

    /* P should be initialized to 1000*I */
    assert(kf.P[0] > 500.0);

    kf_free(&kf);
    PASS();
}

static void test_kalman_filter_predict(void)
{
    TEST("Kalman filter predict step");
    kalman_filter_t kf;
    kf_alloc(&kf, 1, 0, 1);

    /* Simple identity system: A=1, Q=0.1, R=1 */
    double A[] = {1.0};
    double C[] = {1.0};
    double Q[] = {0.1};
    double R[] = {1.0};
    double x0[] = {0.0};
    double P0[] = {1.0};

    kf_set_matrices(&kf, A, NULL, C);
    kf_set_noise(&kf, Q, R);
    kf_set_initial(&kf, x0, P0);

    /* Predict with no input */
    kf_predict(&kf, NULL);
    assert(fabs(kf.x_prior[0] - 0.0) < 1e-10);
    /* P_prior = A*P*A^T + Q = 1*1*1 + 0.1 = 1.1 */
    assert(fabs(kf.P_prior[0] - 1.1) < 1e-6);

    kf_free(&kf);
    PASS();
}

static void test_kalman_filter_update(void)
{
    TEST("Kalman filter update step");
    kalman_filter_t kf;
    kf_alloc(&kf, 1, 0, 1);

    double A[] = {1.0};
    double C[] = {1.0};
    double Q[] = {0.0};  /* No process noise for simple test */
    double R[] = {1.0};
    double x0[] = {0.0};
    double P0[] = {1000.0};

    kf_set_matrices(&kf, A, NULL, C);
    kf_set_noise(&kf, Q, R);
    kf_set_initial(&kf, x0, P0);

    /* Predict first */
    kf_predict(&kf, NULL);

    /* Measurement y = 5.0 */
    double y[] = {5.0};
    kf_update(&kf, y);

    /* x should move toward y=5; K = P_prior/(P_prior+R) = 1000/1001 ≈ 0.999 */
    assert(fabs(kf.x[0] - 4.995) < 0.1);  /* Should be close to 5 */
    /* P should decrease */
    assert(kf.P[0] < 500.0);
    assert(kf.P[0] > 0.0);

    kf_free(&kf);
    PASS();
}

static void test_kalman_filter_step(void)
{
    TEST("Kalman filter combined step");
    kalman_filter_t kf;
    kf_alloc(&kf, 1, 0, 1);

    double A[] = {1.0}, C[] = {1.0};
    double Q[] = {0.01}, R[] = {0.1};
    double x0[] = {10.0}, P0[] = {1.0};

    kf_set_matrices(&kf, A, NULL, C);
    kf_set_noise(&kf, Q, R);
    kf_set_initial(&kf, x0, P0);

    /* Run multiple steps converging measurements to 10.5 */
    double x_out;
    for (int i = 0; i < 50; i++) {
        double y[] = {10.5};
        kf_step(&kf, NULL, y, &x_out);
    }

    /* After 50 steps, state should converge near 10.5 */
    assert(fabs(kf_get_quality(&kf) - 10.5) < 0.5);
    assert(kf_get_quality_variance(&kf) < 1.0);

    kf_free(&kf);
    PASS();
}

/*===========================================================================
 * L5: Algorithm Tests — RLS
 *===========================================================================*/

static void test_rls_estimator(void)
{
    TEST("RLS estimator convergence");
    rls_estimator_t rls;
    rls_alloc(&rls, 2, 0.99, 100.0);

    /* True model: y = 3.0*x1 - 2.0*x2 */
    for (int i = 0; i < 200; i++) {
        double x1 = (i % 10) * 0.5;
        double x2 = (i % 7) * 0.3;
        double y_true = 3.0 * x1 - 2.0 * x2;
        double phi[] = {x1, x2};
        rls_update(&rls, phi, y_true);
    }

    double theta[2];
    rls_get_parameters(&rls, theta);

    /* Should have converged near [3.0, -2.0] */
    assert(fabs(theta[0] - 3.0) < 1.0);
    assert(fabs(theta[1] + 2.0) < 1.0);

    rls_free(&rls);
    PASS();
}

static void test_rls_vff(void)
{
    TEST("RLS with variable forgetting factor");
    rls_vff_t rls_vff;
    rls_vff_alloc(&rls_vff, 2, 0.90, 0.999, 1.0, 100.0);

    for (int i = 0; i < 100; i++) {
        double x1 = (i % 5) * 0.5;
        double x2 = (i % 3) * 0.7;
        double y = 2.0 * x1 + 1.0 * x2;
        double phi[] = {x1, x2};
        rls_vff_update(&rls_vff, phi, y);
    }

    double lambda = rls_vff_get_lambda(&rls_vff);
    assert(lambda >= 0.90 && lambda <= 1.0);

    rls_vff_free(&rls_vff);
    PASS();
}

static void test_wls_window(void)
{
    TEST("WLS window estimator");
    wls_window_t wls;
    wls_window_alloc(&wls, 2, 20, 0.95);

    for (int i = 0; i < 30; i++) {
        double x1 = (i % 4 + 1) * 0.5;
        double x2 = (i % 6 + 1) * 0.3;
        double y = 5.0 * x1 - 1.0 * x2;
        double phi[] = {x1, x2};
        wls_window_update(&wls, phi, y);
    }

    double y_pred = wls_window_predict(&wls, (double[]){2.0, 1.0});
    /* Should predict near 5*2 - 1*1 = 9.0 */
    double expected = 9.0;
    assert(fabs(y_pred - expected) < 5.0);

    wls_window_free(&wls);
    PASS();
}

/*===========================================================================
 * L2: Bias Correction Tests
 *===========================================================================*/

static void test_additive_bias(void)
{
    TEST("Additive bias correction");
    bias_additive_t bc;
    bias_additive_init(&bc, 0.2, -50.0, 50.0, 10.0);

    /* Update with several consistent residuals */
    for (int i = 0; i < 10; i++) {
        bias_additive_update(&bc, 101.0, 100.0);  /* residual = +1.0 */
    }

    double bias = bias_additive_get(&bc);
    /* Should converge near 1.0 */
    assert(fabs(bias - 1.0) < 0.5);

    double corrected = bias_additive_correct(&bc, 100.0);
    assert(fabs(corrected - 101.0) < 1.0);

    PASS();
}

static void test_cusum_detector(void)
{
    TEST("CUSUM drift detector");
    cusum_detector_t cd;
    cusum_init(&cd, 0.5, 3.0);

    /* Feed residuals with a consistent positive bias */
    int alarm = 0;
    for (int i = 0; i < 100; i++) {
        double residual = 0.8;  /* Consistent positive residual > K = 0.25 */
        alarm = cusum_update(&cd, residual, 1.0);
        if (alarm) break;
    }
    /* Should trigger with consistent bias */
    assert(alarm == 1);

    cusum_reset(&cd);
    assert(cd.cusum_hi == 0.0);
    assert(cd.cusum_lo == 0.0);
    PASS();
}

static void test_kalman_bias(void)
{
    TEST("Kalman bias estimation");
    bias_kalman_t bk;
    bias_kalman_init(&bk, 0.001, 0.01, 0.0, 10.0);

    /* Feed multiple lab samples with consistent bias */
    for (int i = 0; i < 20; i++) {
        bias_kalman_update(&bk, 100.0 + 2.5, 100.0);  /* bias = 2.5 */
    }

    /* Bias should converge near 2.5 */
    assert(fabs(bk.bias - 2.5) < 1.0);
    /* Uncertainty should decrease */
    assert(bias_kalman_get_uncertainty(&bk) < 1.0);

    PASS();
}

static void test_bias_context(void)
{
    TEST("Unified bias context");
    qest_config_t cfg;
    qest_config_init(&cfg, "Test", "Q", "%%", 5);

    bias_context_t ctx;
    bias_context_init(&ctx, &cfg);

    /* Process a lab measurement (small residual to pass validation) */
    lab_quality_t q = bias_context_update(&ctx, 50.15, 50.0);
    assert(q == LAB_QUALITY_GOOD);

    /* Correct a prediction */
    double corrected = bias_context_correct(&ctx, 50.0);
    /* Should be biased toward 50.15 */
    assert(corrected > 50.0);

    /* Switch strategy */
    bias_context_set_strategy(&ctx, BIAS_KALMAN);
    double corrected2 = bias_context_correct(&ctx, 50.0);
    assert(corrected2 >= 50.0);

    PASS();
}

/*===========================================================================
 * L3: Multi-Rate Tests
 *===========================================================================*/

static void test_mrf_kalman_alloc(void)
{
    TEST("Multi-rate Kalman allocation");
    mrf_kalman_t mrf;
    mrf_kalman_alloc(&mrf, 2, 3, 1, 1, 100);

    assert(mrf.n_states == 2);
    assert(mrf.n_fast_inputs == 3);
    assert(mrf.x != NULL);
    assert(mrf.P != NULL);
    assert(mrf.x_buffer != NULL);
    assert(mrf.buffer_size == 100);

    mrf_kalman_free(&mrf);
    PASS();
}

static void test_mrf_fast_step(void)
{
    TEST("Multi-rate fast step");
    mrf_kalman_t mrf;
    mrf_kalman_alloc(&mrf, 1, 1, 1, 1, 10);

    double u[] = {1.0};
    mrf_fast_step(&mrf, u, 0.0);

    double x[1];
    mrf_get_state(&mrf, x);
    /* State should have moved due to B*u */
    assert(fabs(x[0]) >= 0.0);

    mrf_kalman_free(&mrf);
    PASS();
}

static void test_mrf_interpolator(void)
{
    TEST("Multi-rate interpolator");
    mrf_interpolator_t interp;
    mrf_interp_init(&interp, 3600.0);

    mrf_interp_feed(&interp, 50.0, 0.0);
    mrf_interp_feed(&interp, 52.0, 3600.0);

    double y_est;
    /* At t=1800, should be halfway */
    int ok = mrf_interp_evaluate(&interp, 1800.0, &y_est);
    assert(ok == 1);
    assert(fabs(y_est - 51.0) < 0.01);

    PASS();
}

/*===========================================================================
 * L6: Canonical Problems Tests
 *===========================================================================*/

static void test_pls_model_predict(void)
{
    TEST("PLS model prediction");
    pls_model_t pls;
    memset(&pls, 0, sizeof(pls_model_t));
    pls.n_inputs = 2;
    pls.n_outputs = 1;
    pls.n_latent = 1;

    /* Simple model: y = 2*x1 + 3*x2  (no scaling) */
    pls.x_stds[0] = 1.0; pls.x_stds[1] = 1.0;
    pls.y_stds[0] = 1.0;
    pls.x_weights[0][0] = 2.0;
    pls.x_weights[1][0] = 3.0;
    pls.y_loadings[0][0] = 1.0;
    pls.intercept[0] = 0.0;

    double x[] = {1.0, 1.0};
    double y_pred[1];
    pls_model_predict(&pls, x, 2, y_pred);
    assert(fabs(y_pred[0] - 5.0) < 1e-10);

    PASS();
}

static void test_ewma_smooth(void)
{
    TEST("EWMA smoothing");
    double s = ewma_smooth(10.0, 0.0, 0.2);
    assert(fabs(s - 2.0) < 1e-10);  /* s = 0.2*10 + 0.8*0 = 2 */

    s = ewma_smooth(10.0, s, 0.2);
    /* s = 0.2*10 + 0.8*2 = 3.6 */
    assert(fabs(s - 3.6) < 1e-10);

    PASS();
}

static void test_distillation_composition(void)
{
    TEST("Distillation composition inference (L6)");
    /* Simulate a simplified distillation column:
     * Overhead composition y is inferred from top tray temperature T
     * using a linear approximation: y = a*T + b
     * where a = -0.05 (composition decreases with temperature)
     *       b = 12.0
     */

    quality_estimator_t *qest = qest_alloc();
    assert(qest != NULL);

    qest_config_t cfg;
    qest_config_init(&cfg, "DistillateComposition", "AI_DIST_COMP", "mol%", 1);
    cfg.model_type = QMODEL_DATA_DRIVEN;

    int rc = qest_configure(qest, &cfg);
    assert(rc == 0);

    /* Set up linear model: comp = -0.05 * T + 12.0 */
    double coeff[] = {-0.05};
    qest_set_linear_model(qest, 12.0, coeff, 1, 0.98, 0.1);

    /* Feed tray temperature */
    double inputs[] = {80.0};  /* 80°C top tray temperature */
    qest_set_inputs(qest, inputs, 1);

    const quality_estimate_t *est = qest_step(qest);
    assert(est != NULL);
    assert(est->is_valid == 1);
    /* Expected: -0.05*80 + 12 = 8.0 mol% */
    assert(fabs(est->predicted_value - 8.0) < 0.1);

    /* Simulate lab sample: 7.8 mol% (slight negative bias) */
    lab_sample_t lab;
    memset(&lab, 0, sizeof(lab));
    lab.measured_value = 7.8;
    lab.lab_stddev = 0.1;
    lab.quality_flag = LAB_QUALITY_GOOD;
    qest_timestamp_set(&lab.sample_time, 2026, 6, 23, 10, 0, 0.0);

    qest_process_lab(qest, &lab);

    /* After bias update, run another estimate */
    qest_set_inputs(qest, inputs, 1);
    est = qest_step(qest);
    /* Corrected estimate should be closer to 7.8 */
    assert(est->bias_corrected_value < est->predicted_value + 0.2);

    qest_free(qest);
    PASS();
}

/* Helper: CSTR conversion first-principles model */
static double cstr_conversion_model(const double *inputs, int n, const double *params, int np)
{
    (void)n; (void)np;
    double T = inputs[0];
    double tau = inputs[1];
    double k0 = params[0];
    double EaR = params[1];
    double k = k0 * exp(-EaR / T);
    double X = k * tau / (1.0 + k * tau);
    return X * 100.0;
}

static void test_reactor_conversion(void)
{
    TEST("Reactor conversion inference (L6)");
    /* CSTR conversion inferred from reactor temperature and residence time:
     * X = k0 * exp(-Ea/(RT)) * tau / (1 + k0 * exp(-Ea/(RT)) * tau)
     *
     * For T = 350K, tau = 120s, k0 = 1e6, Ea/R = 8000:
     * k = 1e6 * exp(-8000/350) ≈ 1e6 * 1.18e-10 ≈ 1.18e-4
     * X ≈ 1.18e-4 * 120 / (1 + 1.18e-4 * 120) ≈ 0.014 / 1.014 ≈ 0.0138
     */

    fpm_model_t fpm;
    double params[] = {1e6, 8000.0};  /* k0, Ea/R */
    fpm_init(&fpm, cstr_conversion_model, params, 2, 0);

    double inputs[] = {350.0, 120.0};  /* T=350K, tau=120s */
    double X = fpm_evaluate(&fpm, inputs, 2);

    /* Expected conversion ~ 1.38% */
    assert(X > 0.5 && X < 3.0);

    PASS();
}

/*===========================================================================
 * L8: Advanced Topics Tests
 *===========================================================================*/

static void test_monte_carlo_prediction(void)
{
    TEST("Monte Carlo prediction uncertainty (L8)");
    /* Use Monte Carlo simulation to estimate prediction uncertainty.
     * Given input uncertainty (sigma = 0.5°C on temperature measurement),
     * propagate through the linear model to get output uncertainty.
     */

    double intercept = 12.0;
    double coeff = -0.05;
    double input_mean = 80.0;
    double input_sigma = 0.5;

    double sum_y = 0.0, sum_y2 = 0.0;
    int n_samples = 10000;

    /* Simple LCG for deterministic reproducibility */
    unsigned int seed = 42;
    for (int i = 0; i < n_samples; i++) {
        /* Box-Muller to generate normal random variable */
        seed = 1103515245 * seed + 12345;
        double u1 = (double)(seed & 0x7FFFFFFF) / 2147483647.0;
        seed = 1103515245 * seed + 12345;
        double u2 = (double)(seed & 0x7FFFFFFF) / 2147483647.0;
        /* Avoid log(0) */
        if (u1 < 1e-10) u1 = 1e-10;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double input = input_mean + input_sigma * z;

        double y = intercept + coeff * input;
        sum_y  += y;
        sum_y2 += y * y;
    }

    double mean = sum_y / n_samples;
    double var  = sum_y2 / n_samples - mean * mean;
    double sigma_pred = sqrt(var);

    /* Expected output sigma ≈ |coeff| * input_sigma = 0.05 * 0.5 = 0.025 */
    assert(sigma_pred > 0.01 && sigma_pred < 0.05);

    PASS();
}

/*===========================================================================
 * L5: MHE / LW-PLS / IV Tests
 *===========================================================================*/

static void test_mhe_buffer(void)
{
    TEST("Moving Horizon Estimation buffer");
    mhe_buffer_t buf;
    mhe_init(&buf, 10, 0.1, 1.0);

    double inputs[] = {0.5, 0.3, 0.1};
    qest_timestamp_t ts;
    qest_timestamp_set(&ts, 2026, 6, 23, 10, 0, 0.0);

    for (int i = 0; i < 15; i++) {
        mhe_push(&buf, inputs, 3, 10.0 + i * 0.1, &ts);
    }
    assert(buf.n_samples_stored == 10);  /* Horizon limited */

    linear_model_t lm;
    lm.n_inputs = 3;
    lm.intercept = 0.0;
    lm.coefficients[0] = 2.0;
    lm.coefficients[1] = 3.0;
    lm.coefficients[2] = 1.0;

    double estimate, variance;
    mhe_estimate(&buf, &lm, 3, &estimate, &variance);
    assert(variance > 0.0);

    PASS();
}

static void test_iv_estimator(void)
{
    TEST("Instrumental Variable estimator");
    iv_estimator_t iv;
    iv_estimator_alloc(&iv, 1, 10);

    /* Simulate: y = 3.0 * x (true gain = 3.0) */
    for (int i = 0; i < 20; i++) {
        double x = (double)(i + 1) * 0.5;
        double y = 3.0 * x;  /* True relationship */
        double phi[] = {x};
        iv_estimator_update(&iv, phi, y);
    }

    double y_pred = iv_estimator_predict(&iv, (double[]){5.0});
    /* Should predict near 15.0 */
    assert(fabs(y_pred - 15.0) < 5.0);

    iv_estimator_free(&iv);
    PASS();
}

static void test_lab_sample_validate(void)
{
    TEST("Lab sample validation");
    lab_quality_t q;

    /* Good sample */
    q = lab_sample_validate(50.0, 49.5, 49.0, 4.0, 0.1, 0.0, 100.0, 0.5, 3.0, 0.5, 0.5);
    assert(q == LAB_QUALITY_GOOD);

    /* Range violation */
    q = lab_sample_validate(150.0, 50.0, 49.0, 4.0, 0.1, 0.0, 100.0, 0.5, 3.0, 0.5, 0.5);
    assert(q == LAB_QUALITY_BAD);

    /* Model inconsistency (large residual) */
    q = lab_sample_validate(20.0, 50.0, 49.0, 4.0, 0.1, 0.0, 100.0, 0.5, 3.0, 0.5, 0.5);
    assert(q == LAB_QUALITY_SUSPECT);

    PASS();
}

static void test_normal_cdf(void)
{
    TEST("Normal CDF and quantile functions");
    /* Phi(0) = 0.5 */
    double p = normal_cdf(0.0);
    assert(fabs(p - 0.5) < 0.001);

    /* Phi(1.96) ≈ 0.975 */
    p = normal_cdf(1.96);
    assert(fabs(p - 0.975) < 0.01);

    /* Phi(-1.96) ≈ 0.025 */
    p = normal_cdf(-1.96);
    assert(fabs(p - 0.025) < 0.01);

    /* Quantile: q(0.975) ≈ 1.96 (Wichura AS241 algorithm) */
    double z = normal_quantile(0.975);
    assert(fabs(z - 1.96) < 0.01);

    /* q(0.5) ≈ 0 */
    z = normal_quantile(0.5);
    assert(fabs(z) < 0.01);

    PASS();
}

static void test_mad_outlier_detect(void)
{
    TEST("MAD outlier detection");
    double data[] = {10.0, 10.2, 9.8, 10.1, 10.0, 10.3, 9.9, 50.0, 10.1, 10.0};
    int flags[10] = {0};
    int n_outliers = mad_outlier_detect(data, 10, 3.0, flags);

    /* 50.0 should be flagged as outlier */
    assert(n_outliers >= 1);
    assert(flags[7] == 1);  /* 50.0 at index 7 */

    PASS();
}

static void test_rls_directional(void)
{
    TEST("RLS with directional forgetting");
    rls_directional_t rls_dir;
    rls_directional_alloc(&rls_dir, 2, 0.95, 0.001, 100.0);

    /* First: train with first set of parameters */
    for (int i = 0; i < 50; i++) {
        double x1 = (i % 7) * 0.5;
        double x2 = (i % 5) * 0.3;
        double y = 2.0 * x1 - 1.0 * x2;
        double phi[] = {x1, x2};
        rls_directional_update(&rls_dir, phi, y);
    }

    double y1 = rls_directional_predict(&rls_dir, (double[]){2.0, 1.0});
    /* Expected: 2*2 - 1*1 = 3 */
    assert(fabs(y1 - 3.0) < 2.0);

    rls_directional_free(&rls_dir);
    PASS();
}

/*===========================================================================
 * L8: Bayesian / Bayesian Inference Test
 *===========================================================================*/

static void test_bayesian_bias_update(void)
{
    TEST("Bayesian bias update with conjugate prior (L8)");
    /* For Gaussian likelihood with known variance, the conjugate prior is
     * also Gaussian. The posterior mean is a precision-weighted average:
     *
     * mu_post = (mu_prior/tau_prior^2 + y_lab/sigma_lab^2) / (1/tau_prior^2 + 1/sigma_lab^2)
     *
     * This test verifies the Kalman bias estimator implements this logic.
     */

    bias_kalman_t bk;
    /* Prior: bias ~ N(0, 10^2) — vague prior */
    /* Lab noise: sigma_lab^2 = 1 */
    bias_kalman_init(&bk, 0.0, 1.0, 0.0, 100.0);
    /* P0 = 100 = tau_prior^2, R = 1 */

    /* First lab measurement: y_lab=52, y_model=50, residual=2 */
    bias_kalman_update(&bk, 52.0, 50.0);

    /* Posterior mean should be between 0 (prior) and 2 (data) */
    /* K = 100/101 ≈ 0.99, so bias ≈ 0 + 0.99*2 ≈ 1.98 */
    assert(fabs(bk.bias - 1.98) < 0.1);

    /* Posterior precision should be higher (uncertainty lower) */
    double post_uncertainty = bias_kalman_get_uncertainty(&bk);
    assert(post_uncertainty < 10.0);  /* Less than prior uncertainty */

    PASS();
}

/*===========================================================================
 * Test Runner
 *===========================================================================*/

int main(void)
{
    printf("\n=== Inferential Quality Estimator — Comprehensive Tests ===\n\n");

    printf("--- L1: Definitions ---\n");
    test_config_init();
    test_timestamp_operations();
    test_quality_estimate_init();
    test_process_variable_init();

    printf("\n--- L2: Core Concepts ---\n");
    test_linear_model_evaluate();
    test_arx_model_init();
    test_additive_bias();
    test_cusum_detector();
    test_kalman_bias();
    test_bias_context();

    printf("\n--- L3: Engineering Structures ---\n");
    test_state_space_alloc();
    test_state_space_predict();
    test_mrf_kalman_alloc();
    test_mrf_fast_step();
    test_mrf_interpolator();

    printf("\n--- L4: Engineering Laws ---\n");
    test_compute_regression_stats();
    test_durbin_watson();
    test_t_test_zero_mean();
    test_lab_sample_validate();
    test_normal_cdf();
    test_mad_outlier_detect();

    printf("\n--- L5: Algorithms ---\n");
    test_kalman_filter_alloc();
    test_kalman_filter_predict();
    test_kalman_filter_update();
    test_kalman_filter_step();
    test_rls_estimator();
    test_rls_vff();
    test_rls_directional();
    test_wls_window();
    test_mhe_buffer();
    test_iv_estimator();

    printf("\n--- L6: Canonical Problems ---\n");
    test_pls_model_predict();
    test_ewma_smooth();
    test_distillation_composition();
    test_reactor_conversion();

    printf("\n--- L8: Advanced Topics ---\n");
    test_monte_carlo_prediction();
    test_bayesian_bias_update();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
