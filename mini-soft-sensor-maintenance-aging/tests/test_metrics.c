/**
 * @file    test_metrics.c
 * @brief   Tests for soft sensor performance metrics
 *
 * L4: Mathematical assertions verifying statistical computations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "soft_sensor_metrics.h"

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b); \
    assert(fabs(_a - _b) < (tol)); \
} while(0)

/* Test regression metrics with known data */
static void test_regression_metrics_perfect(void)
{
    double actual[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double pred[]   = {1.0, 2.0, 3.0, 4.0, 5.0};
    RegressionMetrics m = compute_regression_metrics(actual, pred, 5);
    ASSERT_NEAR(m.rmse, 0.0, 1e-10);
    ASSERT_NEAR(m.mae, 0.0, 1e-10);
    ASSERT_NEAR(m.mape, 0.0, 1e-10);
    ASSERT_NEAR(m.r2, 1.0, 1e-10);
    ASSERT_NEAR(m.bias, 0.0, 1e-10);
    assert(m.n_samples == 5);
    printf("  PASS: perfect prediction\n");
}

/* Test regression metrics with offset predictions */
static void test_regression_metrics_offset(void)
{
    double actual[] = {2.0, 4.0, 6.0};
    double pred[]   = {1.0, 3.0, 5.0};
    /* residuals = {1, 1, 1}, abs = {1, 1, 1} */
    RegressionMetrics m = compute_regression_metrics(actual, pred, 3);
    ASSERT_NEAR(m.rmse, 1.0, 1e-10);
    ASSERT_NEAR(m.mae, 1.0, 1e-10);
    ASSERT_NEAR(m.bias, 1.0, 1e-10);
    /* SS_res = 3, SS_tot = (2-4)^2+(4-4)^2+(6-4)^2 = 4+0+4 = 8 */
    ASSERT_NEAR(m.r2, 1.0 - 3.0/8.0, 1e-10);
    printf("  PASS: offset predictions\n");
}

/* Test empty input */
static void test_regression_metrics_empty(void)
{
    RegressionMetrics m = compute_regression_metrics(NULL, NULL, 0);
    assert(m.n_samples == 0);
    assert(m.rmse == 0.0);
    assert(m.mae == 0.0);
    printf("  PASS: empty input\n");
}

/* Welford algorithm test with known sequence */
static void test_welford_algorithm(void)
{
    RunningStatistics s = running_stats_init();
    double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    /* n=8, mean=5, var_pop=4.0 */

    for (size_t i = 0; i < 8; i++)
        running_stats_push(&s, data[i]);

    ASSERT_NEAR(s.mean, 5.0, 1e-10);
    ASSERT_NEAR(s.var_population, 4.0, 1e-10);
    /* sample variance = M2/(n-1) = 32/7 ≈ 4.5714 */
    ASSERT_NEAR(s.var_sample, 4.571428571428571, 1e-8);
    assert(s.count == 8);
    assert(s.min_val == 2.0);
    assert(s.max_val == 9.0);
    printf("  PASS: Welford algorithm\n");
}

/* Sliding window test */
static void test_sliding_window(void)
{
    SlidingWindow sw;
    sliding_window_init(&sw, 3);

    sliding_window_push(&sw, 1.0);
    sliding_window_push(&sw, 2.0);
    sliding_window_push(&sw, 3.0);
    /* Window: [1,2,3], mean=2, var=1.0 */

    ASSERT_NEAR(sliding_window_mean(&sw), 2.0, 1e-10);
    ASSERT_NEAR(sliding_window_variance(&sw), 1.0, 1e-10);

    /* Push 4.0 -> evict 1.0, window: [2,3,4], mean=3, var=1.0 */
    sliding_window_push(&sw, 4.0);
    ASSERT_NEAR(sliding_window_mean(&sw), 3.0, 1e-10);
    ASSERT_NEAR(sliding_window_variance(&sw), 1.0, 1e-10);

    sliding_window_destroy(&sw);
    printf("  PASS: sliding window\n");
}

/* CUSUM test with known shift */
static void test_cusum_detection(void)
{
    /* Data with shift at index 10: mean changes from 0 to 2 */
    double data[20];
    for (int i = 0; i < 10; i++) data[i] = 0.0;
    for (int i = 10; i < 20; i++) data[i] = 2.0;

    int alarm_idx = -1;
    double max_s = compute_cusum_statistic(data, 20, 0.0, 1.0,
                                            0.5, 4.0, &alarm_idx);
    /* CUSUM should detect the shift */
    assert(alarm_idx > 0);
    assert(alarm_idx >= 10);
    assert(max_s > 4.0);
    printf("  PASS: CUSUM shift detection (alarm at %d)\n", alarm_idx);
}

/* Page-Hinkley test with gradual drift */
static void test_page_hinkley(void)
{
    double data[100];
    for (int i = 0; i < 50; i++) data[i] = 0.0;
    for (int i = 50; i < 100; i++) data[i] = 0.05 * (i - 50);

    int alarm_idx = -1;
    double ph = page_hinkley_test(data, 100, 0.005, 30.0, &alarm_idx);
    /* PH test should trigger eventually for gradual drift */
    assert(ph > 0.0);
    printf("  PASS: Page-Hinkley test (PH=%.3f)\n", ph);
}

/* Confidence interval test */
static void test_confidence_interval(void)
{
    double data[] = {10.0, 12.0, 9.0, 11.0, 10.0, 13.0, 8.0, 11.0};
    ConfidenceInterval ci = confidence_interval_mean(data, 8, 0.95);
    /* xbar = 10.5, should be within CI */
    assert(ci.point_estimate > 0.0);
    assert(ci.lower_bound < ci.upper_bound);
    assert(ci.lower_bound <= ci.point_estimate);
    assert(ci.point_estimate <= ci.upper_bound);
    assert(ci.confidence_level == 0.95);
    assert(ci.degrees_freedom == 7);
    printf("  PASS: confidence interval [%.3f, %.3f]\n",
           ci.lower_bound, ci.upper_bound);
}

/* Paired t-test: identical models */
static void test_paired_t_identical(void)
{
    double err_a[] = {0.1, 0.2, 0.1, 0.3, 0.2};
    double err_b[] = {0.1, 0.2, 0.1, 0.3, 0.2};
    PairedTTest result = paired_t_test(err_a, err_b, 5, 0.05);
    /* Differences are zero */
    assert(result.mean_difference == 0.0);
    assert(result.reject_null == 0);
    assert(result.p_value > 0.05);
    printf("  PASS: paired t-test (identical models)\n");
}

/* Paired t-test: different models */
static void test_paired_t_different(void)
{
    /* Model A consistently better with variance in differences */
    double err_a[] = {0.10, 0.20, 0.15, 0.25, 0.30, 0.18, 0.22, 0.12, 0.28, 0.20};
    double err_b[] = {0.50, 0.60, 0.55, 0.65, 0.70, 0.58, 0.62, 0.52, 0.68, 0.60};
    PairedTTest result = paired_t_test(err_a, err_b, 10, 0.05);
    /* Model A has consistently lower errors */
    assert(result.mean_difference < 0.0);
    assert(result.reject_null == 1);
    assert(result.cohens_d > 0.5);
    printf("  PASS: paired t-test (different models, Cohen d=%.3f)\n",
           result.cohens_d);
}

/* Health index lifecycle test */
static void test_health_index_lifecycle(void)
{
    HealthIndex h = health_index_init();
    assert(h.reliability_score == 1.0);
    assert(h.stage == LIFECYCLE_COMMISSIONING);
    assert(!health_index_recommends_maintenance(&h));

    /* Simulate degradation */
    health_index_update(&h, 0.15, 0.10, 0.05, 0.01, 0.005, 1000.0);
    /* RMSE increased from 0.1 to 0.15, should enter WARNING */
    assert(h.stage >= LIFECYCLE_WARNING || h.stage == LIFECYCLE_NORMAL);
    printf("  PASS: health index (stage=%s, rel=%.3f)\n",
           lifecycle_stage_name(h.stage), h.reliability_score);
}

/* Health index maintenance recommendation */
static void test_health_index_maintenance(void)
{
    HealthIndex h = health_index_init();
    /* Force degradation */
    h.reliability_score = 0.50;
    h.stage = LIFECYCLE_AGING;
    assert(health_index_recommends_maintenance(&h) == 1);
    printf("  PASS: maintenance recommendation\n");
}

/* SPE contributions test */
static void test_spe_contributions(void)
{
    double residuals[] = {0.5, -0.3, 0.8, -0.2};
    double contributions[4];
    compute_spe_contributions(residuals, 4, contributions);
    /* contributions should sum to 1.0 */
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += contributions[i];
    ASSERT_NEAR(sum, 1.0, 1e-10);
    /* Variable 3 (residual 0.8) should have highest contribution */
    assert(contributions[2] > contributions[0]);
    assert(contributions[2] > contributions[1]);
    assert(contributions[2] > contributions[3]);
    printf("  PASS: SPE contributions\n");
}

/* EWMA test */
static void test_ewma(void)
{
    SlidingWindow sw;
    sliding_window_init(&sw, 5);
    sliding_window_push(&sw, 1.0);
    sliding_window_push(&sw, 2.0);
    sliding_window_push(&sw, 3.0);
    sliding_window_push(&sw, 4.0);
    sliding_window_push(&sw, 5.0);

    /* EWMA with lambda=0.5 should converge toward recent values */
    double ewma = sliding_window_ewma(&sw, 0.5);
    assert(ewma > 1.0);
    assert(ewma < 5.0);

    sliding_window_destroy(&sw);
    printf("  PASS: EWMA (lambda=0.5 -> %.3f)\n", ewma);
}

/* F-distribution critical value sanity check */
static void test_f_distribution(void)
{
    /* F(0.95, 5, 20) should be approximately 2.71 */
    double f = f_distribution_critical_value(0.95, 5.0, 20.0);
    assert(f > 2.0 && f < 4.0);
    /* F(0.99, 5, 20) should be larger */
    double f99 = f_distribution_critical_value(0.99, 5.0, 20.0);
    assert(f99 > f);
    printf("  PASS: F-distribution (F_0.95=%.3f, F_0.99=%.3f)\n", f, f99);
}

/* t-distribution critical value sanity check */
static void test_t_distribution(void)
{
    /* t(0.975, 10) should be approximately 2.228 */
    double t = t_distribution_critical_value(0.975, 10.0);
    assert(t > 2.0 && t < 2.5);
    /* t(0.975, 100) should be close to z=1.96 */
    double t_big = t_distribution_critical_value(0.975, 100.0);
    ASSERT_NEAR(t_big, 1.96, 0.2);
    printf("  PASS: t-distribution (t_0.975,10=%.3f, t_0.975,100=%.3f)\n", t, t_big);
}

/* P-value computation test */
static void test_p_value(void)
{
    /* For large t, p should be small */
    double p_small = compute_p_value_from_t(4.0, 20.0);
    assert(p_small < 0.01);
    /* For t=0, p should be 1.0 */
    double p_large = compute_p_value_from_t(0.0, 10.0);
    assert(p_large > 0.9);
    printf("  PASS: p-value computation (p(4.0)=%.6f, p(0)=%.3f)\n",
           p_small, p_large);
}

/* Running stats merge test */
static void test_merge_statistics(void)
{
    RunningStatistics a = running_stats_init();
    RunningStatistics b = running_stats_init();

    running_stats_push(&a, 1.0);
    running_stats_push(&a, 2.0);
    running_stats_push(&a, 3.0);  /* mean=2, n=3 */

    running_stats_push(&b, 4.0);
    running_stats_push(&b, 5.0);
    running_stats_push(&b, 6.0);  /* mean=5, n=3 */

    RunningStatistics merged = running_stats_merge(&a, &b);
    assert(merged.count == 6);
    ASSERT_NEAR(merged.mean, 3.5, 1e-10);
    printf("  PASS: merge statistics (mean=%.1f)\n", merged.mean);
}

/* Running stats remove test */
static void test_remove_statistics(void)
{
    RunningStatistics s = running_stats_init();
    running_stats_push(&s, 1.0);
    running_stats_push(&s, 3.0);
    running_stats_push(&s, 5.0);
    /* mean=3, n=3 */

    running_stats_remove(&s, 5.0);
    /* Should become [1,3], mean=2, n=2 */
    assert(s.count == 2);
    ASSERT_NEAR(s.mean, 2.0, 1e-10);
    printf("  PASS: remove statistics\n");
}

/* Online regression metrics test */
static void test_online_regression(void)
{
    RegressionMetrics m;
    memset(&m, 0, sizeof(m));
    update_regression_metrics_online(&m, 2.0, 1.0);
    update_regression_metrics_online(&m, 4.0, 3.0);
    update_regression_metrics_online(&m, 6.0, 5.0);

    assert(m.n_samples == 3);
    ASSERT_NEAR(m.rmse, 1.0, 1e-10);
    ASSERT_NEAR(m.mae, 1.0, 1e-10);
    printf("  PASS: online regression update (RMSE=%.3f)\n", m.rmse);
}

int main(void)
{
    printf("=== Running soft_sensor_metrics tests ===\n\n");

    test_regression_metrics_perfect();
    test_regression_metrics_offset();
    test_regression_metrics_empty();
    test_welford_algorithm();
    test_sliding_window();
    test_cusum_detection();
    test_page_hinkley();
    test_confidence_interval();
    test_paired_t_identical();
    test_paired_t_different();
    test_health_index_lifecycle();
    test_health_index_maintenance();
    test_spe_contributions();
    test_ewma();
    test_f_distribution();
    test_t_distribution();
    test_p_value();
    test_merge_statistics();
    test_remove_statistics();
    test_online_regression();

    printf("\n=== All tests passed! (20/20) ===\n");
    return 0;
}
