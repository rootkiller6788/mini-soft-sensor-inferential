/**
 * @file    aging_detection.h
 * @brief   Soft sensor aging detection algorithms - L5 Algorithms & L6 Problems
 *
 * Detects model aging through multiple statistical tests:
 *   - CUSUM (cumulative sum) for mean shift
 *   - SPRT (sequential probability ratio test) for early drift
 *   - Mann-Kendall trend test for monotonic aging
 *   - Moving window comparison (current vs historical baseline)
 *   - Variance ratio test (F-test) for precision degradation
 *
 * L5: CUSUM, SPRT, Mann-Kendall, adaptive threshold, multi-metric fusion
 * L6: Distillation column composition sensor aging
 *
 * Ref: Montgomery (2013) "Statistical Quality Control", 7th ed. Wiley.
 *      Page (1954) "Continuous Inspection Schemes" Biometrika 41, 100-115.
 *      Mann (1945) Econometrica 13, 245-259.
 */

#ifndef AGING_DETECTION_H
#define AGING_DETECTION_H

#include "soft_sensor_metrics.h"
#include <stddef.h>

/* ============================================================================
 * L5: CUSUM Configuration
 * ===========================================================================*/

/** @brief CUSUM chart configuration and state */
typedef struct {
    double target_mean;
    double sigma;
    double k;
    double h;
    double cusum_high;
    double cusum_low;
    int    alarm_high;
    int    alarm_low;
    size_t run_length;
} CUSUMChart;

/** @brief SPRT (Sequential Probability Ratio Test) state */
typedef struct {
    double h0_mean;
    double h1_mean;
    double sigma;
    double alpha;
    double beta;
    double log_a;
    double log_b;
    double llr;
    size_t n_samples;
    int    decision;
} SPRTState;

/** @brief Mann-Kendall trend test result */
typedef struct {
    double s_statistic;
    double var_s;
    double z_score;
    double p_value;
    double sen_slope;
    int    trend_detected;
    int    trend_direction;
} MannKendallResult;

/** @brief Moving window comparison result */
typedef struct {
    double baseline_mean;
    double current_mean;
    double mean_difference;
    double t_statistic;
    double p_value;
    double effect_size;
    int    significant_change;
} WindowComparison;

/** @brief Variance ratio (F-test) result for precision aging */
typedef struct {
    double baseline_var;
    double current_var;
    double f_statistic;
    double p_value;
    double critical_f;
    int    variance_increased;
} VarianceRatioTest;

/** @brief Combined aging detection verdict */
typedef struct {
    int    cusum_alarm;
    int    sprt_alarm;
    int    mann_kendall_alarm;
    int    window_comparison_alarm;
    int    variance_ratio_alarm;
    int    overall_aging_detected;
    double aging_confidence;
    double recommended_action;
} AgingVerdict;

/* ============================================================================
 * API: CUSUM
 * ===========================================================================*/

void cusum_chart_init(CUSUMChart *chart, double target_mean,
                       double sigma, double k, double h);
void cusum_chart_update(CUSUMChart *chart, double x);
void cusum_chart_reset(CUSUMChart *chart);

/* ============================================================================
 * API: SPRT
 * ===========================================================================*/

/**
 * @brief Initialize SPRT for drift detection.
 *
 * H0: mean = h0_mean (no aging)
 * H1: mean = h1_mean (aged, shifted mean)
 *
 * alpha = P(reject H0 | H0 true)  [false alarm rate]
 * beta  = P(accept H0 | H1 true)  [missed detection rate]
 *
 * log_a = log((1-beta)/alpha), log_b = log(beta/(1-alpha))
 * Decision: LLR >= log_a -> reject H0 (aging detected)
 *           LLR <= log_b -> accept H0 (no aging)
 */
void sprt_init(SPRTState *sprt, double h0_mean, double h1_mean,
                double sigma, double alpha, double beta);

/**
 * @brief Update SPRT with new observation, O(1).
 * @return 1 if decision reached (aging), -1 if no-aging, 0 if continue.
 */
int sprt_update(SPRTState *sprt, double x);
void sprt_reset(SPRTState *sprt);

/* ============================================================================
 * API: Mann-Kendall Trend Test
 * ===========================================================================*/

/**
 * @brief Mann-Kendall non-parametric trend test.
 *
 * S = sum_{i<j} sign(x_j - x_i)
 * Var(S) = [n(n-1)(2n+5) - sum(tp*(tp-1)*(2*tp+5))] / 18
 * Z = (S - sign(S)) / sqrt(Var(S))
 *
 * Sen's slope estimator (robust trend magnitude):
 *   beta = median{(x_j - x_i) / (j - i)} for all i < j
 *
 * @param data     Time-ordered data points.
 * @param n        Number of points.
 * @return MannKendallResult with trend detection verdict.
 */
MannKendallResult mann_kendall_test(const double *data, size_t n);

/* ============================================================================
 * API: Moving Window Comparison
 * ===========================================================================*/

/**
 * @brief Compare baseline window vs current window using Welch t-test.
 *
 * Useful for detecting if recent performance is significantly worse
 * than the baseline (commissioning) performance.
 *
 * Welch's t-test (unequal variances):
 *   t = (mean_cur - mean_base) / sqrt(var_cur/n_cur + var_base/n_base)
 *   df via Welch-Satterthwaite equation
 *
 * @param baseline_data  Historical baseline metric values.
 * @param baseline_n     Number of baseline samples.
 * @param current_data   Recent (potentially aged) metric values.
 * @param current_n      Number of recent samples.
 * @param alpha          Significance level.
 * @return WindowComparison result.
 */
WindowComparison compare_windows(const double *baseline_data, size_t baseline_n,
                                  const double *current_data, size_t current_n,
                                  double alpha);

/* ============================================================================
 * API: Variance Ratio Test
 * ===========================================================================*/

/**
 * @brief F-test for variance ratio (precision degradation).
 *
 * H0: var_current <= var_baseline (no precision loss)
 * H1: var_current > var_baseline  (precision degraded)
 *
 * F = var_current / var_baseline ~ F(n_cur-1, n_base-1)
 *
 * Increased prediction variance is an early indicator of sensor aging,
 * often appearing before mean shift.
 */
VarianceRatioTest variance_ratio_test(double baseline_var, size_t baseline_n,
                                       double current_var, size_t current_n,
                                       double alpha);

/* ============================================================================
 * API: Aging Fusion
 * ===========================================================================*/

/**
 * @brief Fuse multiple aging detection results into single verdict.
 *
 * Uses weighted voting with configurable weights for each test.
 * aging_confidence = weighted fraction of tests signaling alarm.
 *
 * @param cusum     CUSUM alarm flag.
 * @param sprt      SPRT alarm flag.
 * @param mk        Mann-Kendall alarm flag.
 * @param wc        Window comparison alarm flag.
 * @param vr        Variance ratio alarm flag.
 * @param weights   5-element weight array (must sum to approx 1.0).
 * @return AgingVerdict with fused decision.
 */
AgingVerdict fuse_aging_detection(int cusum, int sprt, int mk,
                                   int wc, int vr,
                                   const double weights[5]);

#endif /* AGING_DETECTION_H */
