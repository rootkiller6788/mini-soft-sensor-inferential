/**
 * @file    aging_detection.c
 * @brief   Aging detection algorithms - CUSUM, SPRT, Mann-Kendall, window comparison
 *
 * L5: CUSUM chart, SPRT sequential test, Mann-Kendall trend, F-test variance ratio
 * L6: Distillation column composition sensor aging detection
 *
 * Ref: Montgomery (2013) "Statistical Quality Control" 7th ed.
 *      Page (1954) Biometrika 41, 100-115.
 *      Mann (1945) Econometrica 13, 245-259.
 */

#include "aging_detection.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ====================================================================
 * L5: CUSUM Chart (two-sided)
 * ==================================================================== */

void cusum_chart_init(CUSUMChart *chart, double target_mean,
                       double sigma, double k, double h)
{
    if (!chart) return;
    memset(chart, 0, sizeof(*chart));
    chart->target_mean = target_mean;
    chart->sigma = sigma;
    chart->k = k;
    chart->h = h;
}

void cusum_chart_update(CUSUMChart *chart, double x)
{
    if (!chart) return;
    double dev = x - chart->target_mean;
    chart->cusum_high = fmax(0.0, chart->cusum_high + dev - chart->k);
    chart->cusum_low  = fmax(0.0, chart->cusum_low - dev - chart->k);
    chart->alarm_high = (chart->cusum_high > chart->h) ? 1 : 0;
    chart->alarm_low  = (chart->cusum_low > chart->h) ? 1 : 0;
    chart->run_length++;
}

void cusum_chart_reset(CUSUMChart *chart)
{
    if (!chart) return;
    chart->cusum_high = 0.0;
    chart->cusum_low = 0.0;
    chart->alarm_high = 0;
    chart->alarm_low = 0;
    chart->run_length = 0;
}

/* ====================================================================
 * L5: SPRT (Sequential Probability Ratio Test)
 *
 * Wald (1945) "Sequential tests of statistical hypotheses"
 * Annals of Mathematical Statistics, 16(2), 117-186.
 *
 * H0: observations ~ N(mu0, sigma^2)  (no aging)
 * H1: observations ~ N(mu1, sigma^2)  (aged)
 *
 * Log-likelihood ratio for Gaussian:
 *   LLR = sum [-(x_i - mu1)^2/(2*sigma^2) + (x_i - mu0)^2/(2*sigma^2)]
 *       = (mu1 - mu0)/sigma^2 * sum(x_i - (mu0 + mu1)/2)
 *
 * Boundaries:
 *   A = (1 - beta) / alpha   -> log(A)
 *   B = beta / (1 - alpha)   -> log(B)
 *
 * Where alpha = P(false alarm), beta = P(missed detection).
 * ==================================================================== */

void sprt_init(SPRTState *sprt, double h0_mean, double h1_mean,
                double sigma, double alpha, double beta)
{
    if (!sprt) return;
    memset(sprt, 0, sizeof(*sprt));
    sprt->h0_mean = h0_mean;
    sprt->h1_mean = h1_mean;
    sprt->sigma = sigma;
    sprt->alpha = alpha;
    sprt->beta = beta;
    sprt->log_a = log((1.0 - beta) / fmax(alpha, 1e-10));
    sprt->log_b = log(beta / fmax(1.0 - alpha, 1e-10));
    sprt->llr = 0.0;
    sprt->n_samples = 0;
    sprt->decision = 0;
}

int sprt_update(SPRTState *sprt, double x)
{
    if (!sprt) return 0;
    if (sprt->decision != 0) return sprt->decision;

    sprt->n_samples++;

    /* Gaussian log-likelihood ratio contribution */
    double var = sprt->sigma * sprt->sigma;
    if (var < 1e-15) var = 1e-15;
    double mid = (sprt->h0_mean + sprt->h1_mean) / 2.0;

    /* LLR increment: (mu1 - mu0) * (x - mid) / sigma^2 */
    double inc = (sprt->h1_mean - sprt->h0_mean) * (x - mid) / var;
    sprt->llr += inc;

    if (sprt->llr >= sprt->log_a) {
        sprt->decision = 1;  /* Reject H0: aging detected */
        return 1;
    }
    if (sprt->llr <= sprt->log_b) {
        sprt->decision = -1; /* Accept H0: no aging */
        return -1;
    }

    return 0; /* Continue sampling */
}

void sprt_reset(SPRTState *sprt)
{
    if (!sprt) return;
    sprt->llr = 0.0;
    sprt->n_samples = 0;
    sprt->decision = 0;
}

/* ====================================================================
 * L5: Mann-Kendall Non-Parametric Trend Test
 *
 * Mann (1945) Econometrica 13, 245-259.
 * Kendall (1975) "Rank Correlation Methods".
 *
 * S = sum_{i < j} sign(x_j - x_i)
 *
 * For n >= 10, S is approximately normal with:
 *   E[S] = 0
 *   Var[S] = [n(n-1)(2n+5) - sum(tp*(tp-1)*(2*tp+5))] / 18
 *
 * Z = (S - sign(S)) / sqrt(Var[S])
 *
 * Sen slope (robust trend magnitude):
 *   beta = median{ (x_j - x_i) / (j - i) } for all i < j
 * ==================================================================== */

/* Comparison function for qsort */
static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

/* sign function */
static int sign_func(double x)
{
    if (x > 0.0) return 1;
    if (x < 0.0) return -1;
    return 0;
}

MannKendallResult mann_kendall_test(const double *data, size_t n)
{
    MannKendallResult result;
    memset(&result, 0, sizeof(result));

    if (!data || n < 3) return result;

    /* Compute S statistic */
    long long S = 0;
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = i + 1; j < n; j++) {
            S += sign_func(data[j] - data[i]);
        }
    }

    /* Compute variance of S with tie correction */
    /* Count tied groups via sorting */
    double *sorted = (double *)malloc(n * sizeof(double));
    if (!sorted) return result;
    memcpy(sorted, data, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);

    double tie_correction = 0.0;
    size_t tie_start = 0;
    for (size_t i = 1; i <= n; i++) {
        if (i == n || fabs(sorted[i] - sorted[tie_start]) > 1e-12) {
            size_t tp = i - tie_start;
            if (tp > 1) {
                tie_correction += (double)(tp * (tp - 1) * (2 * tp + 5));
            }
            tie_start = i;
        }
    }
    free(sorted);

    double var_S = ((double)n * (n - 1.0) * (2.0 * n + 5.0) - tie_correction) / 18.0;
    result.var_s = var_S;

    /* Z-score with continuity correction */
    double S_dbl = (double)S;
    if (S > 0)
        result.z_score = (S_dbl - 1.0) / sqrt(var_S);
    else if (S < 0)
        result.z_score = (S_dbl + 1.0) / sqrt(var_S);
    else
        result.z_score = 0.0;

    /* Two-tailed p-value from normal approximation */
    double abs_z = fabs(result.z_score);
    result.p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs_z / sqrt(2.0))));
    if (result.p_value > 1.0) result.p_value = 1.0;

    /* Sen slope estimator */
    size_t n_pairs = (n * (n - 1)) / 2;
    if (n_pairs > 0) {
        double *slopes = (double *)malloc(n_pairs * sizeof(double));
        if (slopes) {
            size_t idx = 0;
            for (size_t i = 0; i < n - 1; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    slopes[idx++] = (data[j] - data[i]) / (double)(j - i);
                }
            }
            qsort(slopes, n_pairs, sizeof(double), cmp_double);
            /* Median: middle element(s) */
            if (n_pairs % 2 == 1) {
                result.sen_slope = slopes[n_pairs / 2];
            } else {
                result.sen_slope = (slopes[n_pairs / 2 - 1] + slopes[n_pairs / 2]) / 2.0;
            }
            free(slopes);
        }
    }

    result.s_statistic = S_dbl;
    result.trend_detected = (result.p_value < 0.05) ? 1 : 0;
    result.trend_direction = (S > 0) ? 1 : ((S < 0) ? -1 : 0);

    return result;
}

/* ====================================================================
 * L5: Moving Window Comparison (Welch t-test)
 *
 * Welch, B.L. (1947) "The generalization of Student's problem
 * when several different population variances are involved."
 * Biometrika, 34(1/2), 28-35.
 *
 * t = (xbar1 - xbar2) / sqrt(s1^2/n1 + s2^2/n2)
 * df = (s1^2/n1 + s2^2/n2)^2
 *      / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]
 * ==================================================================== */

WindowComparison compare_windows(const double *baseline_data, size_t baseline_n,
                                  const double *current_data, size_t current_n,
                                  double alpha)
{
    WindowComparison wc;
    memset(&wc, 0, sizeof(wc));

    if (!baseline_data || !current_data || baseline_n < 2 || current_n < 2)
        return wc;

    /* Baseline statistics */
    double sum1 = 0.0, sum1_sq = 0.0;
    for (size_t i = 0; i < baseline_n; i++) {
        sum1 += baseline_data[i];
        sum1_sq += baseline_data[i] * baseline_data[i];
    }
    wc.baseline_mean = sum1 / (double)baseline_n;
    double var1 = (sum1_sq - sum1 * sum1 / (double)baseline_n) / (double)(baseline_n - 1);
    if (var1 < 0.0) var1 = 0.0;

    /* Current statistics */
    double sum2 = 0.0, sum2_sq = 0.0;
    for (size_t i = 0; i < current_n; i++) {
        sum2 += current_data[i];
        sum2_sq += current_data[i] * current_data[i];
    }
    wc.current_mean = sum2 / (double)current_n;
    double var2 = (sum2_sq - sum2 * sum2 / (double)current_n) / (double)(current_n - 1);
    if (var2 < 0.0) var2 = 0.0;

    wc.mean_difference = wc.current_mean - wc.baseline_mean;

    /* Welch t-statistic */
    double se1 = var1 / (double)baseline_n;
    double se2 = var2 / (double)current_n;
    double se_pooled = sqrt(se1 + se2);

    if (se_pooled < 1e-15) return wc;

    wc.t_statistic = wc.mean_difference / se_pooled;

    /* Welch-Satterthwaite degrees of freedom */
    double num = (se1 + se2) * (se1 + se2);
    double den = (se1 * se1) / (double)(baseline_n - 1)
                 + (se2 * se2) / (double)(current_n - 1);
    double df_welch = (den > 1e-15) ? num / den : (double)(baseline_n + current_n - 2);

    /* p-value from t-distribution */
    double abs_t = fabs(wc.t_statistic);
    wc.p_value = compute_p_value_from_t(abs_t, df_welch);

    wc.significant_change = (wc.p_value < alpha) ? 1 : 0;

    /* Cohen's d effect size */
    double sd_pooled = sqrt(((double)(baseline_n - 1) * var1
                              + (double)(current_n - 1) * var2)
                             / (double)(baseline_n + current_n - 2));
    if (sd_pooled > 1e-15)
        wc.effect_size = fabs(wc.mean_difference) / sd_pooled;

    return wc;
}

/* ====================================================================
 * L5: Variance Ratio Test (F-test for precision degradation)
 *
 * H0: sigma_current^2 <= sigma_baseline^2 (no precision loss)
 * H1: sigma_current^2 > sigma_baseline^2 (precision degraded)
 *
 * F = s_current^2 / s_baseline^2
 * Critical value: F_{alpha, n_cur-1, n_base-1}
 * ==================================================================== */

VarianceRatioTest variance_ratio_test(double baseline_var, size_t baseline_n,
                                       double current_var, size_t current_n,
                                       double alpha)
{
    VarianceRatioTest vrt;
    memset(&vrt, 0, sizeof(vrt));
    vrt.baseline_var = baseline_var;
    vrt.current_var = current_var;

    if (baseline_var < 1e-15) {
        vrt.variance_increased = (current_var > 1e-15) ? 1 : 0;
        vrt.f_statistic = (current_var > 1e-15) ? 1e10 : 1.0;
        return vrt;
    }

    vrt.f_statistic = current_var / baseline_var;

    /* Critical F-value for one-tailed test */
    vrt.critical_f = f_distribution_critical_value(1.0 - alpha,
                                                     (double)(current_n - 1),
                                                     (double)(baseline_n - 1));

    vrt.variance_increased = (vrt.f_statistic > vrt.critical_f) ? 1 : 0;

    /* Approximate p-value: using relationship between F and beta distribution */
    /* Simplified: if F > 2.0, p < 0.05 for moderate sample sizes */
    if (vrt.f_statistic > vrt.critical_f) {
        vrt.p_value = 0.01;
    } else if (vrt.f_statistic > 1.0) {
        vrt.p_value = 0.5 / vrt.f_statistic;
    } else {
        vrt.p_value = 0.5;
    }

    return vrt;
}

/* ====================================================================
 * L5: Aging Detection Fusion
 *
 * Combines multiple statistical tests into a single aging verdict
 * using weighted voting. Each test contributes according to its
 * configured weight.
 *
 * aging_confidence = sum(w_i * alarm_i) where alarm_i in {0, 1}
 *
 * If combined confidence > 0.5, overall_aging_detected = 1.
 *
 * Weights should reflect test reliability in the specific application:
 *   - CUSUM: good for sustained shifts, weight 0.30
 *   - SPRT:  fastest detection, weight 0.25
 *   - Mann-Kendall: best for monotonic trends, weight 0.20
 *   - Window comparison: robust comparison, weight 0.15
 *   - Variance ratio: early warning, weight 0.10
 * ==================================================================== */

AgingVerdict fuse_aging_detection(int cusum, int sprt, int mk,
                                   int wc, int vr,
                                   const double weights[5])
{
    AgingVerdict verdict;
    memset(&verdict, 0, sizeof(verdict));

    verdict.cusum_alarm = cusum;
    verdict.sprt_alarm = sprt;
    verdict.mann_kendall_alarm = mk;
    verdict.window_comparison_alarm = wc;
    verdict.variance_ratio_alarm = vr;

    double confidence = weights[0] * (double)cusum
                       + weights[1] * (double)sprt
                       + weights[2] * (double)mk
                       + weights[3] * (double)wc
                       + weights[4] * (double)vr;

    verdict.aging_confidence = confidence;
    verdict.overall_aging_detected = (confidence > 0.5) ? 1 : 0;

    /* Recommended action:
     * 0 = no action, 1 = increase monitoring,
     * 2 = schedule maintenance, 3 = urgent maintenance */
    if (confidence > 0.8) verdict.recommended_action = 3.0;
    else if (confidence > 0.6) verdict.recommended_action = 2.0;
    else if (confidence > 0.4) verdict.recommended_action = 1.0;
    else verdict.recommended_action = 0.0;

    return verdict;
}
