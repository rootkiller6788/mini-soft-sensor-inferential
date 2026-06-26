/**
 * @file    soft_sensor_metrics.c
 * @brief   Implementation of soft sensor performance metrics.
 * L1: RMSE, MAE, MAPE, R2. L3: Welford stats, sliding window, EWMA.
 * L4: Jackson-Mudholkar SPE limits, F-distribution T2 limits.
 */
#include "soft_sensor_metrics.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ====================================================================
 * L1: Regression Metrics - Batch Computation
 * RMSE = sqrt( (1/n)*sum((y_i - yhat_i)^2) )
 * MAE  = (1/n)*sum(|y_i - yhat_i|)
 * R^2  = 1 - SS_res / SS_tot
 * MAPE = (100/n)*sum(|e_i/y_i|)
 * Edge cases: n==0 (zero), n==1 (R2=NAN), actual[i]==0 (MAPE clamp)
 * ==================================================================== */

RegressionMetrics compute_regression_metrics(const double *actual,
                                              const double *predicted,
                                              size_t n)
{
    RegressionMetrics m;
    memset(&m, 0, sizeof(m));
    if (n == 0) return m;
    if (!actual || !predicted) return m;

    double sum_actual = 0.0, sum_sq_err = 0.0, sum_abs_err = 0.0;
    double sum_mape = 0.0, max_abs_err = 0.0, sum_residual = 0.0;

    for (size_t i = 0; i < n; i++) {
        double residual = actual[i] - predicted[i];
        double abs_err = fabs(residual);
        sum_actual += actual[i];
        sum_sq_err += residual * residual;
        sum_abs_err += abs_err;
        sum_residual += residual;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (fabs(actual[i]) > 1e-10)
            sum_mape += abs_err / fabs(actual[i]);
    }

    double mean_actual = sum_actual / (double)n;
    double ss_tot = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = actual[i] - mean_actual;
        ss_tot += diff * diff;
    }

    m.n_samples = n;
    m.mse = sum_sq_err / (double)n;
    m.rmse = sqrt(m.mse);
    m.mae = sum_abs_err / (double)n;
    m.bias = sum_residual / (double)n;
    m.max_error = max_abs_err;
    m.mape = (sum_mape / (double)n) * 100.0;

    if (ss_tot < 1e-15)
        m.r2 = (m.mse < 1e-15) ? 1.0 : NAN;
    else {
        m.r2 = 1.0 - (sum_sq_err / ss_tot);
        if (m.r2 > 1.0) m.r2 = 1.0;
    }
    return m;
}

/* ====================================================================
 * L1: Online Regression Metrics Update - O(1) incremental formulas
 * ==================================================================== */

void update_regression_metrics_online(RegressionMetrics *state,
                                       double actual, double pred)
{
    if (!state) return;
    double residual = actual - pred;
    double abs_err = fabs(residual);
    size_t n = state->n_samples + 1;
    double old_sse = state->mse * (double)state->n_samples;
    state->mse = (old_sse + residual * residual) / (double)n;
    state->rmse = sqrt(state->mse);
    state->n_samples = n;
    double old_sum_abs = state->mae * (double)(n - 1);
    state->mae = (old_sum_abs + abs_err) / (double)n;
    if (abs_err > state->max_error) state->max_error = abs_err;
    state->bias = state->bias * (double)(n - 1) / (double)n + residual / (double)n;
    if (fabs(actual) > 1e-10) {
        double mc = abs_err / fabs(actual);
        state->mape = state->mape * (double)(n - 1) / (double)n + mc * 100.0 / (double)n;
    }
}

/* ====================================================================
 * L3: Welford Algorithm (1962) - numerically stable running stats
 * Technometrics 4(3), 419-420.
 * delta = x - mean; mean += delta/n; M2 += delta*(x - mean)
 * ==================================================================== */

RunningStatistics running_stats_init(void)
{
    RunningStatistics s;
    memset(&s, 0, sizeof(s));
    s.min_val = DBL_MAX;
    s.max_val = -DBL_MAX;
    return s;
}

void running_stats_push(RunningStatistics *stats, double x)
{
    if (!stats) return;
    stats->count++;
    size_t n = stats->count;
    double delta = x - stats->mean;
    stats->mean += delta / (double)n;
    double delta2 = x - stats->mean;
    stats->m2 += delta * delta2;
    if (n >= 2) stats->var_sample = stats->m2 / (double)(n - 1);
    stats->var_population = (n > 0) ? stats->m2 / (double)n : 0.0;
    if (x < stats->min_val) stats->min_val = x;
    if (x > stats->max_val) stats->max_val = x;
}

/* West (1979) removal algorithm, CACM 22(9), 532-535 */

void running_stats_remove(RunningStatistics *stats, double x)
{
    if (!stats || stats->count == 0) return;
    if (stats->count == 1) {
        memset(stats, 0, sizeof(*stats));
        stats->min_val = DBL_MAX;
        stats->max_val = -DBL_MAX;
        return;
    }
    size_t n = stats->count;
    double mean_nm1 = (n * stats->mean - x) / (double)(n - 1);
    stats->m2 -= (x - mean_nm1) * (x - stats->mean);
    if (stats->m2 < 0.0) stats->m2 = 0.0;
    stats->mean = mean_nm1;
    stats->count = n - 1;
    if (stats->count >= 2)
        stats->var_sample = stats->m2 / (double)(stats->count - 1);
    else
        stats->var_sample = 0.0;
    stats->var_population = stats->m2 / (double)stats->count;
}

/* Chan et al. (1979) parallel variance merge - STAN-CS-79-773 */

RunningStatistics running_stats_merge(const RunningStatistics *a,
                                       const RunningStatistics *b)
{
    RunningStatistics result;
    if (!a && !b) return running_stats_init();
    if (!a || a->count == 0) { result = *b; return result; }
    if (!b || b->count == 0) { result = *a; return result; }
    size_t n_total = a->count + b->count;
    double delta = b->mean - a->mean;
    result.mean = (a->count * a->mean + b->count * b->mean) / (double)n_total;
    result.m2 = a->m2 + b->m2 + delta * delta * (double)(a->count * b->count) / (double)n_total;
    result.count = n_total;
    result.var_sample = (n_total >= 2) ? result.m2 / (double)(n_total - 1) : 0.0;
    result.var_population = result.m2 / (double)n_total;
    result.min_val = (a->min_val < b->min_val) ? a->min_val : b->min_val;
    result.max_val = (a->max_val > b->max_val) ? a->max_val : b->max_val;
    return result;
}

/* ====================================================================
 * L3: Sliding Window - circular buffer, O(1) push/mean/var
 * ==================================================================== */

void sliding_window_init(SlidingWindow *window, size_t capacity)
{
    if (!window) return;
    memset(window, 0, sizeof(*window));
    window->capacity = capacity;
    window->buffer = (double *)calloc(capacity, sizeof(double));
}

void sliding_window_destroy(SlidingWindow *window)
{
    if (!window) return;
    free(window->buffer);
    memset(window, 0, sizeof(*window));
}

void sliding_window_push(SlidingWindow *window, double value)
{
    if (!window || !window->buffer || window->capacity == 0) return;
    if (window->count == window->capacity) {
        double old_val = window->buffer[window->head];
        window->sum -= old_val;
        window->sum_sq -= old_val * old_val;
        window->count--;
    }
    window->buffer[window->head] = value;
    window->sum += value;
    window->sum_sq += value * value;
    window->head = (window->head + 1) % window->capacity;
    window->count++;
    size_t n = window->count;
    window->current_mean = window->sum / (double)n;
    if (n >= 2) {
        double vn = window->sum_sq - window->sum * window->sum / (double)n;
        window->current_var = vn / (double)(n - 1);
        if (window->current_var < 0.0) window->current_var = 0.0;
    } else {
        window->current_var = 0.0;
    }
}

double sliding_window_mean(const SlidingWindow *window)
{
    if (!window || window->count == 0) return 0.0;
    return window->current_mean;
}

double sliding_window_variance(const SlidingWindow *window)
{
    if (!window || window->count < 2) return 0.0;
    return window->current_var;
}

double sliding_window_stddev(const SlidingWindow *window)
{
    return sqrt(sliding_window_variance(window));
}

/* EWMA: Hunter (1986) J. Quality Tech 18(4), 203-210.
 * EWMA_t = lambda*x_t + (1-lambda)*EWMA_{t-1}
 * Control limits: mu0 +/- L*sigma*sqrt(lambda/(2-lambda)) */

double sliding_window_ewma(const SlidingWindow *window, double lambda)
{
    if (!window || window->count == 0) return 0.0;
    if (lambda <= 0.0) lambda = 0.1;
    if (lambda > 1.0) lambda = 1.0;
    double ewma = window->buffer[0];
    for (size_t i = 1; i < window->count; i++) {
        double val = window->buffer[(window->head - window->count + i) % window->capacity];
        ewma = lambda * val + (1.0 - lambda) * ewma;
    }
    return ewma;
}

/* ====================================================================
 * L1: Health Index - composite sensor health assessment
 * Accuracy score: exp(-(rmse_ratio-1)^2 / 0.5)
 * Drift index: 1 - exp(-|bias|/baseline_rmse)
 * Noise index: current_var / baseline_var
 * Reliability: 0.5*accuracy + 0.3*(1-drift) + 0.2*(1/max(noise,1))
 * ==================================================================== */

HealthIndex health_index_init(void)
{
    HealthIndex h;
    memset(&h, 0, sizeof(h));
    h.accuracy_score = 1.0;
    h.drift_index = 0.0;
    h.noise_index = 1.0;
    h.reliability_score = 1.0;
    h.stage = LIFECYCLE_COMMISSIONING;
    h.hours_since_last_cal = 0;
    h.aging_rate = 0.0;
    return h;
}

void health_index_update(HealthIndex *health,
                          double current_rmse, double baseline_rmse,
                          double current_bias, double current_var,
                          double baseline_var, double hours_elapsed)
{
    if (!health) return;
    if (baseline_rmse < 1e-10)
        health->accuracy_score = (current_rmse < 1e-10) ? 1.0 : 0.0;
    else {
        double ratio = current_rmse / baseline_rmse;
        double excess = ratio - 1.0;
        if (excess < 0.0) excess = 0.0;
        health->accuracy_score = exp(-excess * excess / 0.5);
    }
    double norm_bias = (baseline_rmse > 1e-10) ? fabs(current_bias) / baseline_rmse : fabs(current_bias);
    health->drift_index = 1.0 - exp(-norm_bias);
    if (baseline_var < 1e-15)
        health->noise_index = (current_var < 1e-15) ? 1.0 : current_var / 1e-10;
    else
        health->noise_index = current_var / baseline_var;
    health->reliability_score = 0.5 * health->accuracy_score
                               + 0.3 * (1.0 - health->drift_index)
                               + 0.2 * (1.0 / fmax(health->noise_index, 1.0));
    if (health->reliability_score > 1.0) health->reliability_score = 1.0;
    if (health->reliability_score < 0.0) health->reliability_score = 0.0;
    double r = health->reliability_score;
    if (r >= 0.95)      health->stage = LIFECYCLE_NORMAL;
    else if (r >= 0.80) health->stage = LIFECYCLE_WARNING;
    else if (r >= 0.60) health->stage = LIFECYCLE_AGING;
    else if (r >= 0.40) health->stage = LIFECYCLE_MAINTENANCE;
    else                health->stage = LIFECYCLE_FAILURE;
    health->hours_since_last_cal += (uint64_t)hours_elapsed;
    if (health->hours_since_last_cal > 0)
        health->aging_rate = (1.0 - health->reliability_score)
                             * 1000.0 / (double)health->hours_since_last_cal;
}

const char *lifecycle_stage_name(SensorLifecycleStage stage)
{
    switch (stage) {
        case LIFECYCLE_COMMISSIONING: return "Commissioning";
        case LIFECYCLE_NORMAL:        return "Normal";
        case LIFECYCLE_WARNING:       return "Warning";
        case LIFECYCLE_AGING:         return "Aging";
        case LIFECYCLE_MAINTENANCE:   return "Maintenance";
        case LIFECYCLE_FAILURE:       return "Failure";
        case LIFECYCLE_RETIRED:       return "Retired";
        default:                      return "Unknown";
    }
}

int health_index_recommends_maintenance(const HealthIndex *health)
{
    if (!health) return 0;
    return (health->stage >= LIFECYCLE_AGING) ? 1 : 0;
}

/* ====================================================================
 * L4: Q-Statistic (SPE) - Jackson & Mudholkar (1979) Technometrics 21(3), 341-349.
 * SPE = sum(e_i^2)
 * Q_alpha = theta1 * [c_alpha*sqrt(2*theta2*h0^2)/theta1 + 1
 *            + theta2*h0*(h0-1)/theta1^2]^(1/h0)
 * theta_k = sum_{j=A+1}^m lambda_j^k, h0 = 1 - 2*theta1*theta3/(3*theta2^2)
 * ==================================================================== */

QStatistic compute_q_statistic(const double *residuals, size_t m,
                                const double *eigenvalues, size_t n_components)
{
    QStatistic q;
    memset(&q, 0, sizeof(q));
    if (!residuals || m == 0) return q;
    double spe = 0.0;
    for (size_t i = 0; i < m; i++) spe += residuals[i] * residuals[i];
    q.spe_value = spe;
    if (!eigenvalues || n_components >= m) {
        q.spe_limit_95 = 0.0; q.spe_limit_99 = 0.0; q.spe_alarm = 0;
        return q;
    }
    double theta1 = 0.0, theta2 = 0.0, theta3 = 0.0;
    for (size_t j = n_components; j < m; j++) {
        double lam = eigenvalues[j];
        theta1 += lam; theta2 += lam * lam; theta3 += lam * lam * lam;
    }
    if (theta1 < 1e-15) { q.spe_alarm = 0; return q; }
    double h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 * theta2);
    if (h0 <= 0.0) h0 = 0.01;
    double c95 = 1.6448536269514722;  /* z_0.95 */
    double c99 = 2.3263478740408408;  /* z_0.99 */
    double t95 = c95 * sqrt(2.0 * theta2 * h0 * h0) / theta1
                 + 1.0 + theta2 * h0 * (h0 - 1.0) / (theta1 * theta1);
    double t99 = c99 * sqrt(2.0 * theta2 * h0 * h0) / theta1
                 + 1.0 + theta2 * h0 * (h0 - 1.0) / (theta1 * theta1);
    q.spe_limit_95 = theta1 * pow(t95, 1.0 / h0);
    q.spe_limit_99 = theta1 * pow(t99, 1.0 / h0);
    q.spe_alarm = (spe > q.spe_limit_95) ? 1 : 0;
    return q;
}

/* ====================================================================
 * L4: Hotelling T^2 with F-distribution limits
 * Hotelling (1931) Ann. Math. Stat. 2(3), 360-378.
 * T^2 = sum(t_a^2 / lambda_a)
 * Limit: T^2_alpha = A*(N-1)*(N+1) / [N*(N-A)] * F_{A,N-A,alpha}
 * ==================================================================== */

T2Statistic compute_t2_statistic(const double *scores, const double *eigenvalues,
                                  size_t n_components, size_t n_samples,
                                  double alpha)
{
    T2Statistic t2;
    memset(&t2, 0, sizeof(t2));
    if (!scores || !eigenvalues || n_components == 0 || n_samples == 0)
        return t2;
    double t2_val = 0.0;
    for (size_t i = 0; i < n_components; i++) {
        if (fabs(eigenvalues[i]) > 1e-15)
            t2_val += (scores[i] * scores[i]) / eigenvalues[i];
    }
    t2.t2_value = t2_val;
    if (n_samples <= n_components) return t2;
    double A = (double)n_components, N = (double)n_samples;
    double factor = A * (N - 1.0) * (N + 1.0) / (N * (N - A));
    double conf_level = 1.0 - alpha;
    double f_crit = f_distribution_critical_value(conf_level, A, N - A);
    double f99_crit = f_distribution_critical_value(0.99, A, N - A);
    t2.t2_limit_95 = factor * f_crit;
    t2.t2_limit_99 = factor * f99_crit;
    t2.t2_alarm = (t2_val > t2.t2_limit_95) ? 1 : 0;
    return t2;
}

void compute_spe_contributions(const double *residuals, size_t m,
                                double *contributions)
{
    if (!residuals || !contributions || m == 0) return;
    double total = 0.0;
    for (size_t i = 0; i < m && i < 20; i++) {
        contributions[i] = residuals[i] * residuals[i];
        total += contributions[i];
    }
    if (total > 1e-15)
        for (size_t i = 0; i < m && i < 20; i++)
            contributions[i] /= total;
}

/* ====================================================================
 * L5: Combined Drift Detection - CUSUM + Page-Hinkley + trend slope
 * Ref: Bifet & Gavalda (2007) SDM.
 * Classification: CUSUM+PH -> SUDDEN; PH+trend -> INCREMENTAL;
 *                 trend only -> GRADUAL; oscillating -> RECURRING
 * ==================================================================== */

DriftAnalysis detect_drift(const double *metric_stream, size_t n,
                            double confidence)
{
    DriftAnalysis result;
    memset(&result, 0, sizeof(result));
    (void)confidence; /* Used in classification confidence thresholds */
    if (!metric_stream || n < 10) return result;
    size_t baseline_n = n / 2;
    RunningStatistics baseline = running_stats_init();
    for (size_t i = 0; i < baseline_n; i++)
        running_stats_push(&baseline, metric_stream[i]);
    if (baseline.count < 2) return result;
    double target = baseline.mean;
    double sigma = sqrt(baseline.var_sample);
    if (sigma < 1e-10) sigma = 1e-6;
    double k = 0.5 * sigma, h = 5.0 * sigma;
    int cusum_alarm = 0;
    compute_cusum_statistic(metric_stream, n, target, sigma, k, h, &cusum_alarm);
    int ph_alarm = 0;
    page_hinkley_test(metric_stream, n, 0.005, 50.0, &ph_alarm);
    double x_mean = (double)(n - 1) / 2.0;
    double sum_xy = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double xi = (double)i - x_mean;
        sum_xy += xi * metric_stream[i];
        sum_xx += xi * xi;
    }
    result.rate_of_change = (fabs(sum_xx) > 1e-15) ? sum_xy / sum_xx : 0.0;
    if (cusum_alarm > 0 && ph_alarm > 0) {
        result.type = DRIFT_SUDDEN; result.confidence = 0.90;
    } else if (ph_alarm > 0 && fabs(result.rate_of_change) > 1e-3) {
        result.type = DRIFT_INCREMENTAL; result.confidence = 0.75;
    } else if (fabs(result.rate_of_change) > 1e-4) {
        result.type = DRIFT_GRADUAL; result.confidence = 0.60;
    } else {
        result.type = DRIFT_NONE; result.confidence = 0.50;
    }
    double current_mean = 0.0;
    for (size_t i = (n >= 5 ? n - 5 : 0); i < n; i++)
        current_mean += metric_stream[i] / 5.0;
    result.magnitude = fabs(current_mean - baseline.mean) / fmax(sigma, 1e-10);
    return result;
}

/* CUSUM: Page (1954) Biometrika 41, 100-115.
 * S_Hi[t] = max(0, S_Hi[t-1] + x_t - mu0 - k)
 * S_Lo[t] = max(0, S_Lo[t-1] - x_t + mu0 - k)
 * Alarm when S > h */

double compute_cusum_statistic(const double *stream, size_t n,
                                double target, double sigma,
                                double k, double h, int *alarm_idx)
{
    (void)sigma; /* sigma used in k/h parameterization; passed for logging */
    if (alarm_idx) *alarm_idx = -1;
    if (!stream || n == 0) return 0.0;
    double s_hi = 0.0, s_lo = 0.0, max_s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dev = stream[i] - target;
        s_hi = fmax(0.0, s_hi + dev - k);
        s_lo = fmax(0.0, s_lo - dev - k);
        if (s_hi > max_s) max_s = s_hi;
        if ((s_hi > h || s_lo > h) && alarm_idx && *alarm_idx < 0)
            *alarm_idx = (int)i;
    }
    return max_s;
}

/* Page-Hinkley: Hinkley (1971) Biometrika 58(3), 509-523.
 * m_T = sum(x_t - xbar_t - delta), M_T = max(m_t), PH_T = M_T - m_T
 * Alarm when PH_T > lambda */

double page_hinkley_test(const double *stream, size_t n,
                          double delta, double lambda, int *alarm_idx)
{
    if (alarm_idx) *alarm_idx = -1;
    if (!stream || n == 0) return 0.0;
    double sum_x = 0.0, m_t = 0.0, M_t = 0.0, ph_max = 0.0;
    for (size_t t = 0; t < n; t++) {
        sum_x += stream[t];
        double xbar_t = sum_x / (double)(t + 1);
        m_t += stream[t] - xbar_t - delta;
        if (m_t > M_t) M_t = m_t;
        double ph_t = M_t - m_t;
        if (ph_t > ph_max) ph_max = ph_t;
        if (ph_t > lambda && alarm_idx && *alarm_idx < 0)
            *alarm_idx = (int)t;
    }
    return ph_max;
}

/* ====================================================================
 * L4: Confidence Interval for Mean (Student's t)
 * CI = xbar +/- t_{alpha/2, n-1} * s / sqrt(n)
 * ==================================================================== */

ConfidenceInterval confidence_interval_mean(const double *data,
                                              size_t n, double conf_level)
{
    ConfidenceInterval ci;
    memset(&ci, 0, sizeof(ci));
    if (!data || n < 2) return ci;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += data[i];
    double mean = sum / (double)n;
    double m2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = data[i] - mean;
        m2 += diff * diff;
    }
    double s = sqrt(m2 / (double)(n - 1));
    double se = s / sqrt((double)n);
    double alpha = 1.0 - conf_level;
    double t_crit = t_distribution_critical_value(1.0 - alpha / 2.0, (double)(n - 1));
    ci.point_estimate = mean;
    ci.standard_error = se;
    ci.lower_bound = mean - t_crit * se;
    ci.upper_bound = mean + t_crit * se;
    ci.confidence_level = conf_level;
    ci.degrees_freedom = n - 1;
    return ci;
}

/* Paired t-test: d_i = err_a_i - err_b_i, t = dbar / (sd/sqrt(n))
 * Cohen''s d = |dbar|/sd. Reject H0 if |t| > t_crit. */

PairedTTest paired_t_test(const double *errors_a, const double *errors_b,
                           size_t n, double alpha)
{
    PairedTTest result;
    memset(&result, 0, sizeof(result));
    result.alpha = alpha;
    if (!errors_a || !errors_b || n < 2) return result;
    double sum_d = 0.0, sum_d2 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = errors_a[i] - errors_b[i];
        sum_d += d; sum_d2 += d * d;
    }
    double dbar = sum_d / (double)n;
    double var_d = (sum_d2 - sum_d * sum_d / (double)n) / (double)(n - 1);
    if (var_d < 0.0) var_d = 0.0;
    double sd = sqrt(var_d);
    result.mean_difference = dbar;
    if (sd < 1e-15) {
        result.t_statistic = 0.0; result.p_value = 1.0;
        result.reject_null = 0; result.cohens_d = 0.0;
        return result;
    }
    double se = sd / sqrt((double)n);
    result.t_statistic = dbar / se;
    result.cohens_d = fabs(dbar) / sd;
    double df = (double)(n - 1);
    result.critical_value = t_distribution_critical_value(1.0 - alpha / 2.0, df);
    result.p_value = compute_p_value_from_t(result.t_statistic, df);
    result.reject_null = (fabs(result.t_statistic) > result.critical_value) ? 1 : 0;
    return result;
}

/* t critical value: Hill (1970) Algorithm 396, CACM 13(10), 619-620.
 * For large df (>100): normal approx z_p
 * For moderate df: asymptotic expansion
 * For small df (<5): correction factor */

double t_distribution_critical_value(double p, double df)
{
    if (p <= 0.0) return -1e10;
    if (p >= 1.0) return 1e10;
    if (df < 1.0) df = 1.0;
    int sign = 1;
    double prob = p;
    if (prob < 0.5) { prob = 1.0 - prob; sign = -1; }
    /* Normal quantile z_p via Hastings approximation */
    double tep = prob;
    if (tep > 0.9999999) tep = 0.9999999;
    if (tep < 0.5) tep = 1.0 - tep;
    double y = -log(4.0 * tep * (1.0 - tep));
    double z = sqrt(y * (2.0611786 - 5.7262204 / (y + 11.640595)));
    if (prob < 0.5) z = -z;
    if (df > 100.0) return sign * z;
    double z2 = z * z;
    double t_val = z * (1.0 + (z2 + 1.0) / (4.0 * df)
                 + (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / (96.0 * df * df)
                 + (3.0 * z2 * z2 * z2 + 19.0 * z2 * z2 + 17.0 * z2 - 15.0)
                   * z / (384.0 * df * df * df));
    if (df < 5.0) {
        double corr = 1.0 + 0.25 * (z2 - 1.0) / df;
        t_val = z * corr;
        if (df < 2.0) t_val = z * (1.0 + 0.5 * z2 / df);
    }
    return sign * t_val;
}

/* F critical value: Wilson-Hilferty approx, Abramowitz & Stegun 26.6.15
 * F_{nu1,nu2,p} = [(1-a)(1-b) + z*sqrt((1-a)^2*b + a*(1-b)^2 - a*b*z^2)]^3
 *                  / [(1-b)^2 - b*z^2]^2
 * a = 2/(9*nu1), b = 2/(9*nu2) */

double f_distribution_critical_value(double p, double df1, double df2)
{
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1e10;
    if (df1 < 1.0) df1 = 1.0;
    if (df2 < 1.0) df2 = 1.0;
    double prob = (p > 0.9999999) ? 0.9999999 : p;
    double y = -log(4.0 * prob * (1.0 - prob));
    double zp = sqrt(y * (2.0611786 - 5.7262204 / (y + 11.640595)));
    double a = 2.0 / (9.0 * df1);
    double b = 2.0 / (9.0 * df2);
    double num = (1.0 - a) * (1.0 - b)
                 + zp * sqrt((1.0 - a) * (1.0 - a) * b
                             + a * (1.0 - b) * (1.0 - b) - a * b * zp * zp);
    double den = (1.0 - b) * (1.0 - b) - b * zp * zp;
    if (den <= 0.0) {
        double w = zp * sqrt(df1 + df2) / (df1 + df2 - 2.5) / 2.0;
        return exp(2.0 * w);
    }
    double f_val = (num * num * num) / (den * den);
    if (f_val < 0.0) f_val = 1.0;
    return f_val;
}

/* ====================================================================
 * L4: P-Value from t-Statistic via numerical integration
 * t-PDF: f(t;nu) = Gamma((nu+1)/2)/(sqrt(nu*pi)*Gamma(nu/2)) * (1+t^2/nu)^{-(nu+1)/2}
 * Uses Simpson composite integration and log-gamma via Stirling.
 * For large df (>100): normal tail probability approximation.
 * ==================================================================== */

static double log_gamma_stirling(double x)
{
    if (x <= 0.0) return 0.0;
    double pi = 3.14159265358979323846;
    double term = 0.5 * log(2.0 * pi);
    term += (x - 0.5) * log(x);
    term -= x;
    term += 1.0 / (12.0 * x);
    term -= 1.0 / (360.0 * x * x * x);
    term += 1.0 / (1260.0 * x * x * x * x * x);
    return term;
}

static double t_pdf_density(double t, double nu)
{
    double pi = 3.14159265358979323846;
    double log_num = log_gamma_stirling((nu + 1.0) / 2.0);
    double log_den = 0.5 * log(nu * pi) + log_gamma_stirling(nu / 2.0);
    double log_dens = log_num - log_den - ((nu + 1.0) / 2.0) * log(1.0 + t * t / nu);
    return exp(log_dens);
}

double compute_p_value_from_t(double t_stat, double df)
{
    double abs_t = fabs(t_stat);
    if (df < 1.0) df = 1.0;
    /* Normal approximation for large df */
    if (df > 100.0) {
        double x = abs_t;
        double pdf_n = exp(-0.5 * x * x) / sqrt(2.0 * 3.14159265358979323846);
        double tail = pdf_n / x * (1.0 - 1.0 / (x * x)
                      + 3.0 / (x * x * x * x)
                      - 15.0 / (x * x * x * x * x * x));
        double p = 2.0 * tail;
        if (p > 1.0) p = 1.0;
        if (p < 0.0) p = 0.0;
        return p;
    }
    /* Simpson integration from abs_t to abs_t + 20 */
    double a = abs_t;
    double b = abs_t + 20.0;
    if (b < a + 1.0) b = a + 1.0;
    int n_int = 200;
    double h = (b - a) / (double)n_int;
    double integral = t_pdf_density(a, df) + t_pdf_density(b, df);
    for (int i = 1; i < n_int; i++) {
        double x = a + i * h;
        double w = (i % 2 == 0) ? 2.0 : 4.0;
        integral += w * t_pdf_density(x, df);
    }
    integral *= h / 3.0;
    /* Two-tailed p-value */
    double p = 2.0 * integral;
    if (p > 1.0) p = 1.0;
    if (p < 0.0) p = 0.0;
    return p;
}
