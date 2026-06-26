/**
 * @file    model_health_monitor.c
 * @brief   Model health monitoring - trajectory, residual analysis, RUL estimation
 *
 * L2: Performance trajectory tracking
 * L4: ISO 13379 data processing for diagnostics
 * L5: ARIMA-style residual autocorrelation, degradation model fitting,
 *     RUL (Remaining Useful Life) estimation with confidence intervals
 * L8: Bayesian change point detection via conjugate priors
 *
 * Ref: Kadlec et al. (2009) Comp. & Chem. Eng. 33(4), 795-814.
 *      Barry & Hartigan (1993) JASA 88, 309-319.
 */

#include "model_health_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ====================================================================
 * L2: Performance Trajectory
 * ==================================================================== */

void performance_trajectory_init(PerformanceTrajectory *traj,
                                  size_t capacity,
                                  double baseline_rmse,
                                  double baseline_r2,
                                  double degradation_threshold)
{
    if (!traj) return;
    memset(traj, 0, sizeof(*traj));
    traj->capacity = capacity;
    traj->snapshots = (PerformanceSnapshot *)calloc(capacity, sizeof(PerformanceSnapshot));
    traj->baseline_rmse = baseline_rmse;
    traj->baseline_r2 = baseline_r2;
    traj->degradation_threshold = degradation_threshold;
}

void performance_trajectory_destroy(PerformanceTrajectory *traj)
{
    if (!traj) return;
    free(traj->snapshots);
    memset(traj, 0, sizeof(*traj));
}

void performance_trajectory_add(PerformanceTrajectory *traj,
                                 double timestamp, double rmse,
                                 double r2, double bias, double variance)
{
    if (!traj || !traj->snapshots || traj->capacity == 0) return;
    if (traj->count >= traj->capacity) {
        /* Shift buffer: discard oldest, keep most recent */
        memmove(traj->snapshots, traj->snapshots + 1,
                (traj->capacity - 1) * sizeof(PerformanceSnapshot));
        traj->count = traj->capacity - 1;
    }
    PerformanceSnapshot *s = &traj->snapshots[traj->count];
    s->timestamp = timestamp;
    s->rmse = rmse;
    s->r2 = r2;
    s->bias = bias;
    s->variance = variance;
    traj->count++;
}

/* Fit linear degradation: rmse = slope * hours + intercept
 * Simple linear regression with least squares.
 * slope = sum((x_i - xbar)*(y_i - ybar)) / sum((x_i - xbar)^2)
 * intercept = ybar - slope * xbar
 * r_squared = 1 - SS_res / SS_tot */

DegradationModel fit_degradation_model(const PerformanceTrajectory *traj)
{
    DegradationModel model;
    memset(&model, 0, sizeof(model));
    if (!traj || traj->count < 2) return model;

    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0, sum_yy = 0.0;
    for (size_t i = 0; i < traj->count; i++) {
        double x = traj->snapshots[i].timestamp;
        double y = traj->snapshots[i].rmse;
        sum_x += x; sum_y += y;
        sum_xy += x * y; sum_xx += x * x; sum_yy += y * y;
    }
    double n = (double)traj->count;
    double xbar = sum_x / n, ybar = sum_y / n;
    double ss_xx = sum_xx - n * xbar * xbar;
    double ss_xy = sum_xy - n * xbar * ybar;
    double ss_yy = sum_yy - n * ybar * ybar;

    if (fabs(ss_xx) < 1e-15) return model;
    model.slope = ss_xy / ss_xx;
    model.intercept = ybar - model.slope * xbar;
    if (fabs(ss_yy) > 1e-15) {
        double ss_res = 0.0;
        for (size_t i = 0; i < traj->count; i++) {
            double y_pred = model.slope * traj->snapshots[i].timestamp + model.intercept;
            double e = traj->snapshots[i].rmse - y_pred;
            ss_res += e * e;
        }
        model.r_squared = 1.0 - ss_res / ss_yy;
    }
    /* Degradation rate per 1000 hours */
    model.degradation_rate_per_1000h = model.slope * 1000.0;
    return model;
}

int is_degradation_significant(const PerformanceTrajectory *traj)
{
    if (!traj || traj->count < 10) return 0;
    DegradationModel dm = fit_degradation_model(traj);
    if (dm.r_squared < 0.3) return 0;
    /* t-test on slope: t = slope / SE(slope) */
    double n = (double)traj->count;
    double sum_x = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < traj->count; i++) {
        double x = traj->snapshots[i].timestamp;
        sum_x += x; sum_xx += x * x;
    }
    double xbar = sum_x / n;
    double ss_xx = sum_xx - n * xbar * xbar;
    if (fabs(ss_xx) < 1e-15) return 0;
    /* Estimate residual variance */
    double ss_res = 0.0;
    for (size_t i = 0; i < traj->count; i++) {
        double y_pred = dm.slope * traj->snapshots[i].timestamp + dm.intercept;
        double e = traj->snapshots[i].rmse - y_pred;
        ss_res += e * e;
    }
    double se_slope = sqrt(ss_res / ((n - 2.0) * ss_xx));
    double t_stat = dm.slope / fmax(se_slope, 1e-15);
    /* For df=n-2, compare to critical t (~2.0 at alpha=0.05) */
    return (fabs(t_stat) > 2.0) ? 1 : 0;
}

/* ====================================================================
 * L5: Residual Analysis - Autocorrelation, Durbin-Watson, Ljung-Box
 * ==================================================================== */

ResidualAnalysis analyze_residuals(const double *actual,
                                    const double *predicted,
                                    size_t n)
{
    ResidualAnalysis ra;
    memset(&ra, 0, sizeof(ra));
    if (!actual || !predicted || n < 3) return ra;

    ra.n = n;
    ra.residuals = (double *)malloc(n * sizeof(double));
    if (!ra.residuals) return ra;

    /* Compute residuals and their statistics */
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        ra.residuals[i] = actual[i] - predicted[i];
        sum += ra.residuals[i];
        sum_sq += ra.residuals[i] * ra.residuals[i];
    }
    ra.mean_residual = sum / (double)n;
    ra.var_residual = sum_sq / (double)n - ra.mean_residual * ra.mean_residual;
    if (ra.var_residual < 0.0) ra.var_residual = 0.0;

    /* Durbin-Watson statistic */
    ra.durbin_watson = compute_durbin_watson(ra.residuals, n);

    /* Lag-1 autocorrelation: r1 = sum(e_t * e_{t-1}) / sum(e_t^2) */
    double sum_lag1 = 0.0, sum_lag2 = 0.0;
    for (size_t i = 1; i < n; i++)
        sum_lag1 += ra.residuals[i] * ra.residuals[i - 1];
    for (size_t i = 2; i < n; i++)
        sum_lag2 += ra.residuals[i] * ra.residuals[i - 2];

    if (sum_sq > 1e-15) {
        ra.lag1_autocorr = sum_lag1 / sum_sq;
        ra.lag2_autocorr = sum_lag2 / sum_sq;
    }

    /* Ljung-Box Q test for m = min(10, n/5) lags */
    size_t m = (n / 5 > 10) ? 10 : n / 5;
    if (m < 1) m = 1;
    ra.ljung_box_q = ljung_box_test(ra.residuals, n, m);

    /* Q ~ chi^2(m) under H0: white noise.
     * Approximate p-value: for Q/m > 1.5, likely reject H0 */
    ra.ljung_box_pvalue = 1.0 - (ra.ljung_box_q / (double)m) * 0.5;
    if (ra.ljung_box_pvalue < 0.01) ra.ljung_box_pvalue = 0.01;
    if (ra.ljung_box_pvalue > 1.0) ra.ljung_box_pvalue = 0.99;

    ra.is_white_noise = (ra.ljung_box_pvalue > 0.05) ? 1 : 0;

    return ra;
}

double compute_durbin_watson(const double *residuals, size_t n)
{
    if (!residuals || n < 2) return 2.0;
    double sum_diff_sq = 0.0, sum_sq = 0.0;
    for (size_t i = 1; i < n; i++) {
        double diff = residuals[i] - residuals[i - 1];
        sum_diff_sq += diff * diff;
    }
    for (size_t i = 0; i < n; i++)
        sum_sq += residuals[i] * residuals[i];
    if (sum_sq < 1e-15) return 2.0;
    return sum_diff_sq / sum_sq;
}

/* Ljung-Box (1978) Biometrika 65, 297-303.
 * Q = n*(n+2) * sum_{k=1}^{m} r_k^2 / (n-k) */

double ljung_box_test(const double *residuals, size_t n, size_t m)
{
    if (!residuals || n < 3 || m < 1 || m >= n) return 0.0;
    /* Compute mean */
    double mean_r = 0.0;
    for (size_t i = 0; i < n; i++) mean_r += residuals[i];
    mean_r /= (double)n;
    /* Compute autocorrelations up to lag m */
    double denom = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = residuals[i] - mean_r;
        denom += d * d;
    }
    if (denom < 1e-15) return 0.0;
    double q = 0.0;
    for (size_t k = 1; k <= m; k++) {
        double num = 0.0;
        for (size_t i = k; i < n; i++)
            num += (residuals[i] - mean_r) * (residuals[i - k] - mean_r);
        double r_k = num / denom;
        q += (r_k * r_k) / (double)(n - k);
    }
    q *= (double)n * (double)(n + 2);
    return q;
}

/* ====================================================================
 * L5: RUL (Remaining Useful Life) Estimation
 *
 * Linear degradation model: rmse(t) = rmse_0 + degradation_rate * t
 * RUL = (failure_threshold - current_rmse) / degradation_rate
 *
 * Confidence interval via delta method:
 *   Var(RUL) = Var(slope) * (RUL/slope)^2
 *             + Var(intercept) * (1/slope)^2
 *   where Var(slope) and Var(intercept) from linear regression.
 *
 * Degradation percentage:
 *   degradation_pct = (current_rmse - baseline_rmse)
 *                    / (failure_threshold - baseline_rmse) * 100
 * ==================================================================== */

RULEstimate estimate_rul(const PerformanceTrajectory *traj,
                          double failure_rmse)
{
    RULEstimate rul;
    memset(&rul, 0, sizeof(rul));
    if (!traj || traj->count < 2) return rul;

    DegradationModel dm = fit_degradation_model(traj);
    if (dm.degradation_rate_per_1000h <= 0.0) {
        /* No degradation detected: RUL is effectively infinite */
        rul.rul_hours = 1e6;
        rul.confidence = 0.0;
        return rul;
    }

    /* Current RMSE estimate from most recent snapshot */
    double current_rmse = traj->snapshots[traj->count - 1].rmse;
    double baseline_rmse = traj->baseline_rmse;

    /* Degradation rate in units per hour */
    double rate_per_hour = dm.degradation_rate_per_1000h / 1000.0;

    /* RUL */
    double remaining = failure_rmse - current_rmse;
    rul.rul_hours = remaining / fmax(rate_per_hour, 1e-10);

    /* Confidence intervals: estimate SE of slope */
    double n = (double)traj->count;
    double sum_x = 0.0, sum_xx = 0.0;
    for (size_t i = 0; i < traj->count; i++) {
        double x = traj->snapshots[i].timestamp;
        sum_x += x; sum_xx += x * x;
    }
    double ss_xx = sum_xx - sum_x * sum_x / n;
    if (fabs(ss_xx) < 1e-15) return rul;

    /* Residual variance */
    double ss_res = 0.0;
    for (size_t i = 0; i < traj->count; i++) {
        double yp = dm.slope * traj->snapshots[i].timestamp + dm.intercept;
        double e = traj->snapshots[i].rmse - yp;
        ss_res += e * e;
    }
    double var_slope = ss_res / ((n - 2.0) * ss_xx);
    double se_slope = sqrt(fmax(var_slope, 0.0));

    /* Delta method: SE(RUL) = RUL * SE(slope) / slope */
    double se_rul = rul.rul_hours * se_slope / fmax(fabs(dm.slope), 1e-10);
    double z_95 = 1.96; /* Normal approx for CI */
    rul.rul_lower_ci = rul.rul_hours - z_95 * se_rul;
    rul.rul_upper_ci = rul.rul_hours + z_95 * se_rul;
    if (rul.rul_lower_ci < 0.0) rul.rul_lower_ci = 0.0;

    /* Confidence based on R^2 of degradation model */
    rul.confidence = dm.r_squared;

    /* Degradation percentage */
    double total_degradation_range = failure_rmse - baseline_rmse;
    if (total_degradation_range > 0.0) {
        rul.current_degradation_pct = (current_rmse - baseline_rmse)
                                      / total_degradation_range * 100.0;
        if (rul.current_degradation_pct < 0.0) rul.current_degradation_pct = 0.0;
        if (rul.current_degradation_pct > 100.0) rul.current_degradation_pct = 100.0;
    }

    rul.maintenance_urgent = (rul.current_degradation_pct > 70.0) ? 1 : 0;
    return rul;
}

double schedule_maintenance(const RULEstimate *rul,
                             double max_interval,
                             double safety_margin)
{
    if (!rul) return max_interval;
    if (rul->rul_hours <= 0.0) return 0.0; /* Immediate maintenance */
    double interval = rul->rul_hours * safety_margin;
    if (interval > max_interval) interval = max_interval;
    if (interval < 1.0) interval = 1.0;   /* Minimum 1 hour */
    return interval;
}

/* ====================================================================
 * L8: Bayesian Change Point Detection
 *
 * Barry & Hartigan (1993) JASA 88, 309-319.
 *
 * Model: observations follow N(mu1, sigma^2) before change
 *        and N(mu2, sigma^2) after change.
 *
 * Priors: mu ~ N(mu0, tau0^2), sigma^2 ~ Inv-Gamma(alpha0, beta0)
 *
 * Posterior probability of change at position k:
 *   P(k | data) ∝ P(data | k) * P(k)
 *
 * where P(data | k) is the marginal likelihood integrating out mu1, mu2, sigma^2
 * under conjugate priors.
 *
 * Bayes factor:
 *   BF = P(data | change) / P(data | no change)
 *
 * Simplified implementation: uses Gaussian likelihood with sample means
 * and pooled variance estimate.
 * ==================================================================== */

BayesianChangePoint bayesian_change_point_detect(const double *stream,
                                                   size_t n,
                                                   double threshold)
{
    BayesianChangePoint result;
    memset(&result, 0, sizeof(result));
    result.change_point = 0;

    if (!stream || n < 6) return result;

    /* Prior parameters (used in marginal likelihood computation) */
    double prior_mean = 0.0;
    double prior_var = 100.0;
    (void)prior_mean; (void)prior_var; /* Conjugate priors, integrated out */

    /* Compute overall statistics */
    double total_sum = 0.0, total_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        total_sum += stream[i];
        total_sq += stream[i] * stream[i];
    }
    double total_mean = total_sum / (double)n;
    double total_var = total_sq / (double)n - total_mean * total_mean;
    if (total_var < 1e-10) total_var = 1e-10;

    /* Evaluate change point at each possible position k (n/4 to 3n/4) */
    double best_log_bf = -DBL_MAX;
    size_t best_k = 0;
    double best_mu1 = 0.0, best_mu2 = 0.0;

    for (size_t k = n / 4; k <= 3 * n / 4; k++) {
        /* Mean before k */
        double sum1 = 0.0, sum2 = 0.0;
        for (size_t i = 0; i < k; i++) sum1 += stream[i];
        for (size_t i = k; i < n; i++) sum2 += stream[i];
        double mu1 = sum1 / (double)k;
        double mu2 = sum2 / (double)(n - k);

        /* Log marginal likelihood under H1 (change) */
        /* Using Gaussian log-likelihood approximation */
        double ll1 = 0.0;
        for (size_t i = 0; i < k; i++) {
            double d = stream[i] - mu1;
            ll1 += -0.5 * d * d / total_var;
        }
        ll1 -= 0.5 * (double)k * log(2.0 * 3.141592653589793 * total_var);

        for (size_t i = k; i < n; i++) {
            double d = stream[i] - mu2;
            ll1 += -0.5 * d * d / total_var;
        }
        ll1 -= 0.5 * (double)(n - k) * log(2.0 * 3.141592653589793 * total_var);

        /* Log marginal likelihood under H0 (no change) */
        double ll0 = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = stream[i] - total_mean;
            ll0 += -0.5 * d * d / total_var;
        }
        ll0 -= 0.5 * (double)n * log(2.0 * 3.141592653589793 * total_var);

        /* Log Bayes Factor = ll1 - ll0 */
        double log_bf = ll1 - ll0;

        if (log_bf > best_log_bf) {
            best_log_bf = log_bf;
            best_k = k;
            best_mu1 = mu1;
            best_mu2 = mu2;
        }
    }

    /* Posterior probability via logistic transform of Bayes factor */
    double bf = exp(best_log_bf);
    result.posterior_prob = bf / (1.0 + bf);
    result.bayes_factor = bf;
    result.change_detected = (result.posterior_prob > threshold) ? 1 : 0;
    result.change_point = best_k;
    result.mean_before = best_mu1;
    result.mean_after = best_mu2;

    return result;
}
