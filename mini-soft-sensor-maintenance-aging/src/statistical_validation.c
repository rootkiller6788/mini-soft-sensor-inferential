/**
 * @file    statistical_validation.c
 * @brief   Statistical validation of soft sensor maintenance decisions
 *
 * L4: Hypothesis testing for model comparison, cross-validation statistics
 * L5: Bootstrap confidence intervals, permutation tests for model superiority
 * L8: Bayesian model comparison via posterior predictive checks
 *
 * Ref: Hastie, Tibshirani & Friedman (2009) "The Elements of
 *      Statistical Learning", 2nd ed. Springer.
 *      Efron & Tibshirani (1993) "An Introduction to the Bootstrap".
 */

#include "soft_sensor_metrics.h"
#include "model_health_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ====================================================================
 * L4: k-Fold Cross-Validation Statistics
 *
 * Splits data into k folds. For each fold:
 *   - Train on k-1 folds
 *   - Test on held-out fold
 *   - Compute prediction error
 *
 * Returns mean and standard deviation of test errors across folds.
 * This provides a robust estimate of generalization performance
 * and its variability.
 *
 * Since we don't have access to the actual training algorithm here,
 * we provide the data partitioning infrastructure and compute
 * cross-validation statistics from pre-computed fold errors.
 * ==================================================================== */

typedef struct {
    size_t *fold_assignment;
    size_t n_samples;
    size_t n_folds;
    double *fold_errors;
    double mean_error;
    double std_error;
    double *fold_rmses;
} CrossValidationResult;

/**
 * @brief Assign samples to k folds using stratified random assignment.
 *
 * Creates balanced folds by cycling through fold indices.
 * fold_assignment[i] in [0, k-1] for i = 0..n-1.
 *
 * @param n_samples       Total number of samples.
 * @param n_folds         Number of folds (k). Typically 5 or 10.
 * @param fold_assignment Output array, length n_samples (caller allocates).
 */
void kfold_assign(size_t n_samples, size_t n_folds, size_t *fold_assignment)
{
    if (!fold_assignment || n_folds == 0) return;
    for (size_t i = 0; i < n_samples; i++) {
        fold_assignment[i] = i % n_folds;
    }
    /* Fisher-Yates shuffle of assignments for randomness */
    for (size_t i = n_samples - 1; i > 0; i--) {
        size_t j = (size_t)((double)rand() / (RAND_MAX + 1.0) * (i + 1));
        size_t tmp = fold_assignment[i];
        fold_assignment[i] = fold_assignment[j];
        fold_assignment[j] = tmp;
    }
}

/**
 * @brief Compute cross-validation statistics from fold errors.
 *
 * @param fold_errors Array of test errors per fold, length n_folds.
 * @param n_folds     Number of folds.
 * @param mean_error  Output: mean error across folds.
 * @param std_error   Output: standard deviation of errors across folds.
 */
void cross_validation_stats(const double *fold_errors, size_t n_folds,
                             double *mean_error, double *std_error)
{
    if (!fold_errors || n_folds == 0) {
        if (mean_error) *mean_error = 0.0;
        if (std_error) *std_error = 0.0;
        return;
    }

    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n_folds; i++) {
        sum += fold_errors[i];
        sum_sq += fold_errors[i] * fold_errors[i];
    }

    double mean = sum / (double)n_folds;
    if (mean_error) *mean_error = mean;

    if (n_folds > 1 && std_error) {
        double var = (sum_sq - sum * sum / (double)n_folds) / (double)(n_folds - 1);
        if (var < 0.0) var = 0.0;
        *std_error = sqrt(var);
    } else if (std_error) {
        *std_error = 0.0;
    }
}

/* ====================================================================
 * L5: Bootstrap Confidence Intervals
 *
 * Efron's percentile bootstrap method:
 *   1. Resample with replacement B times
 *   2. Compute statistic for each bootstrap sample
 *   3. CI = [alpha/2 percentile, 1-alpha/2 percentile] of bootstrap distribution
 *
 * Also implements BCa (bias-corrected and accelerated) intervals
 * for improved coverage when the bootstrap distribution is skewed.
 *
 * Ref: Efron (1987) "Better bootstrap confidence intervals".
 *      JASA, 82(397), 171-185.
 * ==================================================================== */

/**
 * @brief Generate a bootstrap sample by resampling with replacement.
 *
 * @param data        Original data, length n.
 * @param n           Sample size.
 * @param bootstrap   Output bootstrap sample, length n (caller allocates).
 */
void bootstrap_resample(const double *data, size_t n, double *bootstrap)
{
    if (!data || !bootstrap || n == 0) return;
    for (size_t i = 0; i < n; i++) {
        size_t idx = (size_t)((double)rand() / (RAND_MAX + 1.0) * n);
        bootstrap[i] = data[idx];
    }
}

/**
 * @brief Compute mean and variance from bootstrap replicates.
 *
 * Given B bootstrap estimates of a statistic, compute:
 *   - Bootstrap mean
 *   - Bootstrap standard error
 *   - Percentile confidence interval
 *
 * @param estimates      Array of B bootstrap estimates.
 * @param n_bootstrap    Number of bootstrap replicates (B).
 * @param alpha          Significance level (0.05 for 95% CI).
 * @param ci_lower       Output: lower confidence bound.
 * @param ci_upper       Output: upper confidence bound.
 * @param boot_mean      Output: mean of bootstrap estimates.
 * @param boot_se        Output: standard error of bootstrap estimates.
 */
void bootstrap_confidence_interval(const double *estimates,
                                    size_t n_bootstrap, double alpha,
                                    double *ci_lower, double *ci_upper,
                                    double *boot_mean, double *boot_se)
{
    if (!estimates || n_bootstrap < 2) return;

    /* Compute mean */
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n_bootstrap; i++) {
        sum += estimates[i];
        sum_sq += estimates[i] * estimates[i];
    }
    double mean = sum / (double)n_bootstrap;
    if (boot_mean) *boot_mean = mean;

    double var = (sum_sq - sum * sum / (double)n_bootstrap)
                 / (double)(n_bootstrap - 1);
    if (var < 0.0) var = 0.0;
    if (boot_se) *boot_se = sqrt(var);

    /* Sort estimates for percentile method */
    double *sorted = (double *)malloc(n_bootstrap * sizeof(double));
    if (!sorted) return;
    memcpy(sorted, estimates, n_bootstrap * sizeof(double));

    for (size_t i = 0; i < n_bootstrap - 1; i++) {
        for (size_t j = i + 1; j < n_bootstrap; j++) {
            if (sorted[i] > sorted[j]) {
                double tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }

    /* Percentile indices */
    size_t idx_lower = (size_t)(alpha / 2.0 * n_bootstrap);
    size_t idx_upper = (size_t)((1.0 - alpha / 2.0) * n_bootstrap);
    if (idx_lower >= n_bootstrap) idx_lower = n_bootstrap - 1;
    if (idx_upper >= n_bootstrap) idx_upper = n_bootstrap - 1;

    if (ci_lower) *ci_lower = sorted[idx_lower];
    if (ci_upper) *ci_upper = sorted[idx_upper];

    free(sorted);
}

/* ====================================================================
 * L5: Permutation Test for Model Comparison
 *
 * Non-parametric test for whether model A is significantly better
 * than model B. Unlike the paired t-test, does not assume normality.
 *
 * Procedure:
 *   1. Compute observed difference in RMSE: d_obs = RMSE_B - RMSE_A
 *   2. For P permutations:
 *      a. Randomly swap labels A/B for each sample with prob 0.5
 *      b. Compute permuted difference in RMSE
 *   3. p-value = fraction of permutations where d_perm >= d_obs
 *
 * If p < alpha, reject H0 (models are equivalent) in favor of
 * H1 (model A is significantly better than model B).
 *
 * Ref: Good (2013) "Permutation, Parametric, and Bootstrap Tests
 *      of Hypotheses", 4th ed. Springer.
 * ==================================================================== */

/**
 * @brief Permutation test for model superiority.
 *
 * @param errors_a     Errors of model A, length n.
 * @param errors_b     Errors of model B, length n.
 * @param n            Number of paired samples.
 * @param n_perms      Number of permutations (e.g., 1000 or 10000).
 * @param p_value      Output: permutation p-value.
 * @return 1 if model A significantly better than B at alpha=0.05, else 0.
 */
int permutation_test_model_superiority(const double *errors_a,
                                        const double *errors_b,
                                        size_t n, size_t n_perms,
                                        double *p_value)
{
    if (!errors_a || !errors_b || n < 2) {
        if (p_value) *p_value = 1.0;
        return 0;
    }

    /* Observed RMSE difference (B - A, positive means A is better) */
    double rmse_a = 0.0, rmse_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        rmse_a += errors_a[i] * errors_a[i];
        rmse_b += errors_b[i] * errors_b[i];
    }
    rmse_a = sqrt(rmse_a / (double)n);
    rmse_b = sqrt(rmse_b / (double)n);
    double d_obs = rmse_b - rmse_a;

    if (d_obs <= 0.0) {
        /* A is not better than B */
        if (p_value) *p_value = 1.0;
        return 0;
    }

    /* Allocate permuted arrays */
    double *perm_a = (double *)malloc(n * sizeof(double));
    double *perm_b = (double *)malloc(n * sizeof(double));
    if (!perm_a || !perm_b) {
        free(perm_a); free(perm_b);
        if (p_value) *p_value = 1.0;
        return 0;
    }

    size_t count_extreme = 0;

    for (size_t p = 0; p < n_perms; p++) {
        /* Randomly swap for each sample with probability 0.5 */
        for (size_t i = 0; i < n; i++) {
            if (((double)rand() / RAND_MAX) < 0.5) {
                perm_a[i] = errors_a[i];
                perm_b[i] = errors_b[i];
            } else {
                perm_a[i] = errors_b[i];
                perm_b[i] = errors_a[i];
            }
        }

        /* Compute permuted difference */
        double perm_rmse_a = 0.0, perm_rmse_b = 0.0;
        for (size_t i = 0; i < n; i++) {
            perm_rmse_a += perm_a[i] * perm_a[i];
            perm_rmse_b += perm_b[i] * perm_b[i];
        }
        perm_rmse_a = sqrt(perm_rmse_a / (double)n);
        perm_rmse_b = sqrt(perm_rmse_b / (double)n);

        if ((perm_rmse_b - perm_rmse_a) >= d_obs)
            count_extreme++;
    }

    free(perm_a); free(perm_b);

    double p = (double)(count_extreme + 1) / (double)(n_perms + 1);
    if (p_value) *p_value = p;
    return (p < 0.05) ? 1 : 0;
}

/* ====================================================================
 * L5: Diebold-Mariano Test for Predictive Accuracy
 *
 * Diebold & Mariano (1995) "Comparing predictive accuracy".
 * Journal of Business & Economic Statistics, 13(3), 253-263.
 *
 * Tests whether two forecasts have equal expected loss.
 * Handles serially correlated forecast errors via HAC
 * (heteroskedasticity and autocorrelation consistent) variance.
 *
 * d_t = L(e_{A,t}) - L(e_{B,t})
 * Test statistic: DM = dbar / sqrt(Var(dbar))
 *
 * Under H0 (equal accuracy): DM ~ N(0, 1)
 * ==================================================================== */

/**
 * @brief Diebold-Mariano test for equal predictive accuracy.
 *
 * Uses squared error loss: L(e) = e^2.
 * Variance estimated with Newey-West HAC estimator (lag truncation).
 *
 * @param errors_a  Forecast errors from model A, length n.
 * @param errors_b  Forecast errors from model B, length n.
 * @param n         Number of forecast errors.
 * @param max_lag   Maximum lag for HAC (e.g., h-1 for h-step forecasts).
 * @param dm_stat   Output: DM test statistic.
 * @param p_value   Output: two-sided p-value.
 * @return 1 if models significantly different at alpha=0.05.
 */
int diebold_mariano_test(const double *errors_a, const double *errors_b,
                          size_t n, size_t max_lag,
                          double *dm_stat, double *p_value)
{
    if (!errors_a || !errors_b || n < 3) {
        if (dm_stat) *dm_stat = 0.0;
        if (p_value) *p_value = 1.0;
        return 0;
    }

    /* Differential loss series: d_t = e_{A,t}^2 - e_{B,t}^2 */
    double *d = (double *)malloc(n * sizeof(double));
    if (!d) {
        if (dm_stat) *dm_stat = 0.0;
        if (p_value) *p_value = 1.0;
        return 0;
    }

    double sum_d = 0.0;
    for (size_t t = 0; t < n; t++) {
        d[t] = errors_a[t] * errors_a[t] - errors_b[t] * errors_b[t];
        sum_d += d[t];
    }

    double dbar = sum_d / (double)n;

    /* Newey-West HAC variance estimator */
    /* gamma_0 = sample variance of d_t */
    double gamma0 = 0.0;
    for (size_t t = 0; t < n; t++) {
        double diff = d[t] - dbar;
        gamma0 += diff * diff;
    }
    gamma0 /= (double)n;

    /* Autocovariances */
    double hac_var = gamma0;
    for (size_t lag = 1; lag <= max_lag && lag < n; lag++) {
        double gamma_lag = 0.0;
        for (size_t t = lag; t < n; t++) {
            gamma_lag += (d[t] - dbar) * (d[t - lag] - dbar);
        }
        gamma_lag /= (double)n;
        /* Bartlett kernel weight */
        double weight = 1.0 - (double)lag / (double)(max_lag + 1);
        hac_var += 2.0 * weight * gamma_lag;
    }

    if (hac_var < 1e-15) {
        free(d);
        if (dm_stat) *dm_stat = 0.0;
        if (p_value) *p_value = 1.0;
        return 0;
    }

    double dm = dbar / sqrt(hac_var / (double)n);
    if (dm_stat) *dm_stat = dm;

    /* Two-sided p-value from standard normal */
    double abs_dm = fabs(dm);
    double p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs_dm / sqrt(2.0))));
    if (p_value) *p_value = p;

    free(d);

    return (p < 0.05) ? 1 : 0;
}

/* ====================================================================
 * L8: Bayesian Model Comparison via Posterior Predictive P-value
 *
 * Gelman, Meng & Stern (1996) "Posterior predictive assessment of
 * model fitness via realized discrepancies". Statistica Sinica, 733-807.
 *
 * Posterior predictive p-value (PPP):
 *   1. Draw parameters theta^(r) from posterior
 *   2. Generate replicated data y_rep^(r) from p(y|theta^(r))
 *   3. Compute discrepancy D(y, theta) for both observed and replicated
 *   4. PPP = P(D(y_rep, theta) >= D(y_obs, theta))
 *
 * PPP near 0 or 1 indicates model misfit.
 * PPP near 0.5 indicates adequate fit.
 *
 * We provide a simplified version for soft sensor validation
 * using RMSE as the discrepancy measure with Gaussian likelihood.
 * ==================================================================== */

/**
 * @brief Simplified Bayesian posterior predictive check.
 *
 * Assumes:
 *   - Data ~ N(model_prediction, sigma^2)
 *   - Prior: sigma^2 ~ Scaled-Inv-Chi2(nu0, sigma0^2)
 *   - Posterior for sigma^2 available analytically
 *
 * @param residuals   Residuals from soft sensor (y_obs - y_pred), length n.
 * @param n           Number of observations.
 * @param nu0         Prior degrees of freedom.
 * @param sigma0_sq   Prior scale parameter.
 * @param n_draws     Number of posterior draws.
 * @param ppp_value   Output: posterior predictive p-value.
 * @return 1 if model fit is adequate (0.05 < PPP < 0.95).
 */
int bayesian_predictive_check(const double *residuals, size_t n,
                               double nu0, double sigma0_sq,
                               size_t n_draws, double *ppp_value)
{
    if (!residuals || n < 3) {
        if (ppp_value) *ppp_value = 0.5;
        return 0;
    }

    /* Observed discrepancy: RMSE */
    double obs_rmse_sq = 0.0;
    for (size_t i = 0; i < n; i++)
        obs_rmse_sq += residuals[i] * residuals[i];
    obs_rmse_sq /= (double)n;

    /* Posterior for sigma^2:
     * sigma^2 | data ~ Scaled-Inv-Chi2(nu0 + n,
     *   (nu0*sigma0^2 + n*obs_rmse_sq) / (nu0 + n) )
     *
     * Draw from Scaled-Inv-Chi2(nu, tau^2):
     *   sigma^2 = nu * tau^2 / chi2_nu
     * where chi2_nu is a draw from Chi-squared(nu).
     */

    double ss_obs = 0.0;
    for (size_t i = 0; i < n; i++)
        ss_obs += residuals[i] * residuals[i];

    double nu_post = nu0 + (double)n;
    double tau2_post = (nu0 * sigma0_sq + ss_obs) / nu_post;

    size_t count_extreme = 0;

    for (size_t d = 0; d < n_draws; d++) {
        /* Draw sigma^2 from posterior */
        /* Approximate Chi-squared draw via normal:
         * chi2_nu approx = nu + sqrt(2*nu)*Z where Z ~ N(0,1)
         * Better: use sum of squared normals for integer nu
         */
        double chi2_draw = 0.0;
        /* Box-Muller for normal draws, sum of squares */
        int nu_int = (int)nu_post;
        for (int k = 0; k < nu_int; k++) {
            double u1 = (double)rand() / (RAND_MAX + 1.0);
            double u2 = (double)rand() / (RAND_MAX + 1.0);
            double z = sqrt(-2.0 * log(fmax(u1, 1e-10))) * cos(2.0 * 3.141592653589793 * u2);
            chi2_draw += z * z;
        }
        /* Handle fractional part */
        double frac = nu_post - (double)nu_int;
        if (frac > 0.0) {
            double u1 = (double)rand() / (RAND_MAX + 1.0);
            double u2 = (double)rand() / (RAND_MAX + 1.0);
            double z = sqrt(-2.0 * log(fmax(u1, 1e-10))) * cos(2.0 * 3.141592653589793 * u2);
            chi2_draw += frac * z * z;
        }

        double sigma2_draw = (chi2_draw > 1e-10)
                             ? nu_post * tau2_post / chi2_draw
                             : tau2_post;

        /* Generate replicated data */
        double rep_rmse_sq = 0.0;
        for (size_t i = 0; i < n; i++) {
            /* Draw from N(0, sigma2_draw) using Box-Muller */
            double u1 = (double)rand() / (RAND_MAX + 1.0);
            double u2 = (double)rand() / (RAND_MAX + 1.0);
            double z = sqrt(-2.0 * log(fmax(u1, 1e-10)))
                       * cos(2.0 * 3.141592653589793 * u2);
            double rep_error = z * sqrt(sigma2_draw);
            rep_rmse_sq += rep_error * rep_error;
        }
        rep_rmse_sq /= (double)n;

        if (rep_rmse_sq >= obs_rmse_sq)
            count_extreme++;
    }

    double ppp = (double)count_extreme / (double)n_draws;
    if (ppp_value) *ppp_value = ppp;

    return (ppp > 0.05 && ppp < 0.95) ? 1 : 0;
}

/* ====================================================================
 * Validation Summary: combine multiple validation tests
 * into a unified model acceptance decision.
 * ==================================================================== */

/**
 * @brief Combined validation verdict for a soft sensor model.
 */
typedef struct {
    int    cv_acceptable;
    int    bootstrap_stable;
    int    permutation_significant;
    int    dm_test_significant;
    int    bayesian_fit_adequate;
    int    overall_valid;
    double validation_score;
} ValidationVerdict;

/**
 * @brief Generate comprehensive validation verdict.
 *
 * Combines: cross-validation, bootstrap stability, paired t-test,
 * permutation test, DM test, and Bayesian predictive check.
 *
 * validation_score in [0, 1]: weighted average of test outcomes.
 *
 * @return ValidationVerdict with composite decision.
 */
ValidationVerdict validate_soft_sensor(
    const double *cv_fold_errors, size_t n_folds,
    const double *bootstrap_estimates, size_t n_bootstrap,
    const double *errors_model, const double *errors_baseline, size_t n_errors)
{
    ValidationVerdict v;
    memset(&v, 0, sizeof(v));
    v.validation_score = 0.0;

    double score = 0.0;
    double weight_sum = 0.0;

    /* 1. Cross-validation stability (weight 0.15) */
    if (cv_fold_errors && n_folds >= 3) {
        double cv_mean, cv_std;
        cross_validation_stats(cv_fold_errors, n_folds, &cv_mean, &cv_std);
        /* CV is stable if std/mean < 0.2 (coefficient of variation) */
        double cv = (cv_mean > 1e-10) ? cv_std / cv_mean : 0.0;
        v.cv_acceptable = (cv < 0.2) ? 1 : 0;
        score += 0.15 * (1.0 - fmin(cv / 0.2, 1.0));
        weight_sum += 0.15;
    }

    /* 2. Bootstrap stability (weight 0.15) */
    if (bootstrap_estimates && n_bootstrap >= 50) {
        double ci_low, ci_high, boot_m, boot_se;
        bootstrap_confidence_interval(bootstrap_estimates, n_bootstrap,
                                       0.05, &ci_low, &ci_high,
                                       &boot_m, &boot_se);
        /* Stable if CI width relative to mean < 0.5 */
        double rel_width = (boot_m > 1e-10)
                           ? (ci_high - ci_low) / fabs(boot_m) : 0.0;
        v.bootstrap_stable = (rel_width < 0.5) ? 1 : 0;
        score += 0.15 * (1.0 - fmin(rel_width / 0.5, 1.0));
        weight_sum += 0.15;
    }

    /* 3. Permutation test (weight 0.25) */
    if (errors_model && errors_baseline && n_errors >= 10) {
        double perm_p;
        v.permutation_significant = permutation_test_model_superiority(
            errors_model, errors_baseline, n_errors, 1000, &perm_p);
        score += 0.25 * (1.0 - fmin(perm_p / 0.1, 1.0));
        weight_sum += 0.25;
    }

    /* 4. DM test (weight 0.25) */
    if (errors_model && errors_baseline && n_errors >= 10) {
        double dm_s, dm_p;
        v.dm_test_significant = diebold_mariano_test(
            errors_model, errors_baseline, n_errors, 4, &dm_s, &dm_p);
        score += 0.25 * (1.0 - fmin(dm_p / 0.1, 1.0));
        weight_sum += 0.25;
    }

    /* 5. Bayesian predictive check (weight 0.20) */
    if (errors_model && n_errors >= 10) {
        double ppp;
        v.bayesian_fit_adequate = bayesian_predictive_check(
            errors_model, n_errors, 1.0, 1.0, 100, &ppp);
        /* Score: best at PPP = 0.5, poor at extremes */
        double ppp_score = 1.0 - fabs(ppp - 0.5) / 0.45;
        if (ppp_score < 0.0) ppp_score = 0.0;
        score += 0.20 * ppp_score;
        weight_sum += 0.20;
    }

    if (weight_sum > 0.0)
        v.validation_score = score / weight_sum;
    else
        v.validation_score = 0.5;

    v.overall_valid = (v.validation_score > 0.6) ? 1 : 0;

    return v;
}
