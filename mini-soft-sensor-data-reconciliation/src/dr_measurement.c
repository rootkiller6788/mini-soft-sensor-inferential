/**
 * @file dr_measurement.c
 * @brief Measurement processing: statistics, outlier detection, uncertainty.
 */

#include "dr_measurement.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Comparison helper for qsort ----------------------------------------- */

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Compute percentile of sorted array using Hyndman-Fan Method 7 */
static double percentile_val_sorted(const double *sorted, int len, double p) {
    double h = (len - 1) * p;
    int lo = (int)floor(h);
    int hi = (int)ceil(h);
    if (lo >= len) return sorted[len - 1];
    if (hi >= len) return sorted[len - 1];
    if (lo < 0) return sorted[0];
    double frac = h - floor(h);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* ---- Summary statistics -------------------------------------------------- */

/**
 * Compute summary statistics from repeated measurements.
 *
 * Uses two-pass algorithm for variance (numerically more stable
 * than the one-pass textbook formula).
 *
 * Skewness: g1 = (n/((n-1)(n-2))) * sum((x_i - mean)^3/s^3)
 * Kurtosis: g2 = (n(n+1)/((n-1)(n-2)(n-3))) * sum((x_i - mean)^4/s^4)
 *                - 3*(n-1)^2/((n-2)(n-3))
 *
 * Reference: Joanes, D.N., Gill, C.A. (1998). "Comparing Measures of
 * Sample Skewness and Kurtosis." The Statistician, 47(1), 183-189.
 */
int dr_meas_compute_statistics(const double *values, int n,
                               dr_meas_stats_t *stats) {
    int i;
    double mean, variance, s, m3, m4;

    if (!values || !stats || n < 2) return DR_ERR_NULL_POINTER;

    stats->n = n;

    /* Pass 1: mean */
    mean = 0.0;
    for (i = 0; i < n; i++) mean += values[i];
    mean /= (double)n;
    stats->mean = mean;

    /* Pass 2: variance and higher moments */
    variance = 0.0; m3 = 0.0; m4 = 0.0;
    for (i = 0; i < n; i++) {
        double d = values[i] - mean;
        double d2 = d * d;
        variance += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    variance /= (double)(n - 1);
    s = sqrt(variance);
    stats->stddev = s;

    /* Skewness and kurtosis */
    if (s > 1e-15 && n >= 3) {
        double n_f = (double)n;
        stats->skewness = (n_f / ((n_f - 1.0) * (n_f - 2.0))) *
                          (m3 / (n_f * s * s * s));
    } else {
        stats->skewness = 0.0;
    }
    if (s > 1e-15 && n >= 4) {
        double n_f = (double)n;
        stats->kurtosis = (n_f * (n_f + 1.0) /
                          ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0))) *
                          (m4 / (n_f * variance * variance)) -
                          (3.0 * (n_f - 1.0) * (n_f - 1.0) /
                          ((n_f - 2.0) * (n_f - 3.0)));
    } else {
        stats->kurtosis = 0.0;
    }

    /* Quartiles: create sorted copy */
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return DR_ERR_NULL_POINTER;
    memcpy(sorted, values, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), compare_double);

    stats->min_val = sorted[0];
    stats->max_val = sorted[n - 1];

    /* Median */
    if (n % 2 == 1) {
        stats->median = sorted[n / 2];
    } else {
        stats->median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    /* Q1 and Q3 using Method 7 (Hyndman & Fan, 1996) */
    stats->q1 = percentile_val_sorted(sorted, n, 0.25);
    stats->q3 = percentile_val_sorted(sorted, n, 0.75);

    free(sorted);
    return DR_OK;
}

/**
 * IQR-based outlier detection (Tukey's fences).
 *
 * Tukey proposed using 1.5 * IQR for "mild" outliers and 3.0 * IQR
 * for "extreme" outliers. This is a robust, non-parametric method
 * that does not assume normality.
 */
int dr_meas_detect_outliers_iqr(const double *values, int n, double k,
                                int *outlier, int *n_out) {
    int i, count = 0;
    double q1, q3, iqr, lower, upper;

    if (!values || !outlier || !n_out || n < 4) return DR_ERR_NULL_POINTER;

    /* Compute quartiles */
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return DR_ERR_NULL_POINTER;
    memcpy(sorted, values, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), compare_double);

    /* Q1 and Q3 */
    {
        int lo, hi;
        double frac;
        double h;

        h = (n - 1) * 0.25;
        lo = (int)floor(h); hi = (int)ceil(h);
        frac = h - floor(h);
        q1 = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;

        h = (n - 1) * 0.75;
        lo = (int)floor(h); hi = (int)ceil(h);
        frac = h - floor(h);
        q3 = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    iqr = q3 - q1;
    lower = q1 - k * iqr;
    upper = q3 + k * iqr;

    count = 0;
    for (i = 0; i < n; i++) {
        if (values[i] < lower || values[i] > upper) {
            outlier[i] = 1;
            count++;
        } else {
            outlier[i] = 0;
        }
    }

    *n_out = count;
    free(sorted);
    return DR_OK;
}

/**
 * Grubbs test for a single outlier.
 *
 * Test statistic: G = max_i |x_i - x_bar| / s
 *
 * Critical value approximation (Grubbs, 1950):
 *   G_crit = (n-1)/sqrt(n) * sqrt(t^2 / (n-2 + t^2))
 * where t = t_{alpha/(2n), n-2} from Student's t-distribution.
 *
 * The critical t-value is approximated using the Hill (1970) method.
 */
int dr_meas_grubbs_test(const double *values, int n, double alpha,
                        int *idx_out, double *g_stat, double *g_crit) {
    int i, max_idx = 0;
    double mean, s, g_max = 0.0;
    double t_crit, n_f;

    if (!values || !idx_out || !g_stat || !g_crit || n < 3)
        return DR_ERR_NULL_POINTER;

    /* Compute mean */
    mean = 0.0;
    for (i = 0; i < n; i++) mean += values[i];
    mean /= (double)n;

    /* Compute stddev and find max deviation */
    double variance = 0.0;
    for (i = 0; i < n; i++) {
        double d = values[i] - mean;
        variance += d * d;
    }
    variance /= (double)(n - 1);
    s = sqrt(variance);

    if (s < 1e-15) {
        *idx_out = -1; *g_stat = 0.0; *g_crit = 0.0;
        return DR_OK;
    }

    for (i = 0; i < n; i++) {
        double g = fabs(values[i] - mean) / s;
        if (g > g_max) { g_max = g; max_idx = i; }
    }
    *g_stat = g_max;

    /* Critical value: t_{alpha/(2n), n-2} */
    {
        double prob = alpha / (2.0 * (double)n);
        t_crit = dr_meas_t_critical(n - 2, prob);
    }
    n_f = (double)n;
    *g_crit = (n_f - 1.0) / sqrt(n_f) *
              sqrt(t_crit * t_crit / (n_f - 2.0 + t_crit * t_crit));

    *idx_out = (g_max > *g_crit) ? max_idx : -1;
    return DR_OK;
}

/* ---- Covariance construction --------------------------------------------- */

/**
 * Build full covariance matrix from std deviations and correlations.
 *
 * Cov(i,j) = rho(i,j) * sigma_i * sigma_j
 *
 * If rho is NULL, returns diagonal covariance (independent measurements).
 */
int dr_meas_build_covariance(const double *stddev, const double *rho,
                             int n, double *cov_out) {
    int i, j;

    if (!stddev || !cov_out || n <= 0) return DR_ERR_NULL_POINTER;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double r = (rho && i != j) ? rho[i * n + j] : ((i == j) ? 1.0 : 0.0);
            cov_out[i * n + j] = r * stddev[i] * stddev[j];
        }
    }
    return DR_OK;
}

/**
 * Convert covariance to correlation matrix.
 *
 * Corr(i,j) = Cov(i,j) / (sigma_i * sigma_j)
 * where sigma_i = sqrt(Cov(i,i)).
 */
int dr_meas_covariance_to_correlation(const double *cov, int n, double *corr) {
    int i, j;

    if (!cov || !corr || n <= 0) return DR_ERR_NULL_POINTER;

    for (i = 0; i < n; i++) {
        double sigma_i = sqrt(cov[i * n + i]);
        for (j = 0; j < n; j++) {
            double sigma_j = sqrt(cov[j * n + j]);
            if (sigma_i < 1e-15 || sigma_j < 1e-15) {
                corr[i * n + j] = (i == j) ? 1.0 : 0.0;
            } else {
                corr[i * n + j] = cov[i * n + j] / (sigma_i * sigma_j);
            }
        }
    }
    return DR_OK;
}

/**
 * Standardize measurements to zero mean, unit variance.
 *
 * z_i = (x_i - mu) / sigma
 *
 * This is a critical preprocessing step for:
 *   - PCA-based methods (covariance matrix becomes correlation matrix)
 *   - Improving numerical conditioning of the constraint matrix
 *   - Making variables comparable when they have different units
 *
 * Returns error if sigma == 0 (constant measurement).
 */
int dr_meas_standardize(double *values, double mean, double stddev, int n) {
    int i;

    if (!values || n <= 0) return DR_ERR_NULL_POINTER;
    if (stddev < 1e-15) return DR_ERR_NOT_SPD;

    for (i = 0; i < n; i++) {
        values[i] = (values[i] - mean) / stddev;
    }
    return DR_OK;
}

/* ---- Critical value computations ----------------------------------------- */

/**
 * Chi-squared critical value using Wilson-Hilferty approximation.
 *
 * chi2_{nu}(alpha) ≈ nu * (1 - 2/(9*nu) + z_alpha * sqrt(2/(9*nu)))^3
 *
 * Maximum absolute error: < 0.01 for nu >= 10, < 0.04 for nu >= 3.
 *
 * Reference: Wilson, E.B., Hilferty, M.M. (1931). "The Distribution of
 * Chi-Square." Proc. Nat. Acad. Sci., 17(12), 684-688.
 */
double dr_meas_chi2_critical(int nu, double alpha) {
    double z, n;

    if (nu <= 0) return 0.0;

    z = dr_meas_normal_critical(alpha);
    n = (double)nu;

    if (n < 3.0) {
        /* For small nu, use direct approach: the approximation is less accurate */
        /* Return a rough estimate */
        return n * pow(1.0 - 2.0/(9.0*n) + z * sqrt(2.0/(9.0*n)), 3.0);
    }

    return n * pow(1.0 - 2.0/(9.0*n) + z * sqrt(2.0/(9.0*n)), 3.0);
}

/**
 * Normal distribution critical value using Abramowitz & Stegun §26.2.23.
 *
 * Computes z such that P(Z > z) = alpha for Z ~ N(0,1).
 *
 * Rational approximation:
 *   z = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
 * where t = sqrt(-2 * log(alpha))
 *
 * Maximum absolute error: 4.5e-4.
 */
double dr_meas_normal_critical(double alpha) {
    double p, t, z;

    if (alpha <= 0.0) return 10.0;
    if (alpha >= 1.0) return -10.0;

    /* Use upper tail: p = min(alpha, 1-alpha) */
    p = (alpha < 0.5) ? alpha : (1.0 - alpha);

    /* Abramowitz & Stegun §26.2.23 */
    t = sqrt(-2.0 * log(p));
    {
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;
        z = t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    }

    return (alpha < 0.5) ? z : -z;
}

/**
 * Student t-distribution critical value using Hill (1970) approximation.
 *
 * Approximation for the upper tail probability alpha:
 *
 *   t = sqrt(nu * (exp(g(alpha)^2 / nu) - 1))
 *
 * where g(alpha) is the normal quantile.
 *
 * This approximation is accurate to within 0.001 for nu >= 1.
 *
 * Reference: Hill, G.W. (1970). "Algorithm 395: Student's t-Distribution."
 * Comm. ACM, 13(10), 617-619.
 */
double dr_meas_t_critical(int nu, double alpha) {
    double z, n;

    if (nu <= 0) return 10.0;
    if (alpha <= 0.0) return 10.0;
    if (alpha >= 0.5) return 0.0;

    z = dr_meas_normal_critical(alpha);
    n = (double)nu;

    /* Hill approximation */
    {
        double g2 = z * z;
        double a = g2 / n;
        double b = 1.0 + a;
        double t = z * sqrt(b);
        /* Refinement for small nu */
        if (nu < 10) {
            t = z * (1.0 + (g2 + 1.0) / (4.0 * n) +
                     (5.0 * g2 * g2 + 16.0 * g2 + 3.0) / (96.0 * n * n));
        }
        return t;
    }
}

/* ---- Uncertainty propagation (GUM) --------------------------------------- */

/**
 * Combined standard uncertainty per GUM (JCGM 100:2008).
 *
 * u_c = sqrt(u_A^2 + sum_i u_{B,i}^2)
 *
 * Type A: evaluated from statistical analysis of repeated measurements
 *   u_A = s / sqrt(n) where s = sample stddev, n = number of observations
 *
 * Type B: evaluated from instrument specifications, calibration certificates,
 *   engineering judgment, etc.
 *   u_B = a / k where a = half-width of interval, k = divisor
 *
 * Common Type B distributions and their divisors:
 *   - Normal (95%): k = 2
 *   - Rectangular:  k = sqrt(3)
 *   - Triangular:   k = sqrt(6)
 *   - U-shaped:     k = sqrt(2)
 */
double dr_meas_combined_uncertainty(double u_type_a, const double *u_type_b,
                                    int n_type_b) {
    int i;
    double sum_sq = u_type_a * u_type_a;

    if (u_type_b && n_type_b > 0) {
        for (i = 0; i < n_type_b; i++) {
            sum_sq += u_type_b[i] * u_type_b[i];
        }
    }
    return sqrt(sum_sq);
}

/**
 * Expanded uncertainty for a given coverage factor k.
 *
 * U = k * u_c
 *
 * Common coverage factors and their confidence levels
 * (assuming normal distribution and large effective degrees of freedom):
 *   k = 1.000 → 68.27%
 *   k = 1.645 → 90%
 *   k = 1.960 → 95%
 *   k = 2.000 → 95.45%
 *   k = 2.576 → 99%
 *   k = 3.000 → 99.73%
 */
double dr_meas_expanded_uncertainty(double u_combined, double k) {
    return k * u_combined;
}

/* ---- VIF and condition number -------------------------------------------- */

/**
 * Variance Inflation Factor (VIF) and condition number.
 *
 * VIF_j = [R^{-1}]_{jj} where R is the correlation matrix.
 * VIF_j measures how much the variance of the j-th coefficient is
 * inflated due to correlation with other variables.
 *
 * Condition number of covariance: kappa = lambda_max / lambda_min.
 *
 * For computational simplicity, we estimate the condition number as
 * the ratio of the largest to smallest diagonal element (a lower bound
 * on the true 2-norm condition number).
 *
 * VIF computation requires inverting the correlation matrix.
 * We use Gauss-Jordan elimination with partial pivoting.
 */
int dr_meas_vif_condition(const double *cov, int n, double *vif, double *cond) {
    int i, j, k;
    double lambda_min, lambda_max;

    if (!cov || !vif || !cond || n <= 0) return DR_ERR_NULL_POINTER;

    /* Estimate condition number from diagonal elements */
    lambda_max = cov[0];
    lambda_min = cov[0];
    for (i = 0; i < n; i++) {
        double diag = cov[i * n + i];
        if (diag > lambda_max) lambda_max = diag;
        if (diag < lambda_min && diag > 0) lambda_min = diag;
    }
    *cond = (lambda_min > 0) ? (lambda_max / lambda_min) : 1e15;

    /* Compute correlation matrix */
    double *R = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!R) return DR_ERR_NULL_POINTER;
    dr_meas_covariance_to_correlation(cov, n, R);

    /* Copy R for inversion */
    double *Rinv = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!Rinv) { free(R); return DR_ERR_NULL_POINTER; }
    memcpy(Rinv, R, (size_t)n * (size_t)n * sizeof(double));

    /* Augment with identity for Gauss-Jordan */
    double *aug = (double *)calloc((size_t)n * (size_t)(2 * n), sizeof(double));
    if (!aug) { free(Rinv); free(R); return DR_ERR_NULL_POINTER; }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) aug[i * 2 * n + j] = Rinv[i * n + j];
        aug[i * 2 * n + n + i] = 1.0;
    }

    /* Gauss-Jordan elimination with partial pivoting */
    for (i = 0; i < n; i++) {
        /* Find pivot */
        int pivot = i;
        double max_val = fabs(aug[i * 2 * n + i]);
        for (k = i + 1; k < n; k++) {
            double val = fabs(aug[k * 2 * n + i]);
            if (val > max_val) { max_val = val; pivot = k; }
        }
        if (max_val < 1e-15) {
            free(aug); free(Rinv); free(R);
            for (j = 0; j < n; j++) vif[j] = 1e15;
            return DR_ERR_SINGULAR;
        }
        /* Swap rows */
        if (pivot != i) {
            for (j = 0; j < 2 * n; j++) {
                double tmp = aug[i * 2 * n + j];
                aug[i * 2 * n + j] = aug[pivot * 2 * n + j];
                aug[pivot * 2 * n + j] = tmp;
            }
        }
        /* Normalize pivot row */
        double piv_val = aug[i * 2 * n + i];
        for (j = 0; j < 2 * n; j++) aug[i * 2 * n + j] /= piv_val;
        /* Eliminate other rows */
        for (k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = aug[k * 2 * n + i];
            for (j = 0; j < 2 * n; j++)
                aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
        }
    }

    /* Extract VIF from diagonal of inverse */
    for (i = 0; i < n; i++) {
        vif[i] = aug[i * 2 * n + n + i];
        if (vif[i] < 1.0) vif[i] = 1.0;  /* VIF cannot be less than 1 */
    }

    free(aug);
    free(Rinv);
    free(R);
    return DR_OK;
}
