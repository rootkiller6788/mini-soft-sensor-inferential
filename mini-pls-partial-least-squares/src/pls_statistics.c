#include "pls_statistics.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

double matrix_sum_of_squares(const Matrix *A) {
    if (!A || !A->data) return 0.0;
    double ss = 0.0;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++)
        ss += SQ(A->data[i]);
    return ss;
}

double compute_R2X(const Matrix *X_orig, const Matrix *X_resid) {
    if (!X_orig || !X_resid) return -1.0;
    double SS_X = matrix_sum_of_squares(X_orig);
    double SS_E = matrix_sum_of_squares(X_resid);
    if (SS_X < DBL_EPSILON) return 1.0;
    return 1.0 - SS_E / SS_X;
}

double compute_R2Y(const Matrix *Y_orig, const Matrix *Y_resid) {
    if (!Y_orig || !Y_resid) return -1.0;
    double SS_Y = matrix_sum_of_squares(Y_orig);
    double SS_F = matrix_sum_of_squares(Y_resid);
    if (SS_Y < DBL_EPSILON) return 1.0;
    return 1.0 - SS_F / SS_Y;
}

double compute_Q2(double PRESS, double SSY_total) {
    if (SSY_total < DBL_EPSILON) return 1.0;
    return 1.0 - PRESS / SSY_total;
}

/* ---- Hotelling T2 ---- */

double compute_T2_single(const Vector *t_scores, const Vector *score_variances) {
    if (!t_scores || !score_variances || t_scores->len != score_variances->len)
        return 0.0;
    double T2 = 0.0;
    for (size_t i = 0; i < t_scores->len; i++) {
        if (score_variances->data[i] > DBL_EPSILON)
            T2 += SQ(t_scores->data[i]) / score_variances->data[i];
    }
    return T2;
}

Vector* compute_T2_batch(const Matrix *T_scores, const Vector *score_variances) {
    if (!T_scores || !score_variances || T_scores->cols != score_variances->len)
        return NULL;
    Vector *T2s = vector_alloc(T_scores->rows);
    if (!T2s) return NULL;
    for (size_t i = 0; i < T_scores->rows; i++) {
        double T2 = 0.0;
        for (size_t a = 0; a < T_scores->cols; a++) {
            if (score_variances->data[a] > DBL_EPSILON)
                T2 += SQ(MATRIX_AT(T_scores, i, a)) / score_variances->data[a];
        }
        T2s->data[i] = T2;
    }
    return T2s;
}

Vector* compute_score_variances(const Matrix *T_scores) {
    if (!T_scores || T_scores->rows < 2) return NULL;
    Vector *vars = vector_alloc(T_scores->cols);
    if (!vars) return NULL;
    for (size_t a = 0; a < T_scores->cols; a++) {
        double mean = 0.0, m2 = 0.0;
        for (size_t i = 0; i < T_scores->rows; i++)
            mean += MATRIX_AT(T_scores, i, a);
        mean /= (double)T_scores->rows;
        for (size_t i = 0; i < T_scores->rows; i++) {
            double d = MATRIX_AT(T_scores, i, a) - mean;
            m2 += d * d;
        }
        vars->data[a] = m2 / (double)(T_scores->rows - 1);
        if (vars->data[a] < DBL_EPSILON) vars->data[a] = 1e-12;
    }
    return vars;
}

/* ---- Normal Quantile (Abramowitz & Stegun 26.2.23) ---- */

double normal_quantile(double p) {
    if (p <= 0.0) return -10.0;
    if (p >= 1.0) return  10.0;
    /* Rational approximation, relative error < 4.5e-4 */
    double q = p - 0.5;
    double r;
    if (fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        return q * (((((((2.5090809287301226 * r + 3.3430575583588128) * r
                    + 6.7265770927008701) * r + 4.5919177025199998) * r
                    + 1.3709225039999999) * r + 0.24256483640000001) * r
                    + 0.021869503800000001) * r + 0.00044367843700000001)
                    / (((((((1.0 * r + 1.3754825980999999) * r
                    + 0.80385127249999996) * r + 0.25721213930000002) * r
                    + 0.048786882970000001) * r + 0.0054980050700000001) * r
                    + 0.00034089465500000001) * r + 0.0000096797283999999997);
    } else {
        r = (q > 0) ? 1.0 - p : p;
        r = sqrt(-log(r));
        double val;
        if (r <= 5.0) {
            r -= 1.6;
            val = (((((((7.7454501427834144e-4 * r + 0.022723844989269184) * r
                      + 0.2417807251774506) * r + 1.2704582524523684) * r
                      + 3.6478483247632045) * r + 5.7694972214606914) * r
                      + 4.6303378461565453) * r + 1.4234371107496835);
        } else {
            r -= 5.0;
            val = (((((((1.0507500716444168e-9 * r + 5.475938084995345e-4) * r
                      + 0.015198666563616457) * r + 0.14810397642748008) * r
                      + 0.68976733498510001) * r + 1.6763848301838038) * r
                      + 2.0531916266377588) * r + 1.0);
        }
        return (q > 0) ? val : -val;
    }
}

/* ---- F-distribution Critical Value (Wilson-Hilferty approx) ---- */

double f_distribution_critical(double d1, double d2, double alpha) {
    if (d1 <= 0 || d2 <= 0 || alpha <= 0 || alpha >= 1) return 1.0;
    /* Wilson-Hilferty transformation: F ~ chi-square / d1
     * Convert to normal approximation then invert */
    double z = normal_quantile(1.0 - alpha);

    /* Use the approximation from Paulson (1942) / Severo-Zelen (1960) */
    double a __attribute__((unused)) = d2 / 2.0;
    double b __attribute__((unused)) = d1 / 2.0;
    double lambda = (pow(z, 2.0) - 3.0) / 6.0;
    double h = 2.0 / (1.0 / (d1 - 1.0) + 1.0 / (d2 - 1.0));
    double w = z * sqrt(h + lambda) / h -
               (1.0 / (d2 - 1.0) - 1.0 / (d1 - 1.0)) * (lambda + 5.0 / 6.0 - 2.0 / (3.0 * h));

    return exp(2.0 * w);
}

double compute_T2_limit(size_t n_samples, size_t n_lvs, double alpha) {
    if (n_samples <= n_lvs || n_lvs == 0) return 0.0;
    double A = (double)n_lvs;
    double n = (double)n_samples;
    double F = f_distribution_critical(A, n - A, alpha);
    return A * (n * n - 1.0) / (n * (n - A)) * F;
}

/* ---- SPE (Q-statistic) ---- */

double compute_SPE_single(const Vector *x_residual) {
    if (!x_residual) return 0.0;
    double spe = 0.0;
    for (size_t i = 0; i < x_residual->len; i++)
        spe += SQ(x_residual->data[i]);
    return spe;
}

Vector* compute_SPE_batch(const Matrix *X_residuals) {
    if (!X_residuals) return NULL;
    Vector *spes = vector_alloc(X_residuals->rows);
    if (!spes) return NULL;
    for (size_t i = 0; i < X_residuals->rows; i++) {
        double spe = 0.0;
        for (size_t j = 0; j < X_residuals->cols; j++)
            spe += SQ(MATRIX_AT(X_residuals, i, j));
        spes->data[i] = spe;
    }
    return spes;
}

void compute_SPE_theta(const Matrix *X_residuals,
                       double *theta1, double *theta2, double *theta3) {
    if (!X_residuals || !theta1 || !theta2 || !theta3) return;
    size_t n = X_residuals->rows, p = X_residuals->cols;

    /* theta1 = mean of SPE across training samples */
    double t1 = 0.0, t2 = 0.0, t3 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double spe_i = 0.0;
        for (size_t j = 0; j < p; j++)
            spe_i += SQ(MATRIX_AT(X_residuals, i, j));
        t1 += spe_i;
        t2 += spe_i * spe_i;
        t3 += spe_i * spe_i * spe_i;
    }
    *theta1 = t1 / (double)n;
    *theta2 = t2 / (double)n - SQ(*theta1);
    *theta3 = t3 / (double)n - 3.0 * (*theta1) * (*theta2) - SQ(*theta1) * (*theta1);
    if (*theta2 < 0) *theta2 = 0.0;
    if (*theta3 < 0) *theta3 = 0.0;
}

double compute_SPE_limit(double theta1, double theta2, double theta3, double alpha) {
    if (theta2 < DBL_EPSILON) return theta1 * 2.0;
    double h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * SQ(theta2));
    if (h0 < 0.1) h0 = 0.1;
    double ca = normal_quantile(1.0 - alpha);
    double term1 = ca * sqrt(2.0 * theta2 * SQ(h0)) / theta1;
    double term2 = theta2 * h0 * (h0 - 1.0) / SQ(theta1);
    double base = 1.0 + term1 + term2;
    if (base < 0) base = 0.0;
    return theta1 * pow(base, 1.0 / h0);
}

/* ---- VIP ---- */

Vector* compute_SSY_per_LV(const PLSModel *model) {
    if (!model || !model->T || !model->B_inner) return NULL;
    Vector *ssy = vector_alloc(model->a_lvs);
    if (!ssy) return NULL;
    for (size_t a = 0; a < model->a_lvs; a++) {
        double ta_norm_sq = 0.0;
        for (size_t i = 0; i < model->n_samples; i++)
            ta_norm_sq += SQ(MATRIX_AT(model->T, i, a));
        double ba = MATRIX_AT(model->B_inner, a, a);
        ssy->data[a] = SQ(ba) * ta_norm_sq;
    }
    return ssy;
}

Vector* compute_VIP(const PLSModel *model) {
    if (!model || !model->W) return NULL;
    Vector *ssy = compute_SSY_per_LV(model);
    if (!ssy) return NULL;

    double ssy_total = 0.0;
    for (size_t a = 0; a < model->a_lvs; a++)
        ssy_total += ssy->data[a];

    Vector *vip = vector_alloc(model->p_vars);
    if (!vip) { vector_free(ssy); return NULL; }

    for (size_t j = 0; j < model->p_vars; j++) {
        double sum_w = 0.0;
        for (size_t a = 0; a < model->a_lvs; a++) {
            double w_ja = MATRIX_AT(model->W, j, a);
            sum_w += SQ(w_ja) * ssy->data[a];
        }
        if (ssy_total > DBL_EPSILON)
            vip->data[j] = sqrt((double)model->p_vars * sum_w / ssy_total);
        else
            vip->data[j] = 0.0;
    }
    vector_free(ssy);
    return vip;
}

/* ---- DModX ---- */

double compute_DModX_single(const Vector *x_residual, size_t p_vars, size_t a_lvs) {
    if (!x_residual || p_vars <= a_lvs) return 0.0;
    double spe = compute_SPE_single(x_residual);
    return sqrt(spe / (double)(p_vars - a_lvs));
}

Vector* compute_DModX_batch(const Matrix *X_residuals, size_t a_lvs) {
    if (!X_residuals || X_residuals->cols <= a_lvs) return NULL;
    Vector *dm = vector_alloc(X_residuals->rows);
    if (!dm) return NULL;
    size_t denom = X_residuals->cols - a_lvs;
    for (size_t i = 0; i < X_residuals->rows; i++) {
        double spe = 0.0;
        for (size_t j = 0; j < X_residuals->cols; j++)
            spe += SQ(MATRIX_AT(X_residuals, i, j));
        dm->data[i] = sqrt(spe / (double)denom);
    }
    return dm;
}

double compute_DModX_critical(const Matrix *X_residuals, size_t a_lvs, double alpha) {
    if (!X_residuals) return 0.0;
    size_t n = X_residuals->rows, p = X_residuals->cols;
    if (p <= a_lvs) return 0.0;
    /* Pooled DModX across training samples as baseline */
    double sum_dm = 0.0;
    for (size_t i = 0; i < n; i++) {
        double spe = 0.0;
        for (size_t j = 0; j < p; j++)
            spe += SQ(MATRIX_AT(X_residuals, i, j));
        sum_dm += sqrt(spe / (double)(p - a_lvs));
    }
    double mean_dm = sum_dm / (double)n;
    /* Critical = mean_DModX * F_factor (heuristic) */
    double F = f_distribution_critical((double)(p - a_lvs), (double)n, alpha);
    return mean_dm * sqrt(F);
}

/* ---- Prediction Error Metrics ---- */

double compute_PRESS(const Matrix *Y_true, const Matrix *Y_pred) {
    if (!Y_true || !Y_pred || Y_true->rows != Y_pred->rows ||
        Y_true->cols != Y_pred->cols) return -1.0;
    double press = 0.0;
    for (size_t i = 0; i < Y_true->rows; i++)
        for (size_t j = 0; j < Y_true->cols; j++) {
            double e = MATRIX_AT(Y_true, i, j) - MATRIX_AT(Y_pred, i, j);
            press += e * e;
        }
    return press;
}

double compute_RMSEC(const Matrix *Y_true, const Matrix *Y_pred) {
    if (!Y_true || !Y_pred || Y_true->rows != Y_pred->rows ||
        Y_true->cols != Y_pred->cols) return -1.0;
    double press = compute_PRESS(Y_true, Y_pred);
    return sqrt(press / (double)(Y_true->rows * Y_true->cols));
}

double compute_RMSEP(const Matrix *Y_true, const Matrix *Y_pred) {
    return compute_RMSEC(Y_true, Y_pred);  /* Same formula, different data context */
}

Vector* compute_bias(const Matrix *Y_true, const Matrix *Y_pred) {
    if (!Y_true || !Y_pred || Y_true->rows != Y_pred->rows ||
        Y_true->cols != Y_pred->cols) return NULL;
    Vector *bias = vector_alloc(Y_true->cols);
    if (!bias) return NULL;
    for (size_t j = 0; j < Y_true->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < Y_true->rows; i++)
            sum += MATRIX_AT(Y_pred, i, j) - MATRIX_AT(Y_true, i, j);
        bias->data[j] = sum / (double)Y_true->rows;
    }
    return bias;
}

Vector* compute_SEP(const Matrix *Y_true, const Matrix *Y_pred) {
    if (!Y_true || !Y_pred || Y_true->rows != Y_pred->rows ||
        Y_true->cols != Y_pred->cols || Y_true->rows < 2) return NULL;
    Vector *bias = compute_bias(Y_true, Y_pred);
    if (!bias) return NULL;
    Vector *sep = vector_alloc(Y_true->cols);
    if (!sep) { vector_free(bias); return NULL; }
    for (size_t j = 0; j < Y_true->cols; j++) {
        double ss = 0.0;
        for (size_t i = 0; i < Y_true->rows; i++) {
            double e = MATRIX_AT(Y_pred, i, j) - MATRIX_AT(Y_true, i, j)
                      - bias->data[j];
            ss += e * e;
        }
        sep->data[j] = sqrt(ss / (double)(Y_true->rows - 1));
    }
    vector_free(bias);
    return sep;
}

Vector* compute_RPD(const Matrix *Y_true, const Matrix *Y_pred) {
    if (!Y_true || !Y_pred) return NULL;
    Vector *sep = compute_SEP(Y_true, Y_pred);
    if (!sep) return NULL;
    Vector *rpd = vector_alloc(Y_true->cols);
    if (!rpd) { vector_free(sep); return NULL; }
    for (size_t j = 0; j < Y_true->cols; j++) {
        double mean = 0.0;
        for (size_t i = 0; i < Y_true->rows; i++)
            mean += MATRIX_AT(Y_true, i, j);
        mean /= (double)Y_true->rows;
        double ss = 0.0;
        for (size_t i = 0; i < Y_true->rows; i++) {
            double d = MATRIX_AT(Y_true, i, j) - mean;
            ss += d * d;
        }
        double sd = sqrt(ss / (double)(Y_true->rows - 1));
        rpd->data[j] = (sep->data[j] > DBL_EPSILON) ? sd / sep->data[j] : 999.0;
    }
    vector_free(sep);
    return rpd;
}
