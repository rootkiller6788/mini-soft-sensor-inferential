#include "pls_preprocessing.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])

/* Comparison function for qsort */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

PreprocessingConfig preproc_config_default(void) {
    PreprocessingConfig c;
    c.x_method = PREPROC_MEAN_CENTER;
    c.y_method = PREPROC_MEAN_CENTER;
    return c;
}

int preprocess_fit(Matrix *X, Matrix *Y,
                   const PreprocessingConfig *config,
                   Vector *x_mean, Vector *x_std,
                   Vector *y_mean, Vector *y_std) {
    if (!X || !Y || !config) return -1;
    if (!x_mean || !x_std || !y_mean || !y_std) return -1;
    if (x_mean->len != X->cols || y_mean->len != Y->cols) return -1;

    /* Compute and store parameters for X */
    for (size_t j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < X->rows; i++) sum += MATRIX_AT(X, i, j);
        x_mean->data[j] = sum / (double)X->rows;
    }
    for (size_t j = 0; j < X->cols; j++) {
        double ss = 0.0;
        for (size_t i = 0; i < X->rows; i++) {
            double d = MATRIX_AT(X, i, j) - x_mean->data[j];
            ss += d * d;
        }
        x_std->data[j] = sqrt(ss / (double)X->rows);
    }

    /* Apply X preprocessing in-place */
    if (config->x_method == PREPROC_MEAN_CENTER || config->x_method == PREPROC_AUTOSCALE ||
        config->x_method == PREPROC_PARETO || config->x_method == PREPROC_RANGE) {
        matrix_center_columns(X, x_mean);
    }
    if (config->x_method == PREPROC_AUTOSCALE)
        matrix_scale_columns(X, x_std);
    else if (config->x_method == PREPROC_PARETO) {
        for (size_t j = 0; j < X->cols; j++) {
            double sd = sqrt(x_std->data[j]);
            if (sd < DBL_EPSILON) continue;
            for (size_t i = 0; i < X->rows; i++) MATRIX_AT(X, i, j) /= sd;
        }
    }

    /* Compute and store parameters for Y */
    for (size_t j = 0; j < Y->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < Y->rows; i++) sum += MATRIX_AT(Y, i, j);
        y_mean->data[j] = sum / (double)Y->rows;
    }
    for (size_t j = 0; j < Y->cols; j++) {
        double ss = 0.0;
        for (size_t i = 0; i < Y->rows; i++) {
            double d = MATRIX_AT(Y, i, j) - y_mean->data[j];
            ss += d * d;
        }
        y_std->data[j] = sqrt(ss / (double)Y->rows);
    }

    /* Apply Y preprocessing in-place */
    if (config->y_method == PREPROC_MEAN_CENTER || config->y_method == PREPROC_AUTOSCALE) {
        matrix_center_columns(Y, y_mean);
    }
    if (config->y_method == PREPROC_AUTOSCALE)
        matrix_scale_columns(Y, y_std);

    return 0;
}

void preprocess_transform(Matrix *X,
                          const Vector *x_mean, const Vector *x_std,
                          int center_x, int scale_x) {
    if (!X || !x_mean) return;
    if (center_x) matrix_center_columns(X, x_mean);
    if (scale_x && x_std) matrix_scale_columns(X, x_std);
}

void preprocess_inverse(Matrix *Y_scaled,
                        const Vector *y_mean, const Vector *y_std,
                        int center_y, int scale_y) {
    if (!Y_scaled) return;
    if (scale_y && y_std) {
        for (size_t j = 0; j < Y_scaled->cols; j++) {
            double sd = y_std->data[j];
            if (sd < DBL_EPSILON) sd = 1.0;
            for (size_t i = 0; i < Y_scaled->rows; i++)
                MATRIX_AT(Y_scaled, i, j) *= sd;
        }
    }
    if (center_y && y_mean) {
        for (size_t j = 0; j < Y_scaled->cols; j++)
            for (size_t i = 0; i < Y_scaled->rows; i++)
                MATRIX_AT(Y_scaled, i, j) += y_mean->data[j];
    }
}

void mean_center_matrix(Matrix *A) {
    if (!A) return;
    Vector *means = matrix_column_means(A);
    if (!means) return;
    matrix_center_columns(A, means);
    vector_free(means);
}

void autoscale_matrix(Matrix *A) {
    mean_center_matrix(A);
    Vector *stds = matrix_column_stddevs(A);
    if (!stds) return;
    matrix_scale_columns(A, stds);
    vector_free(stds);
}

void pareto_scale_matrix(Matrix *A) {
    if (!A) return;
    Vector *means = matrix_column_means(A);
    if (!means) return;
    matrix_center_columns(A, means);
    for (size_t j = 0; j < A->cols; j++) {
        double sum_sq = 0.0;
        for (size_t i = 0; i < A->rows; i++)
            sum_sq += MATRIX_AT(A, i, j) * MATRIX_AT(A, i, j);
        double sd = sqrt(sum_sq / (double)A->rows);
        double ps = (sd > DBL_EPSILON) ? sqrt(sd) : 1.0;
        for (size_t i = 0; i < A->rows; i++)
            MATRIX_AT(A, i, j) /= ps;
    }
    vector_free(means);
}

void range_scale_matrix(Matrix *A) {
    if (!A || A->rows == 0) return;
    for (size_t j = 0; j < A->cols; j++) {
        double vmin = MATRIX_AT(A, 0, j), vmax = MATRIX_AT(A, 0, j);
        for (size_t i = 1; i < A->rows; i++) {
            double v = MATRIX_AT(A, i, j);
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        double rng = vmax - vmin;
        if (rng < DBL_EPSILON) {
            for (size_t i = 0; i < A->rows; i++) MATRIX_AT(A, i, j) = 0.5;
        } else {
            for (size_t i = 0; i < A->rows; i++)
                MATRIX_AT(A, i, j) = (MATRIX_AT(A, i, j) - vmin) / rng;
        }
    }
}

double compute_mad(const Vector *v) {
    if (!v || v->len == 0) return 0.0;

    /* Copy and sort to find median */
    double *sorted = (double*)malloc(v->len * sizeof(double));
    if (!sorted) return 0.0;
    for (size_t i = 0; i < v->len; i++) sorted[i] = v->data[i];
    qsort(sorted, v->len, sizeof(double), cmp_double);

    double median;
    if (v->len % 2 == 1)
        median = sorted[v->len / 2];
    else
        median = (sorted[v->len / 2 - 1] + sorted[v->len / 2]) / 2.0;

    /* Compute absolute deviations */
    double *absdev = (double*)malloc(v->len * sizeof(double));
    if (!absdev) { free(sorted); return 0.0; }
    for (size_t i = 0; i < v->len; i++)
        absdev[i] = fabs(v->data[i] - median);

    qsort(absdev, v->len, sizeof(double), cmp_double);

    double mad;
    if (v->len % 2 == 1)
        mad = absdev[v->len / 2];
    else
        mad = (absdev[v->len / 2 - 1] + absdev[v->len / 2]) / 2.0;

    free(sorted);
    free(absdev);

    /* Consistency factor for normal distribution: 1.4826 */
    return mad * 1.4826;
}

void robust_scale_matrix(Matrix *A) {
    if (!A || A->rows == 0) return;

    for (size_t j = 0; j < A->cols; j++) {
        Vector *col = matrix_get_column(A, j);
        if (!col) continue;

        /* Find median */
        double *sorted = (double*)malloc(col->len * sizeof(double));
        if (!sorted) { vector_free(col); continue; }
        for (size_t i = 0; i < col->len; i++) sorted[i] = col->data[i];
        qsort(sorted, col->len, sizeof(double), cmp_double);

        double median;
        if (col->len % 2 == 1)
            median = sorted[col->len / 2];
        else
            median = (sorted[col->len / 2 - 1] + sorted[col->len / 2]) / 2.0;
        free(sorted);

        /* Subtract median */
        for (size_t i = 0; i < A->rows; i++)
            MATRIX_AT(A, i, j) -= median;

        /* Scale by MAD */
        double mad = compute_mad(col);
        if (mad > DBL_EPSILON)
            for (size_t i = 0; i < A->rows; i++)
                MATRIX_AT(A, i, j) /= mad;

        vector_free(col);
    }
}
