/**
 * pca_core.c -- PCA Core Matrix Operations and Model Management
 *
 * Knowledge Coverage:
 *   L1: PCA model data structures, eigenvalues, variance explained ratio
 *   L2: Centering, scaling, covariance/correlation computation
 *   L3: Row-major 2D matrix storage, memory management patterns
 *   L5: Kaiser criterion, cumulative variance rule for PC selection
 *
 * Reference: Jolliffe (2002) Ch.3, Ch.6; Golub & Van Loan (2013) Ch.8
 * Course Alignment: MIT 2.171, Stanford ENGR205, CMU 24-677
 */
#include "pca_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ===================================================================
 * L3: Matrix allocation and management
 * Row-major storage: element (i,j) at data[i*cols + j]
 * =================================================================== */

pca_matrix* pca_matrix_alloc(size_t rows, size_t cols)
{
    pca_matrix *mat;
    if (rows == 0 || cols == 0) {
        fprintf(stderr, "pca_matrix_alloc: invalid dims (%zu x %zu)\n", rows, cols);
        return NULL;
    }
    mat = (pca_matrix*)malloc(sizeof(pca_matrix));
    if (!mat) { perror("pca_matrix_alloc"); return NULL; }
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc(rows * cols, sizeof(double));
    if (!mat->data) { free(mat); perror("pca_matrix_alloc data"); return NULL; }
    return mat;
}

void pca_matrix_free(pca_matrix *mat)
{
    if (mat) { free(mat->data); mat->data = NULL; free(mat); }
}

double pca_matrix_get(const pca_matrix *mat, size_t row, size_t col)
{
    if (!mat || !mat->data) return 0.0;
    if (row >= mat->rows || col >= mat->cols) return 0.0;
    return mat->data[row * mat->cols + col];
}

void pca_matrix_set(pca_matrix *mat, size_t row, size_t col, double value)
{
    if (!mat || !mat->data) return;
    if (row >= mat->rows || col >= mat->cols) return;
    mat->data[row * mat->cols + col] = value;
}

pca_matrix* pca_matrix_copy(const pca_matrix *src)
{
    pca_matrix *dst;
    size_t n;
    if (!src || !src->data) return NULL;
    dst = pca_matrix_alloc(src->rows, src->cols);
    if (!dst) return NULL;
    n = src->rows * src->cols;
    memcpy(dst->data, src->data, n * sizeof(double));
    return dst;
}

void pca_matrix_print(const pca_matrix *mat, const char *name)
{
    size_t i, j, mr, mc;
    if (!mat) { printf("%s: (null)\n", name ? name : "matrix"); return; }
    printf("%s [%zu x %zu]:\n", name ? name : "matrix", mat->rows, mat->cols);
    mr = mat->rows < 10 ? mat->rows : 10;
    mc = mat->cols < 10 ? mat->cols : 10;
    for (i = 0; i < mr; i++) {
        printf("  ");
        for (j = 0; j < mc; j++)
            printf("%10.4f ", mat->data[i * mat->cols + j]);
        if (mat->cols > 10) printf("...");
        printf("\n");
    }
    if (mat->rows > 10) printf("  ... (%zu more rows)\n", mat->rows - 10);
}

/* ===================================================================
 * L1/L2: PCA model lifecycle
 * =================================================================== */

pca_model* pca_model_alloc(size_t n_vars)
{
    pca_model *model;
    size_t j;
    if (n_vars == 0) return NULL;
    model = (pca_model*)calloc(1, sizeof(pca_model));
    if (!model) return NULL;
    model->n_vars = n_vars;
    model->n_obs = 0;
    model->use_scaling = 1;
    model->eigenvalues = (double*)calloc(n_vars, sizeof(double));
    model->mean_vec     = (double*)calloc(n_vars, sizeof(double));
    model->std_vec      = (double*)calloc(n_vars, sizeof(double));
    model->var_expl     = (double*)calloc(n_vars, sizeof(double));
    model->cum_var      = (double*)calloc(n_vars, sizeof(double));
    model->loadings = pca_matrix_alloc(n_vars, n_vars);
    if (!model->eigenvalues || !model->mean_vec || !model->std_vec ||
        !model->var_expl || !model->cum_var || !model->loadings) {
        pca_model_free(model); return NULL;
    }
    for (j = 0; j < n_vars; j++) model->std_vec[j] = 1.0;
    return model;
}

void pca_model_free(pca_model *model)
{
    if (!model) return;
    pca_matrix_free(model->loadings);
    pca_matrix_free(model->scores);
    free(model->eigenvalues); model->eigenvalues = NULL;
    free(model->mean_vec);    model->mean_vec = NULL;
    free(model->std_vec);     model->std_vec = NULL;
    free(model->var_expl);    model->var_expl = NULL;
    free(model->cum_var);     model->cum_var = NULL;
    model->loadings = NULL;
    model->scores = NULL;
    free(model);
}

/* ===================================================================
 * L2: Data preprocessing -- centering and scaling
 *
 * Centering: x_ij = x_ij_raw - mean_j
 *   mean_j = (1/N) * sum_i x_ij_raw
 *
 * Auto-scaling (center + scale to unit variance):
 *   std_j = sqrt( sum_i (x_ij_raw - mean_j)^2 / (N-1) )
 *   x_ij_scaled = (x_ij_raw - mean_j) / std_j
 *
 * Auto-scaling makes PCA invariant to measurement units -- all variables
 * contribute equally regardless of their original scale (e.g., temperature
 * in degrees vs pressure in MPa).
 * =================================================================== */

void pca_center_columns(pca_matrix *X, double *mean_vec)
{
    size_t i, j, N, M;
    if (!X || !X->data || !mean_vec) return;
    N = X->rows; M = X->cols;
    /* Compute column means */
    for (j = 0; j < M; j++) {
        double sum = 0.0;
        for (i = 0; i < N; i++) sum += X->data[i * M + j];
        mean_vec[j] = sum / (double)N;
    }
    /* Subtract means */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
            X->data[i * M + j] -= mean_vec[j];
}

void pca_scale_columns(pca_matrix *X, double *std_vec)
{
    size_t i, j, N, M;
    const double eps = 1e-12;
    if (!X || !X->data || !std_vec) return;
    N = X->rows; M = X->cols;
    /* Compute sample standard deviations (assumes X is centered) */
    for (j = 0; j < M; j++) {
        double sum_sq = 0.0;
        for (i = 0; i < N; i++) {
            double v = X->data[i * M + j];
            sum_sq += v * v;
        }
        std_vec[j] = sqrt(sum_sq / (double)(N - 1));
        if (std_vec[j] < eps) std_vec[j] = 1.0; /* avoid division by zero */
    }
    /* Divide by standard deviations */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++)
            X->data[i * M + j] /= std_vec[j];
}

void pca_center_scale(pca_matrix *X, double *mean_vec, double *std_vec)
{
    pca_center_columns(X, mean_vec);
    pca_scale_columns(X, std_vec);
}

/* ===================================================================
 * L2: Covariance and correlation matrix computation
 *
 * Sample covariance matrix S (M x M):
 *   S[j][k] = (1/(N-1)) * sum_{i=1..N} (x_ij - m_j)(x_ik - m_k)
 *
 * For centered X (mean=0 per column):
 *   S = (1/(N-1)) * X^T * X
 *
 * Theorem: S is symmetric positive semi-definite. All eigenvalues >= 0.
 *
 * Correlation matrix R from covariance S and std devs sigma:
 *   R[j][k] = S[j][k] / (sigma_j * sigma_k)
 * =================================================================== */

pca_matrix* pca_compute_covariance(const pca_matrix *X)
{
    size_t N, M, j, k, i;
    pca_matrix *cov;
    if (!X || !X->data) return NULL;
    N = X->rows; M = X->cols;
    if (N < 2) { fprintf(stderr, "covariance: need N>=2\n"); return NULL; }
    cov = pca_matrix_alloc(M, M);
    if (!cov) return NULL;
    /* Compute upper triangle and copy to lower (symmetry) */
    for (j = 0; j < M; j++) {
        for (k = j; k < M; k++) {
            double sum = 0.0;
            for (i = 0; i < N; i++)
                sum += X->data[i * M + j] * X->data[i * M + k];
            cov->data[j * M + k] = sum / (double)(N - 1);
            cov->data[k * M + j] = cov->data[j * M + k];
        }
    }
    return cov;
}

pca_matrix* pca_compute_correlation(const pca_matrix *cov, const double *std_vec)
{
    size_t M, j, k;
    pca_matrix *corr;
    if (!cov || !std_vec) return NULL;
    M = cov->rows;
    corr = pca_matrix_alloc(M, M);
    if (!corr) return NULL;
    for (j = 0; j < M; j++) {
        for (k = j; k < M; k++) {
            corr->data[j * M + k] = cov->data[j * M + k] / (std_vec[j] * std_vec[k]);
            corr->data[k * M + j] = corr->data[j * M + k];
        }
    }
    return corr;
}

/* ===================================================================
 * L1: Variance explained per principal component
 *
 * var_expl[k] = lambda_k / sum(lambda)
 * cum_var[k]  = sum_{i=0..k} var_expl[i]
 *
 * These quantify how much of the total data variability is captured
 * by the first k principal components.
 * =================================================================== */

void pca_compute_variance_explained(double *eigenvalues, size_t M,
                                    double *var_expl, double *cum_var)
{
    size_t k;
    double total = 0.0;
    if (!eigenvalues || !var_expl || !cum_var || M == 0) return;
    for (k = 0; k < M; k++) total += eigenvalues[k];
    if (total < 1e-15) {
        for (k = 0; k < M; k++) { var_expl[k] = 0.0; cum_var[k] = 0.0; }
        return;
    }
    cum_var[0] = eigenvalues[0] / total;
    var_expl[0] = cum_var[0];
    for (k = 1; k < M; k++) {
        var_expl[k] = eigenvalues[k] / total;
        cum_var[k] = cum_var[k-1] + var_expl[k];
    }
}

/* ===================================================================
 * L5: Kaiser criterion (Kaiser, 1960)
 *
 * Retain principal components whose eigenvalue exceeds the average
 * eigenvalue. For correlation-based PCA, eigenvalues sum to M, so
 * the average is 1.0, giving the standard rule: retain PCs with
 * eigenvalue > 1.
 *
 * Intuition: A PC with eigenvalue < 1 explains less variance than
 * a single original standardized variable.
 * =================================================================== */

size_t pca_kaiser_rule(const double *eigenvalues, size_t M)
{
    size_t k, n = 0;
    double mean_eig = 0.0;
    if (!eigenvalues || M == 0) return 0;
    for (k = 0; k < M; k++) mean_eig += eigenvalues[k];
    mean_eig /= (double)M;
    for (k = 0; k < M; k++)
        if (eigenvalues[k] > mean_eig) n++;
    if (n == 0 && eigenvalues[0] > 0.0) n = 1;
    return n;
}

/* ===================================================================
 * L5: Cumulative variance rule for PC selection
 *
 * Select the smallest k such that cum_var[k-1] >= threshold.
 * Common threshold values:
 *   0.85 (85%) -- exploratory data analysis
 *   0.90 (90%) -- soft sensor with moderate accuracy requirements
 *   0.95 (95%) -- process monitoring where reconstruction matters
 *
 * Returns k (1-indexed count of PCs to retain).
 * =================================================================== */

size_t pca_cumvar_rule(const double *cum_var, size_t M, double threshold)
{
    size_t k;
    if (!cum_var || M == 0) return 0;
    if (threshold < 0.0) threshold = 0.0;
    if (threshold > 1.0) threshold = 1.0;
    for (k = 0; k < M; k++)
        if (cum_var[k] >= threshold) return k + 1;
    return M;
}
