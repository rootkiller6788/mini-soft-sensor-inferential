/**
 * pca_kernel.c -- Kernel PCA for Nonlinear Process Monitoring
 *
 * Knowledge Coverage:
 *   L5: Kernel PCA with RBF, polynomial, and linear kernels
 *   L8: Nonlinear process monitoring via feature space mapping
 *   L8: Kernel trick for implicit high-dimensional mapping
 *
 * KPCA extends linear PCA to nonlinear data by:
 *   1. Mapping data to a high-dimensional feature space F via phi(x)
 *   2. Performing linear PCA in F
 *   3. Using the kernel trick: k(x,y) = <phi(x), phi(y)> avoids explicit phi
 *
 * The N x N centered kernel matrix is eigendecomposed:
 *   K_centered * alpha_k = lambda_k * alpha_k
 * where alpha_k are the dual eigenvectors (N-dimensional).
 *
 * New data projection:
 *   t_k(new) = sum_{i=1..N} alpha_{k,i} * k_centered(x_i, x_new)
 *
 * Reference: Scholkopf, Smola & Muller (1998),
 *            Lee, J.M. et al. (2004) "Nonlinear process monitoring using KPCA"
 * Course Alignment: MIT 6.302, Stanford EE392, CMU 24-677
 */
#include "pca_kernel.h"
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ===================================================================
 * L5: Kernel function computation
 *
 * RBF (Gaussian):     k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
 *   sigma controls the width; typical choice: sigma = c * median(dist)
 *
 * Polynomial:         k(x,y) = (gamma*<x,y> + coef0)^degree
 *   degree=2, gamma=1, coef0=1 is common
 *
 * Linear:             k(x,y) = <x,y>
 *   Recovers standard linear PCA
 * =================================================================== */
double kpca_kernel_compute(const double *x, const double *y, size_t n_vars,
                           pca_kernel_type ktype, double sigma,
                           double gamma, double coef0, int degree)
{
    size_t j;
    double dot, dist_sq;

    if (!x || !y) return 0.0;

    switch (ktype) {
    case PCA_KERNEL_RBF:
        dist_sq = 0.0;
        for (j = 0; j < n_vars; j++) {
            double d = x[j] - y[j];
            dist_sq += d * d;
        }
        return exp(-dist_sq / (2.0 * sigma * sigma));

    case PCA_KERNEL_POLYNOMIAL:
        dot = 0.0;
        for (j = 0; j < n_vars; j++)
            dot += x[j] * y[j];
        {
            double base = gamma * dot + coef0;
            double result = 1.0;
            int d;
            if (degree < 0) degree = 0;
            for (d = 0; d < degree; d++) result *= base;
            return result;
        }

    case PCA_KERNEL_LINEAR:
    default:
        dot = 0.0;
        for (j = 0; j < n_vars; j++)
            dot += x[j] * y[j];
        return dot;
    }
}

/* ===================================================================
 * L5: Build N x N kernel (Gram) matrix
 *
 * K[i][j] = k(x_i, x_j) for all i,j in 1..N
 *
 * The kernel matrix is symmetric and positive semi-definite
 * (Mercer's theorem ensures this for valid kernel functions).
 * =================================================================== */
int kpca_build_kernel_matrix(const pca_matrix *X, pca_matrix *K,
                             pca_kernel_type ktype, double sigma,
                             double gamma, double coef0, int degree)
{
    size_t N, M, i, j;

    if (!X || !X->data || !K || !K->data) return -1;
    N = X->rows; M = X->cols;
    if (K->rows != N || K->cols != N) return -1;

    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            double kval = kpca_kernel_compute(
                &X->data[i * M], &X->data[j * M], M,
                ktype, sigma, gamma, coef0, degree);
            K->data[i * N + j] = kval;
            K->data[j * N + i] = kval;  /* symmetry */
        }
    }
    return 0;
}

/* ===================================================================
 * L5: Center the kernel matrix in feature space
 *
 * In feature space F, we need centered data: phi_c(x) = phi(x) - mean_phi
 * where mean_phi = (1/N) * sum_i phi(x_i)
 *
 * The centered kernel matrix is:
 *   K_centered = K - 1_N * K - K * 1_N + 1_N * K * 1_N
 *
 * where 1_N is an N x N matrix with all entries = 1/N.
 *
 * This can be computed efficiently as:
 *   Kc[i][j] = K[i][j] - row_mean[i] - row_mean[j] + grand_mean
 *
 * where row_mean[i] = (1/N) * sum_j K[i][j]
 *       grand_mean   = (1/N^2) * sum_i sum_j K[i][j]
 *
 * Reference: Scholkopf et al. (1998), Eq. 5
 * =================================================================== */
int kpca_center_kernel_matrix(pca_matrix *K, double *row_mean, double *grand_mean)
{
    size_t N, i, j;
    double gm = 0.0;

    if (!K || !K->data) return -1;
    N = K->rows;

    /* Compute row means and grand mean */
    for (i = 0; i < N; i++) {
        double sum = 0.0;
        for (j = 0; j < N; j++)
            sum += K->data[i * N + j];
        row_mean[i] = sum / (double)N;
        gm += row_mean[i];
    }
    gm /= (double)N;
    if (grand_mean) *grand_mean = gm;

    /* Center: Kc[i][j] = K[i][j] - rm[i] - rm[j] + gm */
    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            double kc = K->data[i * N + j] - row_mean[i] - row_mean[j] + gm;
            K->data[i * N + j] = kc;
            K->data[j * N + i] = kc;
        }
    }

    return 0;
}

/* ===================================================================
 * L5/L8: Train Kernel PCA model
 *
 * Steps:
 *   1. Build kernel matrix from training data X
 *   2. Center the kernel matrix in feature space
 *   3. Eigendecompose the centered kernel matrix Kc:
 *        Kc * alpha_k = lambda_k * alpha_k
 *      Note: eigenvalues of Kc are N times the eigenvalues of the
 *      covariance matrix in feature space. The scores are:
 *        t_k = sqrt(lambda_k) * alpha_k (properly normalized)
 *   4. Normalize eigenvectors: alpha_k = alpha_k / ||alpha_k||
 *      (in feature space: alpha_k^T * alpha_k = 1/lambda_k)
 *   5. Compute monitoring thresholds for T2 and SPE
 *
 * The number of retained kernel PCs (n_pcs) is typically much smaller
 * than N, as KPCA can capture complex nonlinear structure with few PCs.
 * =================================================================== */
int kpca_train(kpca_model *model, const pca_matrix *X,
               size_t max_sweeps, double tol, double alpha_sig)
{
    size_t N, M, i, a;
    pca_matrix *K, *V;
    double *eig_K;

    if (!model || !X || !X->data) return -1;
    N = X->rows; M = X->cols;
    if (N != model->n_train || M != model->n_vars) return -1;

    /* Store training data (needed for online projection) */
    if (model->X_train) pca_matrix_free(model->X_train);
    model->X_train = pca_matrix_copy(X);
    if (!model->X_train) return -1;

    /* Step 1: Build kernel matrix */
    K = pca_matrix_alloc(N, N);
    if (!K) return -1;
    kpca_build_kernel_matrix(X, K, model->ktype, model->sigma,
                             model->gamma, model->coef0, model->degree);

    /* Step 2: Center kernel matrix */
    {
        double *rm = (double*)calloc(N, sizeof(double));
        double gm = 0.0;
        if (!rm) { pca_matrix_free(K); return -1; }
        kpca_center_kernel_matrix(K, rm, &gm);
        /* Store centering parameters for online use */
        if (model->row_mean_K) free(model->row_mean_K);
        model->row_mean_K = rm;
        model->grand_mean_K = gm;
    }

    /* Step 3: Eigendecompose Kc */
    eig_K = (double*)calloc(N, sizeof(double));
    V = pca_matrix_alloc(N, N);
    if (!eig_K || !V) {
        free(eig_K); pca_matrix_free(V); pca_matrix_free(K); return -1;
    }

    if (pca_jacobi_eigen(K, eig_K, V, max_sweeps, tol) != 0) {
        pca_jacobi_eigen(K, eig_K, V, max_sweeps * 2, tol * 10);
    }

    /* Sort eigenvalues/vectors descending */
    {
        size_t ii, jj;
        for (ii = 0; ii < N - 1; ii++) {
            size_t max_idx = ii;
            for (jj = ii + 1; jj < N; jj++)
                if (eig_K[jj] > eig_K[max_idx]) max_idx = jj;
            if (max_idx != ii) {
                double tmp = eig_K[ii];
                eig_K[ii] = eig_K[max_idx];
                eig_K[max_idx] = tmp;
                for (a = 0; a < N; a++) {
                    double tv = V->data[a * N + ii];
                    V->data[a * N + ii] = V->data[a * N + max_idx];
                    V->data[a * N + max_idx] = tv;
                }
            }
        }
    }

    /* Store results */
    if (model->eigenvalues) free(model->eigenvalues);
    model->eigenvalues = eig_K;

    /* Step 4: Normalize eigenvectors in feature space
     * alpha_k = alpha_k / sqrt(lambda_k) for lambda_k > 0 */
    if (model->alpha) pca_matrix_free(model->alpha);
    model->alpha = V;

    for (a = 0; a < N; a++) {
        double lam = model->eigenvalues[a];
        if (lam > 1e-12) {
            double inv_sqrt = 1.0 / sqrt(lam);
            for (i = 0; i < N; i++)
                model->alpha->data[i * N + a] *= inv_sqrt;
        } else {
            /* Zero out eigenvectors for near-zero eigenvalues */
            for (i = 0; i < N; i++)
                model->alpha->data[i * N + a] = 0.0;
        }
    }

    /* Store centered kernel matrix */
    if (model->K_centered) pca_matrix_free(model->K_centered);
    model->K_centered = K;

    /* Step 5: Compute thresholds */
    kpca_compute_thresholds(model, alpha_sig);

    return 0;
}

/* ===================================================================
 * L5: KPCA model lifecycle
 * =================================================================== */
kpca_model* kpca_model_alloc(size_t n_train, size_t n_vars, size_t n_pcs,
                             pca_kernel_type ktype)
{
    kpca_model *model;
    if (n_train == 0 || n_vars == 0) return NULL;

    model = (kpca_model*)calloc(1, sizeof(kpca_model));
    if (!model) return NULL;

    model->n_train = n_train;
    model->n_vars = n_vars;
    model->n_pcs = n_pcs;
    model->ktype = ktype;

    /* Default kernel parameters */
    model->sigma = 1.0;
    model->gamma = 1.0;
    model->coef0 = 1.0;
    model->degree = 2;

    model->X_train = NULL;
    model->eigenvalues = NULL;
    model->alpha = NULL;
    model->K_centered = NULL;
    model->row_mean_K = NULL;
    model->grand_mean_K = 0.0;
    model->t2_threshold = 0.0;
    model->spe_threshold = 0.0;

    return model;
}

void kpca_model_free(kpca_model *model)
{
    if (!model) return;
    pca_matrix_free(model->X_train);
    pca_matrix_free(model->alpha);
    pca_matrix_free(model->K_centered);
    free(model->eigenvalues);
    free(model->row_mean_K);
    model->X_train = NULL;
    model->alpha = NULL;
    model->K_centered = NULL;
    model->eigenvalues = NULL;
    model->row_mean_K = NULL;
    free(model);
}

/* ===================================================================
 * L5: Project new observation onto kernel PCs
 *
 * For a new point x_new, the k-th kernel PC score is:
 *   t_k = sum_{i=1..N} alpha_{k,i} * k_centered(x_i, x_new)
 *
 * where the centered kernel vector is:
 *   k_centered(x_i, x_new) = k(x_i, x_new) - row_mean[i]
 *     - (1/N) * sum_{j=1..N} k(x_j, x_new) + grand_mean
 *
 * This implements the centering correction for out-of-sample data.
 * =================================================================== */
int kpca_project(const kpca_model *model, const double *x_new, double *t_scores)
{
    size_t N, M, i, a;
    double *k_vec, sum_k, *kc_vec;

    if (!model || !x_new || !t_scores || !model->X_train) return -1;
    N = model->n_train;
    M = model->n_vars;

    k_vec = (double*)malloc(N * sizeof(double));
    kc_vec = (double*)malloc(N * sizeof(double));
    if (!k_vec || !kc_vec) { free(k_vec); free(kc_vec); return -1; }

    /* Compute raw kernel values */
    sum_k = 0.0;
    for (i = 0; i < N; i++) {
        k_vec[i] = kpca_kernel_compute(
            &model->X_train->data[i * M], x_new, M,
            model->ktype, model->sigma, model->gamma, model->coef0, model->degree);
        sum_k += k_vec[i];
    }

    /* Center the kernel vector */
    for (i = 0; i < N; i++) {
        kc_vec[i] = k_vec[i] - model->row_mean_K[i]
                    - sum_k / (double)N + model->grand_mean_K;
    }

    /* Project onto each retained PC */
    for (a = 0; a < model->n_pcs; a++) {
        double t = 0.0;
        for (i = 0; i < N; i++) {
            t += model->alpha->data[i * N + a] * kc_vec[i];
        }
        t_scores[a] = t;
    }

    free(k_vec); free(kc_vec);
    return 0;
}

/* ===================================================================
 * L5: KPCA T2 statistic
 *
 * T2 = sum_{k=1..A} t_k^2 / lambda_k
 *
 * For KPCA, lambda_k are the eigenvalues of the centered kernel matrix
 * divided by N (to match the scale of the feature-space covariance).
 *
 * The T2 threshold uses the same F-distribution formula as linear PCA,
 * applied to the retained kernel PCs.
 * =================================================================== */
double kpca_t2_statistic(const kpca_model *model, const double *t_scores)
{
    size_t a;
    double t2 = 0.0;

    if (!model || !t_scores) return 0.0;

    for (a = 0; a < model->n_pcs; a++) {
        double lam = model->eigenvalues[a];
        if (lam < 1e-12) lam = 1e-12;
        t2 += (t_scores[a] * t_scores[a]) / lam;
    }

    return t2;
}

/* ===================================================================
 * L5: KPCA SPE statistic
 *
 * SPE = ||phi(x) - phi_hat(x)||^2
 *     = k(x, x)_centered - sum_{k=1..A} t_k^2
 *
 * where k(x,x)_centered = k(x,x) - 2*sum_i k(x,x_i)/N + grand_mean
 *
 * This measures the distance between the full feature-space
 * representation and its projection onto the retained PCs.
 * =================================================================== */
double kpca_spe_statistic(const kpca_model *model, const double *x_new,
                          const double *t_scores)
{
    size_t N, M, i, a;
    double kxx, sum_k, kxx_centered, sum_t2;

    if (!model || !x_new || !t_scores) return 0.0;
    N = model->n_train;
    M = model->n_vars;

    /* k(x_new, x_new) */
    kxx = kpca_kernel_compute(x_new, x_new, M, model->ktype,
                              model->sigma, model->gamma, model->coef0, model->degree);

    /* Average kernel to training data */
    sum_k = 0.0;
    for (i = 0; i < N; i++) {
        sum_k += kpca_kernel_compute(
            &model->X_train->data[i * M], x_new, M,
            model->ktype, model->sigma, model->gamma, model->coef0, model->degree);
    }

    /* Centered self-kernel */
    kxx_centered = kxx - 2.0 * sum_k / (double)N + model->grand_mean_K;
    if (kxx_centered < 0.0) kxx_centered = 0.0;

    /* Sum of squared scores */
    sum_t2 = 0.0;
    for (a = 0; a < model->n_pcs; a++) {
        sum_t2 += t_scores[a] * t_scores[a];
    }

    return kxx_centered - sum_t2;
}

/* ===================================================================
 * L5: Compute T2 and SPE thresholds for KPCA monitoring
 *
 * T2 threshold: same F-distribution formula as linear PCA
 *   T2_lim = A*(N-1)*(N+1)/(N*(N-A)) * F(alpha, A, N-A)
 *
 * SPE threshold: based on the residual eigenvalues of Kc/N
 *   using the Jackson-Mudholkar approximation on the
 *   feature-space residual eigenvalues.
 * =================================================================== */
void kpca_compute_thresholds(kpca_model *model, double alpha_sig)
{
    size_t N, A;
    (void)alpha_sig; /* reserved for future alpha-dependent threshold tuning */
    double *residual_eig;
    size_t i;

    if (!model || !model->eigenvalues) return;
    N = model->n_train;
    A = model->n_pcs;
    if (A == 0) A = 1;

    /* T2 threshold via F-distribution */
    {
        double d1 = (double)A;
        double d2 = (double)(N - A);
        if (d2 <= 0.0) d2 = 1.0;
        /* Simplified: use chi-squared approximation */
        model->t2_threshold = d1 * (double)(N - 1) * (double)(N + 1)
                              / ((double)N * d2);
        /* Multiply by approximate F-critical value (~2.0 for typical alpha) */
        model->t2_threshold *= 2.0;
    }

    /* SPE threshold from residual eigenvalues */
    residual_eig = (double*)calloc(N - A, sizeof(double));
    if (!residual_eig) return;

    for (i = A; i < N; i++) {
        double lam = model->eigenvalues[i];
        residual_eig[i - A] = (lam > 0.0) ? lam : 0.0;
    }

    /* Jackson-Mudholkar on residual eigenvalues */
    {
        double theta1 = 0.0, theta2 = 0.0, theta3 = 0.0;
        double h0, spe_lim;
        size_t j;

        for (j = 0; j < N - A; j++) {
            double lam = residual_eig[j];
            theta1 += lam;
            theta2 += lam * lam;
            theta3 += lam * lam * lam;
        }

        if (theta2 > 1e-15) {
            h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * theta2 * theta2);
            spe_lim = theta1 * pow(1.0 + 2.0 * sqrt(theta2) / theta1, 1.0 / h0);
            model->spe_threshold = spe_lim;
        } else {
            model->spe_threshold = 0.0;
        }
    }

    free(residual_eig);
}
