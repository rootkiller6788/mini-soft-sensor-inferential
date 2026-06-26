/**
 * pca_inferential.c -- PCA-based Soft Sensor / Inferential Measurement
 *
 * Knowledge Coverage:
 *   L1: Soft sensor definition, primary/secondary variables, PCR
 *   L2: Latent variable regression, data-driven inferential sensing
 *   L5: PCR training & prediction, cross-validation, VIP scores
 *   L6: Distillation composition estimation, reactor concentration
 *
 * PCR (Principal Component Regression):
 *   y = X * beta where beta = V_A * inv(D_A) * V_A^T * X^T * Y / (N-1)
 *   V_A = M x A truncated loading matrix
 *   D_A = A x A diag(eigenvalues)
 *
 * Reference: Jolliffe (2002) Ch.8, Kresta, MacGregor & Marlin (1991)
 * Course Alignment: MIT 2.171, Stanford ENGR205, Purdue ME 575
 */
#include "pca_inferential.h"
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Allocate soft sensor model */
pca_soft_sensor* pca_soft_sensor_alloc(size_t n_secondary, size_t n_primary, size_t n_pcs)
{
    pca_soft_sensor *ss;
    if (n_secondary == 0 || n_primary == 0) return NULL;
    ss = (pca_soft_sensor*)calloc(1, sizeof(pca_soft_sensor));
    if (!ss) return NULL;
    ss->n_secondary = n_secondary;
    ss->n_primary = n_primary;
    ss->n_pcs = n_pcs;
    ss->beta = (double*)calloc(n_secondary * n_primary, sizeof(double));
    ss->y_mean = (double*)calloc(n_primary, sizeof(double));
    ss->y_std = (double*)calloc(n_primary, sizeof(double));
    ss->pca = pca_model_alloc(n_secondary);
    if (!ss->beta || !ss->y_mean || !ss->y_std || !ss->pca) {
        pca_soft_sensor_free(ss); return NULL;
    }
    return ss;
}

void pca_soft_sensor_free(pca_soft_sensor *ss)
{
    if (!ss) return;
    pca_model_free(ss->pca);
    free(ss->beta); ss->beta = NULL;
    free(ss->y_mean); ss->y_mean = NULL;
    free(ss->y_std); ss->y_std = NULL;
    free(ss);
}

/* ===================================================================
 * L5: PCR Training
 *
 * Given:
 *   X = N x M secondary variable matrix (raw values)
 *   Y = N x P primary variable matrix (raw values)
 *
 * Steps:
 *   1. Center and scale X, storing mean/std in ss->pca
 *   2. Center and scale Y, storing mean/std in ss->y_mean/y_std
 *   3. Fit PCA on X_scaled: get V (loadings) and D (eigenvalues)
 *   4. Compute beta = V_A * inv(D_A) * V_A^T * (X_scaled^T * Y_scaled) / (N-1)
 *      where V_A = first A columns of V, D_A = diag(lambda_1..lambda_A)
 *
 * Equivalently, PCR solution using latent variables:
 *   T_A = X_scaled * V_A          (N x A score matrix)
 *   theta = inv(T_A^T * T_A) * T_A^T * Y_scaled   (A x P)
 *   beta = V_A * theta             (M x P)
 *
 * The regression coefficient matrix beta maps M secondary variables
 * to P primary variables: y_pred = x_scaled * beta
 * =================================================================== */

int pca_pcr_train(const pca_matrix *X, const pca_matrix *Y, size_t n_pcs, pca_soft_sensor *ss)
{
    size_t N, M, P, i, j, a, k;
    pca_matrix *X_scaled, *Y_scaled, *cov;
    double *theta, *invD;

    if (!X || !X->data || !Y || !Y->data || !ss) return -1;
    N = X->rows; M = X->cols; P = Y->cols;
    if (N != Y->rows || M != ss->n_secondary || P != ss->n_primary) return -1;
    if (n_pcs > M) n_pcs = M;
    ss->n_pcs = n_pcs;

    /* Step 1: Scale X */
    X_scaled = pca_matrix_copy(X);
    if (!X_scaled) return -1;
    pca_center_scale(X_scaled, ss->pca->mean_vec, ss->pca->std_vec);
    ss->pca->n_obs = N;
    ss->pca->use_scaling = 1;

    /* Step 2: Center and scale Y */
    Y_scaled = pca_matrix_copy(Y);
    if (!Y_scaled) { pca_matrix_free(X_scaled); return -1; }
    {
        double *y_mean_tmp = (double*)calloc(P, sizeof(double));
        double *y_std_tmp = (double*)calloc(P, sizeof(double));
        if (!y_mean_tmp || !y_std_tmp) {
            free(y_mean_tmp); free(y_std_tmp);
            pca_matrix_free(X_scaled); pca_matrix_free(Y_scaled);
            return -1;
        }
        pca_center_columns(Y_scaled, y_mean_tmp);
        for (j = 0; j < P; j++) ss->y_mean[j] = y_mean_tmp[j];
        pca_scale_columns(Y_scaled, y_std_tmp);
        for (j = 0; j < P; j++) ss->y_std[j] = y_std_tmp[j];
        free(y_mean_tmp); free(y_std_tmp);
    }

    /* Step 3: Fit PCA on X_scaled */
    cov = pca_compute_covariance(X_scaled);
    if (!cov) { pca_matrix_free(X_scaled); pca_matrix_free(Y_scaled); return -1; }
    {
        pca_matrix *V = pca_matrix_alloc(M, M);
        if (!V) {
            pca_matrix_free(cov); pca_matrix_free(X_scaled);
            pca_matrix_free(Y_scaled); return -1;
        }
        if (pca_jacobi_eigen(cov, ss->pca->eigenvalues, V, 50, 1e-10) != 0) {
            pca_jacobi_eigen(cov, ss->pca->eigenvalues, V, 100, 1e-8);
        }
        /* Sort descending and store loadings */
        {
            size_t ii, jj;
            for (ii = 0; ii < M - 1; ii++) {
                size_t max_idx = ii;
                for (jj = ii + 1; jj < M; jj++)
                    if (ss->pca->eigenvalues[jj] > ss->pca->eigenvalues[max_idx])
                        max_idx = jj;
                if (max_idx != ii) {
                    double tmp = ss->pca->eigenvalues[ii];
                    ss->pca->eigenvalues[ii] = ss->pca->eigenvalues[max_idx];
                    ss->pca->eigenvalues[max_idx] = tmp;
                    for (k = 0; k < M; k++) {
                        double tv = V->data[k * M + ii];
                        V->data[k * M + ii] = V->data[k * M + max_idx];
                        V->data[k * M + max_idx] = tv;
                    }
                }
            }
        }
        for (j = 0; j < M; j++)
            for (i = 0; i < M; i++)
                ss->pca->loadings->data[i * M + j] = V->data[i * M + j];
        pca_matrix_free(V);
    }
    pca_compute_variance_explained(ss->pca->eigenvalues, M,
        ss->pca->var_expl, ss->pca->cum_var);

    /* Step 4: Compute beta = V_A * inv(D_A) * V_A^T * X^T * Y / (N-1)
     * First: T = X_scaled * V_A (scores) */
    invD = (double*)calloc(n_pcs, sizeof(double));
    theta = (double*)calloc(n_pcs * P, sizeof(double));
    if (!invD || !theta) {
        free(invD); free(theta);
        pca_matrix_free(cov); pca_matrix_free(X_scaled);
        pca_matrix_free(Y_scaled); return -1;
    }
    for (a = 0; a < n_pcs; a++) {
        double lam = ss->pca->eigenvalues[a];
        invD[a] = (lam > 1e-12) ? 1.0 / lam : 0.0;
    }

    /* Compute X^T * Y: M x P matrix */
    {
        double *XtY = (double*)calloc(M * P, sizeof(double));
        if (!XtY) {
            free(invD); free(theta);
            pca_matrix_free(cov); pca_matrix_free(X_scaled);
            pca_matrix_free(Y_scaled); return -1;
        }
        for (i = 0; i < N; i++) {
            for (j = 0; j < M; j++) {
                double x_ij = X_scaled->data[i * M + j];
                size_t kk;
                for (kk = 0; kk < P; kk++) {
                    XtY[j * P + kk] += x_ij * Y_scaled->data[i * P + kk];
                }
            }
        }
        /* T^T * Y = V_A^T * (X^T * Y): A x P */
        for (a = 0; a < n_pcs; a++) {
            for (k = 0; k < P; k++) {
                double sum = 0.0;
                for (j = 0; j < M; j++) {
                    sum += ss->pca->loadings->data[j * M + a] * XtY[j * P + k];
                }
                theta[a * P + k] = sum / (double)(N - 1);
            }
        }
        free(XtY);
    }

    /* theta = inv(D_A) * (T^T * Y) */
    for (a = 0; a < n_pcs; a++) {
        for (k = 0; k < P; k++) {
            theta[a * P + k] *= invD[a];
        }
    }

    /* beta = V_A * theta: M x P */
    for (j = 0; j < M; j++) {
        for (k = 0; k < P; k++) {
            double sum = 0.0;
            for (a = 0; a < n_pcs; a++) {
                sum += ss->pca->loadings->data[j * M + a] * theta[a * P + k];
            }
            ss->beta[j * P + k] = sum;
        }
    }

    free(invD); free(theta);
    pca_matrix_free(cov);
    pca_matrix_free(X_scaled);
    pca_matrix_free(Y_scaled);
    return 0;
}

/* ===================================================================
 * L5: Online PCR prediction
 *
 * Given a new raw secondary measurement x_raw (length M):
 *   1. Scale: x_scaled[j] = (x_raw[j] - mean[j]) / std[j]
 *   2. Predict: y_scaled[k] = sum_j x_scaled[j] * beta[j,k]
 *   3. Unscale: y_pred[k] = y_scaled[k] * y_std[k] + y_mean[k]
 *
 * Complexity: O(M * P)
 * =================================================================== */
int pca_pcr_predict(const pca_soft_sensor *ss, const double *x_raw, double *y_pred)
{
    size_t j, k, M, P;
    double *x_scaled;

    if (!ss || !x_raw || !y_pred || !ss->beta || !ss->pca) return -1;
    M = ss->n_secondary;
    P = ss->n_primary;

    x_scaled = (double*)malloc(M * sizeof(double));
    if (!x_scaled) return -1;

    /* Scale input */
    for (j = 0; j < M; j++) {
        double std_j = ss->pca->std_vec[j];
        if (std_j < 1e-12) std_j = 1.0;
        x_scaled[j] = (x_raw[j] - ss->pca->mean_vec[j]) / std_j;
    }

    /* Predict in scaled space */
    for (k = 0; k < P; k++) {
        double sum = 0.0;
        for (j = 0; j < M; j++) {
            sum += x_scaled[j] * ss->beta[j * P + k];
        }
        y_pred[k] = sum * ss->y_std[k] + ss->y_mean[k];
    }

    free(x_scaled);
    return 0;
}

/* Batch prediction: X_new (N_new x M) -> Y_pred (N_new x P) */
int pca_pcr_predict_batch(const pca_soft_sensor *ss, const pca_matrix *X_new, pca_matrix *Y_pred)
{
    size_t i, N;
    if (!ss || !X_new || !X_new->data || !Y_pred || !Y_pred->data) return -1;
    if (X_new->cols != ss->n_secondary || Y_pred->cols != ss->n_primary) return -1;
    if (X_new->rows != Y_pred->rows) return -1;
    N = X_new->rows;
    for (i = 0; i < N; i++) {
        if (pca_pcr_predict(ss, &X_new->data[i * X_new->cols], &Y_pred->data[i * Y_pred->cols]) != 0)
            return -1;
    }
    return 0;
}

/* ===================================================================
 * L5: Leave-One-Out Cross-Validation for PC selection
 *
 * For each candidate number of PCs (1..max_pcs):
 *   For each sample i:
 *     Train PCR on all samples except i
 *     Predict y_i using trained model
 *   Compute RMSECV = sqrt(mean((y_i - y_pred_i)^2))
 *   Compute R2CV = 1 - SSE / SST
 *
 * Returns RMSECV and R2CV arrays of length max_pcs.
 * The optimal number of PCs minimizes RMSECV.
 * =================================================================== */
int pca_pcr_cross_validate(const pca_matrix *X, const pca_matrix *Y,
                           size_t max_pcs, double *rmsecv, double *r2cv)
{
    size_t N, M, P, a, i;
    double y_global_mean, sst;

    if (!X || !X->data || !Y || !Y->data || !rmsecv || !r2cv) return -1;
    N = X->rows; M = X->cols; P = Y->cols;
    if (N < 3 || max_pcs == 0) return -1;
    if (max_pcs > M) max_pcs = M;

    /* Compute global mean of Y (first column for R2CV) */
    y_global_mean = 0.0;
    for (i = 0; i < N; i++) y_global_mean += Y->data[i * P + 0];
    y_global_mean /= (double)N;

    sst = 0.0;
    for (i = 0; i < N; i++) {
        double d = Y->data[i * P + 0] - y_global_mean;
        sst += d * d;
    }

    for (a = 0; a < max_pcs; a++) {
        double sse = 0.0;
        size_t n_valid = 0;

        for (i = 0; i < N; i++) {
            /* Build training set without sample i */
            pca_matrix *X_train = pca_matrix_alloc(N - 1, M);
            pca_matrix *Y_train = pca_matrix_alloc(N - 1, P);
            pca_soft_sensor *ss;
            size_t r, src_row;
            double x_test[M], y_pred_val[8];
            int ok = 1;

            if (!X_train || !Y_train) {
                pca_matrix_free(X_train); pca_matrix_free(Y_train);
                continue;
            }

            for (r = 0; r < N; r++) {
                if (r == i) {
                    /* Save as test sample */
                    size_t j;
                    for (j = 0; j < M; j++)
                        x_test[j] = X->data[r * M + j];
                    /* No y_test needed for cv evaluation here */
                    continue;
                }
                src_row = (r < i) ? r : r - 1;
                {
                    size_t j;
                    for (j = 0; j < M; j++)
                        X_train->data[src_row * M + j] = X->data[r * M + j];
                    for (j = 0; j < P; j++)
                        Y_train->data[src_row * P + j] = Y->data[r * P + j];
                }
            }

            ss = pca_soft_sensor_alloc(M, P, a + 1);
            if (!ss) {
                pca_matrix_free(X_train); pca_matrix_free(Y_train);
                continue;
            }

            if (pca_pcr_train(X_train, Y_train, a + 1, ss) != 0) ok = 0;
            if (ok && pca_pcr_predict(ss, x_test, y_pred_val) != 0) ok = 0;

            if (ok) {
                double e = y_pred_val[0] - Y->data[i * P + 0];
                sse += e * e;
                n_valid++;
            }

            pca_soft_sensor_free(ss);
            pca_matrix_free(X_train);
            pca_matrix_free(Y_train);
        }

        if (n_valid > 0) {
            rmsecv[a] = sqrt(sse / (double)n_valid);
            r2cv[a] = 1.0 - sse / sst;
        } else {
            rmsecv[a] = 1e10;
            r2cv[a] = 0.0;
        }
    }
    return 0;
}

/* Score-based regression: y = T * theta */
int pca_score_regression(const pca_matrix *T, const pca_matrix *Y,
                         double *theta, size_t n_pcs, size_t n_primary)
{
    size_t N, i, a, k;
    double *TtY;

    if (!T || !T->data || !Y || !Y->data || !theta) return -1;
    N = T->rows;

    /* T^T * T = diag(t_a^T * t_a) since PCs are orthogonal */
    /* For NIPALS-extracted T, this is approximately diagonal.
     * For general T, compute explicitly. */
    TtY = (double*)calloc(n_pcs * n_primary, sizeof(double));
    if (!TtY) return -1;

    for (a = 0; a < n_pcs; a++) {
        double t_norm_sq = 0.0;
        for (i = 0; i < N; i++) {
            double ta = T->data[i * T->cols + a];
            t_norm_sq += ta * ta;
        }
        if (t_norm_sq < 1e-12) { free(TtY); return -1; }
        for (k = 0; k < n_primary; k++) {
            double sum = 0.0;
            for (i = 0; i < N; i++) {
                sum += T->data[i * T->cols + a] * Y->data[i * Y->cols + k];
            }
            TtY[a * n_primary + k] = sum;
            theta[a * n_primary + k] = sum / t_norm_sq;
        }
    }

    free(TtY);
    return 0;
}

/* ===================================================================
 * L5: Variable Importance in Projection (VIP)
 *
 * VIP_j = sqrt( M * sum_{a=1..A} (w_{ja}^2 * SSY_a) / sum_{a=1..A} SSY_a )
 *
 * where w_{ja} = loading of variable j on PC a,
 *       SSY_a = sum of squares of Y explained by PC a
 *
 * VIP > 1.0 indicates an important variable for predicting Y.
 * VIP < 0.5 indicates a variable with little predictive value.
 *
 * Reference: Wold, S. et al. (1993) "PLS: Partial Least Squares"
 *             adapted here for PCR.
 * =================================================================== */
void pca_vip_scores(const pca_model *pca, const double *ssq_y, size_t n_pcs,
                    size_t n_primary, size_t n_secondary, double *vip)
{
    size_t a, j, k;
    double total_ssy, *ssy_per_pc;

    if (!pca || !pca->loadings || !ssq_y || !vip) return;

    /* SSY per PC: for PCR, can be approximated from eigenvalue contributions */
    ssy_per_pc = (double*)calloc(n_pcs, sizeof(double));
    if (!ssy_per_pc) return;

    total_ssy = 0.0;
    for (a = 0; a < n_pcs; a++) {
        for (k = 0; k < n_primary; k++) {
            ssy_per_pc[a] += ssq_y[k] * pca->var_expl[a]; /* proportional allocation */
        }
        total_ssy += ssy_per_pc[a];
    }

    if (total_ssy < 1e-12) {
        for (j = 0; j < n_secondary; j++) vip[j] = 0.0;
        free(ssy_per_pc);
        return;
    }

    for (j = 0; j < n_secondary; j++) {
        double sum = 0.0;
        for (a = 0; a < n_pcs; a++) {
            double w = pca->loadings->data[j * pca->loadings->cols + a];
            sum += w * w * ssy_per_pc[a];
        }
        vip[j] = sqrt((double)n_secondary * sum / total_ssy);
    }

    free(ssy_per_pc);
}

/* ===================================================================
 * L5: Reconstruction error per variable
 *
 * For input x, compute how well each variable is reconstructed
 * using the PCA model. High reconstruction error suggests
 * a faulty sensor or process anomaly for that variable.
 * =================================================================== */
void pca_reconstruction_error(const double *x, const pca_model *pca, size_t n_pcs,
                              double *recon_error)
{
    size_t a, j, M;

    if (!x || !pca || !pca->loadings || !recon_error) return;
    M = pca->n_vars;

    for (j = 0; j < M; j++) recon_error[j] = 0.0;

    /* Reconstruct: x_hat = sum_a t_a * p_a */
    for (a = 0; a < n_pcs; a++) {
        double t_a = 0.0;
        for (j = 0; j < M; j++)
            t_a += x[j] * pca->loadings->data[j * pca->loadings->cols + a];
        for (j = 0; j < M; j++)
            recon_error[j] += t_a * pca->loadings->data[j * pca->loadings->cols + a];
    }

    /* Error = (x - x_hat)^2 per variable */
    for (j = 0; j < M; j++) {
        double e = x[j] - recon_error[j];
        recon_error[j] = e * e;
    }
}

/* ===================================================================
 * L2: Model adequacy metrics — R2 and Q2
 *
 * R2 (coefficient of determination):
 *   R2 = 1 - SSE / SST
 *   where SSE = sum_i (y_i - y_hat_i)^2, SST = sum_i (y_i - mean(y))^2
 *
 * Q2 (predictive R2 via cross-validation):
 *   Q2 = 1 - PRESS / SST
 *   where PRESS = sum_i (y_i - y_hat_cv,i)^2
 *
 * Q2 > 0.5 indicates good predictive ability.
 * Q2 > 0.9 indicates excellent predictive ability.
 * =================================================================== */

double pca_pcr_r2(const pca_matrix *Y_true, const pca_matrix *Y_pred)
{
    size_t N, i;
    double sse = 0.0, sst = 0.0, y_mean = 0.0;

    if (!Y_true || !Y_pred || !Y_true->data || !Y_pred->data) return -1.0;
    N = Y_true->rows;
    if (N == 0 || Y_true->cols == 0) return -1.0;

    /* Compute mean of first column */
    for (i = 0; i < N; i++)
        y_mean += Y_true->data[i * Y_true->cols + 0];
    y_mean /= (double)N;

    for (i = 0; i < N; i++) {
        double y = Y_true->data[i * Y_true->cols + 0];
        double y_hat = Y_pred->data[i * Y_pred->cols + 0];
        sse += (y - y_hat) * (y - y_hat);
        sst += (y - y_mean) * (y - y_mean);
    }

    if (sst < 1e-15) return 0.0;
    return 1.0 - sse / sst;
}

double pca_pcr_q2(const pca_matrix *Y_true, const pca_matrix *Y_pred, const double *y_mean)
{
    size_t N, i;
    double press = 0.0, sst = 0.0;

    if (!Y_true || !Y_pred || !Y_true->data || !Y_pred->data || !y_mean) return -1.0;
    N = Y_true->rows;
    if (N == 0) return -1.0;

    for (i = 0; i < N; i++) {
        double y = Y_true->data[i * Y_true->cols + 0];
        double y_hat = Y_pred->data[i * Y_pred->cols + 0];
        press += (y - y_hat) * (y - y_hat);
        sst += (y - y_mean[0]) * (y - y_mean[0]);
    }

    if (sst < 1e-15) return 0.0;
    return 1.0 - press / sst;
}

/* RMSE: root mean square error */
double pca_pcr_rmse(const pca_matrix *Y_true, const pca_matrix *Y_pred)
{
    size_t N, i, k, P;
    double mse = 0.0;

    if (!Y_true || !Y_pred || !Y_true->data || !Y_pred->data) return -1.0;
    N = Y_true->rows; P = Y_true->cols;
    if (N == 0 || P == 0) return -1.0;

    for (i = 0; i < N; i++) {
        for (k = 0; k < P; k++) {
            double e = Y_true->data[i * P + k] - Y_pred->data[i * P + k];
            mse += e * e;
        }
    }

    return sqrt(mse / (double)(N * P));
}
