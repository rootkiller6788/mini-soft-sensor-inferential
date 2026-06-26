/**
 * pca_adaptive.c -- Recursive and Moving Window PCA for Time-Varying Processes
 *
 * Knowledge Coverage:
 *   L5: Recursive PCA (RPCA) with forgetting factor
 *   L5: Moving Window PCA (MWPCA) with sliding window
 *   L8: Adaptive process monitoring for time-varying industrial processes
 *
 * RPCA: Updates mean, covariance, and eigenstructure as each new observation
 * arrives, using exponential forgetting. Avoids storing full history.
 *
 * MWPCA: Maintains a fixed-length sliding window. As new data enters,
 * old data is discarded. Recomputes model when significant change detected.
 *
 * Welford's algorithm is used for numerically stable recursive variance.
 *
 * Reference: Li et al. (2000), Jeng (2010)
 * Course Alignment: Georgia Tech ECE 6550, RWTH Aachen
 */
#include "pca_adaptive.h"
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================================================================
 * L5: RPCA model allocation/management
 * =================================================================== */

rpca_model* rpca_model_alloc(size_t n_vars, double forgetting_factor, int use_scaling)
{
    rpca_model *model;
    if (n_vars == 0) return NULL;
    model = (rpca_model*)calloc(1, sizeof(rpca_model));
    if (!model) return NULL;
    model->n_vars = n_vars;
    model->n_obs_total = 0;
    model->forgetting_factor = forgetting_factor;
    model->use_scaling = use_scaling;
    if (model->forgetting_factor <= 0.0 || model->forgetting_factor > 1.0)
        model->forgetting_factor = 1.0;
    model->mean_vec = (double*)calloc(n_vars, sizeof(double));
    model->cov_matrix = (double*)calloc(n_vars * n_vars, sizeof(double));
    model->eigenvalues = (double*)calloc(n_vars, sizeof(double));
    model->std_vec = (double*)calloc(n_vars, sizeof(double));
    model->loadings = pca_matrix_alloc(n_vars, n_vars);
    if (!model->mean_vec || !model->cov_matrix || !model->eigenvalues ||
        !model->std_vec || !model->loadings) {
        rpca_model_free(model); return NULL;
    }
    {
        size_t j;
        for (j = 0; j < n_vars; j++) model->std_vec[j] = 1.0;
    }
    return model;
}

void rpca_model_free(rpca_model *model)
{
    if (!model) return;
    pca_matrix_free(model->loadings);
    free(model->mean_vec);    model->mean_vec = NULL;
    free(model->cov_matrix);  model->cov_matrix = NULL;
    free(model->eigenvalues); model->eigenvalues = NULL;
    free(model->std_vec);     model->std_vec = NULL;
    model->loadings = NULL;
    free(model);
}

/* ===================================================================
 * L5: Forgetting factor helper functions
 *
 * Exponential forgetting with factor lambda in (0,1]:
 *   Exponentially Weighted Moving Average (EWMA):
 *     mu_k = lambda * mu_{k-1} + (1-lambda) * x_k
 *
 *   Exponentially Weighted Moving Variance (EWMV):
 *     sigma^2_k = lambda * sigma^2_{k-1} +
 *                 (1-lambda) * (x_k - mu_{k-1}) * (x_k - mu_k)
 *   (using Welford's formulation for numerical stability)
 *
 *   Exponentially Weighted Moving Covariance (EWMC):
 *     cov_k(i,j) = lambda * cov_{k-1}(i,j) +
 *                  (1-lambda) * (x_k(i) - mu_k(i)) * (x_k(j) - mu_k(j))
 * =================================================================== */

double rpca_forgetting_mean_update(double old_mean, double new_val, double lambda)
{
    return lambda * old_mean + (1.0 - lambda) * new_val;
}

double rpca_forgetting_var_update(double old_var, double old_mean,
                                   double new_mean, double new_val, double lambda)
{
    /* Welford-style: use the difference from both old and new means */
    double delta_old = new_val - old_mean;
    double delta_new = new_val - new_mean;
    return lambda * old_var + (1.0 - lambda) * delta_old * delta_new;
}

double rpca_forgetting_cov_update(double old_cov, double xi_old, double xj_old,
                                   double xi_new, double xj_new, double lambda)
{
    (void)xi_old; (void)xj_old; /* kept for API completeness */
    return lambda * old_cov + (1.0 - lambda) * (xi_new) * (xj_new);
}

/* ===================================================================
 * L5: RPCA recursive mean update (Welford-style for stability)
 *
 * mu_k = lambda * mu_{k-1} + (1-lambda) * x_k
 *
 * This is an exponentially weighted moving average (EWMA).
 * When lambda = 1, no forgetting occurs (mu stays constant after init).
 * When lambda = 0, only the most recent observation matters.
 * =================================================================== */
int rpca_update_mean(rpca_model *model, const double *x_new)
{
    size_t j;
    if (!model || !x_new) return -1;
    for (j = 0; j < model->n_vars; j++) {
        model->mean_vec[j] = rpca_forgetting_mean_update(
            model->mean_vec[j], x_new[j], model->forgetting_factor);
    }
    return 0;
}

/* ===================================================================
 * L5: RPCA recursive covariance update
 *
 * After updating mean, update covariance:
 *   C_k(i,j) = lambda * C_{k-1}(i,j) +
 *              (1-lambda) * (x_i - mu_i) * (x_j - mu_j)
 *
 * This maintains a rank-1 update structure.
 * Only the upper triangle is computed (symmetry).
 * =================================================================== */
int rpca_update_covariance(rpca_model *model, const double *x_new)
{
    size_t i, j, n;
    double *x_centered;

    if (!model || !x_new) return -1;
    n = model->n_vars;

    x_centered = (double*)malloc(n * sizeof(double));
    if (!x_centered) return -1;

    /* Center the new observation using CURRENT mean (before update) */
    for (j = 0; j < n; j++) {
        x_centered[j] = x_new[j] - model->mean_vec[j];
    }

    /* Update mean first, then use centered values for covariance */
    rpca_update_mean(model, x_new);

    /* Now x_centered used with old mean is fine for the recursive formula */
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            double old_cov = model->cov_matrix[i * n + j];
            model->cov_matrix[i * n + j] = rpca_forgetting_cov_update(
                old_cov, 0.0, 0.0, x_centered[i], x_centered[j],
                model->forgetting_factor);
            model->cov_matrix[j * n + i] = model->cov_matrix[i * n + j];
        }
    }

    free(x_centered);
    return 0;
}

/* ===================================================================
 * L5: RPCA recursive standard deviation via Welford's algorithm
 *
 * Welford (1962): numerically stable one-pass variance.
 * Extended here with forgetting factor.
 *
 * M_k = M_{k-1} + (x_k - M_{k-1}) / k          (for lambda=1, k = n_obs)
 * S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
 * var = S_k / (k-1)
 *
 * For forgetting factor lambda:
 *   effective_k = 1 / (1 - lambda)   (asymptotic equivalent sample size)
 *   M_k = lambda * M_{k-1} + (1-lambda) * x_k
 *   S_k = lambda * S_{k-1} + lambda * (1-lambda) * (x_k - M_{k-1})^2
 *   sigma^2 = S_k
 * =================================================================== */
int rpca_update_std(rpca_model *model, const double *x_new)
{
    size_t j;
    double lam = model->forgetting_factor;
    double one_minus_lam = 1.0 - lam;

    if (!model || !x_new) return -1;

    for (j = 0; j < model->n_vars; j++) {
        double delta = x_new[j] - model->mean_vec[j];
        /* Update variance via EWMA formula */
        model->std_vec[j] = lam * model->std_vec[j] +
                            lam * one_minus_lam * delta * delta;
        /* Convert to std dev */
        double var = model->std_vec[j];
        model->std_vec[j] = (var > 0.0) ? sqrt(var) : 1.0;
    }

    return 0;
}

/* ===================================================================
 * L5: Full RPCA update — mean, std, covariance, re-eigendecompose
 *
 * After updating, recomputes eigenstructure via Jacobi.
 * max_sweeps and tol control convergence of the eigen decomposition.
 *
 * The eigendecomposition is the most expensive step: O(M^3) per call.
 * In practice, it may be done every K observations rather than every
 * single observation to reduce computational burden.
 * =================================================================== */
int rpca_update_full(rpca_model *model, const double *x_new,
                     size_t max_sweeps, double tol)
{
    size_t n, i, j;
    pca_matrix *cov_mat;
    int ret;

    if (!model || !x_new) return -1;
    n = model->n_vars;

    /* Update all recursive statistics */
    rpca_update_mean(model, x_new);
    rpca_update_covariance(model, x_new);
    rpca_update_std(model, x_new);

    model->n_obs_total++;

    /* Build symmetric covariance matrix for Jacobi */
    cov_mat = pca_matrix_alloc(n, n);
    if (!cov_mat) return -1;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            cov_mat->data[i * n + j] = model->cov_matrix[i * n + j];
        }
    }

    /* Re-eigendecompose */
    ret = pca_jacobi_eigen(cov_mat, model->eigenvalues, model->loadings,
                           max_sweeps, tol);

    pca_matrix_free(cov_mat);
    return ret;
}

/* Extract current RPCA state into a standard pca_model */
int rpca_get_pca_model(const rpca_model *rpca, pca_model *pca_out)
{
    size_t i, j, n;

    if (!rpca || !pca_out) return -1;
    n = rpca->n_vars;
    if (pca_out->n_vars != n) return -1;

    pca_out->n_obs = rpca->n_obs_total;
    for (j = 0; j < n; j++) {
        pca_out->mean_vec[j] = rpca->mean_vec[j];
        pca_out->std_vec[j] = rpca->std_vec[j];
        pca_out->eigenvalues[j] = rpca->eigenvalues[j];
    }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            pca_out->loadings->data[i * n + j] = rpca->loadings->data[i * n + j];

    pca_compute_variance_explained(pca_out->eigenvalues, n,
                                    pca_out->var_expl, pca_out->cum_var);
    return 0;
}

/* ===================================================================
 * L5: Moving Window PCA (MWPCA)
 *
 * Maintains a fixed-size window of the most recent w observations.
 * When the window is full, each new observation replaces the oldest.
 * The PCA model is recomputed from the current window contents.
 *
 * Advantages over RPCA:
 *   - Explicit control of adaptation speed via window size
 *   - Equal weight to all observations in the window
 *   - Better handling of abrupt changes (old data quickly flushed)
 *
 * Disadvantages:
 *   - Higher memory: O(w * M) vs O(M^2) for RPCA
 *   - Higher recomputation cost when window is full
 *   - Discontinuity when observations are dropped
 * =================================================================== */

mwpca_model* mwpca_model_alloc(size_t n_vars, size_t window_size, int use_scaling)
{
    mwpca_model *model;
    if (n_vars == 0 || window_size == 0) return NULL;
    model = (mwpca_model*)calloc(1, sizeof(mwpca_model));
    if (!model) return NULL;
    model->n_vars = n_vars;
    model->window_size = window_size;
    model->count = 0;
    model->write_pos = 0;
    model->use_scaling = use_scaling;
    model->window_data = pca_matrix_alloc(window_size, n_vars);
    model->mean_vec = (double*)calloc(n_vars, sizeof(double));
    model->std_vec = (double*)calloc(n_vars, sizeof(double));
    model->cov_matrix = (double*)calloc(n_vars * n_vars, sizeof(double));
    model->eigenvalues = (double*)calloc(n_vars, sizeof(double));
    model->loadings = pca_matrix_alloc(n_vars, n_vars);
    if (!model->window_data || !model->mean_vec || !model->std_vec ||
        !model->cov_matrix || !model->eigenvalues || !model->loadings) {
        mwpca_model_free(model); return NULL;
    }
    {
        size_t j;
        for (j = 0; j < n_vars; j++) model->std_vec[j] = 1.0;
    }
    return model;
}

void mwpca_model_free(mwpca_model *model)
{
    if (!model) return;
    pca_matrix_free(model->window_data);
    pca_matrix_free(model->loadings);
    free(model->mean_vec);
    free(model->std_vec);
    free(model->cov_matrix);
    free(model->eigenvalues);
    model->window_data = NULL;
    model->loadings = NULL;
    free(model);
}

/* Add observation to circular buffer and optionally recompute model */
int mwpca_add_observation(mwpca_model *model, const double *x_new,
                          size_t max_sweeps, double tol)
{
    size_t j, n;
    if (!model || !x_new) return -1;
    n = model->n_vars;

    /* Insert into circular buffer at write_pos */
    for (j = 0; j < n; j++)
        model->window_data->data[model->write_pos * n + j] = x_new[j];

    model->write_pos = (model->write_pos + 1) % model->window_size;
    if (model->count < model->window_size) model->count++;

    /* Recompute model if window is full */
    if (model->count >= model->window_size) {
        return mwpca_recompute(model, max_sweeps, tol);
    }
    return 0;
}

/* Force recompute of PCA model from current window */
int mwpca_recompute(mwpca_model *model, size_t max_sweeps, double tol)
{
    size_t i, j, k, n, w, actual_count;
    pca_matrix *work, *cov;
    double *col_mean, *col_std;

    if (!model) return -1;
    n = model->n_vars;
    w = model->window_size;
    actual_count = model->count;

    if (actual_count < 2) return -1;

    /* Create working copy from circular buffer in chronological order */
    work = pca_matrix_alloc(actual_count, n);
    if (!work) return -1;

    for (i = 0; i < actual_count; i++) {
        /* Read from circular buffer: oldest data at write_pos (if full)
         * or at position 0 (if not yet wrapped) */
        size_t read_pos;
        if (actual_count < w) {
            read_pos = i; /* not yet wrapped */
        } else {
            read_pos = (model->write_pos + i) % w;
        }
        for (j = 0; j < n; j++) {
            work->data[i * n + j] = model->window_data->data[read_pos * n + j];
        }
    }

    /* Center and scale */
    col_mean = (double*)calloc(n, sizeof(double));
    col_std  = (double*)calloc(n, sizeof(double));
    if (!col_mean || !col_std) {
        free(col_mean); free(col_std);
        pca_matrix_free(work); return -1;
    }

    if (model->use_scaling) {
        pca_center_scale(work, col_mean, col_std);
    } else {
        pca_center_columns(work, col_mean);
        for (j = 0; j < n; j++) col_std[j] = 1.0;
    }

    for (j = 0; j < n; j++) {
        model->mean_vec[j] = col_mean[j];
        model->std_vec[j] = col_std[j];
    }

    /* Compute covariance */
    cov = pca_compute_covariance(work);
    if (!cov) {
        free(col_mean); free(col_std);
        pca_matrix_free(work); return -1;
    }

    /* Store covariance */
    for (i = 0; i < n * n; i++) model->cov_matrix[i] = cov->data[i];

    /* Eigen decomposition */
    {
        pca_matrix *V = pca_matrix_alloc(n, n);
        if (!V) {
            pca_matrix_free(cov); free(col_mean); free(col_std);
            pca_matrix_free(work); return -1;
        }
        if (pca_jacobi_eigen(cov, model->eigenvalues, V, max_sweeps, tol) != 0) {
            pca_jacobi_eigen(cov, model->eigenvalues, V, max_sweeps * 2, tol * 10);
        }

        /* Sort descending */
        for (i = 0; i < n - 1; i++) {
            size_t max_idx = i;
            for (j = i + 1; j < n; j++)
                if (model->eigenvalues[j] > model->eigenvalues[max_idx])
                    max_idx = j;
            if (max_idx != i) {
                double tmp = model->eigenvalues[i];
                model->eigenvalues[i] = model->eigenvalues[max_idx];
                model->eigenvalues[max_idx] = tmp;
                for (k = 0; k < n; k++) {
                    double tv = V->data[k * n + i];
                    V->data[k * n + i] = V->data[k * n + max_idx];
                    V->data[k * n + max_idx] = tv;
                }
            }
        }

        /* Copy to loadings */
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                model->loadings->data[i * n + j] = V->data[i * n + j];

        pca_matrix_free(V);
    }

    pca_matrix_free(cov);
    free(col_mean); free(col_std);
    pca_matrix_free(work);
    return 0;
}

/* Extract MWPCA state into standard pca_model */
int mwpca_get_pca_model(const mwpca_model *mwpca, pca_model *pca_out)
{
    size_t i, j, n;
    if (!mwpca || !pca_out) return -1;
    n = mwpca->n_vars;
    if (pca_out->n_vars != n) return -1;

    pca_out->n_obs = mwpca->count;
    for (j = 0; j < n; j++) {
        pca_out->mean_vec[j] = mwpca->mean_vec[j];
        pca_out->std_vec[j] = mwpca->std_vec[j];
        pca_out->eigenvalues[j] = mwpca->eigenvalues[j];
    }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            pca_out->loadings->data[i * n + j] = mwpca->loadings->data[i * n + j];

    pca_compute_variance_explained(pca_out->eigenvalues, n,
                                    pca_out->var_expl, pca_out->cum_var);
    return 0;
}
