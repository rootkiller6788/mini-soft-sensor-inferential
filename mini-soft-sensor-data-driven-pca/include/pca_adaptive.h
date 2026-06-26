/**
 * pca_adaptive.h - Adaptive and Recursive PCA for Time-Varying Processes
 */

#ifndef PCA_ADAPTIVE_H
#define PCA_ADAPTIVE_H

#include "pca_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RPCA model: updates mean, covariance, eigenstructure online */
typedef struct {
    size_t n_vars;
    size_t n_obs_total;
    double *mean_vec;
    double *cov_matrix;
    double *eigenvalues;
    pca_matrix *loadings;
    double forgetting_factor;
    double *std_vec;
    int use_scaling;
} rpca_model;

rpca_model* rpca_model_alloc(size_t n_vars, double forgetting_factor, int use_scaling);
void rpca_model_free(rpca_model *model);
int rpca_update_mean(rpca_model *model, const double *x_new);
int rpca_update_covariance(rpca_model *model, const double *x_new);
int rpca_update_std(rpca_model *model, const double *x_new);
int rpca_update_full(rpca_model *model, const double *x_new, size_t max_sweeps, double tol);
int rpca_get_pca_model(const rpca_model *rpca, pca_model *pca_out);

/* MWPCA model: sliding window approach */
typedef struct {
    size_t n_vars;
    size_t window_size;
    size_t count;
    pca_matrix *window_data;
    size_t write_pos;
    double *mean_vec;
    double *std_vec;
    double *cov_matrix;
    double *eigenvalues;
    pca_matrix *loadings;
    int use_scaling;
} mwpca_model;

mwpca_model* mwpca_model_alloc(size_t n_vars, size_t window_size, int use_scaling);
void mwpca_model_free(mwpca_model *model);
int mwpca_add_observation(mwpca_model *model, const double *x_new, size_t max_sweeps, double tol);
int mwpca_recompute(mwpca_model *model, size_t max_sweeps, double tol);
int mwpca_get_pca_model(const mwpca_model *mwpca, pca_model *pca_out);

/* Forgetting factor recursive statistics */
double rpca_forgetting_mean_update(double old_mean, double new_val, double lambda);
double rpca_forgetting_var_update(double old_var, double old_mean, double new_mean, double new_val, double lambda);
double rpca_forgetting_cov_update(double old_cov, double xi_old, double xj_old, double xi_new, double xj_new, double lambda);

#ifdef __cplusplus
}
#endif

#endif
