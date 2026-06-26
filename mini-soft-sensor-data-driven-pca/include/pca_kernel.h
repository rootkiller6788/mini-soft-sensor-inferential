/**
 * pca_kernel.h - Kernel PCA for Nonlinear Process Monitoring
 */

#ifndef PCA_KERNEL_H
#define PCA_KERNEL_H

#include "pca_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PCA_KERNEL_RBF,
    PCA_KERNEL_POLYNOMIAL,
    PCA_KERNEL_LINEAR
} pca_kernel_type;

typedef struct {
    size_t n_train;
    size_t n_vars;
    size_t n_pcs;
    pca_kernel_type ktype;
    double sigma;
    double gamma;
    double coef0;
    int degree;
    pca_matrix *X_train;
    double *eigenvalues;
    pca_matrix *alpha;
    pca_matrix *K_centered;
    double *row_mean_K;
    double grand_mean_K;
    double t2_threshold;
    double spe_threshold;
} kpca_model;

kpca_model* kpca_model_alloc(size_t n_train, size_t n_vars, size_t n_pcs, pca_kernel_type ktype);
void kpca_model_free(kpca_model *model);
double kpca_kernel_compute(const double *x, const double *y, size_t n_vars,
                           pca_kernel_type ktype, double sigma,
                           double gamma, double coef0, int degree);
int kpca_build_kernel_matrix(const pca_matrix *X, pca_matrix *K,
                             pca_kernel_type ktype, double sigma,
                             double gamma, double coef0, int degree);
int kpca_center_kernel_matrix(pca_matrix *K, double *row_mean, double *grand_mean);
int kpca_train(kpca_model *model, const pca_matrix *X, size_t max_sweeps, double tol, double alpha_sig);
int kpca_project(const kpca_model *model, const double *x_new, double *t_scores);
double kpca_t2_statistic(const kpca_model *model, const double *t_scores);
double kpca_spe_statistic(const kpca_model *model, const double *x_new, const double *t_scores);
void kpca_compute_thresholds(kpca_model *model, double alpha_sig);

#ifdef __cplusplus
}
#endif

#endif
