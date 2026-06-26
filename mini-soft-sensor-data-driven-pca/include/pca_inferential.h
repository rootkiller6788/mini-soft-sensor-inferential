/**
 * pca_inferential.h ¡ª PCA-based Inferential Sensor (Soft Sensor)
 *
 * Knowledge Coverage:
 *   L1 Definitions: Soft sensor, inferential measurement, primary variable y,
 *                   secondary variable x, prediction model.
 *   L2 Core Concepts: Data-driven soft sensing using PCA regression,
 *                     latent variable regression, inverse vs forward modeling.
 *   L5 Algorithms: PCR (Principal Component Regression),
 *                   training from historical data, online prediction.
 *   L6 Canonical Problems: Distillation column composition estimation,
 *                           chemical reactor concentration inference.
 *
 * Reference: Kresta, MacGregor & Marlin (1991), Qin (2003)
 *            Jolliffe (2002) Ch.8
 * Course Alignment: MIT 2.171, Stanford ENGR205, CMU 24-677, Purdue ME 575
 */

#ifndef PCA_INFERENTIAL_H
#define PCA_INFERENTIAL_H

#include "pca_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Soft sensor regression model: y = X * beta */
typedef struct {
    size_t n_secondary;      /* M: number of secondary (easy-to-measure) variables */
    size_t n_primary;        /* P: number of primary (hard-to-measure) variables */
    size_t n_pcs;            /* A: number of retained PCs */
    double *beta;            /* M x P regression coefficients, row-major */
    pca_model *pca;          /* PCA model trained on secondary variables X */
    double *y_mean;          /* length P, mean of primary variables in training */
    double *y_std;           /* length P, std of primary variables in training */
} pca_soft_sensor;

/* Allocate soft sensor model */
pca_soft_sensor* pca_soft_sensor_alloc(size_t n_secondary, size_t n_primary, size_t n_pcs);

/* Free soft sensor model */
void pca_soft_sensor_free(pca_soft_sensor *ss);

/*
 * PCR training: y_hat = X * beta where beta = V_A * inv(D_A) * V_A^T * X^T * Y/(N-1)
 * X = N x M secondary variables (centered/scaled)
 * Y = N x P primary variables (centered/scaled)
 * V_A = M x A truncated loading matrix
 * D_A = A x A diagonal matrix of eigenvalues
 */
int pca_pcr_train(const pca_matrix *X, const pca_matrix *Y, size_t n_pcs, pca_soft_sensor *ss);

/* Online prediction: given new secondary observation x (length M, raw),
 * predict primary variable values y_pred (length P). Returns 0 on success. */
int pca_pcr_predict(const pca_soft_sensor *ss, const double *x_raw, double *y_pred);

/* Batch prediction: X_new is N_new x M raw observations, Y_pred is N_new x P output */
int pca_pcr_predict_batch(const pca_soft_sensor *ss, const pca_matrix *X_new, pca_matrix *Y_pred);

/* Leave-one-out cross-validation for selecting number of PCs */
int pca_pcr_cross_validate(const pca_matrix *X, const pca_matrix *Y,
                           size_t max_pcs, double *rmsecv, double *r2cv);

/* Score-based prediction: y = T_A * theta where theta = (T_A^T*T_A)^{-1}*T_A^T*Y */
int pca_score_regression(const pca_matrix *T, const pca_matrix *Y, double *theta, size_t n_pcs, size_t n_primary);

/* Variable importance in projection (VIP) for soft sensor interpretation */
void pca_vip_scores(const pca_model *pca, const double *ssq_y, size_t n_pcs,
                    size_t n_primary, size_t n_secondary, double *vip);

/* Reconstruction-based contribution for inferential sensor diagnosis */
void pca_reconstruction_error(const double *x, const pca_model *pca, size_t n_pcs,
                              double *recon_error);

/* Model adequacy: compute R^2 and Q^2 for the soft sensor */
double pca_pcr_r2(const pca_matrix *Y_true, const pca_matrix *Y_pred);
double pca_pcr_q2(const pca_matrix *Y_true, const pca_matrix *Y_pred, const double *y_mean);

/* RMSE computation */
double pca_pcr_rmse(const pca_matrix *Y_true, const pca_matrix *Y_pred);

#ifdef __cplusplus
}
#endif

#endif /* PCA_INFERENTIAL_H */
