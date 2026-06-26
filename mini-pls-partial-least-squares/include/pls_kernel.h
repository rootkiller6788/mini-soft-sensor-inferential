#ifndef PLS_KERNEL_H
#define PLS_KERNEL_H
#include "pls_model.h"
#include "matrix_ops.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_kernel.h — Kernel PLS for Nonlinear Regression
 *
 * Extends PLS to nonlinear relationships by mapping data into a
 * high-dimensional feature space via kernel functions, then applying
 * linear PLS in that feature space (the "kernel trick").
 *
 * Knowledge Coverage:
 *   L5 — Algorithms: Kernel PLS algorithm, kernel matrix computation,
 *        centering in feature space, RBF and polynomial kernels
 *   L2 — Core Concepts: kernel trick — avoids explicit mapping phi(x)
 *        by using k(x_i, x_j) = <phi(x_i), phi(x_j)>
 *   L8 — Advanced Topics: nonlinear PLS, kernel methods in chemometrics
 *
 * Kernel Functions:
 *   - Linear:       k(x, z) = x^T * z + c
 *   - Polynomial:   k(x, z) = (gamma * x^T * z + c)^d
 *   - RBF/Gaussian: k(x, z) = exp(-gamma * ||x - z||^2)
 *   - Sigmoid:      k(x, z) = tanh(gamma * x^T * z + c)
 *
 * The kernel PLS algorithm (Rosipal & Trejo, 2001):
 *   1. Compute kernel matrix K (n x n) where K_{ij} = k(x_i, x_j)
 *   2. Center K in feature space: K = (I - 1_n*1_n^T/n) * K * (I - 1_n*1_n^T/n)
 *   3. Apply PLS to (K, Y) — K acts as the new "X" matrix
 *   4. For prediction: compute k_new = [k(x_new, x_i)] for all training i
 *      then Y_pred = k_new_centered * Beta_K
 *
 * References:
 *   Rosipal, R., Trejo, L.J. "Kernel Partial Least Squares Regression
 *   in Reproducing Kernel Hilbert Space", J. Machine Learning Research,
 *   2:97-123, 2001.
 *   Scholkopf, B., Smola, A. "Learning with Kernels", MIT Press, 2002.
 *   Bennett, K.P., Embrechts, M.J. "An optimization perspective on kernel
 *   partial least squares regression", in Advances in Learning Theory,
 *   IOS Press, 2003.
 * ================================================================= */

/*
 * Kernel function type enumeration.
 */
typedef enum {
    KERNEL_LINEAR     = 0,
    KERNEL_POLYNOMIAL = 1,
    KERNEL_RBF        = 2,
    KERNEL_SIGMOID    = 3
} KernelType;

/*
 * Kernel PLS configuration.
 */
typedef struct {
    KernelType  kernel_type;    /* Kernel function selection                     */
    double      gamma;          /* Kernel width parameter (RBF, polynomial, sigmoid) */
    double      coef0;          /* Additive constant (polynomial, sigmoid)       */
    int         degree;         /* Polynomial degree                             */
    size_t      a_lvs;          /* Number of latent variables                    */
    double      tolerance;      /* NIPALS convergence tolerance                  */
    int         max_iterations; /* Max NIPALS iterations per LV                  */
} KernelPLSConfig;

KernelPLSConfig kernel_pls_config_default(void);

/*
 * Kernel PLS model.
 * Stores both the training data (for prediction kernels) and the
 * PLS decomposition in kernel feature space.
 */
typedef struct {
    /* Training data (needed to compute k(x_new, x_i) for prediction) */
    Matrix     *X_train;        /* Training predictors (n x p), original space   */
    size_t      n_train;        /* Number of training samples                    */
    size_t      p_vars;         /* Number of X variables                        */

    /* Kernel parameters */
    KernelType  kernel_type;
    double      gamma;
    double      coef0;
    int         degree;

    /* Kernel PLS model in feature space */
    Matrix     *K;              /* Centered kernel matrix (n x n)                */
    Matrix     *Beta_K;         /* PLS coefficients in kernel space (n x q)      */
    Vector     *row_mean_K;     /* Row means of K for centering                  */
    Vector     *col_mean_K;     /* Column means of K for centering               */
    double      total_mean_K;   /* Overall mean of K                             */

    /* Preprocessing */
    Vector     *X_mean;
    Vector     *X_std;
    Vector     *Y_mean;
    Vector     *Y_std;
    int         center_x;
    int         scale_x;
    int         center_y;
    int         scale_y;

    /* Dimensions and fit quality */
    size_t      q_vars;
    size_t      a_lvs;
    double      R2Y_cum;
    double      Q2_cum;
} KernelPLSModel;

/* ---- Kernel Function Evaluation ---- */

/*
 * Evaluate kernel function k(x, z) for two vectors.
 * Complexity: O(p) for linear, polynomial, sigmoid; O(p) + exp for RBF.
 */
double kernel_evaluate(const Vector *x, const Vector *z,
                       KernelType type, double gamma,
                       double coef0, int degree);

/*
 * Compute the full kernel matrix K where K_{ij} = k(x_i, x_j).
 * Complexity: O(n^2 * p).
 * Returns (n x n) matrix. Caller frees.
 */
Matrix* kernel_compute_matrix(const Matrix *X,
                              KernelType type, double gamma,
                              double coef0, int degree);

/*
 * Center a kernel matrix in feature space:
 * K_centered = (I - 1_n/n) * K * (I - 1_n/n)
 *
 * This is equivalent to centering the feature-mapped data:
 *   phi_centered(x) = phi(x) - mean(phi)
 *
 * Complexity: O(n^2).
 * Reference: Scholkopf & Smola, "Learning with Kernels", Sec. 14.2.
 */
void kernel_center_matrix(Matrix *K);

/*
 * Compute the kernel vector for a new sample against all training samples.
 * k_new = [k(x_new, x_1), ..., k(x_new, x_n)]
 *
 * Returns new Vector (n x 1). Caller frees.
 */
Vector* kernel_compute_new(const Vector *x_new, const Matrix *X_train,
                           KernelType type, double gamma,
                           double coef0, int degree);

/*
 * Center a new kernel vector using training statistics.
 * k_centered = k_new - row_mean_K - col_mean_K * 1 + total_mean_K * 1
 */
void kernel_center_new(Vector *k_new, const Vector *row_mean_K,
                       const Vector *col_mean_K, double total_mean_K);

/* ---- Kernel PLS Model Lifecycle (L2) ---- */

KernelPLSModel* kernel_pls_model_alloc(size_t n_train, size_t p_vars,
                                        size_t q_vars, size_t a_lvs);
void            kernel_pls_model_free(KernelPLSModel *model);

/*
 * kernel_pls_fit: Fit a Kernel PLS model.
 *
 * Combines kernel matrix computation, centering, and PLS fitting
 * into a single call. The kernel matrix K (n x n) acts as the
 * predictor block in PLS.
 *
 * @param X       training predictors (n x p)
 * @param Y       training responses (n x q)
 * @param config  kernel PLS configuration
 * @param model   output model, pre-allocated
 * @return        0 on success, -1 on error
 *
 * Complexity: O(n^2 * p + n^3) dominated by kernel matrix + PLS on K
 */
int kernel_pls_fit(const Matrix *X, const Matrix *Y,
                   const KernelPLSConfig *config,
                   KernelPLSModel *model);

/*
 * kernel_pls_predict: Predict Y for new X data.
 *
 * @param model   fitted kernel PLS model
 * @param X_new   new predictors (m x p)
 * @param Y_pred  output predictions (m x q), caller-allocated
 * @return        0 on success
 */
int kernel_pls_predict(const KernelPLSModel *model,
                       const Matrix *X_new, Matrix *Y_pred);

/*
 * kernel_pls_predict_single: Predict Y for a single new sample.
 * y_pred must be pre-allocated (q x 1).
 */
int kernel_pls_predict_single(const KernelPLSModel *model,
                              const Vector *x_new, Vector *y_pred);

/*
 * kernel_pls_compute_scores: Compute latent variable scores from
 * the kernel matrix. Useful for diagnostic plots.
 * Returns (n x a_lvs) matrix.
 */
Matrix* kernel_pls_compute_scores(const KernelPLSModel *model);

#ifdef __cplusplus
}
#endif
#endif /* PLS_KERNEL_H */
