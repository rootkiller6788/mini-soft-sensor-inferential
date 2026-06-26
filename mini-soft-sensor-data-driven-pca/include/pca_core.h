/**
 * pca_core.h — PCA Core Data Structures and Basic Operations
 *
 * Knowledge Coverage:
 *   L1 Definitions: PCA model (loading matrix, score matrix, eigenvalues),
 *                   data matrix X (N x M), mean vector, std deviation vector,
 *                   variance explained ratio, cumulative variance explained.
 *   L2 Core Concepts: Centering and scaling, covariance vs correlation PCA,
 *                      dimensionality reduction via latent variables.
 *   L3 Engineering Structures: Row-major 2D matrix storage, double-precision
 *                               floating point for numerical stability.
 *
 * Reference: Jolliffe (2002) §3.1-§3.5, §6.1
 * Course Alignment: MIT 2.171, Stanford ENGR205, CMU 24-677
 */

#ifndef PCA_CORE_H
#define PCA_CORE_H

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * L1: Core type definitions
 * --------------------------------------------------------------------------- */

/**
 * pca_matrix — 2D matrix in row-major order
 * rows: number of rows (observations/samples N)
 * cols: number of columns (variables/features M)
 * data: row-major flat array, data[i*cols + j] = element at row i, column j
 */
typedef struct {
    size_t rows;
    size_t cols;
    double *data;   /* owned, row-major */
} pca_matrix;

/**
 * pca_model — Principal Component Analysis model
 *
 * When trained on an N x M data matrix X:
 *   loadings  : M x M orthonormal matrix (eigenvectors of X'X or correlation)
 *   scores    : N x M matrix, T = X * loadings  (after centering)
 *   eigenvalues: length M, variances along each PC direction
 *   mean_vec  : length M, column means of training data
 *   std_vec   : length M, column standard deviations (1.0 for covariance PCA)
 *   var_expl  : length M, proportion of variance explained by each PC
 *   cum_var   : length M, cumulative proportion of variance explained
 *   n_obs     : number of training observations
 *   n_vars    : number of process variables
 *   use_scaling: 1 if correlation-based PCA (auto-scaling), 0 if covariance
 */
typedef struct {
    size_t n_obs;           /* N: number of training samples */
    size_t n_vars;          /* M: number of variables */
    pca_matrix *loadings;   /* M x M orthonormal loading matrix */
    pca_matrix *scores;     /* N x M score matrix */
    double *eigenvalues;    /* length M, sorted descending */
    double *mean_vec;       /* length M, column means */
    double *std_vec;        /* length M, column std deviations */
    double *var_expl;       /* length M, variance explained per PC */
    double *cum_var;        /* length M, cumulative variance explained */
    int use_scaling;        /* 1 = correlation PCA, 0 = covariance PCA */
} pca_model;

/* ---------------------------------------------------------------------------
 * L3: Matrix operations (engineering structures)
 * --------------------------------------------------------------------------- */

/**
 * pca_matrix_alloc — allocate a rows-by-cols zero-initialized matrix
 * Complexity: O(rows * cols) time, O(rows * cols) space
 */
pca_matrix* pca_matrix_alloc(size_t rows, size_t cols);

/**
 * pca_matrix_free — deallocate matrix memory
 * Complexity: O(1)
 */
void pca_matrix_free(pca_matrix *mat);

/**
 * pca_matrix_get — get element at (row, col), bounds-checked
 * Returns 0.0 and prints warning if out of bounds.
 */
double pca_matrix_get(const pca_matrix *mat, size_t row, size_t col);

/**
 * pca_matrix_set — set element at (row, col), bounds-checked
 */
void pca_matrix_set(pca_matrix *mat, size_t row, size_t col, double value);

/**
 * pca_matrix_copy — deep copy of a matrix
 * Complexity: O(rows * cols)
 */
pca_matrix* pca_matrix_copy(const pca_matrix *src);

/**
 * pca_matrix_print — print matrix to stdout (up to 10x10 for readability)
 */
void pca_matrix_print(const pca_matrix *mat, const char *name);

/* ---------------------------------------------------------------------------
 * L1/L2: PCA model lifecycle
 * --------------------------------------------------------------------------- */

/**
 * pca_model_alloc — allocate a PCA model for M variables
 * Does NOT train; caller must populate via fit function.
 */
pca_model* pca_model_alloc(size_t n_vars);

/**
 * pca_model_free — deallocate entire PCA model including sub-matrices
 */
void pca_model_free(pca_model *model);

/* ---------------------------------------------------------------------------
 * L2: Data preprocessing — center (subtract mean) and scale (divide by std)
 * --------------------------------------------------------------------------- */

/**
 * pca_center_columns — subtract column mean from each column in-place
 * Updates mean_vec in the model.
 * Complexity: O(N * M)
 */
void pca_center_columns(pca_matrix *X, double *mean_vec);

/**
 * pca_scale_columns — divide each column by its standard deviation in-place
 * Updates std_vec. If any std is 0, uses 1.0 to avoid division by zero.
 * Complexity: O(N * M)
 */
void pca_scale_columns(pca_matrix *X, double *std_vec);

/**
 * pca_center_scale — center then scale in-place (auto-scaling)
 * Complexity: O(N * M)
 */
void pca_center_scale(pca_matrix *X, double *mean_vec, double *std_vec);

/* ---------------------------------------------------------------------------
 * L2: Covariance and correlation matrix computation
 * --------------------------------------------------------------------------- */

/**
 * pca_compute_covariance — compute sample covariance matrix (M x M)
 *   Cov[j][k] = (1/(N-1)) * sum_i (X[i][j] - mean_j) * (X[i][k] - mean_k)
 * Assumes X is already centered.
 * Complexity: O(N * M^2)
 */
pca_matrix* pca_compute_covariance(const pca_matrix *X);

/**
 * pca_compute_correlation — compute correlation matrix (M x M) from covariance
 *   Corr[j][k] = Cov[j][k] / (std_j * std_k)
 * Complexity: O(M^2)
 */
pca_matrix* pca_compute_correlation(const pca_matrix *cov, const double *std_vec);

/* ---------------------------------------------------------------------------
 * L1/L2: Variance explained calculations
 * --------------------------------------------------------------------------- */

/**
 * pca_compute_variance_explained — compute var_expl and cum_var from eigenvalues
 *   var_expl[k] = lambda_k / sum(lambda_i)
 *   cum_var[k]  = sum_{i=0..k} var_expl[i]
 * Complexity: O(M)
 */
void pca_compute_variance_explained(double *eigenvalues, size_t M,
                                    double *var_expl, double *cum_var);

/* ---------------------------------------------------------------------------
 * L5: Kaiser criterion — number of PCs with eigenvalue > 1.0
 * --------------------------------------------------------------------------- */

/**
 * pca_kaiser_rule — return number of PCs to retain per Kaiser criterion
 *   Retain all PCs with eigenvalue > avg(eigenvalues)
 *   For correlation PCA (scaled data), avg = 1.0, so > 1.0 is standard.
 * Complexity: O(M)
 */
size_t pca_kaiser_rule(const double *eigenvalues, size_t M);

/**
 * pca_cumvar_rule — return number of PCs needed to exceed cumulative threshold
 *   Typically threshold = 0.85 or 0.95 (85% or 95% variance explained)
 * Complexity: O(M)
 */
size_t pca_cumvar_rule(const double *cum_var, size_t M, double threshold);

#ifdef __cplusplus
}
#endif

#endif /* PCA_CORE_H */
