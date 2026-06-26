#ifndef PLS_PREPROCESSING_H
#define PLS_PREPROCESSING_H
#include "matrix_ops.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_preprocessing.h — Data Preprocessing for PLS
 *
 * Implements the standard preprocessing methods required before PLS
 * modelling. Proper preprocessing is critical for PLS performance.
 *
 * Knowledge Coverage:
 *   L3 — Engineering Structures: mean-centering, auto-scaling (unit variance),
 *         Pareto scaling, range scaling, median-centering, robust auto-scaling
 *   L4 — Engineering Laws: mean-centering removes the need for an explicit
 *         intercept term, auto-scaling gives all variables equal weight
 *
 * Preprocessing Methods:
 *   1. Mean-centering:      X_ij := X_ij - mean(X_j)
 *   2. Auto-scaling (UV):   X_ij := (X_ij - mean_j) / stddev_j
 *   3. Pareto scaling:      X_ij := (X_ij - mean_j) / sqrt(stddev_j)
 *   4. Range scaling:       X_ij := (X_ij - min_j) / (max_j - min_j)
 *   5. Median-centering:    X_ij := X_ij - median(X_j)
 *   6. Robust auto-scaling: X_ij := (X_ij - median_j) / MAD_j
 *      where MAD = median(|X_ij - median_j|) * 1.4826
 *
 * References:
 *   Eriksson, L. et al. "Multi- and Megavariate Data Analysis",
 *   Umetrics Academy, 2006.
 *   van den Berg, R.A. et al. "Centering, scaling, and transformations:
 *   improving the biological information content of metabolomics data",
 *   BMC Genomics, 7:142, 2006.
 * ================================================================= */

/*
 * Preprocessing method enumeration.
 */
typedef enum {
    PREPROC_NONE          = 0,
    PREPROC_MEAN_CENTER   = 1,
    PREPROC_AUTOSCALE     = 2,
    PREPROC_PARETO        = 3,
    PREPROC_RANGE         = 4,
    PREPROC_MEDIAN_CENTER = 5,
    PREPROC_ROBUST_SCALE  = 6
} PreprocessingMethod;

/*
 * Preprocessing configuration.
 * Specifies what preprocessing to apply to X and Y blocks.
 */
typedef struct {
    PreprocessingMethod  x_method;
    PreprocessingMethod  y_method;
} PreprocessingConfig;

PreprocessingConfig preproc_config_default(void);

/*
 * preprocess_fit: Compute preprocessing parameters from training data.
 *
 * Computes and stores the centering/scaling parameters needed for
 * consistent preprocessing of future data.
 *
 * @param X       training predictor data (n x p), modified in-place
 * @param Y       training response data (n x q), modified in-place
 * @param config  preprocessing configuration
 * @param x_mean  output: column means of X (p x 1), caller-allocated
 * @param x_std   output: column std devs of X (p x 1), caller-allocated
 * @param y_mean  output: column means of Y (q x 1), caller-allocated
 * @param y_std   output: column std devs of Y (q x 1), caller-allocated
 * @return        0 on success
 */
int preprocess_fit(Matrix *X, Matrix *Y,
                   const PreprocessingConfig *config,
                   Vector *x_mean, Vector *x_std,
                   Vector *y_mean, Vector *y_std);

/*
 * preprocess_transform: Apply preprocessing to new data using stored parameters.
 *
 * Applies the same centering/scaling as was determined during training.
 *
 * @param X          data to preprocess (m x p), modified in-place
 * @param x_mean     column means from training (p x 1)
 * @param x_std      column std devs from training (p x 1)
 * @param center_x   1 to mean-center
 * @param scale_x    1 to scale to unit variance
 */
void preprocess_transform(Matrix *X,
                          const Vector *x_mean, const Vector *x_std,
                          int center_x, int scale_x);

/*
 * preprocess_inverse: Reverse preprocessing to get predictions in original units.
 *
 * Y_original = Y_scaled * y_std + y_mean
 *
 * @param Y_scaled  scaled predictions (m x q), modified in-place
 * @param y_mean    column means from training (q x 1)
 * @param y_std     column std devs from training (q x 1)
 * @param center_y  1 if mean-centering was applied
 * @param scale_y   1 if scaling was applied
 */
void preprocess_inverse(Matrix *Y_scaled,
                        const Vector *y_mean, const Vector *y_std,
                        int center_y, int scale_y);

/*
 * mean_center_matrix: Center each column to zero mean (in-place).
 * Complexity: O(rows * cols).
 */
void mean_center_matrix(Matrix *A);

/*
 * autoscale_matrix: Mean-center and scale each column to unit variance (in-place).
 * Complexity: O(rows * cols).
 */
void autoscale_matrix(Matrix *A);

/*
 * pareto_scale_matrix: Mean-center and scale by sqrt(stddev) (in-place).
 * Pareto scaling is a compromise between no scaling and UV scaling,
 * reducing the dominance of large-variance variables without giving
 * all variables equal weight.
 * Complexity: O(rows * cols).
 */
void pareto_scale_matrix(Matrix *A);

/*
 * range_scale_matrix: Scale each column to [0, 1] range (in-place).
 * X_ij := (X_ij - min_j) / (max_j - min_j)
 * Handles constant columns: sets to 0.5 (mid-range).
 * Complexity: O(rows * cols).
 */
void range_scale_matrix(Matrix *A);

/*
 * robust_scale_matrix: Center by median and scale by MAD (in-place).
 * Robust to outliers — uses median instead of mean, MAD instead of std.
 * X_ij := (X_ij - median_j) / (MAD_j * 1.4826)
 * The factor 1.4826 makes MAD consistent with std for normal data.
 * Complexity: O(rows * cols * log(cols)) due to median sort.
 */
void robust_scale_matrix(Matrix *A);

/*
 * compute_mad: Compute Median Absolute Deviation for a vector.
 * MAD = median(|x_i - median(x)|)
 * Returns MAD * 1.4826 (consistency factor for normal distribution).
 * Complexity: O(n log n) due to sorting.
 */
double compute_mad(const Vector *v);

#ifdef __cplusplus
}
#endif
#endif /* PLS_PREPROCESSING_H */
