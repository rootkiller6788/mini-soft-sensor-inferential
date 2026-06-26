#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * matrix_ops.h — Fundamental Matrix Operations for PLS Computations
 *
 * Linear algebra backbone for Partial Least Squares algorithms.
 * Implements operations required by NIPALS, SIMPLS, Kernel PLS.
 *
 * Key Operations (L3: Engineering Structures):
 *   - Matrix/Vector lifecycle (alloc, copy, free)
 *   - GEMM (General Matrix Multiply) — BLAS Level 3
 *   - Transpose, outer product, dot product
 *   - Cholesky decomposition LDL^T
 *   - QR decomposition via Householder reflections
 *   - Eigenvalue via power iteration
 *   - SVD via Golub-Kahan bidiagonalization
 *   - Pseudo-inverse via SVD
 *
 * References:
 *   Golub & Van Loan, "Matrix Computations", 4th ed., 2013
 *   Stewart, "Matrix Algorithms Vol I", SIAM, 1998
 *   Dongarra et al., "BLAS Technical Forum Standard", 2002
 * ================================================================= */

/* ---- L1: Definitions — Core Matrix and Vector Types ---- */

typedef struct {
    double  *data;
    size_t   rows;
    size_t   cols;
    int      owns_data;
} Matrix;

typedef struct {
    double *data;
    size_t  len;
    int     owns_data;
} Vector;

/* ---- L2: Core Concepts — Lifecycle Operations ---- */

Matrix* matrix_alloc(size_t rows, size_t cols);
Matrix* matrix_from_array(size_t rows, size_t cols, const double *data);
Matrix* matrix_copy(const Matrix *A);

/*
 * matrix_view: Create a matrix view (no data copy) into a sub-block.
 * The view shares memory with the original — the original must outlive the view.
 * Useful for extracting sub-blocks without allocation overhead.
 * Complexity: O(1). L3: zero-copy engineering pattern.
 */
Matrix* matrix_view(Matrix *A, size_t row_start, size_t col_start,
                    size_t rows, size_t cols);
void    matrix_free(Matrix *A);
void    matrix_fill(Matrix *A, double value);
void    matrix_eye(Matrix *A);

Vector* vector_alloc(size_t n);
Vector* vector_from_array(size_t n, const double *data);
Vector* vector_copy(const Vector *v);
void    vector_free(Vector *v);
void    vector_fill(Vector *v, double value);

/* ---- Column/Row Extraction and Assignment ---- */

Vector* matrix_get_column(const Matrix *A, size_t j);
Vector* matrix_get_row(const Matrix *A, size_t i);
void    matrix_set_column(Matrix *A, size_t j, const Vector *v);
void    matrix_set_row(Matrix *A, size_t i, const Vector *v);

/* ---- L3: Engineering Structures — Matrix Arithmetic ---- */

/*
 * C := alpha * A * B + beta * C  (DGEMM, BLAS Level 3)
 * Dimensions: A(m x k), B(k x n), C(m x n)
 * Complexity: O(m * n * k)
 * Implements the fundamental triple-loop matrix multiply.
 * Reference: Dongarra, J., "Basic Linear Algebra Subprograms
 * Technical Forum Standard", Int. J. High Perform. Appl. 16(1), 2002.
 */
void    matrix_gemm(Matrix *C, double alpha, const Matrix *A,
                    const Matrix *B, double beta);

Matrix* matrix_multiply(const Matrix *A, const Matrix *B);
Vector* matrix_vector_multiply(const Matrix *A, const Vector *x);
Matrix* matrix_transpose(const Matrix *A);

/*
 * Outer product: C = u * v^T, C(i,j) = u[i] * v[j]
 * Dimensions: C(u->len x v->len)
 * Complexity: O(m * n)
 * Used extensively in NIPALS deflation steps.
 */
Matrix* vector_outer_product(const Vector *u, const Vector *v);

double  vector_dot(const Vector *u, const Vector *v);
double  vector_norm_l2(const Vector *v);
double  matrix_norm_frobenius(const Matrix *A);
double  vector_mean(const Vector *v);
double  vector_variance(const Vector *v);
double  vector_stddev(const Vector *v);
void    matrix_add_inplace(Matrix *A, const Matrix *B);
void    matrix_scale_inplace(Matrix *A, double scalar);

/*
 * elementwise_multiply: Hadamard (element-wise) product C = A .* B
 * All three matrices must have identical dimensions.
 * Complexity: O(rows * cols)
 * Used in kernel PLS for kernel matrix operations.
 */
Matrix* matrix_elementwise_multiply(const Matrix *A, const Matrix *B);

/* ---- L3: Matrix Decompositions ---- */

/*
 * Cholesky decomposition: A = L * L^T where L is lower-triangular.
 * A must be symmetric positive-definite (n x n).
 * On output, the lower triangle of A contains L.
 *
 * Theorem (Cholesky): Any SPD matrix A has a unique Cholesky factor L
 * with positive diagonal. Complexity: O(n^3 / 3).
 * Reference: Golub & Van Loan, Algorithm 4.2.2.
 *
 * Returns 0 on success, -1 if a pivot <= 0 (not positive-definite).
 */
int     matrix_cholesky_decomp(Matrix *A);

/*
 * Forward substitution: solve L * x = b for lower-triangular L.
 * Complexity: O(n^2).
 */
void    matrix_forward_substitution(const Matrix *L, Vector *b);

/*
 * Backward substitution: solve U * x = b for upper-triangular U.
 * Complexity: O(n^2).
 */
void    matrix_backward_substitution(const Matrix *U, Vector *b);

/*
 * QR decomposition via Householder reflections: A = Q * R.
 *
 * A (m x n, m >= n) is decomposed in-place. Upper triangle -> R,
 * lower triangle -> Householder vectors v_k.
 *
 * Theorem (Householder, 1958): Any real matrix can be factored as
 * A = Q * R where Q is orthogonal and R is upper-triangular.
 *
 * Complexity: O(m * n^2). Returns 0 on success.
 * Reference: Golub & Van Loan, Algorithm 5.2.1.
 */
int     matrix_qr_decomp(Matrix *A);

/*
 * Power iteration for the dominant eigenvalue of a symmetric matrix.
 *
 * Algorithm: v_{k+1} = A * v_k / ||A * v_k||_2
 *            lambda_k = (v_k^T * A * v_k) / (v_k^T * v_k)
 *
 * Convergence rate: |lambda_2 / lambda_1| (linear).
 * Complexity: O(max_its * n^2).
 * Reference: Golub & Van Loan, Sec 8.2.1 — The Power Method.
 */
double  matrix_power_iteration(const Matrix *A, int max_its, double tol);

/*
 * SVD via Golub-Kahan bidiagonalization.
 * A (m x n) = U * S * V^T. Returns singular values in S (length min(m,n)).
 *
 * Theorem (Eckart-Young-Mirsky): The best rank-k approximation to A
 * in Frobenius norm is the truncated SVD with k largest singular values.
 *
 * Complexity: O(m * n^2 + n^3).
 * Reference: Golub & Kahan, J. SIAM Numer. Anal. B, 2(2):205-224, 1965.
 */
int     matrix_svd_values(const Matrix *A, Vector *S, size_t sv_count);

/*
 * Moore-Penrose pseudo-inverse: A^+ = V * S^+ * U^T.
 * Complexity: O(m * n^2 + n^3).
 */
Matrix* matrix_pinv(const Matrix *A);

/*
 * Solve Ax = b via QR decomposition (least-squares for m >= n).
 * Complexity: O(m * n^2).
 */
Vector* matrix_solve_qr(const Matrix *A, const Vector *b);

/*
 * Compute column-wise means of a matrix.
 * Returns a new Vector of length A->cols.
 * Complexity: O(rows * cols).
 */
Vector* matrix_column_means(const Matrix *A);

/*
 * Compute column-wise standard deviations (population formula).
 * Returns a new Vector of length A->cols.
 * Complexity: O(rows * cols).
 */
Vector* matrix_column_stddevs(const Matrix *A);

/*
 * Center columns of a matrix by subtracting column means (in-place).
 * Complexity: O(rows * cols).
 */
void    matrix_center_columns(Matrix *A, const Vector *means);

/*
 * Scale columns to unit variance (in-place).
 * Complexity: O(rows * cols).
 */
void    matrix_scale_columns(Matrix *A, const Vector *stddevs);

#ifdef __cplusplus
}
#endif
#endif /* MATRIX_OPS_H */
