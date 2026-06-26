/**
 * @file kalman_matrix_ops.h
 * @brief Matrix Operations for Kalman Filtering — Linear Algebra Primitives
 *
 * L3 Engineering Structures: Matrix multiplication, inversion, Cholesky,
 *   eigenvalue computation, QR decomposition
 * L4 Engineering Laws: Numerical stability properties, condition numbers
 *
 * This module provides the linear algebra primitives needed by all Kalman
 * filter variants. Implementations are self-contained (no external BLAS/LAPACK)
 * to ensure portability across embedded and industrial platforms.
 *
 * All matrices are stored in row-major order.
 *
 * Course alignment: MIT 6.302 (linear algebra for control), CMU 18-771
 */
#ifndef KALMAN_MATRIX_OPS_H
#define KALMAN_MATRIX_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Small value for numerical stability checks */
#define MAT_EPSILON 1e-15

/** Maximum dimension supported */
#define MAT_MAX_DIM 24

/* --------------------------------------------------------------------------
 * L3: Matrix-Vector Operations
 * -------------------------------------------------------------------------- */

/**
 * Matrix-vector multiply: y = A * x
 *
 * y[i] = sum_j A[i][j] * x[j]
 *
 * @param A    Matrix, rows x cols, row-major
 * @param x    Vector, size cols
 * @param y    Output vector, size rows
 * @param rows Number of rows
 * @param cols Number of columns
 *
 * Complexity: O(rows * cols)
 */
void mat_vec_mul(const double *A, const double *x, double *y,
                 uint8_t rows, uint8_t cols);

/**
 * Vector-vector outer product: A = x * y'
 *
 * A[i][j] = x[i] * y[j]
 *
 * @param x    Vector, size rows
 * @param y    Vector, size cols
 * @param A    Output matrix, rows x cols, row-major
 * @param rows Length of x
 * @param cols Length of y
 *
 * Complexity: O(rows * cols)
 */
void vec_outer_product(const double *x, const double *y, double *A,
                       uint8_t rows, uint8_t cols);

/**
 * Vector inner (dot) product: d = x' * y = sum_i x[i]*y[i]
 *
 * @param x  Vector, size n
 * @param y  Vector, size n
 * @param n  Dimension
 * @return Dot product
 *
 * Complexity: O(n)
 */
double vec_dot(const double *x, const double *y, uint8_t n);

/**
 * Scaled vector addition: y = alpha * x + y (y += alpha * x)
 *
 * @param alpha  Scalar multiplier
 * @param x      Source vector
 * @param y      Accumulator vector (modified in-place)
 * @param n      Dimension
 *
 * Complexity: O(n)
 */
void vec_axpy(double alpha, const double *x, double *y, uint8_t n);

/**
 * Vector Euclidean norm (L2): ||x||_2 = sqrt(sum(x[i]^2))
 *
 * @param x  Vector
 * @param n  Dimension
 * @return L2 norm
 *
 * Complexity: O(n)
 */
double vec_norm(const double *x, uint8_t n);

/* --------------------------------------------------------------------------
 * L3: Matrix-Matrix Operations
 * -------------------------------------------------------------------------- */

/**
 * Matrix multiply: C = A * B (standard O(n^3))
 *
 * C[i][j] = sum_k A[i][k] * B[k][j]
 *
 * @param A      Matrix, m x k, row-major
 * @param B      Matrix, k x n, row-major
 * @param C      Output matrix, m x n, row-major
 * @param m      Rows of A and C
 * @param k      Cols of A, rows of B (inner dimension)
 * @param n      Cols of B and C
 *
 * Complexity: O(m * k * n)
 */
void mat_mul(const double *A, const double *B, double *C,
             uint8_t m, uint8_t k, uint8_t n);

/**
 * Matrix multiply with transposed first argument: C = A' * B
 *
 * C[i][j] = sum_k A[k][i] * B[k][j]
 *
 * Complexity: O(m * k * n) where A is k x m, B is k x n
 */
void mat_mul_AT_B(const double *A, const double *B, double *C,
                  uint8_t k, uint8_t m, uint8_t n);

/**
 * Matrix multiply with transposed second argument: C = A * B'
 *
 * C[i][j] = sum_k A[i][k] * B[j][k]
 *
 * Complexity: O(m * k * n) where A is m x k, B is n x k
 */
void mat_mul_A_BT(const double *A, const double *B, double *C,
                  uint8_t m, uint8_t k, uint8_t n);

/**
 * Triple product: C = A * B * A' (frequently used in covariance propagation)
 *
 * Optimized to avoid storing the intermediate product.
 *
 * Complexity: O(m^2 * k) where A is n x k, B is k x k
 * Reference: Kailath (1980) "Linear Systems"
 */
void mat_triple_ATA(const double *A, const double *B, double *C,
                    uint8_t n, uint8_t k);

/**
 * Matrix addition: C = A + B
 *
 * Complexity: O(m * n)
 */
void mat_add(const double *A, const double *B, double *C,
             uint8_t m, uint8_t n);

/**
 * Matrix subtraction: C = A - B
 *
 * Complexity: O(m * n)
 */
void mat_sub(const double *A, const double *B, double *C,
             uint8_t m, uint8_t n);

/**
 * Scale matrix: A = alpha * A (in-place)
 *
 * Complexity: O(m * n)
 */
void mat_scale(double alpha, double *A, uint8_t m, uint8_t n);

/**
 * Copy matrix: B = A
 *
 * Complexity: O(m * n)
 */
void mat_copy(const double *A, double *B, uint8_t m, uint8_t n);

/**
 * Matrix trace: tr(A) = sum_i A[i][i]
 *
 * Complexity: O(m)
 */
double mat_trace(const double *A, uint8_t m);

/**
 * Set matrix to identity: A = I_m (m x m identity)
 *
 * Complexity: O(m^2)
 */
void mat_identity(double *A, uint8_t m);

/**
 * Set matrix to zero: A = 0
 *
 * Complexity: O(m * n)
 */
void mat_zero(double *A, uint8_t m, uint8_t n);

/* --------------------------------------------------------------------------
 * L3: Matrix Decompositions — essential for Kalman filtering
 * -------------------------------------------------------------------------- */

/**
 * Cholesky decomposition: A = L * L' where L is lower triangular.
 *
 * Requires A to be symmetric positive definite.
 * L is stored in the lower triangle of A (L[i][j] for j <= i).
 * Upper triangle is untouched.
 *
 * @param A  Input: SPD matrix, n x n, row-major
 *           Output: lower triangle contains L
 * @param n  Dimension
 * @return 1 on success, 0 if matrix is not positive definite
 *
 * Complexity: O(n^3 / 3)
 * Reference: Golub & Van Loan (2013) "Matrix Computations", Algorithm 4.2.1
 */
int mat_cholesky(double *A, uint8_t n);

/**
 * Forward substitution: solve L * x = b where L is lower triangular.
 *
 * @param L  Lower triangular matrix (unit diagonal or from Cholesky), n x n
 * @param b  Right-hand side vector, size n
 * @param x  Solution vector, size n
 * @param n  Dimension
 *
 * Complexity: O(n^2)
 */
void mat_forward_sub(const double *L, const double *b, double *x, uint8_t n);

/**
 * Back substitution: solve U * x = b where U is upper triangular.
 *
 * @param U  Upper triangular matrix, n x n
 * @param b  Right-hand side vector, size n
 * @param x  Solution vector, size n
 * @param n  Dimension
 *
 * Complexity: O(n^2)
 */
void mat_back_sub(const double *U, const double *b, double *x, uint8_t n);

/**
 * Solve linear system A * x = b using Cholesky decomposition.
 *
 * Steps: A = L*L' (Cholesky), L*y = b (forward sub), L'*x = y (back sub)
 *
 * @param A  Input: SPD matrix, n x n (preserved)
 * @param b  Right-hand side vector, size n
 * @param x  Solution vector, size n
 * @param n  Dimension
 * @return 1 on success, 0 if singular
 *
 * Complexity: O(n^3 / 3)
 */
int mat_solve_cholesky(const double *A, const double *b, double *x, uint8_t n);

/**
 * Compute matrix inverse via Cholesky decomposition.
 *
 * A_inv = (L')^{-1} * L^{-1}
 *
 * @param A     Input: SPD matrix, n x n
 * @param A_inv Output: inverse, n x n
 * @param n     Dimension
 * @return 1 on success, 0 if singular
 *
 * Complexity: O(n^3)
 */
int mat_inverse_cholesky(const double *A, double *A_inv, uint8_t n);

/**
 * Compute matrix determinant via Cholesky: |A| = |L|^2 = prod(L[i][i]^2)
 *
 * @param A  SPD matrix, n x n
 * @param n  Dimension
 * @return log(|A|) — using log for numerical stability
 *
 * Complexity: O(n^3)
 */
double mat_logdet_cholesky(double *A, uint8_t n);

/**
 * QR decomposition using modified Gram-Schmidt (numerically stable).
 *
 * Decomposes A = Q * R where Q is m x n orthogonal and R is n x n upper triangular.
 *
 * @param A  Input: m x n matrix, m >= n, row-major
 *           Output: upper triangle stores R
 * @param Q  Output: orthogonal matrix Q, m x n, row-major
 * @param m  Number of rows of A
 * @param n  Number of columns of A
 *
 * Complexity: O(m * n^2)
 * Reference: Golub & Van Loan (2013), Algorithm 5.2.6
 */
void mat_qr_mgs(const double *A, double *Q, double *R, uint8_t m, uint8_t n);

/**
 * QR decomposition using Householder reflections (more stable for large m).
 *
 * @param A     Input: m x n matrix, row-major
 *              Output: upper triangle + Householder vectors
 * @param tau   Output: Householder coefficients, size n
 * @param m     Number of rows
 * @param n     Number of columns
 *
 * Complexity: O(m * n^2)
 */
void mat_qr_householder(double *A, double *tau, uint8_t m, uint8_t n);

/**
 * Rank-1 update of Cholesky factor: given L such that A = L*L',
 * compute new_L such that A + alpha*x*x' = new_L*new_L'.
 *
 * Uses the stable algorithm from:
 * @param L    Existing Cholesky factor, lower triangular, n x n
 * @param x    Update vector, size n
 * @param alpha Scaling
 * @param n    Dimension
 * @return 1 on success, 0 if update causes loss of positive definiteness
 *
 * Complexity: O(n^2)
 * Reference: Gill, Golub, Murray, Saunders (1974)
 *   "Methods for Modifying Matrix Factorizations"
 */
int mat_cholesky_rank1_update(double *L, const double *x, double alpha,
                               uint8_t n);

/**
 * Symmetric rank-k update: C = alpha * A * A' + C  (C += alpha * A * A')
 *
 * Used in covariance propagation.
 *
 * Complexity: O(m * n * k)
 */
void mat_syrk(double alpha, const double *A, double *C,
              uint8_t m, uint8_t k);

/**
 * Compute the Frobenius norm: ||A||_F = sqrt(sum_{i,j} A[i][j]^2)
 *
 * Complexity: O(m * n)
 */
double mat_frobenius_norm(const double *A, uint8_t m, uint8_t n);

/**
 * Compute the condition number estimate of a symmetric matrix.
 *
 * Uses the ratio of largest to smallest magnitude eigenvalue from
 * power iteration + inverse power iteration.
 *
 * Complexity: O(n^2 * iterations)
 */
double mat_cond_estimate_sym(const double *A, uint8_t n, uint8_t max_iters);

/**
 * Check if matrix is symmetric within tolerance.
 *
 * @return 1 if symmetric, 0 otherwise
 *
 * Complexity: O(n^2)
 */
int mat_is_symmetric(const double *A, uint8_t n, double tol);

/**
 * Check if matrix is positive definite.
 *
 * Uses Cholesky attempt — SPD iff Cholesky succeeds.
 *
 * Complexity: O(n^3)
 */
int mat_is_positive_definite(const double *A, uint8_t n);

/**
 * Compute eigenvalues of a 2x2 symmetric matrix analytically.
 *
 * For A = [[a, b], [b, c]]:
 * lambda = (a+c)/2 +/- sqrt(((a-c)/2)^2 + b^2)
 *
 * @param A        2x2 symmetric matrix, row-major
 * @param lambda   Output: eigenvalues (lambda[0] >= lambda[1])
 *
 * Complexity: O(1)
 */
void mat_eigen_2x2(const double *A, double *lambda);

/**
 * Power iteration: estimate the dominant eigenvalue and eigenvector.
 *
 * x_{k+1} = A * x_k / ||A * x_k||
 * lambda = (x' * A * x) / (x' * x)
 *
 * @param A        n x n matrix
 * @param x_init   Initial vector (random if NULL), size n
 * @param x_eig    Output: dominant eigenvector, size n
 * @param lambda   Output: dominant eigenvalue
 * @param n        Dimension
 * @param max_iter Maximum iterations
 *
 * Complexity: O(n^2 * iterations)
 */
void mat_power_iteration(const double *A, const double *x_init,
                         double *x_eig, double *lambda,
                         uint8_t n, uint8_t max_iter);

/**
 * Compute sqrt(P) for a symmetric positive definite matrix via Cholesky,
 * then form the lower-triangular matrix square root S where P = S * S'.
 *
 * @param P  SPD matrix, n x n
 * @param S  Output: lower triangular sqrt, n x n
 * @param n  Dimension
 * @return 1 on success
 *
 * Complexity: O(n^3)
 */
int mat_sqrtm_cholesky(const double *P, double *S, uint8_t n);

/**
 * Form the symmetric product: C = A * B * A' in-place using workspace.
 *
 * Steps: T = B * A', C = A * T
 *
 * Complexity: O(m * n * k + m * m * k) where A is m x k, B is k x k
 */
void mat_similarity_transform(const double *A, const double *B, double *C,
                               uint8_t m, uint8_t k,
                               double *workspace);

#ifdef __cplusplus
}
#endif
#endif /* KALMAN_MATRIX_OPS_H */
