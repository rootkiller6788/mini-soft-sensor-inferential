/**
 * pca_decomposition.h — Eigenvalue and Singular Value Decomposition for PCA
 *
 * Knowledge Coverage:
 *   L3 Engineering Structures: Symmetric tridiagonal form, Givens rotation,
 *                               QR decomposition via Householder reflections.
 *   L4 Engineering Laws: Spectral theorem for symmetric matrices (real eigenvalues,
 *                         orthogonal eigenvectors).
 *   L5 Algorithms: Jacobi eigenvalue algorithm for symmetric matrices,
 *                   NIPALS algorithm for sequential PC extraction,
 *                   Power iteration for dominant eigenpair,
 *                   SVD via Golub-Reinsch bidiagonalization.
 *
 * Reference: Golub & Van Loan "Matrix Computations" (4th ed, 2013) §8.4, §8.6
 *            Jolliffe (2002) §3.5
 * Course Alignment: MIT 6.302, Stanford ENGR205, Berkeley ME233
 */

#ifndef PCA_DECOMPOSITION_H
#define PCA_DECOMPOSITION_H

#include "pca_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * L5: Jacobi Eigenvalue Decomposition for real symmetric matrices
 *
 * Computes all eigenvalues and eigenvectors of a symmetric M x M matrix.
 * Uses classical cyclic Jacobi method with Givens rotations.
 *
 * Theorem (Spectral Theorem): Every real symmetric matrix A has real eigenvalues
 *   and an orthonormal basis of eigenvectors, so A = V * D * V^T.
 *
 * Input:  A = M x M symmetric matrix (upper triangle used, lower ignored)
 * Output: eigenvalues = length M, sorted descending
 *         eigenvectors = M x M matrix, columns = eigenvectors (orthonormal)
 * Complexity: O(M^3) per sweep, typical ~6-10 sweeps for convergence
 */
int pca_jacobi_eigen(const pca_matrix *A, double *eigenvalues,
                     pca_matrix *eigenvectors, size_t max_sweeps, double tol);

/* ---------------------------------------------------------------------------
 * L5: Power Iteration for dominant eigenvalue/eigenvector
 *
 * Computes the largest-magnitude eigenvalue and corresponding eigenvector
 * using the power method: v_{k+1} = A * v_k / ||A * v_k||
 *
 * Theorem: If |lambda_1| > |lambda_2|, power iteration converges to the
 *   dominant eigenpair at a rate |lambda_2 / lambda_1|^k.
 *
 * Complexity: O(k * M^2) for k iterations
 */
int pca_power_iteration(const pca_matrix *A, double *eigenvalue,
                        double *eigenvector, size_t max_iter, double tol);

/* ---------------------------------------------------------------------------
 * L5: Deflation — remove the effect of the dominant eigenpair from A
 *
 * After finding (lambda_1, v_1), compute:
 *   A' = A - lambda_1 * v_1 * v_1^T
 * Then power iteration on A' yields the second eigenpair.
 * Complexity: O(M^2)
 */
void pca_deflate_matrix(pca_matrix *A, double eigenvalue,
                        const double *eigenvector);

/* ---------------------------------------------------------------------------
 * L5: NIPALS (Nonlinear Iterative Partial Least Squares) for PCA
 *
 * Sequentially extracts principal components without computing the full
 * covariance matrix. Useful for high-dimensional data with few PCs needed.
 *
 * Algorithm (for each PC a=1..A):
 *   1. Initialize t_a as any column of X
 *   2. p_a = X^T * t_a / (t_a^T * t_a)    (loading)
 *   3. p_a = p_a / ||p_a||                 (normalize)
 *   4. t_a = X * p_a / (p_a^T * p_a)       (score)
 *   5. Repeat 2-4 until convergence of t_a
 *   6. lambda_a = t_a^T * t_a / (N-1)
 *   7. X = X - t_a * p_a^T                  (deflate)
 *
 * Complexity: O(A * N * M) per PC per iteration, typically 5-20 iterations
 *
 * Reference: Wold, H. (1966) "Estimation of principal components and related
 *   models by iterative least squares"
 */
int pca_nipals(pca_matrix *X, size_t n_components,
               pca_matrix *loadings, pca_matrix *scores,
               double *eigenvalues, size_t max_iter, double tol);

/* ---------------------------------------------------------------------------
 * L5: SVD-based PCA (Singular Value Decomposition)
 *
 * Computes PCA via SVD of centered data matrix X (N x M):
 *   X = U * Sigma * V^T
 * Then: loadings = V, eigenvalues = diag(Sigma)^2 / (N-1), scores = U * Sigma
 *
 * Uses Golub-Reinsch bidiagonalization followed by implicit-shift QR.
 *
 * Complexity: O(N * M * min(N,M)) for the bidiagonal reduction
 */
int pca_svd_decomposition(const pca_matrix *X,
                          pca_matrix *U, double *S, pca_matrix *Vt);

/* ---------------------------------------------------------------------------
 * L4: QR Decomposition via Householder Reflections
 *
 * Decompose A = Q * R where Q is orthogonal (Q^T * Q = I) and R is upper
 * triangular. Householder reflection: H = I - 2*v*v^T / (v^T*v)
 *
 * Theorem: Every real matrix A has a QR decomposition (not necessarily unique).
 *
 * Complexity: O(M^2 * N) for M x N matrix
 */
int pca_householder_qr(const pca_matrix *A, pca_matrix *Q, pca_matrix *R);

/* ---------------------------------------------------------------------------
 * L3: Tridiagonalization of symmetric matrix via Householder
 *
 * Reduces a symmetric M x M matrix A to tridiagonal form T:
 *   T = Q^T * A * Q  where Q is orthogonal
 * This is the first phase of the implicit-shift QR eigenvalue algorithm.
 *
 * Complexity: O(M^3)
 */
int pca_tridiagonalize(const pca_matrix *A, pca_matrix *Q, pca_matrix *T);

/* ---------------------------------------------------------------------------
 * L3: Givens rotation helper for Jacobi method
 *
 * Given a 2x2 symmetric submatrix [[a, b], [b, c]],
 * compute rotation angle such that G^T * [[a,b],[b,c]] * G is diagonal.
 *
 * cos_theta and sin_theta are returned via output parameters.
 */
void pca_givens_rotation(double a, double b, double c,
                         double *cos_theta, double *sin_theta);

/* ---------------------------------------------------------------------------
 * L5: Full PCA fitting via Jacobi eigen-decomposition
 *
 * Complete pipeline:
 *   1. Center (and optionally scale) X
 *   2. Compute covariance or correlation matrix
 *   3. Jacobi eigen-decomposition
 *   4. Sort eigenvalues/vectors descending
 *   5. Compute scores: T = X * loadings
 *   6. Compute variance explained
 *
 * Complexity: O(N*M^2 + M^3) for covariance computation + eigen decomposition
 */
int pca_fit_full(pca_matrix *X, pca_model *model, size_t max_sweeps, double tol);

#ifdef __cplusplus
}
#endif

#endif /* PCA_DECOMPOSITION_H */
