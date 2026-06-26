/**
 * @file dr_matrix.h
 * @brief Dense matrix operations for data reconciliation.
 *
 * Provides core linear algebra operations needed by DR solvers:
 * matrix multiplication, QR decomposition, Cholesky factorization,
 * triangular solvers, and matrix inversion.
 *
 * All matrices are stored in row-major order:
 *   A(i,j) = data[i * ncol + j]
 *
 * Numerical methods implemented:
 *   - Householder QR (Golub & Van Loan Section 5.2)
 *   - Cholesky-Banachiewicz (Golub & Van Loan Section 4.2)
 *   - Forward/back substitution (Golub & Van Loan Section 3.1)
 *   - Matrix inversion via QR (Golub & Van Loan Section 3.2)
 *
 * Reference:
 *   Golub, G.H., Van Loan, C.F. (2013). "Matrix Computations," 4th ed.
 *   Johns Hopkins University Press.
 *
 *   Higham, N.J. (2002). "Accuracy and Stability of Numerical Algorithms,"
 *   2nd ed. SIAM.
 *
 *   Lawson, C.L., Hanson, R.J. (1995). "Solving Least Squares Problems."
 *   SIAM Classics in Applied Mathematics.
 */

#ifndef DR_MATRIX_H
#define DR_MATRIX_H

#include <stddef.h>

/**
 * @brief Dense matrix in row-major storage.
 */
typedef struct {
    int     rows;
    int     cols;
    int     ld;
    double *data;
} dr_matrix_t;

/* Lifecycle */
dr_matrix_t *dr_mat_alloc(int rows, int cols);
void dr_mat_free(dr_matrix_t *A);
void dr_mat_zero(dr_matrix_t *A);
void dr_mat_identity(dr_matrix_t *A);
int  dr_mat_copy(dr_matrix_t *A, const dr_matrix_t *B);

/* Core operations */
int dr_mat_gemm(double alpha, const dr_matrix_t *A, const dr_matrix_t *B,
                double beta, dr_matrix_t *C);
int dr_mat_gemv(double alpha, const dr_matrix_t *A, const double *x,
                double beta, double *y);
int dr_mat_transpose(const dr_matrix_t *A, dr_matrix_t *B);

/* Triangular solvers */
int dr_mat_forward_sub(const dr_matrix_t *L, const double *b, double *x);
int dr_mat_back_sub(const dr_matrix_t *U, const double *b, double *x);

/* Factorizations */
int dr_mat_cholesky(dr_matrix_t *A);
int dr_mat_cholesky_solve(const dr_matrix_t *L, const double *b, double *x);
int dr_mat_qr(dr_matrix_t *A, double *tau);
int dr_mat_qr_form_q(dr_matrix_t *A, const double *tau);
int dr_mat_qr_solve(const dr_matrix_t *QR, const double *tau,
                    const double *b, double *x);

/* Utility */
int    dr_mat_inverse_via_qr(dr_matrix_t *A, double *work);
double dr_mat_norm_L2(const double *x, int n);
int    dr_mat_diag_inv(double *D, int n);
int    dr_mat_cholesky_raw(double *A, int n);
int    dr_mat_cholesky_solve_raw(const double *L, const double *b, double *x, int n);
double dr_mat_det_triangular(const double *A, int n, int *sign);

#endif /* DR_MATRIX_H */
