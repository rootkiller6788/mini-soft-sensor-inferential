/**
 * @file kalman_matrix_ops.c
 * @brief Matrix Operations for Kalman Filtering — Linear Algebra Implementation
 *
 * Self-contained linear algebra library providing all matrix operations
 * needed by Kalman filter variants. No external BLAS/LAPACK dependency.
 *
 * L3 Engineering Structures: Matrix multiply, Cholesky, QR, inverse
 * L4 Engineering Laws: numerical stability, condition numbers
 *
 * References:
 *   Golub & Van Loan (2013) "Matrix Computations", 4th ed.
 *   Trefethen & Bau (1997) "Numerical Linear Algebra"
 *   Higham (2002) "Accuracy and Stability of Numerical Algorithms"
 */
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ===================================================================
 * Vector Operations
 * =================================================================== */

void mat_vec_mul(const double *A, const double *x, double *y,
                 uint8_t rows, uint8_t cols)
{
    if (!A || !x || !y) return;
    for (uint8_t i = 0; i < rows; i++) {
        double sum = 0.0;
        for (uint8_t j = 0; j < cols; j++)
            sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

void vec_outer_product(const double *x, const double *y, double *A,
                       uint8_t rows, uint8_t cols)
{
    if (!x || !y || !A) return;
    for (uint8_t i = 0; i < rows; i++)
        for (uint8_t j = 0; j < cols; j++)
            A[i * cols + j] = x[i] * y[j];
}

double vec_dot(const double *x, const double *y, uint8_t n)
{
    if (!x || !y) return 0.0;
    double s = 0.0;
    for (uint8_t i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

void vec_axpy(double alpha, const double *x, double *y, uint8_t n)
{
    if (!x || !y) return;
    for (uint8_t i = 0; i < n; i++) y[i] += alpha * x[i];
}

double vec_norm(const double *x, uint8_t n)
{
    if (!x) return 0.0;
    double s = 0.0;
    for (uint8_t i = 0; i < n; i++) s += x[i] * x[i];
    return sqrt(s);
}

/* ===================================================================
 * Matrix-Matrix Multiply
 * =================================================================== */

void mat_mul(const double *A, const double *B, double *C,
             uint8_t m, uint8_t k, uint8_t n)
{
    if (!A || !B || !C) return;
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += A[i * k + kk] * B[kk * n + j];
            C[i * n + j] = sum;
        }
    }
}

void mat_mul_AT_B(const double *A, const double *B, double *C,
                  uint8_t k, uint8_t m, uint8_t n)
{
    /* A is k x m, B is k x n, C = A' * B is m x n */
    if (!A || !B || !C) return;
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += A[kk * m + i] * B[kk * n + j];
            C[i * n + j] = sum;
        }
    }
}

void mat_mul_A_BT(const double *A, const double *B, double *C,
                  uint8_t m, uint8_t k, uint8_t n)
{
    /* A is m x k, B is n x k, C = A * B' is m x n */
    if (!A || !B || !C) return;
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += A[i * k + kk] * B[j * k + kk];
            C[i * n + j] = sum;
        }
    }
}

void mat_triple_ATA(const double *A, const double *B, double *C,
                    uint8_t n, uint8_t k)
{
    if (!A || !B || !C) return;
    /* C = A * B * A', dimensions: A(nxk), B(kxk), C(nxn) */
    double BA[MAT_MAX_DIM * MAT_MAX_DIM];  /* B * A' (k x n) */
    for (uint8_t i = 0; i < k; i++) {
        for (uint8_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += B[i * k + kk] * A[j * k + kk];
            BA[i * MAT_MAX_DIM + j] = sum;
        }
    }
    for (uint8_t i = 0; i < n; i++) {
        for (uint8_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += A[i * k + kk] * BA[kk * MAT_MAX_DIM + j];
            C[i * n + j] = sum;
        }
    }
}

void mat_add(const double *A, const double *B, double *C,
             uint8_t m, uint8_t n)
{
    if (!A || !B || !C) return;
    for (uint8_t i = 0; i < m * n; i++) C[i] = A[i] + B[i];
}

void mat_sub(const double *A, const double *B, double *C,
             uint8_t m, uint8_t n)
{
    if (!A || !B || !C) return;
    for (uint8_t i = 0; i < m * n; i++) C[i] = A[i] - B[i];
}

void mat_scale(double alpha, double *A, uint8_t m, uint8_t n)
{
    if (!A) return;
    for (uint8_t i = 0; i < m * n; i++) A[i] *= alpha;
}

void mat_copy(const double *A, double *B, uint8_t m, uint8_t n)
{
    if (!A || !B) return;
    memcpy(B, A, (size_t)m * n * sizeof(double));
}

double mat_trace(const double *A, uint8_t m)
{
    if (!A) return 0.0;
    double t = 0.0;
    for (uint8_t i = 0; i < m; i++) t += A[i * m + i];
    return t;
}

void mat_identity(double *A, uint8_t m)
{
    if (!A) return;
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j < m; j++)
            A[i * m + j] = (i == j) ? 1.0 : 0.0;
    }
}

void mat_zero(double *A, uint8_t m, uint8_t n)
{
    if (!A) return;
    memset(A, 0, (size_t)m * n * sizeof(double));
}

/* ===================================================================
 * Cholesky Decomposition: A = L * L'
 * Algorithm 4.2.1 from Golub & Van Loan
 * =================================================================== */

int mat_cholesky(double *A, uint8_t n)
{
    if (!A || n == 0) return 0;
    for (uint8_t j = 0; j < n; j++) {
        /* Diagonal: A[j][j] = sqrt(A[j][j] - sum_{k<j} A[j][k]^2) */
        double sum_diag = 0.0;
        for (uint8_t k = 0; k < j; k++) {
            double ljk = A[j * n + k];
            sum_diag += ljk * ljk;
        }
        double diag = A[j * n + j] - sum_diag;
        if (diag <= MAT_EPSILON) return 0;  /* Not positive definite */
        A[j * n + j] = sqrt(diag);

        /* Below diagonal: A[i][j] = (A[i][j] - sum A[i][k]*A[j][k]) / A[j][j] */
        for (uint8_t i = j + 1; i < n; i++) {
            double sum = 0.0;
            for (uint8_t k = 0; k < j; k++)
                sum += A[i * n + k] * A[j * n + k];
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
    }
    return 1;
}

void mat_forward_sub(const double *L, const double *b, double *x, uint8_t n)
{
    if (!L || !b || !x) return;
    for (uint8_t i = 0; i < n; i++) {
        double sum = b[i];
        for (uint8_t j = 0; j < i; j++)
            sum -= L[i * n + j] * x[j];
        x[i] = sum / L[i * n + i];
    }
}

void mat_back_sub(const double *U, const double *b, double *x, uint8_t n)
{
    if (!U || !b || !x) return;
    for (int i_int = n - 1; i_int >= 0; i_int--) {
        uint8_t i = (uint8_t)i_int;
        double sum = b[i];
        for (uint8_t j = i + 1; j < n; j++)
            sum -= U[i * n + j] * x[j];
        x[i] = sum / U[i * n + i];
    }
}

int mat_solve_cholesky(const double *A, const double *b, double *x, uint8_t n)
{
    if (!A || !b || !x || n == 0) return 0;
    double L[MAT_MAX_DIM * MAT_MAX_DIM];
    memcpy(L, A, (size_t)n * n * sizeof(double));
    if (!mat_cholesky(L, n)) return 0;

    double y[MAT_MAX_DIM];
    mat_forward_sub(L, b, y, n);

    /* Form U = L' (upper triangular) from L */
    double U[MAT_MAX_DIM * MAT_MAX_DIM];
    for (uint8_t i = 0; i < n; i++)
        for (uint8_t j = 0; j < n; j++)
            U[i * n + j] = (j >= i) ? L[j * n + i] : 0.0;

    mat_back_sub(U, y, x, n);
    return 1;
}

int mat_inverse_cholesky(const double *A, double *A_inv, uint8_t n)
{
    if (!A || !A_inv || n == 0) return 0;

    /* Cholesky: A = L * L' */
    double L[MAT_MAX_DIM * MAT_MAX_DIM];
    memcpy(L, A, (size_t)n * n * sizeof(double));
    if (!mat_cholesky(L, n)) return 0;

    /* Invert L via forward substitution for each column of I */
    double L_inv[MAT_MAX_DIM * MAT_MAX_DIM];
    memset(L_inv, 0, sizeof(L_inv));
    for (uint8_t col = 0; col < n; col++) {
        double e[MAT_MAX_DIM] = {0};
        e[col] = 1.0;
        double col_vec[MAT_MAX_DIM];
        mat_forward_sub(L, e, col_vec, n);
        for (uint8_t i = 0; i < n; i++) L_inv[i * n + col] = col_vec[i];
    }

    /* A_inv = L_inv' * L_inv */
    mat_mul_AT_B(L_inv, L_inv, A_inv, n, n, n);
    return 1;
}

double mat_logdet_cholesky(double *A, uint8_t n)
{
    if (!A || n == 0) return -DBL_MAX;
    if (!mat_cholesky(A, n)) return -DBL_MAX;
    double logdet = 0.0;
    for (uint8_t i = 0; i < n; i++) {
        double d = A[i * n + i];
        if (d <= 0.0) return -DBL_MAX;
        logdet += log(d);
    }
    return 2.0 * logdet;
}

/* ===================================================================
 * QR Decomposition — Modified Gram-Schmidt
 * =================================================================== */

void mat_qr_mgs(const double *A, double *Q, double *R, uint8_t m, uint8_t n)
{
    if (!A || !Q || !R || m < n) return;
    memset(R, 0, (size_t)n * n * sizeof(double));

    /* Copy A into Q initially */
    for (uint8_t i = 0; i < m; i++)
        for (uint8_t j = 0; j < n; j++)
            Q[i * n + j] = A[i * n + j];

    for (uint8_t k = 0; k < n; k++) {
        /* Compute norm of column k */
        double norm_k = 0.0;
        for (uint8_t i = 0; i < m; i++) norm_k += Q[i * n + k] * Q[i * n + k];
        norm_k = sqrt(norm_k);
        R[k * n + k] = norm_k;

        if (norm_k > MAT_EPSILON) {
            double inv_norm = 1.0 / norm_k;
            for (uint8_t i = 0; i < m; i++) Q[i * n + k] *= inv_norm;
        }

        /* Orthogonalize remaining columns */
        for (uint8_t j = k + 1; j < n; j++) {
            double dot = 0.0;
            for (uint8_t i = 0; i < m; i++) dot += Q[i * n + k] * Q[i * n + j];
            R[k * n + j] = dot;
            for (uint8_t i = 0; i < m; i++) Q[i * n + j] -= dot * Q[i * n + k];
        }
    }
}

/* ===================================================================
 * QR Decomposition — Householder reflections
 * =================================================================== */

void mat_qr_householder(double *A, double *tau, uint8_t m, uint8_t n)
{
    if (!A || !tau || m < n) return;

    for (uint8_t k = 0; k < n; k++) {
        /* Compute Householder vector for column k below diagonal */
        double sigma = 0.0;
        for (uint8_t i = k; i < m; i++) sigma += A[i * n + k] * A[i * n + k];

        double x1 = A[k * n + k];
        double alpha = (x1 >= 0.0) ? -sqrt(sigma) : sqrt(sigma);
        tau[k] = (sigma - x1 * alpha) > MAT_EPSILON
                 ? (sigma - x1 * alpha) / (alpha * alpha) : 0.0;

        A[k * n + k] = x1 - alpha;

        /* Apply Householder reflection to remaining columns */
        if (tau[k] > MAT_EPSILON) {
            for (uint8_t j = k + 1; j < n; j++) {
                double dot = 0.0;
                for (uint8_t i = k; i < m; i++) dot += A[i * n + k] * A[i * n + j];
                double beta = tau[k] * dot;
                for (uint8_t i = k; i < m; i++) A[i * n + j] -= beta * A[i * n + k];
            }
        }
    }
}

/* ===================================================================
 * Cholesky Rank-1 Update (Gill, Golub, Murray, Saunders)
 * =================================================================== */

int mat_cholesky_rank1_update(double *L, const double *x, double alpha,
                               uint8_t n)
{
    if (!L || !x || n == 0) return 0;
    if (alpha <= 0.0) return 0;  /* Only positive updates supported here */

    /* Make local writable copy of x for the rotation algorithm */
    double x_copy[MAT_MAX_DIM];
    for (uint8_t i = 0; i < n; i++) x_copy[i] = x[i];

    /* Use the stable algorithm from LINPACK's dchud (Gill, Golub, Murray, Saunders 1974) */
    for (uint8_t j = 0; j < n; j++) {
        double tj = L[j * n + j];
        double uj = sqrt(alpha) * x_copy[j];
        double r = sqrt(tj * tj + uj * uj);
        double c = r / tj;
        double s = uj / tj;
        L[j * n + j] = r;

        for (uint8_t i = j + 1; i < n; i++) {
            double L_ij = L[i * n + j];
            L[i * n + j] = (c * L_ij + s * sqrt(alpha) * x_copy[i]) / (1.0 + c);
        }

        /* Update remaining x for Givens rotation continuation */
        for (uint8_t i = j + 1; i < n; i++) {
            x_copy[i] = x_copy[i] - s * (L[i * n + j]);
        }
        alpha = 1.0;  /* After first step, equivalent alpha absorbed */
    }
    return 1;
}

/* ===================================================================
 * Matrix Norms and Condition Number
 * =================================================================== */

void mat_syrk(double alpha, const double *A, double *C,
              uint8_t m, uint8_t k)
{
    if (!A || !C) return;
    for (uint8_t i = 0; i < m; i++) {
        for (uint8_t j = 0; j <= i; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += A[i * k + kk] * A[j * k + kk];
            C[i * m + j] += alpha * sum;
            if (i != j) C[j * m + i] += alpha * sum;
        }
    }
}

double mat_frobenius_norm(const double *A, uint8_t m, uint8_t n)
{
    if (!A) return 0.0;
    double s = 0.0;
    for (uint8_t i = 0; i < m * n; i++) s += A[i] * A[i];
    return sqrt(s);
}

double mat_cond_estimate_sym(const double *A, uint8_t n, uint8_t max_iters)
{
    if (!A || n == 0) return 1.0;
    if (max_iters == 0) max_iters = 20;

    /* Power iteration for largest eigenvalue */
    double v[MAT_MAX_DIM];
    for (uint8_t i = 0; i < n; i++) v[i] = 1.0 / sqrt((double)n);

    double lambda_max = 0.0;
    for (uint8_t iter = 0; iter < max_iters; iter++) {
        double Av[MAT_MAX_DIM];
        mat_vec_mul(A, v, Av, n, n);
        double norm = vec_norm(Av, n);
        if (norm < MAT_EPSILON) break;
        lambda_max = vec_dot(v, Av, n);
        for (uint8_t i = 0; i < n; i++) v[i] = Av[i] / norm;
    }
    lambda_max = fabs(lambda_max);

    /* Use trace to estimate lambda_min for condition number */
    double trace_A = mat_trace(A, n);
    double lambda_min = trace_A / (double)n;
    if (lambda_min < MAT_EPSILON) lambda_min = MAT_EPSILON;

    return lambda_max / lambda_min;
}

/* ===================================================================
 * Symmetry and Positive Definiteness Checks
 * =================================================================== */

int mat_is_symmetric(const double *A, uint8_t n, double tol)
{
    if (!A) return 0;
    for (uint8_t i = 0; i < n; i++) {
        for (uint8_t j = i + 1; j < n; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > tol)
                return 0;
        }
    }
    return 1;
}

int mat_is_positive_definite(const double *A, uint8_t n)
{
    if (!A || n == 0) return 0;
    double L[MAT_MAX_DIM * MAT_MAX_DIM];
    memcpy(L, A, (size_t)n * n * sizeof(double));
    return mat_cholesky(L, n);
}

/* ===================================================================
 * Eigenvalues of 2x2 symmetric matrix
 * =================================================================== */

void mat_eigen_2x2(const double *A, double *lambda)
{
    if (!A || !lambda) return;
    /* A = [[a, b], [b, c]] */
    double a = A[0], b = A[1], c = A[3];
    double tr = a + c;
    double det = a * c - b * b;
    double disc = tr * tr - 4.0 * det;
    disc = (disc > 0.0) ? sqrt(disc) : 0.0;
    lambda[0] = 0.5 * (tr + disc);  /* larger */
    lambda[1] = 0.5 * (tr - disc);  /* smaller */
}

/* ===================================================================
 * Power Iteration
 * =================================================================== */

void mat_power_iteration(const double *A, const double *x_init,
                         double *x_eig, double *lambda,
                         uint8_t n, uint8_t max_iter)
{
    if (!A || !x_eig || !lambda || n == 0) return;
    if (max_iter == 0) max_iter = 100;

    if (x_init) {
        memcpy(x_eig, x_init, n * sizeof(double));
    } else {
        for (uint8_t i = 0; i < n; i++) x_eig[i] = cos((double)i + 0.5);
    }

    /* Normalize initial */
    double nrm = vec_norm(x_eig, n);
    if (nrm < MAT_EPSILON) { for (uint8_t i = 0; i < n; i++) x_eig[i] = 1.0; nrm = sqrt((double)n); }
    for (uint8_t i = 0; i < n; i++) x_eig[i] /= nrm;

    for (uint8_t iter = 0; iter < max_iter; iter++) {
        double Ax[MAT_MAX_DIM];
        mat_vec_mul(A, x_eig, Ax, n, n);
        double nrm2 = vec_norm(Ax, n);
        if (nrm2 < MAT_EPSILON) { *lambda = 0.0; return; }
        for (uint8_t i = 0; i < n; i++) x_eig[i] = Ax[i] / nrm2;
    }

    /* Rayleigh quotient */
    double Ax[MAT_MAX_DIM];
    mat_vec_mul(A, x_eig, Ax, n, n);
    *lambda = vec_dot(x_eig, Ax, n);
}

/* ===================================================================
 * Matrix Square Root via Cholesky
 * =================================================================== */

int mat_sqrtm_cholesky(const double *P, double *S, uint8_t n)
{
    if (!P || !S || n == 0) return 0;
    memcpy(S, P, (size_t)n * n * sizeof(double));
    if (!mat_cholesky(S, n)) return 0;
    /* S now contains L such that P = L * L' */
    return 1;
}

/* ===================================================================
 * Similarity transform helper
 * =================================================================== */

void mat_similarity_transform(const double *A, const double *B, double *C,
                               uint8_t m, uint8_t k,
                               double *workspace)
{
    if (!A || !B || !C) return;
    /* C = A * B * A', where A is m x k, B is k x k */
    double *T = workspace;
    mat_mul_A_BT(A, B, T, m, k, k);  /* T = A * B (m x k) — wait, that's wrong */
    /* Actually: first compute B * A' (k x m), then A * (B * A') (m x m) */
    double *BA = workspace;
    for (uint8_t i = 0; i < k; i++) {
        for (uint8_t j = 0; j < m; j++) {
            double sum = 0.0;
            for (uint8_t kk = 0; kk < k; kk++)
                sum += B[i * k + kk] * A[j * k + kk];
            BA[i * m + j] = sum;
        }
    }
    mat_mul(A, BA, C, m, k, m);
}
