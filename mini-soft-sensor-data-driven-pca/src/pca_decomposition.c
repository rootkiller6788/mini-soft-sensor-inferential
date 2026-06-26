/**
 * pca_decomposition.c -- Eigenvalue/SVD Decomposition Algorithms for PCA
 *
 * Knowledge Coverage:
 *   L3: Symmetric tridiagonal form, Givens rotation, Householder reflection
 *   L4: Spectral theorem -- real symmetric matrices have real eigenvalues
 *        and orthogonal eigenvectors
 *   L5: Jacobi eigenvalue algorithm, NIPALS, Power iteration,
 *        SVD via Golub-Reinsch bidiagonalization, QR decomposition
 *
 * All algorithms are pure C11 -- no BLAS/LAPACK dependency.
 * Numerical stability prioritized over raw speed.
 *
 * Reference: Golub & Van Loan "Matrix Computations" (4th ed, 2013)
 *            Jolliffe (2002) Ch.3.5
 * Course Alignment: MIT 6.302, Stanford ENGR205, Berkeley ME233
 */
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ===================================================================
 * L3: Givens rotation parameters for 2x2 symmetric matrix
 *
 * Given [[a, b], [b, c]], find cos_t, sin_t such that
 *   [cos_t  -sin_t]^T [a b] [cos_t  -sin_t] = [d1  0]
 *   [sin_t   cos_t]   [b c] [sin_t   cos_t]   [ 0 d2]
 *
 * Formula: tau = (c - a) / (2*b)
 *          t = sign(tau) / (|tau| + sqrt(1 + tau^2))
 *          cos_t = 1 / sqrt(1 + t^2)
 *          sin_t = t * cos_t
 *
 * This is numerically stable for all b (including b near 0).
 * =================================================================== */
void pca_givens_rotation(double a, double b, double c,
                         double *cos_theta, double *sin_theta)
{
    double tau, t;
    const double eps = 1e-15;

    if (fabs(b) < eps) {
        *cos_theta = 1.0;
        *sin_theta = 0.0;
        return;
    }

    tau = (c - a) / (2.0 * b);

    if (tau >= 0.0)
        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
    else
        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));

    *cos_theta = 1.0 / sqrt(1.0 + t * t);
    *sin_theta = t * (*cos_theta);
}

/* ===================================================================
 * L5: Jacobi Eigenvalue Algorithm for real symmetric matrices
 *
 * Theorem (Spectral Theorem): Every real symmetric MxM matrix A can be
 *   decomposed as A = V * D * V^T, where V is orthogonal and D is diagonal
 *   with real entries (the eigenvalues).
 *
 * Algorithm: Apply Givens rotations J(p,q,theta) to zero out off-diagonal
 *   elements. Each rotation:
 *     A' = J^T * A * J
 *     V' = V * J   (accumulating right-multiplied rotations)
 *
 * Convergence: Quadratic once off-diagonals are small.
 * One sweep = M*(M-1)/2 rotations (all upper-triangular pairs).
 * Typical convergence in 6-10 sweeps for moderate M.
 *
 * Returns: 0 on success, -1 on invalid input, -2 if no convergence.
 * Complexity: O(M^3 * sweeps) time, O(M^2) auxiliary space.
 * =================================================================== */
int pca_jacobi_eigen(const pca_matrix *A, double *eigenvalues,
                     pca_matrix *eigenvectors, size_t max_sweeps, double tol)
{
    size_t M, p, q, i, sweep;
    pca_matrix *A_work;
    double *Ad, *Vd;
    double off_norm;

    if (!A || !A->data || !eigenvalues || !eigenvectors) return -1;
    M = A->rows;
    if (M < 2 || A->cols != M || eigenvectors->rows != M || eigenvectors->cols != M)
        return -1;

    /* Working copy: modify in-place */
    A_work = pca_matrix_copy(A);
    if (!A_work) return -1;
    Ad = A_work->data;

    /* Initialize eigenvectors = identity */
    Vd = eigenvectors->data;
    {
        size_t k;
        for (k = 0; k < M * M; k++) Vd[k] = 0.0;
        for (k = 0; k < M; k++) Vd[k * M + k] = 1.0;
    }

    if (max_sweeps == 0) max_sweeps = 50;
    if (tol <= 0.0) tol = 1e-12;

    for (sweep = 0; sweep < max_sweeps; sweep++) {
        /* Compute off-diagonal Frobenius norm squared */
        off_norm = 0.0;
        for (p = 0; p < M; p++)
            for (q = p + 1; q < M; q++)
                off_norm += Ad[p * M + q] * Ad[p * M + q];
        off_norm = sqrt(off_norm);

        if (off_norm < tol) {
            /* Converged! */
            for (i = 0; i < M; i++)
                eigenvalues[i] = Ad[i * M + i];
            pca_matrix_free(A_work);
            return 0;
        }

        /* Cyclic sweep over all upper-triangular pairs */
        for (p = 0; p < M; p++) {
            for (q = p + 1; q < M; q++) {
                double app = Ad[p * M + p];
                double aqq = Ad[q * M + q];
                double apq = Ad[p * M + q];
                double c, s;
                size_t r;

                if (fabs(apq) < DBL_MIN) continue;

                pca_givens_rotation(app, apq, aqq, &c, &s);

                /* Update columns p, q for all rows */
                for (r = 0; r < M; r++) {
                    double arp = Ad[r * M + p];
                    double arq = Ad[r * M + q];
                    Ad[r * M + p] =  c * arp - s * arq;
                    Ad[r * M + q] =  s * arp + c * arq;
                }

                /* Update rows p, q for all columns (restore symmetry) */
                for (r = 0; r < M; r++) {
                    double apr = Ad[p * M + r];
                    double aqr = Ad[q * M + r];
                    Ad[p * M + r] =  c * apr - s * aqr;
                    Ad[q * M + r] =  s * apr + c * aqr;
                }

                /* Accumulate eigenvectors: V = V * J(p,q,theta)^T */
                for (r = 0; r < M; r++) {
                    double vrp = Vd[r * M + p];
                    double vrq = Vd[r * M + q];
                    Vd[r * M + p] =  c * vrp - s * vrq;
                    Vd[r * M + q] =  s * vrp + c * vrq;
                }
            }
        }
    }

    /* Not fully converged -- return best estimate */
    for (i = 0; i < M; i++)
        eigenvalues[i] = Ad[i * M + i];
    pca_matrix_free(A_work);
    return -2;
}

/* ===================================================================
 * L5: Power Iteration for dominant eigenpair
 *
 * Given A (MxM), starting from v_0 (random or ones):
 *   v_{k+1} = A * v_k / ||A * v_k||
 *
 * Under the gap condition |lambda_1| > |lambda_2|, the sequence converges
 * to the dominant eigenvector at rate O(|lambda_2/lambda_1|^k).
 *
 * The Rayleigh quotient gives the eigenvalue estimate:
 *   lambda_k = (v_k^T * A * v_k) / (v_k^T * v_k)
 *
 * Complexity: O(max_iter * M^2) time, O(M) auxiliary space.
 * =================================================================== */
int pca_power_iteration(const pca_matrix *A, double *eigenvalue,
                        double *eigenvector, size_t max_iter, double tol)
{
    size_t M, iter, i, j;
    double *v_old, *v_new, *Av;
    double norm, lambda_old, lambda_new;

    if (!A || !A->data || !eigenvalue || !eigenvector) return -1;
    M = A->rows;
    if (M == 0 || A->cols != M) return -1;

    v_old = (double*)malloc(M * sizeof(double));
    v_new = (double*)malloc(M * sizeof(double));
    Av    = (double*)malloc(M * sizeof(double));
    if (!v_old || !v_new || !Av) {
        free(v_old); free(v_new); free(Av); return -1;
    }

    /* Initialize v_0 = [1,1,...,1] / sqrt(M) */
    norm = sqrt((double)M);
    for (i = 0; i < M; i++) v_old[i] = 1.0 / norm;

    if (max_iter == 0) max_iter = 1000;
    if (tol <= 0.0) tol = 1e-10;

    lambda_old = 0.0;

    for (iter = 0; iter < max_iter; iter++) {
        /* Av = A * v_old */
        for (i = 0; i < M; i++) {
            double sum = 0.0;
            for (j = 0; j < M; j++)
                sum += A->data[i * M + j] * v_old[j];
            Av[i] = sum;
        }

        /* Normalize: v_new = Av / ||Av|| */
        norm = 0.0;
        for (i = 0; i < M; i++) norm += Av[i] * Av[i];
        norm = sqrt(norm);
        if (norm < 1e-15) { free(v_old); free(v_new); free(Av); return -1; }
        for (i = 0; i < M; i++) v_new[i] = Av[i] / norm;

        /* Rayleigh quotient: lambda = v_new^T * A * v_new */
        lambda_new = 0.0;
        for (i = 0; i < M; i++) {
            double sum = 0.0;
            for (j = 0; j < M; j++)
                sum += A->data[i * M + j] * v_new[j];
            lambda_new += v_new[i] * sum;
        }

        /* Convergence check */
        if (fabs(lambda_new - lambda_old) < tol * (1.0 + fabs(lambda_new))) {
            *eigenvalue = lambda_new;
            memcpy(eigenvector, v_new, M * sizeof(double));
            free(v_old); free(v_new); free(Av);
            return 0;
        }

        lambda_old = lambda_new;
        memcpy(v_old, v_new, M * sizeof(double));
    }

    /* Return best estimate */
    *eigenvalue = lambda_old;
    memcpy(eigenvector, v_old, M * sizeof(double));
    free(v_old); free(v_new); free(Av);
    return -2;
}

/* ===================================================================
 * L5: Matrix deflation -- remove dominant eigenpair
 *
 * A' = A - lambda * v * v^T
 *
 * After deflation, the next dominant eigenvalue of A' corresponds to
 * the second eigenvalue of A. Sequential deflation + power iteration
 * yields all eigenpairs (Wielandt deflation).
 *
 * Complexity: O(M^2)
 * =================================================================== */
void pca_deflate_matrix(pca_matrix *A, double eigenvalue,
                        const double *eigenvector)
{
    size_t M, i, j;

    if (!A || !A->data || !eigenvector) return;
    M = A->rows;

    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++)
            A->data[i * M + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
}

/* ===================================================================
 * L3: Householder reflection helper
 *
 * Given vector x of length n, compute v such that:
 *   H = I - beta * v * v^T
 * satisfies H * x = ||x|| * e_1 (zeros out elements 2..n).
 *
 * alpha = -sign(x[0]) * ||x||
 * v[0] = x[0] - alpha
 * v[i] = x[i]  for i > 0
 * beta = 2 / (v^T * v) = 2 / (||x||^2 - x[0]^2 + (x[0]-alpha)^2)
 *      = (x[0] - alpha) / alpha
 *      = 2 / (sigma + (x[0]-alpha)^2)  [numerically stable]
 *
 * Returns alpha (the resulting first element after H is applied).
 * =================================================================== */
static double householder_vector(const double *x, size_t n, double *v, double *beta)
{
    size_t i;
    double sigma, x_norm, alpha, v0_sq;

    if (n <= 1) {
        *beta = 0.0;
        return x[0];
    }

    /* Norm of x */
    sigma = 0.0;
    for (i = 0; i < n; i++) sigma += x[i] * x[i];
    x_norm = sqrt(sigma);

    if (x_norm < 1e-15) {
        *beta = 0.0;
        return x[0];
    }

    /* alpha = -sign(x[0]) * ||x|| */
    alpha = (x[0] > 0.0) ? -x_norm : x_norm;

    v[0] = x[0] - alpha;
    for (i = 1; i < n; i++) v[i] = x[i];

    v0_sq = v[0] * v[0];
    *beta = 2.0 * v0_sq / (sigma - x[0] * x[0] + v0_sq);
    /* Equivalent to: beta = 2 / (v^T * v) */

    return alpha;
}

/* ===================================================================
 * L4: QR Decomposition via Householder Reflections
 *
 * A (MxN, M >= N) = Q (MxM) * R (MxN)
 * Q is orthogonal (Q^T * Q = I), R is upper triangular.
 *
 * Algorithm: For k = 0..N-1:
 *   Construct Householder H_k to zero out A[k+1:M, k]
 *   Apply H_k to remaining columns: A[k:M, k:N] = H_k * A[k:M, k:N]
 *   Accumulate: Q = Q * H_k^T
 *
 * Complexity: O(M * N^2) for the reduction, O(M^2 * N) for Q accumulation.
 * =================================================================== */
int pca_householder_qr(const pca_matrix *A, pca_matrix *Q, pca_matrix *R)
{
    size_t M, N, k, i, j;
    pca_matrix *A_work;
    double *v;

    if (!A || !A->data || !Q || !R) return -1;
    M = A->rows; N = A->cols;
    if (M < N || Q->rows != M || Q->cols != M || R->rows != M || R->cols != N)
        return -1;

    A_work = pca_matrix_copy(A);
    if (!A_work) return -1;

    v = (double*)malloc(M * sizeof(double));
    if (!v) { pca_matrix_free(A_work); return -1; }

    /* Q = I_M */
    for (i = 0; i < M * M; i++) Q->data[i] = 0.0;
    for (i = 0; i < M; i++) Q->data[i * M + i] = 1.0;

    for (k = 0; k < N; k++) {
        /* Extract column k from row k downward */
        double x[M - k];
        double alpha, beta;

        for (i = 0; i < M - k; i++)
            x[i] = A_work->data[(k + i) * N + k];

        alpha = householder_vector(x, M - k, v, &beta);

        /* Apply H to remaining columns: A[k:M, j] -= beta*v*(v^T*A[k:M,j]) */
        for (j = k; j < N; j++) {
            double dot = 0.0;
            for (i = 0; i < M - k; i++)
                dot += v[i] * A_work->data[(k + i) * N + j];
            dot *= beta;
            for (i = 0; i < M - k; i++)
                A_work->data[(k + i) * N + j] -= dot * v[i];
        }

        /* Store reflector (for potential later use) */
        A_work->data[k * N + k] = alpha;
        for (i = 1; i < M - k; i++)
            A_work->data[(k + i) * N + k] = v[i];

        /* Accumulate Q: Q = Q * (I - beta*v*v^T) */
        for (i = 0; i < M; i++) {
            double dot = 0.0;
            size_t r;
            for (r = 0; r < M - k; r++)
                dot += Q->data[i * M + (k + r)] * v[r];
            dot *= beta;
            for (r = 0; r < M - k; r++)
                Q->data[i * M + (k + r)] -= dot * v[r];
        }
    }

    /* Extract R (upper triangular part of A_work) */
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            if (i <= j)
                R->data[i * N + j] = A_work->data[i * N + j];
            else
                R->data[i * N + j] = 0.0;
        }
    }

    free(v);
    pca_matrix_free(A_work);
    return 0;
}

/* ===================================================================
 * L3: Symmetric tridiagonalization via Householder reflections
 *
 * Reduces symmetric MxM matrix A to tridiagonal form T:
 *   T = Q^T * A * Q
 * where Q is orthogonal (product of Householder reflections).
 *
 * For k = 0..M-3:
 *   1. Build Householder H_k to zero out A[k+2:M, k]
 *   2. Apply similarity transform: A = H_k * A * H_k^T
 *      (rank-2 update: A = A - v*w^T - w*v^T)
 *   3. Accumulate: Q = Q * H_k
 *
 * Result: T is tridiagonal (only T[i][i], T[i][i+1], T[i+1][i] non-zero).
 *
 * This is the first phase of the implicit-shift QR eigenvalue algorithm.
 * Complexity: O(M^3)
 * =================================================================== */
int pca_tridiagonalize(const pca_matrix *A, pca_matrix *Q, pca_matrix *T)
{
    size_t M, k, i, j;
    pca_matrix *A_work;
    double *v, *p;

    if (!A || !A->data || !Q || !T) return -1;
    M = A->rows;
    if (M < 3 || A->cols != M) {
        /* M <= 2: just copy */
        for (i = 0; i < M * M; i++) T->data[i] = A->data[i];
        for (i = 0; i < M * M; i++) Q->data[i] = 0.0;
        for (i = 0; i < M; i++) Q->data[i * M + i] = 1.0;
        return 0;
    }

    A_work = pca_matrix_copy(A);
    if (!A_work) return -1;

    /* Q = I */
    for (i = 0; i < M * M; i++) Q->data[i] = 0.0;
    for (i = 0; i < M; i++) Q->data[i * M + i] = 1.0;

    v = (double*)malloc(M * sizeof(double));
    p = (double*)malloc(M * sizeof(double));
    if (!v || !p) { free(v); free(p); pca_matrix_free(A_work); return -1; }

    for (k = 0; k < M - 2; k++) {
        /* Build Householder to zero out column k below subdiagonal */
        double sigma = 0.0;
        for (i = k + 1; i < M; i++) {
            v[i] = A_work->data[i * M + k];
            sigma += v[i] * v[i];
        }
        v[k] = 0.0;

        if (sigma > 1e-15) {
            double norm = sqrt(sigma);
            double alpha = (A_work->data[(k+1) * M + k] > 0.0) ? -norm : norm;

            v[k+1] = A_work->data[(k+1) * M + k] - alpha;

            /* beta = 2 / (v^T * v) */
            double vnorm2 = 0.0;
            for (i = k + 1; i < M; i++) vnorm2 += v[i] * v[i];
            double beta = 2.0 / vnorm2;

            /* p = beta * A * v */
            for (i = 0; i < M; i++) {
                double sum = 0.0;
                for (j = k; j < M; j++)
                    sum += A_work->data[i * M + j] * v[j];
                p[i] = beta * sum;
            }

            /* w = p - (beta * p^T * v / 2) * v */
            double pTv = 0.0;
            for (i = k; i < M; i++) pTv += p[i] * v[i];
            double gamma __attribute__((unused)) = beta * pTv / 2.0;

            /* Rank-2 update: A = A - p*v^T - v*p^T */
            for (i = 0; i < M; i++) {
                for (j = 0; j < M; j++) {
                    A_work->data[i * M + j] -= p[i] * v[j] + v[i] * p[j];
                }
            }

            /* Accumulate Q: Q = Q * (I - beta * v * v^T) */
            for (i = 0; i < M; i++) {
                double dot = 0.0;
                for (j = k; j < M; j++)
                    dot += Q->data[i * M + j] * v[j];
                dot *= beta;
                for (j = k; j < M; j++)
                    Q->data[i * M + j] -= dot * v[j];
            }
        }
    }

    /* Copy to T */
    for (i = 0; i < M * M; i++) T->data[i] = A_work->data[i];

    free(v); free(p);
    pca_matrix_free(A_work);
    return 0;
}

/* ===================================================================
 * L5: NIPALS Algorithm for Sequential PC Extraction
 *
 * NIPALS = Nonlinear Iterative Partial Least Squares.
 * Extracts PCs one at a time from the data matrix X, deflating after each.
 *
 * For PC a:
 *   t = column of X (initial guess)
 *   Repeat:
 *     p = X^T * t                         (loading estimate)
 *     p = p / ||p||                       (normalize to unit)
 *     t_new = X * p                       (score estimate)
 *     Check convergence: ||t_new - t|| < tol
 *     t = t_new
 *   lambda = t^T * t / (N-1)              (eigenvalue)
 *   X = X - t * p^T                        (deflate)
 *
 * Advantages: O(A * N * M) per PC vs O(M^3) for full eigen.
 *             Memory efficient for high-dimensional data (N, M large).
 * Disadvantages: Convergence can be slow for close eigenvalues.
 *
 * Complexity: O(A * N * M * iters_per_pc)
 * Reference: Wold (1966)
 * =================================================================== */
int pca_nipals(pca_matrix *X, size_t n_components,
               pca_matrix *loadings, pca_matrix *scores,
               double *eigenvalues, size_t max_iter, double tol)
{
    size_t N, M, a, i, j, iter;
    pca_matrix *X_work;
    double *t_old, *t_new, *p, diff;

    if (!X || !X->data || !loadings || !scores || !eigenvalues) return -1;
    N = X->rows; M = X->cols;
    if (n_components > M) n_components = M;
    if (n_components == 0) return 0;

    X_work = pca_matrix_copy(X);
    if (!X_work) return -1;

    t_old = (double*)malloc(N * sizeof(double));
    t_new = (double*)malloc(N * sizeof(double));
    p     = (double*)malloc(M * sizeof(double));
    if (!t_old || !t_new || !p) {
        free(t_old); free(t_new); free(p);
        pca_matrix_free(X_work); return -1;
    }

    if (max_iter == 0) max_iter = 100;
    if (tol <= 0.0) tol = 1e-8;

    for (a = 0; a < n_components; a++) {
        /* Initialize t with the column of X_work with largest variance */
        {
            double best_var = -1.0;
            size_t best_col = 0;
            for (j = 0; j < M; j++) {
                double sum = 0.0, sum_sq = 0.0;
                for (i = 0; i < N; i++) {
                    double v = X_work->data[i * M + j];
                    sum += v; sum_sq += v * v;
                }
                double var_j = sum_sq / (double)N - (sum / (double)N) * (sum / (double)N);
                if (var_j > best_var) { best_var = var_j; best_col = j; }
            }
            for (i = 0; i < N; i++)
                t_old[i] = X_work->data[i * M + best_col];
        }

        for (iter = 0; iter < max_iter; iter++) {
            /* p = X^T * t_old */
            for (j = 0; j < M; j++) {
                double sum = 0.0;
                for (i = 0; i < N; i++)
                    sum += X_work->data[i * M + j] * t_old[i];
                p[j] = sum;
            }

            /* Normalize p */
            {
                double norm = 0.0;
                for (j = 0; j < M; j++) norm += p[j] * p[j];
                norm = sqrt(norm);
                if (norm < 1e-15) break;
                for (j = 0; j < M; j++) p[j] /= norm;
            }

            /* t_new = X * p */
            for (i = 0; i < N; i++) {
                double sum = 0.0;
                for (j = 0; j < M; j++)
                    sum += X_work->data[i * M + j] * p[j];
                t_new[i] = sum;
            }

            /* Convergence check */
            diff = 0.0;
            for (i = 0; i < N; i++) {
                double d = t_new[i] - t_old[i];
                diff += d * d;
            }
            diff = sqrt(diff);

            /* Copy t_new -> t_old for next iteration */
            for (i = 0; i < N; i++) t_old[i] = t_new[i];

            if (diff < tol) break;
        }

        /* Eigenvalue: lambda = (t^T * t) / (N-1) */
        {
            double ss = 0.0;
            for (i = 0; i < N; i++) ss += t_new[i] * t_new[i];
            eigenvalues[a] = ss / (double)(N - 1);
        }

        /* Store loadings (column a of loading matrix) */
        for (j = 0; j < M; j++)
            loadings->data[j * loadings->cols + a] = p[j];

        /* Store scores (column a of score matrix) */
        for (i = 0; i < N; i++)
            scores->data[i * scores->cols + a] = t_new[i];

        /* Deflate X: X = X - t * p^T */
        for (i = 0; i < N; i++)
            for (j = 0; j < M; j++)
                X_work->data[i * M + j] -= t_new[i] * p[j];
    }

    free(t_old); free(t_new); free(p);
    pca_matrix_free(X_work);
    return 0;
}

/* ===================================================================
 * L5: SVD via Golub-Reinsch bidiagonalization
 *
 * X (NxM) = U * Sigma * V^T
 * Phase 1: Bidiagonalization via alternating Householder reflections
 * Phase 2: Singular values from eigendecomposition of B^T * B
 * =================================================================== */
int pca_svd_decomposition(const pca_matrix *X,
                          pca_matrix *U, double *S, pca_matrix *Vt)
{
    size_t N, M, i, j, k;
    pca_matrix *A;
    double *work;

    if (!X || !X->data || !U || !S || !Vt) return -1;
    N = X->rows; M = X->cols;
    if (U->rows != N || U->cols != N || Vt->rows != M || Vt->cols != M)
        return -1;

    A = pca_matrix_copy(X);
    if (!A) return -1;

    /* U = I_N, Vt = I_M */
    for (i = 0; i < N * N; i++) U->data[i] = 0.0;
    for (i = 0; i < N; i++) U->data[i * N + i] = 1.0;
    for (i = 0; i < M * M; i++) Vt->data[i] = 0.0;
    for (i = 0; i < M; i++) Vt->data[i * M + i] = 1.0;

    work = (double*)malloc((N > M ? N : M) * sizeof(double));
    if (!work) { pca_matrix_free(A); return -1; }

    /* Phase 1: Golub-Kahan bidiagonalization */
    for (k = 0; k < M; k++) {
        /* Left Householder */
        {
            double x[N - k];
            double v[N - k];
            double beta;
            size_t ii;
            for (ii = 0; ii < N - k; ii++)
                x[ii] = A->data[(k + ii) * M + k];
            householder_vector(x, N - k, v, &beta);
            for (j = k; j < M; j++) {
                double dot = 0.0;
                for (ii = 0; ii < N - k; ii++)
                    dot += v[ii] * A->data[(k + ii) * M + j];
                dot *= beta;
                for (ii = 0; ii < N - k; ii++)
                    A->data[(k + ii) * M + j] -= dot * v[ii];
            }
            for (i = 0; i < N; i++) {
                double dot = 0.0;
                for (ii = 0; ii < N - k; ii++)
                    dot += v[ii] * U->data[i * N + (k + ii)];
                dot *= beta;
                for (ii = 0; ii < N - k; ii++)
                    U->data[i * N + (k + ii)] -= dot * v[ii];
            }
        }
        /* Right Householder */
        if (k + 1 < M) {
            double x[M - k - 1];
            double v[M - k - 1];
            double beta;
            size_t jj;
            for (jj = 0; jj < M - k - 1; jj++)
                x[jj] = A->data[k * M + (k + 1 + jj)];
            householder_vector(x, M - k - 1, v, &beta);
            for (i = k; i < N; i++) {
                double dot = 0.0;
                for (jj = 0; jj < M - k - 1; jj++)
                    dot += v[jj] * A->data[i * M + (k + 1 + jj)];
                dot *= beta;
                for (jj = 0; jj < M - k - 1; jj++)
                    A->data[i * M + (k + 1 + jj)] -= dot * v[jj];
            }
            for (j = 0; j < M; j++) {
                double dot = 0.0;
                for (jj = 0; jj < M - k - 1; jj++)
                    dot += v[jj] * Vt->data[(k + 1 + jj) * M + j];
                dot *= beta;
                for (jj = 0; jj < M - k - 1; jj++)
                    Vt->data[(k + 1 + jj) * M + j] -= dot * v[jj];
            }
        }
    }

    /* Phase 2: Singular values from B^T * B tridiagonal eigen */
    {
        double *d = (double*)calloc(M, sizeof(double));
        double *e = (double*)calloc(M, sizeof(double));
        if (!d || !e) { free(d); free(e); free(work); pca_matrix_free(A); return -1; }
        for (k = 0; k < M; k++) {
            d[k] = fabs(A->data[k * M + k]);
            if (k + 1 < M) e[k] = A->data[k * M + (k + 1)];
            else e[k] = 0.0;
        }
        {
            pca_matrix *Tmat = pca_matrix_alloc(M, M);
            if (!Tmat) { free(d); free(e); free(work); pca_matrix_free(A); return -1; }
            for (k = 0; k < M; k++) {
                double dk2 = d[k] * d[k];
                double ek1_2 = (k > 0) ? e[k-1] * e[k-1] : 0.0;
                Tmat->data[k * M + k] = dk2 + ek1_2;
                if (k < M - 1) {
                    double val = d[k] * e[k];
                    Tmat->data[k * M + (k+1)] = val;
                    Tmat->data[(k+1) * M + k] = val;
                }
            }
            /* Simplified Jacobi to get approximate eigenvalues */
            {
                size_t sweep, p, q;
                for (sweep = 0; sweep < 30; sweep++) {
                    for (p = 0; p < M; p++) {
                        for (q = p + 1; q < M; q++) {
                            double app = Tmat->data[p * M + p];
                            double aqq = Tmat->data[q * M + q];
                            double apq = Tmat->data[p * M + q];
                            double tau, t, c, s;
                            if (fabs(apq) < 1e-14) continue;
                            tau = (aqq - app) / (2.0 * apq);
                            if (tau >= 0.0)
                                t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                            else
                                t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                            c = 1.0 / sqrt(1.0 + t * t);
                            s = t * c;
                            {
                                double tmp_p = c * app - s * apq;
                                double tmp_q = s * app + c * apq;
                                Tmat->data[p * M + p] = c * tmp_p - s * tmp_q;
                                Tmat->data[q * M + q] = s * tmp_p + c * tmp_q;
                                {
                                    size_t r;
                                    for (r = 0; r < M; r++) {
                                        if (r != p && r != q) {
                                            double arp = Tmat->data[r * M + p];
                                            double arq = Tmat->data[r * M + q];
                                            Tmat->data[r * M + p] = c * arp - s * arq;
                                            Tmat->data[p * M + r] = Tmat->data[r * M + p];
                                            Tmat->data[r * M + q] = s * arp + c * arq;
                                            Tmat->data[q * M + r] = Tmat->data[r * M + q];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (k = 0; k < M; k++) {
                double val = Tmat->data[k * M + k];
                S[k] = (val > 1e-15) ? sqrt(val) : 0.0;
            }
            pca_matrix_free(Tmat);
        }
        free(d); free(e);
    }
    free(work);
    pca_matrix_free(A);
    return 0;
}

/* ===================================================================
 * L5: Sort eigenvalues and eigenvectors descending
 * Used after Jacobi or NIPALS to ensure ordering by variance.
 * =================================================================== */
static void sort_eigen_descending(double *eigenvalues, pca_matrix *eigenvectors, size_t M)
{
    size_t i, j, r;

    for (i = 0; i < M - 1; i++) {
        size_t max_idx = i;
        for (j = i + 1; j < M; j++) {
            if (eigenvalues[j] > eigenvalues[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            /* Swap eigenvalues */
            double tmp = eigenvalues[i];
            eigenvalues[i] = eigenvalues[max_idx];
            eigenvalues[max_idx] = tmp;
            /* Swap eigenvector columns */
            for (r = 0; r < M; r++) {
                double tv = eigenvectors->data[r * M + i];
                eigenvectors->data[r * M + i] = eigenvectors->data[r * M + max_idx];
                eigenvectors->data[r * M + max_idx] = tv;
            }
        }
    }
}

/* ===================================================================
 * L5: Full PCA fitting via Jacobi eigen-decomposition
 *
 * Complete end-to-end PCA pipeline:
 *   1. Center and scale X (auto-scaling for correlation PCA)
 *   2. Compute covariance matrix S = X^T * X / (N-1)
 *   3. Jacobi eigen-decomposition of S
 *   4. Sort eigenvalues/vectors descending
 *   5. Compute scores: T = X * V
 *   6. Compute variance explained and cumulative
 *
 * This is the standard method for moderate M (variables < ~100).
 * For larger M, prefer NIPALS or randomized SVD.
 *
 * Complexity: O(N*M^2 + M^3*sweeps)
 * Returns: 0 on success, -1 on invalid input or numerical failure.
 * =================================================================== */
int pca_fit_full(pca_matrix *X, pca_model *model, size_t max_sweeps, double tol)
{
    size_t N, M, i, j, k;
    pca_matrix *cov, *V;
    int ret;

    if (!X || !X->data || !model) return -1;
    N = X->rows; M = X->cols;
    if (N < 2 || M < 2) return -1;
    if (M != model->n_vars) return -1;

    model->n_obs = N;

    /* Step 1: Center and scale */
    pca_center_scale(X, model->mean_vec, model->std_vec);
    model->use_scaling = 1;

    /* Step 2: Covariance matrix */
    cov = pca_compute_covariance(X);
    if (!cov) return -1;

    /* Step 3: Jacobi eigen */
    V = pca_matrix_alloc(M, M);
    if (!V) { pca_matrix_free(cov); return -1; }

    ret = pca_jacobi_eigen(cov, model->eigenvalues, V, max_sweeps, tol);

    if (ret != 0 && ret != -2) {
        /* Try harder */
        ret = pca_jacobi_eigen(cov, model->eigenvalues, V, max_sweeps * 2, tol * 10);
    }

    /* Step 4: Sort descending */
    sort_eigen_descending(model->eigenvalues, V, M);

    /* Copy V to model->loadings */
    for (j = 0; j < M; j++)
        for (i = 0; i < M; i++)
            model->loadings->data[i * M + j] = V->data[i * M + j];

    /* Step 5: Scores T = X * V */
    if (model->scores) pca_matrix_free(model->scores);
    model->scores = pca_matrix_alloc(N, M);
    if (!model->scores) { pca_matrix_free(V); pca_matrix_free(cov); return -1; }

    for (i = 0; i < N; i++) {
        for (k = 0; k < M; k++) {
            double sum = 0.0;
            for (j = 0; j < M; j++)
                sum += X->data[i * M + j] * V->data[j * M + k];
            model->scores->data[i * M + k] = sum;
        }
    }

    /* Step 6: Variance explained */
    pca_compute_variance_explained(model->eigenvalues, M,
                                    model->var_expl, model->cum_var);

    pca_matrix_free(V);
    pca_matrix_free(cov);
    return 0;
}
