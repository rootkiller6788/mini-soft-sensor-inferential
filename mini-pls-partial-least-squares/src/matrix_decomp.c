#include "matrix_ops.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

int matrix_cholesky_decomp(Matrix *A) {
    if (!A || !A->data || A->rows != A->cols) return -1;
    size_t n = A->rows;
    for (size_t j = 0; j < n; j++) {
        double sd = 0.0;
        for (size_t k = 0; k < j; k++) sd += SQ(MATRIX_AT(A, j, k));
        double d = MATRIX_AT(A, j, j) - sd;
        if (d <= 0.0) return -1;
        MATRIX_AT(A, j, j) = sqrt(d);
        for (size_t i = j + 1; i < n; i++) {
            double so = 0.0;
            for (size_t k = 0; k < j; k++)
                so += MATRIX_AT(A, i, k) * MATRIX_AT(A, j, k);
            MATRIX_AT(A, i, j) = (MATRIX_AT(A, i, j) - so) / MATRIX_AT(A, j, j);
        }
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++)
            MATRIX_AT(A, i, j) = 0.0;
    return 0;
}

void matrix_forward_substitution(const Matrix *L, Vector *b) {
    if (!L || !L->data || !b || !b->data || L->rows != L->cols || L->rows != b->len)
        return;
    size_t n = L->rows;
    for (size_t i = 0; i < n; i++) {
        double s = 0.0;
        for (size_t j = 0; j < i; j++) s += MATRIX_AT(L, i, j) * b->data[j];
        b->data[i] = (b->data[i] - s) / MATRIX_AT(L, i, i);
    }
}

void matrix_backward_substitution(const Matrix *U, Vector *b) {
    if (!U || !U->data || !b || !b->data || U->rows != U->cols || U->rows != b->len)
        return;
    size_t n = U->rows;
    for (size_t i = n; i-- > 0; ) {
        double s = 0.0;
        for (size_t j = i + 1; j < n; j++) s += MATRIX_AT(U, i, j) * b->data[j];
        b->data[i] = (b->data[i] - s) / MATRIX_AT(U, i, i);
    }
}

int matrix_qr_decomp(Matrix *A) {
    if (!A || !A->data || A->rows < A->cols) return -1;
    size_t m = A->rows, n = A->cols;
    for (size_t k = 0; k < n; k++) {
        double sigma = 0.0;
        for (size_t i = k; i < m; i++) sigma += SQ(MATRIX_AT(A, i, k));
        if (sigma < DBL_EPSILON) continue;
        double akk = MATRIX_AT(A, k, k);
        double alpha = (akk > 0) ? -sqrt(sigma) : sqrt(sigma);
        double vk = akk - alpha;
        MATRIX_AT(A, k, k) = vk;
        double beta = -alpha * vk;
        if (fabs(beta) < DBL_EPSILON) continue;
        for (size_t j = k + 1; j < n; j++) {
            double w = MATRIX_AT(A, k, j);
            for (size_t i = k + 1; i < m; i++)
                w += MATRIX_AT(A, i, k) * MATRIX_AT(A, i, j);
            w /= beta;
            MATRIX_AT(A, k, j) -= w;
            for (size_t i = k + 1; i < m; i++)
                MATRIX_AT(A, i, j) -= w * MATRIX_AT(A, i, k);
        }
        MATRIX_AT(A, k, k) = alpha;
    }
    return 0;
}

double matrix_power_iteration(const Matrix *A, int max_its, double tol) {
    if (!A || !A->data || A->rows != A->cols || A->rows == 0) return 0.0;
    size_t n = A->rows;
    Vector *v = vector_alloc(n);
    if (!v) return 0.0;
    for (size_t i = 0; i < n; i++) v->data[i] = 1.0 + 0.1 * (double)i;
    double lo = 0.0, lam = 0.0;
    for (int it = 0; it < max_its; it++) {
        Vector *w = matrix_vector_multiply(A, v);
        if (!w) break;
        double nr = vector_norm_l2(w);
        if (nr < DBL_EPSILON) { vector_free(w); break; }
        for (size_t i = 0; i < n; i++) v->data[i] = w->data[i] / nr;
        Vector *Av = matrix_vector_multiply(A, v);
        if (!Av) { vector_free(w); break; }
        lam = vector_dot(v, Av) / vector_dot(v, v);
        vector_free(Av); vector_free(w);
        if (fabs(lam - lo) < tol) break;
        lo = lam;
    }
    vector_free(v);
    return lam;
}

int matrix_svd_values(const Matrix *A, Vector *S, size_t sv_count) {
    if (!A || !A->data || !S || !S->data) return -1;
    size_t m = A->rows, n = A->cols, mn = (m < n) ? m : n;
    if (sv_count > mn || sv_count == 0) return -1;
    Matrix *AtA = matrix_alloc(n, n);
    if (!AtA) return -1;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++)
                sum += MATRIX_AT(A, k, i) * MATRIX_AT(A, k, j);
            MATRIX_AT(AtA, i, j) = sum;
        }
    for (size_t sv = 0; sv < sv_count; sv++) {
        double lam = matrix_power_iteration(AtA, 200, 1e-12);
        if (lam < 0) lam = 0.0;
        S->data[sv] = sqrt(lam);
        if (sv < sv_count - 1) {
            Vector *v = vector_alloc(n);
            if (!v) { matrix_free(AtA); return -1; }
            for (size_t i = 0; i < n; i++) v->data[i] = 1.0;
            for (int it = 0; it < 30; it++) {
                Vector *w = matrix_vector_multiply(AtA, v);
                if (!w) break;
                double nr = vector_norm_l2(w);
                if (nr < DBL_EPSILON) { vector_free(w); break; }
                for (size_t i = 0; i < n; i++) v->data[i] = w->data[i] / nr;
                vector_free(w);
            }
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    MATRIX_AT(AtA, i, j) -= lam * v->data[i] * v->data[j];
            vector_free(v);
        }
    }
    matrix_free(AtA);
    return 0;
}

Matrix* matrix_pinv(const Matrix *A) {
    if (!A || !A->data) return NULL;
    size_t m = A->rows, n = A->cols;
    Matrix *AtA = matrix_alloc(n, n);
    if (!AtA) return NULL;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++)
                sum += MATRIX_AT(A, k, i) * MATRIX_AT(A, k, j);
            MATRIX_AT(AtA, i, j) = sum;
        }
    for (size_t i = 0; i < n; i++) MATRIX_AT(AtA, i, i) += 1e-12;
    if (matrix_cholesky_decomp(AtA) != 0) { matrix_free(AtA); return NULL; }
    Matrix *pinv = matrix_alloc(n, m);
    if (!pinv) { matrix_free(AtA); return NULL; }
    for (size_t j = 0; j < m; j++) {
        Vector *b = vector_alloc(n);
        if (!b) continue;
        for (size_t i = 0; i < n; i++) b->data[i] = MATRIX_AT(A, j, i);
        matrix_forward_substitution(AtA, b);
        matrix_backward_substitution(AtA, b);
        for (size_t i = 0; i < n; i++) MATRIX_AT(pinv, i, j) = b->data[i];
        vector_free(b);
    }
    matrix_free(AtA);
    return pinv;
}

Vector* matrix_solve_qr(const Matrix *A, const Vector *b) {
    if (!A || !A->data || !b || !b->data || A->rows != b->len || A->rows < A->cols)
        return NULL;
    Matrix *R = matrix_copy(A);
    if (!R) return NULL;
    if (matrix_qr_decomp(R) != 0) { matrix_free(R); return NULL; }
    size_t n = R->cols;
    Vector *x = vector_alloc(n);
    if (!x) { matrix_free(R); return NULL; }
    for (size_t i = 0; i < n; i++) x->data[i] = b->data[i];
    for (size_t i = n; i-- > 0; ) {
        double s = 0.0;
        for (size_t j = i + 1; j < n; j++)
            s += MATRIX_AT(R, i, j) * x->data[j];
        if (fabs(MATRIX_AT(R, i, i)) < DBL_EPSILON) x->data[i] = 0.0;
        else x->data[i] = (x->data[i] - s) / MATRIX_AT(R, i, i);
    }
    matrix_free(R);
    return x;
}

Vector* matrix_column_means(const Matrix *A) {
    if (!A || !A->data) return NULL;
    Vector *m = vector_alloc(A->cols);
    if (!m) return NULL;
    for (size_t j = 0; j < A->cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < A->rows; i++) sum += MATRIX_AT(A, i, j);
        m->data[j] = sum / (double)A->rows;
    }
    return m;
}

Vector* matrix_column_stddevs(const Matrix *A) {
    Vector *means = matrix_column_means(A);
    if (!means) return NULL;
    Vector *s = vector_alloc(A->cols);
    if (!s) { vector_free(means); return NULL; }
    for (size_t j = 0; j < A->cols; j++) {
        double ss = 0.0;
        for (size_t i = 0; i < A->rows; i++) {
            double d = MATRIX_AT(A, i, j) - means->data[j];
            ss += d * d;
        }
        s->data[j] = sqrt(ss / (double)A->rows);
    }
    vector_free(means);
    return s;
}

void matrix_center_columns(Matrix *A, const Vector *means) {
    if (!A || !A->data || !means || !means->data || A->cols != means->len) return;
    for (size_t j = 0; j < A->cols; j++)
        for (size_t i = 0; i < A->rows; i++)
            MATRIX_AT(A, i, j) -= means->data[j];
}

void matrix_scale_columns(Matrix *A, const Vector *stddevs) {
    if (!A || !A->data || !stddevs || !stddevs->data || A->cols != stddevs->len) return;
    for (size_t j = 0; j < A->cols; j++) {
        double sd = stddevs->data[j];
        if (sd < DBL_EPSILON) continue;
        for (size_t i = 0; i < A->rows; i++) MATRIX_AT(A, i, j) /= sd;
    }
}
