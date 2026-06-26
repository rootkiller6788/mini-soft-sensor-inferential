#include "matrix_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

static int mx_ok(const Matrix *A) {
    return A && (!(A->rows > 0 && A->cols > 0) || A->data) ? 0 : -1;
}
static int vx_ok(const Vector *v) {
    return v && (!(v->len > 0) || v->data) ? 0 : -1;
}

/* ---- Matrix Lifecycle ---- */

Matrix* matrix_alloc(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return NULL;
    Matrix *A = (Matrix*)calloc(1, sizeof(Matrix));
    if (!A) return NULL;
    A->data = (double*)calloc(rows * cols, sizeof(double));
    if (!A->data) { free(A); return NULL; }
    A->rows = rows; A->cols = cols; A->owns_data = 1;
    return A;
}

Matrix* matrix_from_array(size_t rows, size_t cols, const double *data) {
    if (!data || rows == 0 || cols == 0) return NULL;
    Matrix *A = matrix_alloc(rows, cols);
    if (!A) return NULL;
    memcpy(A->data, data, rows * cols * sizeof(double));
    return A;
}

Matrix* matrix_copy(const Matrix *A) {
    if (mx_ok(A) != 0) return NULL;
    return matrix_from_array(A->rows, A->cols, A->data);
}

Matrix* matrix_view(Matrix *A, size_t r0, size_t c0, size_t rs, size_t cs) {
    if (!A || r0 + rs > A->rows || c0 + cs > A->cols) return NULL;
    Matrix *V = (Matrix*)calloc(1, sizeof(Matrix));
    if (!V) return NULL;
    V->data = A->data + r0 * A->cols + c0;
    V->rows = rs; V->cols = cs; V->owns_data = 0;
    return V;
}

void matrix_free(Matrix *A) {
    if (!A) return;
    if (A->owns_data && A->data) free(A->data);
    free(A);
}

void matrix_fill(Matrix *A, double value) {
    if (!A || !A->data) return;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++) A->data[i] = value;
}

void matrix_eye(Matrix *A) {
    if (!A || !A->data || A->rows != A->cols) return;
    matrix_fill(A, 0.0);
    for (size_t i = 0; i < A->rows; i++) MATRIX_AT(A, i, i) = 1.0;
}

/* ---- Vector Lifecycle ---- */

Vector* vector_alloc(size_t n) {
    if (n == 0) return NULL;
    Vector *v = (Vector*)calloc(1, sizeof(Vector));
    if (!v) return NULL;
    v->data = (double*)calloc(n, sizeof(double));
    if (!v->data) { free(v); return NULL; }
    v->len = n; v->owns_data = 1;
    return v;
}

Vector* vector_from_array(size_t n, const double *data) {
    if (!data || n == 0) return NULL;
    Vector *v = vector_alloc(n);
    if (!v) return NULL;
    memcpy(v->data, data, n * sizeof(double));
    return v;
}

Vector* vector_copy(const Vector *v) {
    if (vx_ok(v) != 0) return NULL;
    return vector_from_array(v->len, v->data);
}

void vector_free(Vector *v) {
    if (!v) return;
    if (v->owns_data && v->data) free(v->data);
    free(v);
}

void vector_fill(Vector *v, double value) {
    if (!v || !v->data) return;
    for (size_t i = 0; i < v->len; i++) v->data[i] = value;
}

Vector* matrix_get_column(const Matrix *A, size_t j) {
    if (mx_ok(A) != 0 || j >= A->cols) return NULL;
    Vector *v = vector_alloc(A->rows);
    if (!v) return NULL;
    for (size_t i = 0; i < A->rows; i++) v->data[i] = MATRIX_AT(A, i, j);
    return v;
}

Vector* matrix_get_row(const Matrix *A, size_t i) {
    if (mx_ok(A) != 0 || i >= A->rows) return NULL;
    Vector *v = vector_alloc(A->cols);
    if (!v) return NULL;
    for (size_t j = 0; j < A->cols; j++) v->data[j] = MATRIX_AT(A, i, j);
    return v;
}

void matrix_set_column(Matrix *A, size_t j, const Vector *v) {
    if (mx_ok(A) != 0 || vx_ok(v) != 0 || j >= A->cols || v->len != A->rows) return;
    for (size_t i = 0; i < A->rows; i++) MATRIX_AT(A, i, j) = v->data[i];
}

void matrix_set_row(Matrix *A, size_t i, const Vector *v) {
    if (mx_ok(A) != 0 || vx_ok(v) != 0 || i >= A->rows || v->len != A->cols) return;
    for (size_t j = 0; j < A->cols; j++) MATRIX_AT(A, i, j) = v->data[j];
}

/* ---- Matrix Arithmetic ---- */

void matrix_gemm(Matrix *C, double alpha, const Matrix *A,
                 const Matrix *B, double beta) {
    if (mx_ok(C) || mx_ok(A) || mx_ok(B)) return;
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) return;
    size_t m = A->rows, k = A->cols, n = B->cols;
    if (beta != 1.0)
        for (size_t i = 0; i < m * n; i++) C->data[i] *= beta;
    for (size_t i = 0; i < m; i++)
        for (size_t kk = 0; kk < k; kk++) {
            double aik = alpha * MATRIX_AT(A, i, kk);
            if (fabs(aik) < DBL_MIN) continue;
            for (size_t j = 0; j < n; j++)
                MATRIX_AT(C, i, j) += aik * MATRIX_AT(B, kk, j);
        }
}

Matrix* matrix_multiply(const Matrix *A, const Matrix *B) {
    if (mx_ok(A) || mx_ok(B) || A->cols != B->rows) return NULL;
    Matrix *C = matrix_alloc(A->rows, B->cols);
    if (!C) return NULL;
    matrix_gemm(C, 1.0, A, B, 0.0);
    return C;
}

Vector* matrix_vector_multiply(const Matrix *A, const Vector *x) {
    if (mx_ok(A) || vx_ok(x) || A->cols != x->len) return NULL;
    Vector *y = vector_alloc(A->rows);
    if (!y) return NULL;
    for (size_t i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < A->cols; j++)
            sum += MATRIX_AT(A, i, j) * x->data[j];
        y->data[i] = sum;
    }
    return y;
}

Matrix* matrix_transpose(const Matrix *A) {
    if (mx_ok(A)) return NULL;
    Matrix *At = matrix_alloc(A->cols, A->rows);
    if (!At) return NULL;
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < A->cols; j++)
            MATRIX_AT(At, j, i) = MATRIX_AT(A, i, j);
    return At;
}

Matrix* vector_outer_product(const Vector *u, const Vector *v) {
    if (vx_ok(u) || vx_ok(v)) return NULL;
    Matrix *C = matrix_alloc(u->len, v->len);
    if (!C) return NULL;
    for (size_t i = 0; i < u->len; i++)
        for (size_t j = 0; j < v->len; j++)
            MATRIX_AT(C, i, j) = u->data[i] * v->data[j];
    return C;
}

double vector_dot(const Vector *u, const Vector *v) {
    if (vx_ok(u) || vx_ok(v) || u->len != v->len) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < u->len; i++) sum += u->data[i] * v->data[i];
    return sum;
}

double vector_norm_l2(const Vector *v) {
    if (vx_ok(v)) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < v->len; i++) sum += SQ(v->data[i]);
    return sqrt(sum);
}

double matrix_norm_frobenius(const Matrix *A) {
    if (mx_ok(A)) return 0.0;
    double sum = 0.0;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++) sum += SQ(A->data[i]);
    return sqrt(sum);
}

double vector_mean(const Vector *v) {
    if (vx_ok(v) || v->len == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < v->len; i++) sum += v->data[i];
    return sum / (double)v->len;
}

double vector_variance(const Vector *v) {
    if (vx_ok(v) || v->len < 2) return 0.0;
    double m = vector_mean(v), ss = 0.0;
    for (size_t i = 0; i < v->len; i++) { double d = v->data[i] - m; ss += d * d; }
    return ss / (double)(v->len - 1);
}

double vector_stddev(const Vector *v) { return sqrt(vector_variance(v)); }

void matrix_add_inplace(Matrix *A, const Matrix *B) {
    if (mx_ok(A) || mx_ok(B) || A->rows != B->rows || A->cols != B->cols) return;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++) A->data[i] += B->data[i];
}

void matrix_scale_inplace(Matrix *A, double scalar) {
    if (mx_ok(A)) return;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++) A->data[i] *= scalar;
}

Matrix* matrix_elementwise_multiply(const Matrix *A, const Matrix *B) {
    if (mx_ok(A) || mx_ok(B) || A->rows != B->rows || A->cols != B->cols) return NULL;
    Matrix *C = matrix_alloc(A->rows, A->cols);
    if (!C) return NULL;
    for (size_t i = 0, n = A->rows * A->cols; i < n; i++)
        C->data[i] = A->data[i] * B->data[i];
    return C;
}
