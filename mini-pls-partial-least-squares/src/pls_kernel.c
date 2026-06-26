#include "pls_kernel.h"
#include "pls_nipals.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

KernelPLSConfig kernel_pls_config_default(void) {
    KernelPLSConfig c;
    c.kernel_type = KERNEL_RBF;
    c.gamma = 1.0;
    c.coef0 = 0.0;
    c.degree = 3;
    c.a_lvs = 3;
    c.tolerance = 1e-10;
    c.max_iterations = 200;
    return c;
}

double kernel_evaluate(const Vector *x, const Vector *z,
                       KernelType type, double gamma,
                       double coef0, int degree) {
    if (!x || !z || x->len != z->len) return 0.0;

    double dot = 0.0, sq_dist = 0.0;
    for (size_t i = 0; i < x->len; i++) {
        dot += x->data[i] * z->data[i];
        sq_dist += SQ(x->data[i] - z->data[i]);
    }

    switch (type) {
        case KERNEL_LINEAR:
            return dot + coef0;
        case KERNEL_POLYNOMIAL:
            return pow(gamma * dot + coef0, degree);
        case KERNEL_RBF:
            return exp(-gamma * sq_dist);
        case KERNEL_SIGMOID:
            return tanh(gamma * dot + coef0);
        default:
            return dot;
    }
}

Matrix* kernel_compute_matrix(const Matrix *X,
                              KernelType type, double gamma,
                              double coef0, int degree) {
    if (!X) return NULL;
    size_t n = X->rows;
    Matrix *K = matrix_alloc(n, n);
    if (!K) return NULL;

    for (size_t i = 0; i < n; i++) {
        Vector *xi = matrix_get_row(X, i);
        if (!xi) continue;
        for (size_t j = i; j < n; j++) {
            Vector *xj = matrix_get_row(X, j);
            if (!xj) { vector_free(xi); continue; }
            double kval = kernel_evaluate(xi, xj, type, gamma, coef0, degree);
            MATRIX_AT(K, i, j) = kval;
            MATRIX_AT(K, j, i) = kval;  /* Symmetric */
            vector_free(xj);
        }
        vector_free(xi);
    }
    return K;
}

void kernel_center_matrix(Matrix *K) {
    if (!K || K->rows != K->cols || K->rows == 0) return;
    size_t n = K->rows;

    /* Compute row means, column means, and total mean */
    Vector *row_mean = vector_alloc(n);
    Vector *col_mean = vector_alloc(n);
    if (!row_mean || !col_mean) { vector_free(row_mean); vector_free(col_mean); return; }

    double total_mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        double rsum = 0.0, csum = 0.0;
        for (size_t j = 0; j < n; j++) {
            rsum += MATRIX_AT(K, i, j);
            csum += MATRIX_AT(K, j, i);
        }
        row_mean->data[i] = rsum / (double)n;
        col_mean->data[i] = csum / (double)n;
        total_mean += rsum;
    }
    total_mean /= (double)(n * n);

    /* K = K - 1_n * row_mean^T - col_mean * 1_n^T + total_mean * 1_n * 1_n^T */
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            MATRIX_AT(K, i, j) = MATRIX_AT(K, i, j)
                               - row_mean->data[i]
                               - col_mean->data[j]
                               + total_mean;

    vector_free(row_mean);
    vector_free(col_mean);
}

Vector* kernel_compute_new(const Vector *x_new, const Matrix *X_train,
                           KernelType type, double gamma,
                           double coef0, int degree) {
    if (!x_new || !X_train) return NULL;
    size_t n = X_train->rows;
    Vector *k_new = vector_alloc(n);
    if (!k_new) return NULL;

    for (size_t i = 0; i < n; i++) {
        Vector *xi = matrix_get_row(X_train, i);
        if (!xi) { k_new->data[i] = 0.0; continue; }
        k_new->data[i] = kernel_evaluate(x_new, xi, type, gamma, coef0, degree);
        vector_free(xi);
    }
    return k_new;
}

void kernel_center_new(Vector *k_new, const Vector *row_mean_K,
                       const Vector *col_mean_K, double total_mean_K) {
    if (!k_new || !row_mean_K || !col_mean_K) return;
    if (k_new->len != row_mean_K->len || k_new->len != col_mean_K->len) return;
    /* k_centered = k_new - row_mean_K - sum(k_new)/n * 1 + total_mean_K */
    double k_sum = 0.0;
    for (size_t i = 0; i < k_new->len; i++) k_sum += k_new->data[i];
    double k_mean = k_sum / (double)k_new->len;

    for (size_t i = 0; i < k_new->len; i++)
        k_new->data[i] = k_new->data[i] - row_mean_K->data[i]
                        - k_mean + total_mean_K;
}

KernelPLSModel* kernel_pls_model_alloc(size_t n_train, size_t p_vars,
                                        size_t q_vars, size_t a_lvs) {
    KernelPLSModel *m = (KernelPLSModel*)calloc(1, sizeof(KernelPLSModel));
    if (!m) return NULL;
    m->n_train = n_train; m->p_vars = p_vars;
    m->q_vars = q_vars; m->a_lvs = a_lvs;

    m->X_train = matrix_alloc(n_train, p_vars);
    m->K = matrix_alloc(n_train, n_train);
    m->Beta_K = matrix_alloc(n_train, q_vars);
    m->row_mean_K = vector_alloc(n_train);
    m->col_mean_K = vector_alloc(n_train);
    m->X_mean = vector_alloc(p_vars);
    m->X_std = vector_alloc(p_vars);
    m->Y_mean = vector_alloc(q_vars);
    m->Y_std = vector_alloc(q_vars);

    if (!m->X_train || !m->K || !m->Beta_K ||
        !m->row_mean_K || !m->col_mean_K ||
        !m->X_mean || !m->X_std || !m->Y_mean || !m->Y_std) {
        kernel_pls_model_free(m); return NULL;
    }
    m->kernel_type = KERNEL_RBF; m->gamma = 1.0;
    m->coef0 = 0.0; m->degree = 3;
    m->center_x = 0; m->scale_x = 0;
    m->center_y = 0; m->scale_y = 0;
    m->R2Y_cum = 0.0; m->Q2_cum = 0.0;
    return m;
}

void kernel_pls_model_free(KernelPLSModel *model) {
    if (!model) return;
    matrix_free(model->X_train); matrix_free(model->K);
    matrix_free(model->Beta_K);
    vector_free(model->row_mean_K); vector_free(model->col_mean_K);
    vector_free(model->X_mean); vector_free(model->X_std);
    vector_free(model->Y_mean); vector_free(model->Y_std);
    free(model);
}

int kernel_pls_fit(const Matrix *X, const Matrix *Y,
                   const KernelPLSConfig *config,
                   KernelPLSModel *model) {
    if (!X || !Y || !model || !config) return -1;
    size_t n = X->rows, p = X->cols, q = Y->cols, a = config->a_lvs;

    if (n != model->n_train || p != model->p_vars || q != model->q_vars)
        return -1;

    /* Copy training data */
    memcpy(model->X_train->data, X->data, n * p * sizeof(double));
    model->kernel_type = config->kernel_type;
    model->gamma = config->gamma;
    model->coef0 = config->coef0;
    model->degree = config->degree;

    /* Compute kernel matrix */
    Matrix *K_raw = kernel_compute_matrix(X, config->kernel_type,
                                           config->gamma, config->coef0,
                                           config->degree);
    if (!K_raw) return -1;
    memcpy(model->K->data, K_raw->data, n * n * sizeof(double));

    /* Center kernel matrix and store centering statistics */
    kernel_center_matrix(model->K);

    /* Store centering statistics for prediction */
    /* row_mean: mean of each row before centering (we lost it, approximate) */
    for (size_t i = 0; i < n; i++) {
        double rsum = 0.0, csum = 0.0;
        for (size_t j = 0; j < n; j++) {
            rsum += MATRIX_AT(K_raw, i, j);
            csum += MATRIX_AT(K_raw, j, i);
        }
        model->row_mean_K->data[i] = rsum / (double)n;
        model->col_mean_K->data[i] = csum / (double)n;
        model->total_mean_K += rsum;
    }
    model->total_mean_K /= (double)(n * n);

    /* Fit PLS on (K, Y) using NIPALS */
    PLSModel *pls = pls_model_alloc(n, n, q, a);
    if (!pls) { matrix_free(K_raw); return -1; }

    Matrix *Kw = matrix_copy(model->K);
    Matrix *Yw = matrix_copy(Y);
    if (!Kw || !Yw) {
        matrix_free(Kw); matrix_free(Yw);
        pls_model_free(pls); matrix_free(K_raw); return -1;
    }

    NIPALSConfig nipcfg = nipals_config_default();
    nipcfg.tolerance = config->tolerance;
    nipcfg.max_iterations = config->max_iterations;
    if (nipals_fit(Kw, Yw, a, &nipcfg, pls) == 0) {
        /* Beta_K = W * inv(P^T * W) * Q^T  (n x q) */
        memcpy(model->Beta_K->data, pls->Beta->data, n * q * sizeof(double));
        model->R2Y_cum = pls->R2Y_cum;
    }

    matrix_free(Kw); matrix_free(Yw);
    pls_model_free(pls);
    matrix_free(K_raw);
    return 0;
}

int kernel_pls_predict(const KernelPLSModel *model,
                       const Matrix *X_new, Matrix *Y_pred) {
    if (!model || !X_new || !Y_pred || X_new->cols != model->p_vars ||
        Y_pred->rows != X_new->rows || Y_pred->cols != model->q_vars)
        return -1;

    for (size_t i = 0; i < X_new->rows; i++) {
        Vector *x_i = matrix_get_row(X_new, i);
        if (!x_i) continue;
        Vector *y_i = vector_alloc(model->q_vars);
        if (!y_i) { vector_free(x_i); continue; }
        if (kernel_pls_predict_single(model, x_i, y_i) == 0)
            matrix_set_row(Y_pred, i, y_i);
        vector_free(x_i); vector_free(y_i);
    }
    return 0;
}

int kernel_pls_predict_single(const KernelPLSModel *model,
                              const Vector *x_new, Vector *y_pred) {
    if (!model || !x_new || !y_pred || y_pred->len != model->q_vars) return -1;
    if (x_new->len != model->p_vars) return -1;

    Vector *k_new = kernel_compute_new(x_new, model->X_train,
                                        model->kernel_type, model->gamma,
                                        model->coef0, model->degree);
    if (!k_new) return -1;
    kernel_center_new(k_new, model->row_mean_K, model->col_mean_K,
                      model->total_mean_K);

    /* y_pred = k_new^T * Beta_K */
    for (size_t j = 0; j < model->q_vars; j++) {
        double sum = 0.0;
        for (size_t k = 0; k < model->n_train; k++)
            sum += k_new->data[k] * MATRIX_AT(model->Beta_K, k, j);
        y_pred->data[j] = sum;
    }
    vector_free(k_new);
    return 0;
}

Matrix* kernel_pls_compute_scores(const KernelPLSModel *model) {
    if (!model || !model->K) return NULL;
    /* Scores from the kernel matrix: T = K * R where R is not stored;
     * for now return K as a proxy, or compute via NIPALS on K */
    /* Simplified: return a dummy scores matrix */
    Matrix *S = matrix_alloc(model->n_train, model->a_lvs);
    if (!S) return NULL;
    /* For production use, one would recompute NIPALS on K to get scores */
    for (size_t i = 0; i < model->n_train; i++)
        for (size_t a = 0; a < model->a_lvs; a++)
            MATRIX_AT(S, i, a) = MATRIX_AT(model->K, i, a);  /* first A columns of K as proxy */
    return S;
}
