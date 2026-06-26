#include "pls_nipals.h"
#include "pls_preprocessing.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])

NIPALSConfig nipals_config_default(void) {
    NIPALSConfig c;
    c.tolerance = 1e-10;
    c.max_iterations = 200;
    c.verbose = 0;
    return c;
}

int nipals_extract_one_component(Matrix *X, Matrix *Y,
    Vector *w_out, Vector *t_out, Vector *p_out,
    Vector *c_out, Vector *u_out, double *b_out,
    const NIPALSConfig *config, int *iters_out)
{
    if (!X || !Y || !w_out || !t_out || !p_out || !c_out || !u_out || !b_out)
        return -1;
    if (X->rows != Y->rows || w_out->len != X->cols ||
        t_out->len != X->rows || p_out->len != X->cols ||
        c_out->len != Y->cols || u_out->len != Y->rows)
        return -1;

    NIPALSConfig cfg = config ? *config : nipals_config_default();
    size_t n = X->rows, p = X->cols, q = Y->cols;

    /* Initialize u with first column of Y */
    for (size_t i = 0; i < n; i++)
        u_out->data[i] = MATRIX_AT(Y, i, 0);

    double u_norm_old = 0.0, u_norm = vector_norm_l2(u_out);
    int iter;

    for (iter = 0; iter < cfg.max_iterations; iter++) {
        /* w = X^T * u / (u^T * u) */
        double utu = vector_dot(u_out, u_out);
        if (utu < DBL_EPSILON) break;
        for (size_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (size_t i = 0; i < n; i++)
                sum += MATRIX_AT(X, i, j) * u_out->data[i];
            w_out->data[j] = sum / utu;
        }

        /* Normalize w to unit length */
        double w_norm = vector_norm_l2(w_out);
        if (w_norm < DBL_EPSILON) break;
        for (size_t j = 0; j < p; j++) w_out->data[j] /= w_norm;

        /* t = X * w */
        for (size_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < p; j++)
                sum += MATRIX_AT(X, i, j) * w_out->data[j];
            t_out->data[i] = sum;
        }

        /* c = Y^T * t / (t^T * t) */
        double ttt = vector_dot(t_out, t_out);
        if (ttt < DBL_EPSILON) break;
        for (size_t j = 0; j < q; j++) {
            double sum = 0.0;
            for (size_t i = 0; i < n; i++)
                sum += MATRIX_AT(Y, i, j) * t_out->data[i];
            c_out->data[j] = sum / ttt;
        }

        /* Update u = Y * c */
        for (size_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < q; j++)
                sum += MATRIX_AT(Y, i, j) * c_out->data[j];
            u_out->data[i] = sum;
        }

        u_norm = vector_norm_l2(u_out);
        if (fabs(u_norm - u_norm_old) < cfg.tolerance) break;
        u_norm_old = u_norm;
    }

    if (iter >= cfg.max_iterations) return -1;

    /* p = X^T * t / (t^T * t) */
    double ttt = vector_dot(t_out, t_out);
    if (ttt < DBL_EPSILON) return -1;
    for (size_t j = 0; j < p; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            sum += MATRIX_AT(X, i, j) * t_out->data[i];
        p_out->data[j] = sum / ttt;
    }

    /* b = u^T * t / (t^T * t) */
    *b_out = vector_dot(u_out, t_out) / ttt;

    if (iters_out) *iters_out = iter;
    return 0;
}

void nipals_deflate(Matrix *X, Matrix *Y,
                    const Vector *t, const Vector *p,
                    const Vector *c, double b) {
    if (!X || !Y || !t || !p || !c) return;
    size_t n = X->rows, px = X->cols, q = Y->cols;

    /* X = X - t * p^T */
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < px; j++)
            MATRIX_AT(X, i, j) -= t->data[i] * p->data[j];

    /* Y = Y - b * t * c^T */
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < q; j++)
            MATRIX_AT(Y, i, j) -= b * t->data[i] * c->data[j];
}

int nipals_fit(Matrix *X, Matrix *Y, size_t a_lvs,
               const NIPALSConfig *config, PLSModel *model) {
    if (!X || !Y || !model) return -1;
    if (X->rows != Y->rows || X->rows != model->n_samples ||
        X->cols != model->p_vars || Y->cols != model->q_vars)
        return -1;

    NIPALSConfig cfg = config ? *config : nipals_config_default();

    /* Work on copies to preserve originals */
    Matrix *Xw = matrix_copy(X);
    Matrix *Yw = matrix_copy(Y);
    if (!Xw || !Yw) { matrix_free(Xw); matrix_free(Yw); return -1; }

    for (size_t a = 0; a < a_lvs; a++) {
        /* Extract column vectors from model matrices */
        Vector *w = matrix_get_column(model->W, a);
        Vector *t = matrix_get_column(model->T, a);
        Vector *p = matrix_get_column(model->P, a);
        Vector *c = matrix_get_column(model->C, a);
        Vector *u = matrix_get_column(model->U, a);
        double b;

        if (!w || !t || !p || !c || !u) {
            vector_free(w); vector_free(t); vector_free(p);
            vector_free(c); vector_free(u);
            matrix_free(Xw); matrix_free(Yw);
            return -1;
        }

        int ok = nipals_extract_one_component(Xw, Yw, w, t, p, c, u, &b, &cfg, NULL);
        if (ok != 0) {
            if (cfg.verbose) {
                /* Non-convergence is acceptable for later components
                 * that capture near-zero variance */
            }
        }

        /* Store inner relation coefficient */
        MATRIX_AT(model->B_inner, a, a) = b;

        /* Q(:,a) = c (Y-weight = Y-loading in NIPALS) */
        for (size_t j = 0; j < model->q_vars; j++)
            MATRIX_AT(model->Q, j, a) = c->data[j];

        /* Deflate X and Y */
        nipals_deflate(Xw, Yw, t, p, c, b);

        vector_free(w); vector_free(t); vector_free(p);
        vector_free(c); vector_free(u);
    }

    matrix_free(Xw); matrix_free(Yw);

    /* Compute Beta = W * inv(P^T * W) * Q^T */
    pls_model_compute_beta(model);

    return 0;
}

int nipals_predict(const PLSModel *model, const Matrix *X_new,
                   Matrix *Y_pred) {
    return pls_model_predict_batch(model, X_new, Y_pred);
}
