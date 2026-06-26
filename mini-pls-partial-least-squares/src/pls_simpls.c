#include "pls_simpls.h"
#include "pls_nipals.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])

SIMPLSConfig simpls_config_default(void) {
    SIMPLSConfig c;
    c.tolerance = 1e-10;
    c.max_iterations = 200;
    c.verbose = 0;
    return c;
}

int simpls_compute_weights(const Matrix *S, size_t a_lvs,
                           Matrix *W_out, const SIMPLSConfig *config) {
    if (!S || !W_out || W_out->cols < a_lvs) return -1;
    SIMPLSConfig cfg = config ? *config : simpls_config_default();
    size_t p = S->rows, q = S->cols;

    /* Work with S * S^T (p x p) */
    Matrix *SST = matrix_alloc(p, p);
    if (!SST) return -1;
    for (size_t i = 0; i < p; i++)
        for (size_t j = 0; j < p; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < q; k++)
                sum += MATRIX_AT(S, i, k) * MATRIX_AT(S, j, k);
            MATRIX_AT(SST, i, j) = sum;
        }

    /* Projection accumulator: P_acc = I initially (for orthogonality) */
    Matrix *P_acc = matrix_alloc(p, p);
    if (!P_acc) { matrix_free(SST); return -1; }
    for (size_t i = 0; i < p; i++) MATRIX_AT(P_acc, i, i) = 1.0;

    for (size_t a = 0; a < a_lvs; a++) {
        /* Compute constrained matrix: M = P_acc * SST, then find dominant eigenvector */
        /* For simplicity, use power iteration on P_acc * SST */
        /* Build M = P_acc * SST */
        Matrix *M = matrix_multiply(P_acc, SST);
        if (!M) { matrix_free(SST); matrix_free(P_acc); return -1; }

        /* Find dominant eigenvector via power iteration on M */
        Vector *w = vector_alloc(p);
        if (!w) { matrix_free(M); matrix_free(SST); matrix_free(P_acc); return -1; }
        for (size_t i = 0; i < p; i++) w->data[i] = 1.0;

        double lo = 0.0, lam = 0.0;
        for (int it = 0; it < cfg.max_iterations; it++) {
            Vector *mw = matrix_vector_multiply(M, w);
            if (!mw) break;
            double nr = vector_norm_l2(mw);
            if (nr < DBL_EPSILON) { vector_free(mw); break; }
            for (size_t i = 0; i < p; i++) w->data[i] = mw->data[i] / nr;
            lam = vector_dot(w, mw) / vector_dot(w, w);
            vector_free(mw);
            if (fabs(lam - lo) < cfg.tolerance) break;
            lo = lam;
        }

        /* Store weight vector */
        for (size_t i = 0; i < p; i++)
            MATRIX_AT(W_out, i, a) = w->data[i];

        /* Update projection: P_acc = P_acc * (I - w * w^T) */
        /* This ensures w_{a+1} is orthogonal to previous weights in SST metric */
        Matrix *wwT = vector_outer_product(w, w);
        if (wwT) {
            Matrix *I_mwwT = matrix_alloc(p, p);
            if (I_mwwT) {
                for (size_t i = 0; i < p; i++)
                    for (size_t j = 0; j < p; j++)
                        MATRIX_AT(I_mwwT, i, j) =
                            (i == j ? 1.0 : 0.0) - MATRIX_AT(wwT, i, j);
                Matrix *P_new = matrix_multiply(P_acc, I_mwwT);
                if (P_new) {
                    memcpy(P_acc->data, P_new->data, p * p * sizeof(double));
                    matrix_free(P_new);
                }
                matrix_free(I_mwwT);
            }
            matrix_free(wwT);
        }

        vector_free(w);
        matrix_free(M);
    }

    matrix_free(SST);
    matrix_free(P_acc);
    return 0;
}

int simpls_fit(const Matrix *X, const Matrix *Y, size_t a_lvs,
               const SIMPLSConfig *config, PLSModel *model) {
    if (!X || !Y || !model) return -1;
    if (X->rows != Y->rows || X->rows != model->n_samples ||
        X->cols != model->p_vars || Y->cols != model->q_vars)
        return -1;

    SIMPLSConfig cfg = config ? *config : simpls_config_default();
    size_t n = X->rows, p = X->cols, q = Y->cols;

    /* Compute cross-covariance S = X^T * Y (p x q) */
    Matrix *S = matrix_alloc(p, q);
    if (!S) return -1;
    for (size_t i = 0; i < p; i++)
        for (size_t j = 0; j < q; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++)
                sum += MATRIX_AT(X, k, i) * MATRIX_AT(Y, k, j);
            MATRIX_AT(S, i, j) = sum;
        }

    /* Compute weight vectors */
    if (simpls_compute_weights(S, a_lvs, model->W, &cfg) != 0) {
        matrix_free(S);
        return -1;
    }

    /* Compute scores T = X * W */
    Matrix *T_new = matrix_multiply(X, model->W);
    if (!T_new) { matrix_free(S); return -1; }
    memcpy(model->T->data, T_new->data, n * a_lvs * sizeof(double));
    matrix_free(T_new);

    /* Compute loadings P = X^T * T * inv(T^T * T) */
    Matrix *Xt = matrix_transpose(X);
    if (!Xt) { matrix_free(S); return -1; }
    Matrix *XtT = matrix_multiply(Xt, model->T);
    matrix_free(Xt);
    if (!XtT) { matrix_free(S); return -1; }

    Matrix *Tt = matrix_transpose(model->T);
    if (!Tt) { matrix_free(XtT); matrix_free(S); return -1; }
    Matrix *TtT = matrix_multiply(Tt, model->T);
    matrix_free(Tt);
    if (!TtT) { matrix_free(XtT); matrix_free(S); return -1; }

    Matrix *TtT_inv = matrix_pinv(TtT);
    if (!TtT_inv) { matrix_free(XtT); matrix_free(TtT); matrix_free(S); return -1; }

    Matrix *P_new = matrix_multiply(XtT, TtT_inv);
    matrix_free(XtT); matrix_free(TtT); matrix_free(TtT_inv);
    if (!P_new) { matrix_free(S); return -1; }
    memcpy(model->P->data, P_new->data, p * a_lvs * sizeof(double));
    matrix_free(P_new);

    /* Compute Y-loadings Q */
    Matrix *Yt = matrix_transpose(Y);
    if (!Yt) { matrix_free(S); return -1; }
    Matrix *YtT = matrix_multiply(Yt, model->T);
    matrix_free(Yt);
    if (!YtT) { matrix_free(S); return -1; }

    Tt = matrix_transpose(model->T);
    if (!Tt) { matrix_free(YtT); matrix_free(S); return -1; }
    TtT = matrix_multiply(Tt, model->T);
    matrix_free(Tt);
    if (!TtT) { matrix_free(YtT); matrix_free(S); return -1; }

    TtT_inv = matrix_pinv(TtT);
    if (!TtT_inv) { matrix_free(YtT); matrix_free(TtT); matrix_free(S); return -1; }

    Matrix *Q_new = matrix_multiply(YtT, TtT_inv);
    matrix_free(YtT); matrix_free(TtT); matrix_free(TtT_inv);
    if (!Q_new) { matrix_free(S); return -1; }
    memcpy(model->Q->data, Q_new->data, q * a_lvs * sizeof(double));
    matrix_free(Q_new);

    /* Inner relation: B = diag of regression of Y-scores on X-scores */
    Vector *t_col = vector_alloc(n);
    Vector *y_col = vector_alloc(n);
    if (t_col && y_col) {
        /* Use simplified approach: inner relation from OLS of each Y column on T */
        for (size_t a = 0; a < a_lvs; a++) {
            for (size_t i = 0; i < n; i++)
                t_col->data[i] = MATRIX_AT(model->T, i, a);
            double tt = vector_dot(t_col, t_col);
            /* For each Y variable, compute U score */
            for (size_t j = 0; j < q; j++) {
                for (size_t i = 0; i < n; i++)
                    y_col->data[i] = MATRIX_AT(Y, i, j);
                double ut = vector_dot(y_col, t_col);
                /* Simple inner relation coefficient */
                if (tt > DBL_EPSILON)
                    MATRIX_AT(model->B_inner, a, a) += ut / tt / (double)q;
            }
        }
    }
    vector_free(t_col); vector_free(y_col);

    /* Compute Beta coefficients */
    pls_model_compute_beta(model);

    matrix_free(S);
    return 0;
}

int simpls_predict(const PLSModel *model, const Matrix *X_new,
                   Matrix *Y_pred) {
    return pls_model_predict_batch(model, X_new, Y_pred);
}

int simpls_compute_beta_direct(PLSModel *model) {
    return pls_model_compute_beta(model);
}
