#include "pls_model.h"
#include "matrix_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])

/* =================================================================
 * pls_model.c — PLS Model Lifecycle and Core Operations
 *
 * Knowledge: L1 (PLSModel struct), L2 (lifecycle, predict, scores),
 * L3 (preprocessing configuration, Beta computation).
 * ================================================================= */

PLSModel* pls_model_alloc(size_t n_samples, size_t p_vars,
                          size_t q_vars, size_t a_lvs) {
    if (n_samples == 0 || p_vars == 0 || q_vars == 0 || a_lvs == 0)
        return NULL;
    PLSModel *m = (PLSModel*)calloc(1, sizeof(PLSModel));
    if (!m) return NULL;

    m->n_samples = n_samples;
    m->p_vars = p_vars;
    m->q_vars = q_vars;
    m->a_lvs = a_lvs;

    m->W = matrix_alloc(p_vars, a_lvs);
    m->P = matrix_alloc(p_vars, a_lvs);
    m->T = matrix_alloc(n_samples, a_lvs);
    m->C = matrix_alloc(q_vars, a_lvs);
    m->U = matrix_alloc(n_samples, a_lvs);
    m->Q = matrix_alloc(q_vars, a_lvs);
    m->B_inner = matrix_alloc(a_lvs, a_lvs);
    m->Beta = matrix_alloc(p_vars, q_vars);
    m->b0 = vector_alloc(q_vars);

    m->X_mean = vector_alloc(p_vars);
    m->X_std = vector_alloc(p_vars);
    m->Y_mean = vector_alloc(q_vars);
    m->Y_std = vector_alloc(q_vars);

    /* Check all allocations succeeded */
    if (!m->W || !m->P || !m->T || !m->C || !m->U || !m->Q ||
        !m->B_inner || !m->Beta || !m->b0 ||
        !m->X_mean || !m->X_std || !m->Y_mean || !m->Y_std) {
        pls_model_free(m);
        return NULL;
    }

    m->center_x = 0; m->scale_x = 0;
    m->center_y = 0; m->scale_y = 0;
    m->R2X_cum = 0.0; m->R2Y_cum = 0.0; m->Q2_cum = 0.0;

    return m;
}

PLSModel* pls_model_copy(const PLSModel *model) {
    if (!model) return NULL;
    PLSModel *cp = pls_model_alloc(model->n_samples, model->p_vars,
                                    model->q_vars, model->a_lvs);
    if (!cp) return NULL;

    /* Deep copy all matrices */
    Matrix *matrices_src[] = {model->W, model->P, model->T, model->C,
                               model->U, model->Q, model->B_inner, model->Beta};
    Matrix *matrices_dst[] = {cp->W, cp->P, cp->T, cp->C,
                               cp->U, cp->Q, cp->B_inner, cp->Beta};
    for (int k = 0; k < 8; k++) {
        if (matrices_src[k] && matrices_dst[k])
            memcpy(matrices_dst[k]->data, matrices_src[k]->data,
                   matrices_src[k]->rows * matrices_src[k]->cols * sizeof(double));
    }

    Vector *vecs_src[] = {model->b0, model->X_mean, model->X_std,
                           model->Y_mean, model->Y_std};
    Vector *vecs_dst[] = {cp->b0, cp->X_mean, cp->X_std,
                           cp->Y_mean, cp->Y_std};
    for (int k = 0; k < 5; k++) {
        if (vecs_src[k] && vecs_dst[k])
            memcpy(vecs_dst[k]->data, vecs_src[k]->data,
                   vecs_src[k]->len * sizeof(double));
    }

    cp->center_x = model->center_x; cp->scale_x = model->scale_x;
    cp->center_y = model->center_y; cp->scale_y = model->scale_y;
    cp->R2X_cum = model->R2X_cum;
    cp->R2Y_cum = model->R2Y_cum;
    cp->Q2_cum = model->Q2_cum;

    return cp;
}

void pls_model_free(PLSModel *model) {
    if (!model) return;
    matrix_free(model->W); matrix_free(model->P);
    matrix_free(model->T); matrix_free(model->C);
    matrix_free(model->U); matrix_free(model->Q);
    matrix_free(model->B_inner); matrix_free(model->Beta);
    vector_free(model->b0);
    vector_free(model->X_mean); vector_free(model->X_std);
    vector_free(model->Y_mean); vector_free(model->Y_std);
    free(model);
}

int pls_model_compute_beta(PLSModel *model) {
    if (!model || !model->W || !model->P || !model->Q) return -1;

    /* Compute P^T * W (a_lvs x a_lvs) */
    Matrix *Pt = matrix_transpose(model->P);
    if (!Pt) return -1;
    Matrix *PtW = matrix_multiply(Pt, model->W);
    matrix_free(Pt);
    if (!PtW) return -1;

    /* Invert P^T * W using pseudo-inverse for numerical stability */
    Matrix *PtW_inv = matrix_pinv(PtW);
    matrix_free(PtW);
    if (!PtW_inv) return -1;

    /* Beta = W * inv(P^T * W) * Q^T */
    Matrix *W_PtWinv = matrix_multiply(model->W, PtW_inv);
    matrix_free(PtW_inv);
    if (!W_PtWinv) return -1;

    Matrix *Qt = matrix_transpose(model->Q);
    if (!Qt) { matrix_free(W_PtWinv); return -1; }

    Matrix *Beta_new = matrix_multiply(W_PtWinv, Qt);
    matrix_free(W_PtWinv);
    matrix_free(Qt);
    if (!Beta_new) return -1;

    /* Copy into model->Beta */
    memcpy(model->Beta->data, Beta_new->data,
           Beta_new->rows * Beta_new->cols * sizeof(double));
    matrix_free(Beta_new);

    return 0;
}

void pls_model_set_preprocessing(PLSModel *model,
    const Vector *x_mean, const Vector *x_std, int center_x, int scale_x,
    const Vector *y_mean, const Vector *y_std, int center_y, int scale_y) {
    if (!model) return;
    if (x_mean) memcpy(model->X_mean->data, x_mean->data,
                        model->p_vars * sizeof(double));
    if (x_std)  memcpy(model->X_std->data, x_std->data,
                        model->p_vars * sizeof(double));
    if (y_mean) memcpy(model->Y_mean->data, y_mean->data,
                        model->q_vars * sizeof(double));
    if (y_std)  memcpy(model->Y_std->data, y_std->data,
                        model->q_vars * sizeof(double));
    model->center_x = center_x; model->scale_x = scale_x;
    model->center_y = center_y; model->scale_y = scale_y;
}

void pls_model_set_stats(PLSModel *model, double R2X_cum,
                         double R2Y_cum, double Q2_cum) {
    if (!model) return;
    model->R2X_cum = R2X_cum;
    model->R2Y_cum = R2Y_cum;
    model->Q2_cum = Q2_cum;
}

int pls_model_predict_single(const PLSModel *model,
                             const Vector *x_new, Vector *y_pred) {
    if (!model || !x_new || !y_pred || x_new->len != model->p_vars ||
        y_pred->len != model->q_vars) return -1;

    /* Apply preprocessing: center and scale */
    Vector *xp = vector_copy(x_new);
    if (!xp) return -1;

    if (model->center_x)
        for (size_t j = 0; j < model->p_vars; j++)
            xp->data[j] -= model->X_mean->data[j];
    if (model->scale_x)
        for (size_t j = 0; j < model->p_vars; j++) {
            double sd = model->X_std->data[j];
            if (sd > 1e-15) xp->data[j] /= sd;
        }

    /* y_pred = x_processed^T * Beta + b0 */
    for (size_t j = 0; j < model->q_vars; j++) {
        double sum = model->b0->data[j];
        for (size_t k = 0; k < model->p_vars; k++)
            sum += xp->data[k] * MATRIX_AT(model->Beta, k, j);
        y_pred->data[j] = sum;
    }

    vector_free(xp);
    return 0;
}

int pls_model_predict_batch(const PLSModel *model,
                            const Matrix *X_new, Matrix *Y_pred) {
    if (!model || !X_new || !Y_pred || X_new->cols != model->p_vars ||
        Y_pred->rows != X_new->rows || Y_pred->cols != model->q_vars)
        return -1;

    for (size_t i = 0; i < X_new->rows; i++) {
        Vector *x_i = matrix_get_row(X_new, i);
        if (!x_i) continue;
        Vector *y_i = vector_alloc(model->q_vars);
        if (!y_i) { vector_free(x_i); continue; }
        if (pls_model_predict_single(model, x_i, y_i) == 0)
            matrix_set_row(Y_pred, i, y_i);
        vector_free(x_i);
        vector_free(y_i);
    }
    return 0;
}

Matrix* pls_model_compute_scores(const PLSModel *model, const Matrix *X_new) {
    if (!model || !X_new || X_new->cols != model->p_vars) return NULL;
    Matrix *T_new = matrix_alloc(X_new->rows, model->a_lvs);
    if (!T_new) return NULL;

    for (size_t i = 0; i < X_new->rows; i++) {
        /* Preprocess row i */
        Vector *xp = matrix_get_row(X_new, i);
        if (!xp) continue;
        if (model->center_x)
            for (size_t j = 0; j < model->p_vars; j++)
                xp->data[j] -= model->X_mean->data[j];
        if (model->scale_x)
            for (size_t j = 0; j < model->p_vars; j++) {
                double sd = model->X_std->data[j];
                if (sd > 1e-15) xp->data[j] /= sd;
            }

        /* t = x_processed^T * W */
        for (size_t a = 0; a < model->a_lvs; a++) {
            double sum = 0.0;
            for (size_t j = 0; j < model->p_vars; j++)
                sum += xp->data[j] * MATRIX_AT(model->W, j, a);
            MATRIX_AT(T_new, i, a) = sum;
        }
        vector_free(xp);
    }
    return T_new;
}

Matrix* pls_model_compute_residuals(const PLSModel *model, const Matrix *X_new) {
    if (!model || !X_new || X_new->cols != model->p_vars) return NULL;

    /* Compute scores first */
    Matrix *T_new = pls_model_compute_scores(model, X_new);
    if (!T_new) return NULL;

    /* Reconstruct X_hat = T_new * P^T */
    Matrix *Pt = matrix_transpose(model->P);
    if (!Pt) { matrix_free(T_new); return NULL; }
    Matrix *X_hat = matrix_multiply(T_new, Pt);
    matrix_free(T_new);
    matrix_free(Pt);
    if (!X_hat) return NULL;

    /* Compute residuals E = X_new_processed - X_hat */
    /* First preprocess X_new */
    Matrix *X_processed = matrix_copy(X_new);
    if (!X_processed) { matrix_free(X_hat); return NULL; }
    if (model->center_x) matrix_center_columns(X_processed, model->X_mean);
    if (model->scale_x)  matrix_scale_columns(X_processed, model->X_std);

    Matrix *E = matrix_alloc(X_new->rows, model->p_vars);
    if (!E) { matrix_free(X_hat); matrix_free(X_processed); return NULL; }

    for (size_t i = 0; i < X_new->rows; i++)
        for (size_t j = 0; j < model->p_vars; j++)
            MATRIX_AT(E, i, j) = MATRIX_AT(X_processed, i, j) -
                                 MATRIX_AT(X_hat, i, j);

    matrix_free(X_hat);
    matrix_free(X_processed);
    return E;
}

Vector* pls_model_inner_predict(const PLSModel *model, const Vector *t_score) {
    if (!model || !t_score || t_score->len != model->a_lvs) return NULL;
    Vector *u_pred = vector_alloc(model->a_lvs);
    if (!u_pred) return NULL;
    for (size_t i = 0; i < model->a_lvs; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < model->a_lvs; j++)
            sum += MATRIX_AT(model->B_inner, i, j) * t_score->data[j];
        u_pred->data[i] = sum;
    }
    return u_pred;
}

PLSComponentStats* pls_component_stats_alloc(size_t max_lvs) {
    PLSComponentStats *s = (PLSComponentStats*)calloc(1, sizeof(PLSComponentStats));
    if (!s) return NULL;
    s->n_lvs = max_lvs;
    s->R2X = (double*)calloc(max_lvs + 1, sizeof(double));
    s->R2Y = (double*)calloc(max_lvs + 1, sizeof(double));
    s->Q2 = (double*)calloc(max_lvs + 1, sizeof(double));
    s->PRESS = (double*)calloc(max_lvs + 1, sizeof(double));
    if (!s->R2X || !s->R2Y || !s->Q2 || !s->PRESS) {
        pls_component_stats_free(s);
        return NULL;
    }
    return s;
}

void pls_component_stats_free(PLSComponentStats *stats) {
    if (!stats) return;
    free(stats->R2X); free(stats->R2Y);
    free(stats->Q2); free(stats->PRESS);
    free(stats);
}
