#include "pls_recursive.h"
#include "pls_nipals.h"
#include "pls_statistics.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

RecursivePLSConfig recursive_pls_config_default(void) {
    RecursivePLSConfig c;
    c.forgetting_factor = 0.99;
    c.window_size = 0;
    c.update_interval = 10;
    c.use_exponential = 1;
    c.outlier_threshold = 99.0;
    c.verbose = 0;
    return c;
}

/* ---- Welford's Online Algorithm ---- */

void welford_update(size_t *n, double *mean, double *M2, double x) {
    (*n)++;
    double delta = x - *mean;
    *mean += delta / (double)(*n);
    double delta2 = x - *mean;
    *M2 += delta * delta2;
}

void welford_finalize(size_t n, double mean, double M2,
                      double *variance, double *stddev) {
    if (n < 2) {
        *variance = 0.0;
        *stddev = 0.0;
    } else {
        *variance = M2 / (double)(n - 1);
        *stddev = sqrt(*variance);
    }
}

/* ---- Exponentially Weighted Moving Average ---- */

void ewma_update(double lambda, double *mean, double *M2, double x) {
    double delta = x - *mean;
    *mean = lambda * (*mean) + (1.0 - lambda) * x;
    *M2 = lambda * (*M2) + (1.0 - lambda) * delta * (x - *mean);
}

/* ---- Recursive PLS Lifecycle ---- */

RecursivePLS* recursive_pls_alloc(size_t p_vars, size_t q_vars,
                                   size_t a_lvs,
                                   const RecursivePLSConfig *config) {
    RecursivePLS *r = (RecursivePLS*)calloc(1, sizeof(RecursivePLS));
    if (!r) return NULL;

    r->config = config ? *config : recursive_pls_config_default();
    r->n_total = 0; r->n_effective = 0;
    r->samples_since_update = 0;

    r->XtX = matrix_alloc(p_vars, p_vars);
    r->XtY = matrix_alloc(p_vars, q_vars);
    r->sum_x = vector_alloc(p_vars);
    r->sum_y = vector_alloc(q_vars);
    r->sum_sq_y = 0.0;

    if (r->config.window_size > 0) {
        r->window_X = matrix_alloc(r->config.window_size, p_vars);
        r->window_Y = matrix_alloc(r->config.window_size, q_vars);
        r->window_pos = 0; r->window_count = 0;
    } else {
        r->window_X = NULL; r->window_Y = NULL;
    }

    r->model = pls_model_alloc(1, p_vars, q_vars, a_lvs);
    /* We'll resize model later when we have data */

    if (!r->XtX || !r->XtY || !r->sum_x || !r->sum_y || !r->model) {
        recursive_pls_free(r); return NULL;
    }
    return r;
}

void recursive_pls_free(RecursivePLS *rpls) {
    if (!rpls) return;
    matrix_free(rpls->XtX); matrix_free(rpls->XtY);
    vector_free(rpls->sum_x); vector_free(rpls->sum_y);
    matrix_free(rpls->window_X); matrix_free(rpls->window_Y);
    pls_model_free(rpls->model);
    free(rpls);
}

int recursive_pls_initialize(RecursivePLS *rpls,
                             const Matrix *X, const Matrix *Y) {
    if (!rpls || !X || !Y || X->rows != Y->rows) return -1;
    size_t n = X->rows, p = X->cols, q = Y->cols, a = rpls->model->a_lvs;

    /* Accumulate statistics */
    for (size_t i = 0; i < n; i++) {
        for (size_t j1 = 0; j1 < p; j1++) {
            rpls->sum_x->data[j1] += MATRIX_AT(X, i, j1);
            for (size_t j2 = 0; j2 < p; j2++)
                MATRIX_AT(rpls->XtX, j1, j2) += MATRIX_AT(X, i, j1) *
                                                 MATRIX_AT(X, i, j2);
            for (size_t k = 0; k < q; k++)
                MATRIX_AT(rpls->XtY, j1, k) += MATRIX_AT(X, i, j1) *
                                                MATRIX_AT(Y, i, k);
        }
        for (size_t k = 0; k < q; k++) {
            rpls->sum_y->data[k] += MATRIX_AT(Y, i, k);
            rpls->sum_sq_y += SQ(MATRIX_AT(Y, i, k));
        }
    }
    rpls->n_total = n;
    rpls->n_effective = n;

    /* Fit initial PLS model */
    pls_model_free(rpls->model);
    rpls->model = pls_model_alloc(n, p, q, a);
    if (!rpls->model) return -1;

    Matrix *Xc = matrix_copy(X);
    Matrix *Yc = matrix_copy(Y);
    if (!Xc || !Yc) { matrix_free(Xc); matrix_free(Yc); return -1; }

    Vector *xm = matrix_column_means(X);
    Vector *ym = matrix_column_means(Y);
    if (xm) matrix_center_columns(Xc, xm);
    if (ym) matrix_center_columns(Yc, ym);

    NIPALSConfig cfg = nipals_config_default();
    if (nipals_fit(Xc, Yc, a, &cfg, rpls->model) != 0) {
        matrix_free(Xc); matrix_free(Yc);
        vector_free(xm); vector_free(ym);
        return -1;
    }

    pls_model_set_preprocessing(rpls->model, xm, NULL, 1, 0, ym, NULL, 1, 0);
    pls_model_compute_beta(rpls->model);

    matrix_free(Xc); matrix_free(Yc);
    vector_free(xm); vector_free(ym);
    return 0;
}

int recursive_pls_update(RecursivePLS *rpls,
                         const Vector *x_new, const Vector *y_new) {
    if (!rpls || !x_new) return -1;
    size_t p = rpls->model->p_vars, q = rpls->model->q_vars;

    if (x_new->len != p) return -1;
    if (y_new && y_new->len != q) return -1;

    double lambda = rpls->config.forgetting_factor;

    if (rpls->config.use_exponential) {
        /* Exponential forgetting update */
        rpls->n_effective = (size_t)(lambda * (double)rpls->n_effective + 1.0);
        rpls->n_total++;

        for (size_t j1 = 0; j1 < p; j1++) {
            rpls->sum_x->data[j1] = lambda * rpls->sum_x->data[j1] + x_new->data[j1];
            for (size_t j2 = 0; j2 < p; j2++)
                MATRIX_AT(rpls->XtX, j1, j2) = lambda * MATRIX_AT(rpls->XtX, j1, j2)
                    + x_new->data[j1] * x_new->data[j2];
            if (y_new) {
                for (size_t k = 0; k < q; k++)
                    MATRIX_AT(rpls->XtY, j1, k) = lambda * MATRIX_AT(rpls->XtY, j1, k)
                        + x_new->data[j1] * y_new->data[k];
            }
        }
        if (y_new) {
            for (size_t k = 0; k < q; k++) {
                rpls->sum_y->data[k] = lambda * rpls->sum_y->data[k] + y_new->data[k];
                rpls->sum_sq_y = lambda * rpls->sum_sq_y + SQ(y_new->data[k]);
            }
        }
    } else if (rpls->window_X && rpls->window_Y) {
        /* Moving window: add to ring buffer */
        for (size_t j = 0; j < p; j++)
            MATRIX_AT(rpls->window_X, rpls->window_pos, j) = x_new->data[j];
        if (y_new)
            for (size_t k = 0; k < q; k++)
                MATRIX_AT(rpls->window_Y, rpls->window_pos, k) = y_new->data[k];

        rpls->window_pos = (rpls->window_pos + 1) % rpls->config.window_size;
        if (rpls->window_count < rpls->config.window_size)
            rpls->window_count++;
        rpls->n_total++;
    }

    rpls->samples_since_update++;

    /* Trigger model update if interval reached */
    if (rpls->samples_since_update >= rpls->config.update_interval) {
        recursive_pls_refit(rpls);
        rpls->samples_since_update = 0;
        return 1;  /* Model was updated */
    }

    return 0;
}

int recursive_pls_predict(const RecursivePLS *rpls,
                          const Vector *x_new, Vector *y_pred) {
    if (!rpls || !rpls->model || !x_new || !y_pred) return -1;
    return pls_model_predict_single(rpls->model, x_new, y_pred);
}

int recursive_pls_refit(RecursivePLS *rpls) {
    if (!rpls) return -1;
    size_t p = rpls->model->p_vars, q = rpls->model->q_vars, a = rpls->model->a_lvs;

    /* Use current XtX and XtY to build pseudo data and re-fit */
    /* This is an approximate re-fit using the moment representation */
    size_t n = rpls->n_effective > 0 ? rpls->n_effective : rpls->n_total;
    if (n < 3) return -1;
    if (n > 10000) n = 10000;  /* Cap for memory */

    /* Build centered XtX and XtY */
    Matrix *XtXc = matrix_copy(rpls->XtX);
    Matrix *XtYc = matrix_copy(rpls->XtY);
    if (!XtXc || !XtYc) {
        matrix_free(XtXc); matrix_free(XtYc); return -1;
    }

    /* Center: remove mean effects */
    for (size_t i = 0; i < p; i++) {
        double xi_mean = rpls->sum_x->data[i] / (double)rpls->n_total;
        for (size_t j = 0; j < p; j++) {
            double xj_mean = rpls->sum_x->data[j] / (double)rpls->n_total;
            MATRIX_AT(XtXc, i, j) -= xi_mean * rpls->sum_x->data[j];
            MATRIX_AT(XtXc, i, j) -= xj_mean * rpls->sum_x->data[i];
            MATRIX_AT(XtXc, i, j) += (double)rpls->n_total * xi_mean * xj_mean;
        }
        xi_mean = rpls->sum_x->data[i] / (double)rpls->n_total;
        for (size_t k = 0; k < q; k++) {
            double yk_mean = rpls->sum_y->data[k] / (double)rpls->n_total;
            MATRIX_AT(XtYc, i, k) -= xi_mean * rpls->sum_y->data[k];
            MATRIX_AT(XtYc, i, k) -= yk_mean * rpls->sum_x->data[i];
            MATRIX_AT(XtYc, i, k) += (double)rpls->n_total * xi_mean * yk_mean;
        }
    }

    /* Use NIPALS on the moment representation
     * This is an approximation: we do power iteration directly on XtXc
     * to extract PLS weights (simplified recursive PLS approach) */
    Matrix *W = matrix_alloc(p, a);
    if (!W) { matrix_free(XtXc); matrix_free(XtYc); return -1; }

    /* Approximate PLS via eigen-decomposition of XtYc * XtYc^T */
    Matrix *S = matrix_multiply(XtYc, matrix_transpose(XtYc));
    Matrix *YtX = matrix_transpose(XtYc);  /* q x p */
    if (!S || !YtX) {
        matrix_free(W); matrix_free(XtXc); matrix_free(XtYc);
        matrix_free(S); matrix_free(YtX); return -1;
    }

    /* Simple: use power iteration on XtXc * S for each component */
    Matrix *M = matrix_multiply(XtXc, S);
    if (M) {
        for (size_t a_lv = 0; a_lv < a; a_lv++) {
            Vector *w = vector_alloc(p);
            if (!w) break;
            for (size_t i = 0; i < p; i++) w->data[i] = 1.0;
            double lo = 0.0, lam = 0.0;
            for (int it = 0; it < 100; it++) {
                Vector *mw = matrix_vector_multiply(M, w);
                if (!mw) break;
                double nr = vector_norm_l2(mw);
                if (nr < DBL_EPSILON) { vector_free(mw); break; }
                for (size_t i = 0; i < p; i++) w->data[i] = mw->data[i] / nr;
                lam = vector_dot(w, mw) / vector_dot(w, w);
                vector_free(mw);
                if (fabs(lam - lo) < 1e-10) break;
                lo = lam;
            }
            for (size_t i = 0; i < p; i++) MATRIX_AT(W, i, a_lv) = w->data[i];
            /* Deflate M */
            for (size_t i = 0; i < p; i++)
                for (size_t j = 0; j < p; j++)
                    MATRIX_AT(M, i, j) -= lam * w->data[i] * w->data[j];
            vector_free(w);
        }
        matrix_free(M);
    }

    /* Compute Beta = W * inv(P^T * W) * Q^T (simplified) */
    memcpy(rpls->model->W->data, W->data, p * a * sizeof(double));

    /* P = XtXc * W * inv(W^T * XtXc * W) */
    Matrix *XtXcW = matrix_multiply(XtXc, W);
    Matrix *Wt = matrix_transpose(W);
    if (XtXcW && Wt) {
        Matrix *WtXtXcW = matrix_multiply(Wt, XtXcW);
        if (WtXtXcW) {
            memcpy(rpls->model->P->data, XtXcW->data, p * a * sizeof(double));
            /* Q = XtYc^T * W * inv(W^T * XtXc * W) */
            Matrix *YtX_W = matrix_multiply(YtX, W);
            if (YtX_W) {
                memcpy(rpls->model->Q->data, YtX_W->data, q * a * sizeof(double));
                matrix_free(YtX_W);
            }
            matrix_free(WtXtXcW);
        }
        matrix_free(XtXcW);
    }
    matrix_free(Wt);

    pls_model_compute_beta(rpls->model);

    matrix_free(W); matrix_free(XtXc); matrix_free(XtYc);
    matrix_free(S); matrix_free(YtX);
    return 0;
}
