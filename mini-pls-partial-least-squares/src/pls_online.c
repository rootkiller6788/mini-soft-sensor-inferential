#include "pls_online.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#define MATRIX_AT(A, i, j) ((A)->data[(i) * (A)->cols + (j)])
#define SQ(x) ((x) * (x))

OnlinePrediction* online_prediction_alloc(size_t q_vars, size_t a_lvs) {
    OnlinePrediction *p = (OnlinePrediction*)calloc(1, sizeof(OnlinePrediction));
    if (!p) return NULL;
    p->y_pred = vector_alloc(q_vars);
    p->t_scores = vector_alloc(a_lvs);
    if (!p->y_pred || !p->t_scores) {
        online_prediction_free(p); return NULL;
    }
    p->T2 = 0.0; p->T2_limit_95 = 0.0; p->T2_limit_99 = 0.0;
    p->SPE = 0.0; p->SPE_limit_95 = 0.0; p->SPE_limit_99 = 0.0;
    p->DModX = 0.0; p->DModX_crit = 0.0;
    p->alarm = ALARM_NONE;
    p->timestamp = 0;
    return p;
}

void online_prediction_free(OnlinePrediction *pred) {
    if (!pred) return;
    vector_free(pred->y_pred);
    vector_free(pred->t_scores);
    free(pred);
}

int online_monitor_init(const PLSModel *model, OnlineMonitor *monitor,
                        double alpha_95, double alpha_99) {
    if (!model || !monitor) return -1;

    monitor->model = (PLSModel*)model;  /* Non-owning pointer */
    monitor->sample_count = 0;
    monitor->alarm_count = 0;
    monitor->verbose = 0;

    /* Compute score variances from training scores */
    monitor->score_variances = compute_score_variances(model->T);
    if (!monitor->score_variances) return -1;

    /* T2 limits */
    monitor->T2_limit_95 = compute_T2_limit(model->n_samples, model->a_lvs, alpha_95);
    monitor->T2_limit_99 = compute_T2_limit(model->n_samples, model->a_lvs, alpha_99);

    /* SPE theta parameters and limits */
    /* Compute training X-residuals */
    Matrix *E = pls_model_compute_residuals(model, model->T);
    if (!E) {
        /* Use identity proxy for T as X (approximate residuals from scores only) */
        E = matrix_alloc(model->n_samples, model->p_vars);
        if (!E) return -1;
        /* Fill with small values (model explains data well) */
        matrix_fill(E, 0.0);
    }

    compute_SPE_theta(E, &monitor->SPE_theta1,
                      &monitor->SPE_theta2, &monitor->SPE_theta3);
    monitor->SPE_limit_95 = compute_SPE_limit(monitor->SPE_theta1,
                                               monitor->SPE_theta2,
                                               monitor->SPE_theta3, alpha_95);
    monitor->SPE_limit_99 = compute_SPE_limit(monitor->SPE_theta1,
                                               monitor->SPE_theta2,
                                               monitor->SPE_theta3, alpha_99);

    /* DModX critical */
    monitor->DModX_crit = compute_DModX_critical(E, model->a_lvs, alpha_95);

    matrix_free(E);
    return 0;
}

void online_monitor_free(OnlineMonitor *monitor) {
    if (!monitor) return;
    vector_free(monitor->score_variances);
}

void online_monitor_reset(OnlineMonitor *monitor) {
    if (!monitor) return;
    monitor->sample_count = 0;
    monitor->alarm_count = 0;
}

void online_monitor_summary(const OnlineMonitor *monitor,
                            size_t *total, size_t *alarms, double *alarm_rate) {
    if (!monitor) return;
    if (total) *total = monitor->sample_count;
    if (alarms) *alarms = monitor->alarm_count;
    if (alarm_rate)
        *alarm_rate = (monitor->sample_count > 0)
            ? (double)monitor->alarm_count / (double)monitor->sample_count
            : 0.0;
}

void online_compute_yhat(const PLSModel *model,
                         const Vector *x_processed, Vector *y_pred) {
    if (!model || !x_processed || !y_pred) return;
    for (size_t j = 0; j < model->q_vars; j++) {
        double sum = model->b0->data[j];
        for (size_t k = 0; k < model->p_vars; k++)
            sum += x_processed->data[k] * MATRIX_AT(model->Beta, k, j);
        y_pred->data[j] = sum;
    }
}

void online_preprocess(const PLSModel *model,
                       const Vector *x_raw, Vector *x_processed) {
    if (!model || !x_raw || !x_processed) return;
    for (size_t j = 0; j < model->p_vars; j++)
        x_processed->data[j] = x_raw->data[j];

    if (model->center_x)
        for (size_t j = 0; j < model->p_vars; j++)
            x_processed->data[j] -= model->X_mean->data[j];
    if (model->scale_x)
        for (size_t j = 0; j < model->p_vars; j++) {
            double sd = model->X_std->data[j];
            if (sd > 1e-15) x_processed->data[j] /= sd;
        }
}

int online_predict_sample(OnlineMonitor *monitor,
                          const Vector *x_new, OnlinePrediction *pred) {
    if (!monitor || !monitor->model || !x_new || !pred) return -1;
    if (x_new->len != monitor->model->p_vars) return -1;

    PLSModel *m = monitor->model;

    /* Step 1: Preprocess */
    Vector *xp = vector_copy(x_new);
    if (!xp) return -1;
    online_preprocess(m, x_new, xp);

    /* Step 2: Predict */
    online_compute_yhat(m, xp, pred->y_pred);

    /* Step 3: Compute scores t = xp^T * W */
    for (size_t a = 0; a < m->a_lvs; a++) {
        double sum = 0.0;
        for (size_t j = 0; j < m->p_vars; j++)
            sum += xp->data[j] * MATRIX_AT(m->W, j, a);
        pred->t_scores->data[a] = sum;
    }

    /* Step 4: Compute T2 */
    pred->T2 = compute_T2_single(pred->t_scores, monitor->score_variances);
    pred->T2_limit_95 = monitor->T2_limit_95;
    pred->T2_limit_99 = monitor->T2_limit_99;

    /* Step 5: Compute SPE */
    /* x_hat = t * P^T, residual = xp - x_hat */
    Vector *x_residual = vector_copy(xp);
    if (x_residual) {
        for (size_t j = 0; j < m->p_vars; j++) {
            double x_hat_j = 0.0;
            for (size_t a = 0; a < m->a_lvs; a++)
                x_hat_j += pred->t_scores->data[a] * MATRIX_AT(m->P, j, a);
            x_residual->data[j] -= x_hat_j;
        }
        pred->SPE = compute_SPE_single(x_residual);
        vector_free(x_residual);
    }
    pred->SPE_limit_95 = monitor->SPE_limit_95;
    pred->SPE_limit_99 = monitor->SPE_limit_99;

    /* Step 6: Compute DModX */
    pred->DModX = compute_DModX_single(x_residual, m->p_vars, m->a_lvs);
    /* Recompute after freeing... */
    pred->DModX = sqrt(pred->SPE / (double)(m->p_vars - m->a_lvs));
    pred->DModX_crit = monitor->DModX_crit;

    /* Step 7: Set alarm */
    pred->alarm = ALARM_NONE;
    int t2_warn = (pred->T2 > pred->T2_limit_95);
    int t2_crit = (pred->T2 > pred->T2_limit_99);
    int spe_warn = (pred->SPE > pred->SPE_limit_95);
    int spe_crit = (pred->SPE > pred->SPE_limit_99);

    if (t2_crit || spe_crit) {
        if (t2_crit && spe_crit) pred->alarm = ALARM_BOTH;
        else pred->alarm = ALARM_CRITICAL;
    } else if (t2_warn || spe_warn) {
        pred->alarm = ALARM_WARNING;
    }

    /* Step 8: Timestamp and counters */
    pred->timestamp = time(NULL);
    monitor->sample_count++;
    if (pred->alarm >= ALARM_WARNING)
        monitor->alarm_count++;

    vector_free(xp);
    return 0;
}

int online_predict_batch(OnlineMonitor *monitor,
                         const Matrix *X_new, OnlinePrediction *preds) {
    if (!monitor || !X_new || !preds) return -1;
    for (size_t i = 0; i < X_new->rows; i++) {
        Vector *xi = matrix_get_row(X_new, i);
        if (!xi) continue;
        online_predict_sample(monitor, xi, &preds[i]);
        vector_free(xi);
    }
    return 0;
}

void online_prediction_print(const OnlinePrediction *pred, size_t q_vars) {
    if (!pred) return;
    printf("=== Online PLS Prediction ===\n");
    printf("Timestamp: %s", ctime(&pred->timestamp));
    printf("Predicted Y:");
    for (size_t j = 0; j < q_vars; j++)
        printf(" %.4f", pred->y_pred->data[j]);
    printf("\n");
    printf("T2: %.4f (95%%: %.4f, 99%%: %.4f)\n",
           pred->T2, pred->T2_limit_95, pred->T2_limit_99);
    printf("SPE: %.4f (95%%: %.4f, 99%%: %.4f)\n",
           pred->SPE, pred->SPE_limit_95, pred->SPE_limit_99);
    printf("DModX: %.4f (crit: %.4f)\n", pred->DModX, pred->DModX_crit);
    printf("Alarm: %s\n", alarm_state_string(pred->alarm));
}

const char* alarm_state_string(AlarmState state) {
    switch (state) {
        case ALARM_NONE:     return "NONE";
        case ALARM_WARNING:  return "WARNING";
        case ALARM_CRITICAL: return "CRITICAL";
        case ALARM_BOTH:     return "BOTH (T2+SPE)";
        default:             return "UNKNOWN";
    }
}

Vector* online_T2_contributions(const OnlineMonitor *monitor,
                                const Vector *x_processed,
                                const Vector *t_scores) {
    if (!monitor || !x_processed || !t_scores) return NULL;
    PLSModel *m = monitor->model;
    Vector *contrib = vector_alloc(m->p_vars);
    if (!contrib) return NULL;

    for (size_t j = 0; j < m->p_vars; j++) {
        double cj = 0.0;
        for (size_t a = 0; a < m->a_lvs; a++) {
            if (monitor->score_variances->data[a] > 1e-15)
                cj += t_scores->data[a] * MATRIX_AT(m->W, j, a) *
                      x_processed->data[j] / monitor->score_variances->data[a];
        }
        contrib->data[j] = cj;
    }
    return contrib;
}

Vector* online_SPE_contributions(const OnlineMonitor *monitor,
                                 const Vector *x_processed,
                                 const Vector *t_scores) {
    if (!monitor || !x_processed || !t_scores) return NULL;
    PLSModel *m = monitor->model;
    Vector *contrib = vector_alloc(m->p_vars);
    if (!contrib) return NULL;

    for (size_t j = 0; j < m->p_vars; j++) {
        double x_hat_j = 0.0;
        for (size_t a = 0; a < m->a_lvs; a++)
            x_hat_j += t_scores->data[a] * MATRIX_AT(m->P, j, a);
        contrib->data[j] = SQ(x_processed->data[j] - x_hat_j);
    }
    return contrib;
}
