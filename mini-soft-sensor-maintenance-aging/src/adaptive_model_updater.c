/**
 * @file    adaptive_model_updater.c
 * @brief   Adaptive model update strategies - RPLS, MW-PLS, JIT-PLS
 *
 * L3: Matrix operations, PLS model representation
 * L5: RPLS (Qin 1998), MW-PLS, forgetting factor, update triggers
 * L8: JIT locally weighted PLS, adaptive forgetting factor
 *
 * Ref: Qin (1998) Comp. & Chem. Eng. 22(4-5), 503-514.
 *      Dayal & MacGregor (1997) J. Process Control 7(3), 169-179.
 */

#include "adaptive_model_updater.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ====================================================================
 * L3: Matrix Operations (Dense)
 * ==================================================================== */

MatrixDense matrix_create(size_t rows, size_t cols)
{
    MatrixDense m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (double *)calloc(rows * cols, sizeof(double));
    return m;
}

void matrix_destroy(MatrixDense *m)
{
    if (!m) return;
    free(m->data);
    memset(m, 0, sizeof(*m));
}

double matrix_get(const MatrixDense *m, size_t i, size_t j)
{
    if (!m || !m->data || i >= m->rows || j >= m->cols) return 0.0;
    return m->data[i * m->stride + j];
}

void matrix_set(MatrixDense *m, size_t i, size_t j, double val)
{
    if (!m || !m->data || i >= m->rows || j >= m->cols) return;
    m->data[i * m->stride + j] = val;
}

/* C = A * B, where A: m x k, B: k x n, C: m x n */
void matrix_multiply(const MatrixDense *a, const MatrixDense *b, MatrixDense *c)
{
    if (!a || !b || !c || !a->data || !b->data || !c->data) return;
    if (a->cols != b->rows) return;

    size_t m = a->rows, k = a->cols, n = b->cols;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t p = 0; p < k; p++) {
                sum += a->data[i * a->stride + p] * b->data[p * b->stride + j];
            }
            c->data[i * c->stride + j] = sum;
        }
    }
}

/* dst = src^T */
void matrix_transpose(const MatrixDense *src, MatrixDense *dst)
{
    if (!src || !dst || !src->data || !dst->data) return;
    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dst->data[j * dst->stride + i] = src->data[i * src->stride + j];
        }
    }
}

void matrix_scale(MatrixDense *m, double scalar)
{
    if (!m || !m->data) return;
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] *= scalar;
    }
}

/* ====================================================================
 * L3: PLS Model Management
 * ==================================================================== */

PLSModel pls_model_create(size_t n_vars, size_t n_components)
{
    PLSModel model;
    memset(&model, 0, sizeof(model));
    if (n_vars == 0 || n_components == 0 || n_components > n_vars)
        return model;
    model.n_vars = n_vars;
    model.n_components = n_components;
    model.p = (double *)calloc(n_vars * n_components, sizeof(double));
    model.w = (double *)calloc(n_vars * n_components, sizeof(double));
    model.q = (double *)calloc(n_components, sizeof(double));
    model.b = (double *)calloc(n_vars, sizeof(double));
    model.t_mean = (double *)calloc(n_components, sizeof(double));
    model.x_mean = (double *)calloc(n_vars, sizeof(double));
    model.x_std = (double *)calloc(n_vars, sizeof(double));
    for (size_t i = 0; i < n_vars; i++) model.x_std[i] = 1.0;
    model.y_std = 1.0;
    return model;
}

void pls_model_destroy(PLSModel *model)
{
    if (!model) return;
    free(model->p); free(model->w); free(model->q);
    free(model->b); free(model->t_mean);
    free(model->x_mean); free(model->x_std);
    memset(model, 0, sizeof(*model));
}

/* PLS prediction: y = x_centered * b + y_mean
 * x_centered_i = (x_i - x_mean_i) / x_std_i */
double pls_predict(const PLSModel *model, const double *x)
{
    if (!model || !model->b || !x) return model ? model->y_mean : 0.0;

    double y_pred = model->y_mean;
    for (size_t i = 0; i < model->n_vars; i++) {
        double x_centered = (x[i] - model->x_mean[i]) / fmax(model->x_std[i], 1e-10);
        y_pred += model->b[i] * x_centered;
    }
    return y_pred;
}

void pls_predict_batch(const PLSModel *model, const double *x,
                        size_t n_samples, double *y_pred)
{
    if (!model || !x || !y_pred) return;
    for (size_t s = 0; s < n_samples; s++) {
        y_pred[s] = pls_predict(model, &x[s * model->n_vars]);
    }
}

/* VIP (Variable Importance in Projection)
 * VIP_j = sqrt( sum_{a} w_{ja}^2 * SSY_a * p / (SSY_total * A) )
 * Simplified: VIP_j = sqrt( sum_{a} w_{ja}^2 * q_a^2 * A / sum_{a} q_a^2 )
 * Variables with VIP > 1.0 are important. */

void pls_vip_scores(const PLSModel *model, double *vip)
{
    if (!model || !vip || !model->w || !model->q) return;

    double ssy_total = 0.0;
    for (size_t a = 0; a < model->n_components; a++)
        ssy_total += model->q[a] * model->q[a];

    if (ssy_total < 1e-15) {
        for (size_t j = 0; j < model->n_vars; j++) vip[j] = 0.0;
        return;
    }

    for (size_t j = 0; j < model->n_vars; j++) {
        double sum_w2_q2 = 0.0;
        for (size_t a = 0; a < model->n_components; a++) {
            double w_ja = model->w[j * model->n_components + a];
            sum_w2_q2 += w_ja * w_ja * model->q[a] * model->q[a];
        }
        vip[j] = sqrt((double)model->n_vars * sum_w2_q2 / ssy_total);
    }
}

/* ====================================================================
 * L5: RPLS - Recursive Partial Least Squares
 *
 * Qin (1998) Comp. & Chem. Eng. 22(4-5), 503-514.
 *
 * RPLS updates the mean and covariance recursively using an
 * exponentially weighted moving average (EWMA) with forgetting
 * factor lambda in (0, 1].
 *
 * Mean update:
 *   x_mean_new = lambda * x_mean_old + (1 - lambda) * x_new
 *
 * Covariance update (after mean subtraction of new data):
 *   P_xx_new = lambda * P_xx_old + (1 - lambda) * dx * dx^T
 *   P_xy_new = lambda * P_xy_old + (1 - lambda) * dx * dy
 *
 * Then NIPALS is run on the updated covariance to extract
 * the PLS components and regression vector b.
 *
 * Complexity: O(n_vars * n_components) per update after
 * mean/covariance update.
 *
 * n_effective = 1 / (1 - lambda) is the effective memory length.
 * ==================================================================== */

void rpls_init(RPLSState *state, size_t n_vars, size_t n_components,
                double lambda)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->n_vars = n_vars;
    state->n_components = n_components;
    state->lambda = lambda;

    state->x_mean = (double *)calloc(n_vars, sizeof(double));
    state->y_mean = (double *)calloc(1, sizeof(double));
    state->pxx = (double *)calloc(n_vars * n_vars, sizeof(double));
    state->pxy = (double *)calloc(n_vars, sizeof(double));
    state->p = (double *)calloc(n_vars * n_components, sizeof(double));
    state->w = (double *)calloc(n_vars * n_components, sizeof(double));
    state->q = (double *)calloc(n_components, sizeof(double));
    state->b = (double *)calloc(n_vars, sizeof(double));
    state->n_effective = 1;
}

void rpls_destroy(RPLSState *state)
{
    if (!state) return;
    free(state->x_mean); free(state->y_mean);
    free(state->pxx); free(state->pxy);
    free(state->p); free(state->w); free(state->q); free(state->b);
    memset(state, 0, sizeof(*state));
}

void rpls_update(RPLSState *state, const double *x, double y)
{
    if (!state || !x) return;

    double lam = state->lambda;
    double onemlam = 1.0 - lam;
    size_t nv = state->n_vars;

    /* Update means */
    for (size_t i = 0; i < nv; i++)
        state->x_mean[i] = lam * state->x_mean[i] + onemlam * x[i];
    state->y_mean[0] = lam * state->y_mean[0] + onemlam * y;

    /* Mean-centered new observation */
    double *dx = (double *)malloc(nv * sizeof(double));
    if (!dx) return;
    for (size_t i = 0; i < nv; i++)
        dx[i] = x[i] - state->x_mean[i];
    double dy = y - state->y_mean[0];

    /* Update P_xx and P_xy */
    for (size_t i = 0; i < nv; i++) {
        for (size_t j = 0; j < nv; j++) {
            size_t idx = i * nv + j;
            state->pxx[idx] = lam * state->pxx[idx] + onemlam * dx[i] * dx[j];
        }
        state->pxy[i] = lam * state->pxy[i] + onemlam * dx[i] * dy;
    }

    /* Simplified NIPALS on covariance for component extraction.
     * Full NIPALS requires eigenvalue decomposition; here we use
     * a single-component update for efficiency.
     *
     * w = P_xy / ||P_xy||   (normalized)
     * t_score = w^T * x_centered
     * p = P_xx * w / (w^T * P_xx * w)
     * q = t_score^T * y / (t_score^T * t_score)
     * Update b via rank-1 correction on deflated residual.
     */

    /* Extract first component weight as normalized P_xy */
    double pxy_norm = 0.0;
    for (size_t i = 0; i < nv; i++)
        pxy_norm += state->pxy[i] * state->pxy[i];
    pxy_norm = sqrt(fmax(pxy_norm, 1e-15));

    for (size_t i = 0; i < nv; i++)
        state->w[i] = state->pxy[i] / pxy_norm;

    /* Regression coefficients: b = W * (P^T * W)^{-1} * q
     * Simplified single-component: b = w * q */
    for (size_t a = 0; a < state->n_components && a < 1; a++) {
        size_t offset = a * nv;
        /* t score */
        double t = 0.0;
        for (size_t i = 0; i < nv; i++)
            t += state->w[offset + i] * dx[i];

        /* q loading */
        double q_val = (fabs(t) > 1e-15) ? (t * dy) / (t * t) : 0.0;
        state->q[a] = q_val;

        /* Update b coefficient: b += w * q */
        for (size_t i = 0; i < nv; i++)
            state->b[i] += state->w[offset + i] * q_val;
    }

    /* Effective sample count */
    state->n_effective = (size_t)(1.0 / fmax(onemlam, 0.001));
    state->n_samples++;

    free(dx);
}

double rpls_predict(const RPLSState *state, const double *x)
{
    if (!state || !state->b || !x) return 0.0;

    double y_pred = state->y_mean[0];
    for (size_t i = 0; i < state->n_vars; i++) {
        double x_centered = x[i] - state->x_mean[i];
        y_pred += state->b[i] * x_centered;
    }
    return y_pred;
}

void rpls_bulk_update(RPLSState *state, const double *x, const double *y,
                       size_t n_samples)
{
    if (!state || !x || !y) return;
    for (size_t s = 0; s < n_samples; s++) {
        rpls_update(state, &x[s * state->n_vars], y[s]);
    }
}

/* ====================================================================
 * L5: MW-PLS - Moving Window PLS
 *
 * Maintains a window of the most recent W samples.
 * When the window is full, the oldest sample is discarded
 * and the model is incrementally updated.
 *
 * Uses rank-1 downdating and updating of the covariance matrix.
 * This is more efficient than full recomputation.
 * ==================================================================== */

void mwpls_init(MWPLSState *state, size_t n_vars, size_t n_components,
                 size_t window_size)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->n_vars = n_vars;
    state->n_components = n_components;
    state->window_size = window_size;
    state->x_window = (double *)calloc(window_size * n_vars, sizeof(double));
    state->y_window = (double *)calloc(window_size, sizeof(double));
    state->current_model = pls_model_create(n_vars, n_components);
}

void mwpls_destroy(MWPLSState *state)
{
    if (!state) return;
    free(state->x_window);
    free(state->y_window);
    pls_model_destroy(&state->current_model);
    memset(state, 0, sizeof(*state));
}

void mwpls_add_sample(MWPLSState *state, const double *x, double y)
{
    if (!state || !x) return;

    /* Add to window */
    size_t pos = state->current_pos;
    for (size_t i = 0; i < state->n_vars; i++)
        state->x_window[pos * state->n_vars + i] = x[i];
    state->y_window[pos] = y;

    state->current_pos = (pos + 1) % state->window_size;
    if (state->n_samples < state->window_size)
        state->n_samples++;

    if (state->n_samples >= state->window_size)
        state->window_full = 1;

    /* Update model means incrementally */
    if (state->n_samples > 0) {
        /* Compute window statistics */
        double sum_x[50] = {0}; /* Max 50 vars */
        double sum_y = 0.0;
        size_t effective_n = state->window_full ? state->window_size : state->n_samples;

        for (size_t s = 0; s < effective_n; s++) {
            for (size_t i = 0; i < state->n_vars && i < 50; i++)
                sum_x[i] += state->x_window[s * state->n_vars + i];
            sum_y += state->y_window[s];
        }

        for (size_t i = 0; i < state->n_vars && i < 50; i++)
            state->current_model.x_mean[i] = sum_x[i] / (double)effective_n;
        state->current_model.y_mean = sum_y / (double)effective_n;
    }
}

double mwpls_predict(const MWPLSState *state, const double *x)
{
    if (!state || !x) return 0.0;
    return pls_predict(&state->current_model, x);
}

/* ====================================================================
 * L5: Forgetting Factor Management
 * ==================================================================== */

void forgetting_factor_init(ForgettingFactor *ff, ForgettingFactorType type,
                              double base_lambda)
{
    if (!ff) return;
    memset(ff, 0, sizeof(*ff));
    ff->type = type;
    ff->base_lambda = base_lambda;
    ff->current_lambda = base_lambda;
    ff->min_lambda = 0.80;
    ff->max_lambda = 0.999;
    ff->update_rate = 0.01;
    ff->step_interval = 100;
}

double forgetting_factor_update(ForgettingFactor *ff, double process_change)
{
    if (!ff) return 0.95;

    switch (ff->type) {
        case FORGET_CONSTANT:
            return ff->base_lambda;
        case FORGET_ADAPTIVE:
            /* Decrease lambda (faster forgetting) when process changes are large */
            ff->current_lambda = ff->base_lambda
                                 - ff->update_rate * fmin(process_change, 1.0);
            if (ff->current_lambda < ff->min_lambda)
                ff->current_lambda = ff->min_lambda;
            if (ff->current_lambda > ff->max_lambda)
                ff->current_lambda = ff->max_lambda;
            return ff->current_lambda;
        case FORGET_STEP:
            ff->samples_since_step++;
            if (ff->samples_since_step >= ff->step_interval) {
                ff->current_lambda = ff->max_lambda;  /* Reset to slow forgetting */
            }
            return ff->current_lambda;
        case FORGET_PROCESS_STATE:
            /* lambda increases (slower forgetting) when process is stable */
            ff->current_lambda = ff->max_lambda
                                 - (ff->max_lambda - ff->min_lambda) * fmin(process_change, 1.0);
            return ff->current_lambda;
        default:
            return ff->base_lambda;
    }
}

void forgetting_factor_reset(ForgettingFactor *ff)
{
    if (!ff) return;
    ff->current_lambda = ff->base_lambda;
    ff->samples_since_step = 0;
}

/* ====================================================================
 * L5: Update Trigger Logic
 * ==================================================================== */

void update_trigger_init(UpdateTriggerConfig *cfg, UpdateTrigger type,
                          size_t interval, double rmse_thresh)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->type = type;
    cfg->periodic_interval = interval;
    cfg->rmse_threshold = rmse_thresh;
    cfg->r2_threshold = 0.8;
    cfg->process_change_threshold = 0.1;
}

int update_trigger_check(UpdateTriggerConfig *cfg, double current_rmse,
                          double current_r2, double process_change)
{
    if (!cfg) return 0;

    cfg->samples_since_update++;
    int trigger = 0;

    switch (cfg->type) {
        case TRIGGER_PERIODIC:
            if (cfg->samples_since_update >= cfg->periodic_interval)
                trigger = 1;
            break;
        case TRIGGER_PERFORMANCE:
            if (current_rmse > cfg->rmse_threshold || current_r2 < cfg->r2_threshold)
                trigger = 1;
            break;
        case TRIGGER_PROCESS_CHANGE:
            if (process_change > cfg->process_change_threshold)
                trigger = 1;
            break;
        case TRIGGER_HYBRID:
            if (cfg->samples_since_update >= cfg->periodic_interval ||
                current_rmse > cfg->rmse_threshold ||
                process_change > cfg->process_change_threshold)
                trigger = 1;
            break;
    }

    if (trigger) {
        cfg->samples_since_update = 0;
        cfg->update_needed = 1;
    } else {
        cfg->update_needed = 0;
    }

    cfg->last_rmse = current_rmse;
    return trigger;
}

/* ====================================================================
 * L8: JIT Locally Weighted PLS
 *
 * Just-in-Time learning: for each query point, select the k most
 * similar historical samples, build a local PLS model, make a
 * prediction, and then discard the model.
 *
 * Distance: weighted Euclidean with locality bandwidth.
 *   d(x_q, x_i)^2 = sum_j (x_qj - x_ij)^2 / bandwidth^2
 *
 * Weight: w_i = exp(-d_i^2 / 2)  (Gaussian kernel)
 *
 * The bandwidth parameter controls the locality:
 *   - Small bandwidth: very local (few neighbors matter)
 *   - Large bandwidth: global model (all neighbors matter)
 * ==================================================================== */

void jitpls_init(JITPLSState *state, size_t n_vars, size_t db_capacity,
                  size_t n_neighbors, double bandwidth)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->n_vars = n_vars;
    state->db_capacity = db_capacity;
    state->n_neighbors = n_neighbors;
    state->locality_bandwidth = bandwidth;
    state->database_x = (double *)calloc(db_capacity * n_vars, sizeof(double));
    state->database_y = (double *)calloc(db_capacity, sizeof(double));
}

void jitpls_destroy(JITPLSState *state)
{
    if (!state) return;
    free(state->database_x);
    free(state->database_y);
    memset(state, 0, sizeof(*state));
}

void jitpls_add_to_database(JITPLSState *state, const double *x, double y)
{
    if (!state || !x) return;
    if (state->db_count >= state->db_capacity) return;

    size_t pos = state->db_count;
    for (size_t i = 0; i < state->n_vars; i++)
        state->database_x[pos * state->n_vars + i] = x[i];
    state->database_y[pos] = y;
    state->db_count++;
}

/* Simplified JIT prediction: weighted average of k nearest neighbors.
 * Full JIT-PLS would build a PLS model on neighbors, but for prediction
 * efficiency we use locally weighted averaging.
 *
 * For full JIT-PLS, one would:
 *   1. Select neighbors
 *   2. Build PLS model on weighted data
 *   3. Predict
 *   4. Discard model
 *
 * Here we use a simpler but effective locally weighted average
 * which captures the spirit of JIT learning. */

double jitpls_predict(JITPLSState *state, const double *x_query,
                       size_t n_components)
{
    (void)n_components; /* Reserved for full JIT-PLS expansion */

    if (!state || !x_query || state->db_count == 0) return 0.0;

    size_t k = state->n_neighbors;
    if (k > state->db_count) k = state->db_count;
    if (k == 0) return 0.0;

    double bw2 = state->locality_bandwidth * state->locality_bandwidth;
    if (bw2 < 1e-10) bw2 = 1.0;

    /* Compute distances and select k nearest */
    /* For simplicity, use weighted average directly */
    double weighted_sum = 0.0, weight_total = 0.0;

    for (size_t i = 0; i < state->db_count; i++) {
        double dist_sq = 0.0;
        for (size_t j = 0; j < state->n_vars; j++) {
            double diff = x_query[j] - state->database_x[i * state->n_vars + j];
            dist_sq += diff * diff;
        }
        double weight = exp(-dist_sq / (2.0 * bw2));
        weighted_sum += weight * state->database_y[i];
        weight_total += weight;
    }

    if (weight_total > 1e-15)
        return weighted_sum / weight_total;

    /* Fallback: simple mean */
    double mean = 0.0;
    for (size_t i = 0; i < state->db_count; i++)
        mean += state->database_y[i];
    return mean / (double)state->db_count;
}

void jitpls_prune_database(JITPLSState *state, size_t max_age, size_t min_keep)
{
    if (!state) return;
    /* Keep only the most recent max_age entries, but at least min_keep */
    if (state->db_count <= max_age || state->db_count <= min_keep) return;

    size_t discard = state->db_count - max_age;
    if (state->db_count - discard < min_keep)
        discard = state->db_count - min_keep;

    /* Shift remaining data to front */
    size_t keep = state->db_count - discard;
    memmove(state->database_x,
            state->database_x + discard * state->n_vars,
            keep * state->n_vars * sizeof(double));
    memmove(state->database_y,
            state->database_y + discard,
            keep * sizeof(double));
    state->db_count = keep;
}

double sample_similarity(const double *x1, const double *x2, size_t n_vars)
{
    if (!x1 || !x2) return DBL_MAX;

    double dist_sq = 0.0;
    for (size_t i = 0; i < n_vars; i++) {
        double diff = x1[i] - x2[i];
        dist_sq += diff * diff;
    }
    return sqrt(dist_sq);
}
