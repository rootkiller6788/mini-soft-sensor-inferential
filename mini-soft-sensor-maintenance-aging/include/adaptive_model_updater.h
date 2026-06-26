/**
 * @file    adaptive_model_updater.h
 * @brief   Adaptive model update strategies - L5 Algorithms & L8 Advanced
 *
 * Implements recursive and windowed model update methods for
 * maintaining soft sensor accuracy over time:
 *   - RPLS: Recursive Partial Least Squares (Helland et al. 1992)
 *   - MW-PLS: Moving Window PLS
 *   - REW-PLS: Recursive Exponentially Weighted PLS
 *   - JIT-LW-PLS: Just-in-Time Locally Weighted PLS
 *   - Forgetting factor strategies (constant, adaptive)
 *   - Model update trigger logic
 *
 * L5: RPLS, MW-PLS, REW-PLS, JIT modeling, forgetting factor
 * L8: Adaptive forgetting factor based on process state
 *
 * Ref: Qin, S.J. (1998) "Recursive PLS algorithms for adaptive data
 *      monitoring". Comp. & Chem. Eng., 22(4-5), 503-514.
 *      Dayal & MacGregor (1997) "Recursive exponentially weighted PLS".
 *      J. Process Control, 7(3), 169-179.
 */

#ifndef ADAPTIVE_MODEL_UPDATER_H
#define ADAPTIVE_MODEL_UPDATER_H

#include <stddef.h>

/* ============================================================================
 * L3: Matrix/Vector types for model representation
 * ===========================================================================*/

/** @brief Dense matrix structure for PLS model components */
typedef struct {
    double *data;
    size_t rows;
    size_t cols;
    size_t stride;
} MatrixDense;

/** @brief PLS model components (Loadings, Weights, Regression coefficients) */
typedef struct {
    size_t n_vars;
    size_t n_components;
    double *p;         /**< X-loadings matrix [n_vars * n_components] */
    double *w;         /**< X-weights matrix [n_vars * n_components] */
    double *q;         /**< Y-loadings vector [n_components] */
    double *b;         /**< Regression coefficients [n_vars] */
    double *t_mean;    /**< Mean of X scores [n_components] */
    double *x_mean;    /**< Mean of X variables [n_vars] */
    double y_mean;     /**< Mean of Y */
    double *x_std;     /**< Standard deviation of X [n_vars] */
    double y_std;      /**< Standard deviation of Y */
} PLSModel;

/* ============================================================================
 * L5: Forgetting Factor Strategy
 * ===========================================================================*/

/** @brief Forgetting factor types */
typedef enum {
    FORGET_CONSTANT      = 0,
    FORGET_ADAPTIVE      = 1,
    FORGET_STEP          = 2,
    FORGET_PROCESS_STATE = 3
} ForgettingFactorType;

/** @brief Forgetting factor configuration */
typedef struct {
    ForgettingFactorType type;
    double base_lambda;
    double current_lambda;
    double min_lambda;
    double max_lambda;
    double update_rate;
    size_t step_interval;
    size_t samples_since_step;
} ForgettingFactor;

/* ============================================================================
 * L5: Update Trigger Logic
 * ===========================================================================*/

/** @brief Model update trigger strategy */
typedef enum {
    TRIGGER_PERIODIC        = 0,
    TRIGGER_PERFORMANCE     = 1,
    TRIGGER_PROCESS_CHANGE  = 2,
    TRIGGER_HYBRID          = 3
} UpdateTrigger;

/** @brief Update trigger configuration */
typedef struct {
    UpdateTrigger type;
    size_t periodic_interval;
    double rmse_threshold;
    double r2_threshold;
    double process_change_threshold;
    size_t samples_since_update;
    int    update_needed;
    double last_rmse;
} UpdateTriggerConfig;

/* ============================================================================
 * L5: RPLS State
 * ===========================================================================*/

/** @brief RPLS recursive state (maintains covariance matrices) */
typedef struct {
    size_t n_vars;
    size_t n_components;
    double *x_mean;
    double *y_mean;
    double *pxx;
    double *pxy;
    double *p;
    double *w;
    double *q;
    double *b;
    double lambda;
    size_t n_samples;
    size_t n_effective;
} RPLSState;

/* ============================================================================
 * L5: Moving Window PLS State
 * ===========================================================================*/

/** @brief MW-PLS maintains a window of recent samples */
typedef struct {
    double *x_window;
    double *y_window;
    size_t window_size;
    size_t n_vars;
    size_t current_pos;
    size_t n_samples;
    int    window_full;
    PLSModel current_model;
    size_t n_components;
} MWPLSState;

/* ============================================================================
 * L8: JIT Locally Weighted PLS State
 * ===========================================================================*/

/** @brief JIT modeling: select relevant samples for each query */
typedef struct {
    double *database_x;
    double *database_y;
    size_t db_capacity;
    size_t db_count;
    size_t n_vars;
    size_t n_neighbors;
    double locality_bandwidth;
} JITPLSState;

/* ============================================================================
 * API: Matrix Operations
 * ===========================================================================*/

MatrixDense matrix_create(size_t rows, size_t cols);
void matrix_destroy(MatrixDense *m);
double matrix_get(const MatrixDense *m, size_t i, size_t j);
void matrix_set(MatrixDense *m, size_t i, size_t j, double val);
void matrix_multiply(const MatrixDense *a, const MatrixDense *b, MatrixDense *c);
void matrix_transpose(const MatrixDense *src, MatrixDense *dst);
void matrix_scale(MatrixDense *m, double scalar);

/* ============================================================================
 * API: PLS Model Management
 * ===========================================================================*/

PLSModel pls_model_create(size_t n_vars, size_t n_components);
void pls_model_destroy(PLSModel *model);
double pls_predict(const PLSModel *model, const double *x);
void pls_predict_batch(const PLSModel *model, const double *x,
                        size_t n_samples, double *y_pred);

/**
 * @brief Compute variable importance in projection (VIP) scores.
 * VIP_j = sqrt( sum_{a=1}^A (w_{ja}^2 * SSY_a * p) / (SSY_total * A) )
 * Variables with VIP > 1.0 are considered important.
 */
void pls_vip_scores(const PLSModel *model, double *vip);

/* ============================================================================
 * API: RPLS
 * ===========================================================================*/

void rpls_init(RPLSState *state, size_t n_vars, size_t n_components,
                double lambda);
void rpls_destroy(RPLSState *state);

/**
 * @brief Recursive PLS update with new sample pair (x, y).
 *
 * Algorithm (Qin 1998):
 *   1. Mean update: x_mean = lambda*x_mean + (1-lambda)*x_new
 *   2. Covariance update via rank-1 modification
 *   3. PLS decomposition on updated covariance
 *   4. Regression coefficient extraction
 *
 * O(n_vars * n_components) per update.
 */
void rpls_update(RPLSState *state, const double *x, double y);
double rpls_predict(const RPLSState *state, const double *x);

/**
 * @brief RPLS bulk update: process a batch of new samples.
 * More efficient than individual updates for large batches.
 */
void rpls_bulk_update(RPLSState *state, const double *x, const double *y,
                       size_t n_samples);

/* ============================================================================
 * API: MW-PLS
 * ===========================================================================*/

void mwpls_init(MWPLSState *state, size_t n_vars, size_t n_components,
                 size_t window_size);
void mwpls_destroy(MWPLSState *state);
void mwpls_add_sample(MWPLSState *state, const double *x, double y);
double mwpls_predict(const MWPLSState *state, const double *x);

/* ============================================================================
 * API: Forgetting Factor
 * ===========================================================================*/

void forgetting_factor_init(ForgettingFactor *ff, ForgettingFactorType type,
                              double base_lambda);
double forgetting_factor_update(ForgettingFactor *ff, double process_change);
void forgetting_factor_reset(ForgettingFactor *ff);

/* ============================================================================
 * API: Update Trigger
 * ===========================================================================*/

void update_trigger_init(UpdateTriggerConfig *cfg, UpdateTrigger type,
                          size_t interval, double rmse_thresh);
int update_trigger_check(UpdateTriggerConfig *cfg, double current_rmse,
                          double current_r2, double process_change);

/* ============================================================================
 * L8: JIT Locally Weighted PLS API
 * ===========================================================================*/

void jitpls_init(JITPLSState *state, size_t n_vars, size_t db_capacity,
                  size_t n_neighbors, double bandwidth);
void jitpls_destroy(JITPLSState *state);
void jitpls_add_to_database(JITPLSState *state, const double *x, double y);

/**
 * @brief Just-in-Time prediction: select k-nearest neighbors,
 *        build local PLS model, predict, discard model.
 *
 * Distance metric: weighted Euclidean with locality bandwidth.
 * weight_i = exp(-d_i^2 / bandwidth^2)
 */
double jitpls_predict(JITPLSState *state, const double *x_query,
                       size_t n_components);

/* ============================================================================
 * API: Model Database Maintenance
 * ===========================================================================*/

/**
 * @brief Remove outdated samples from JIT database based on age.
 * Samples older than max_age are pruned. Maintains at least min_keep.
 */
void jitpls_prune_database(JITPLSState *state, size_t max_age, size_t min_keep);

/**
 * @brief Compute similarity between two sample vectors.
 * Returns 0 (identical) to infinity (completely different).
 */
double sample_similarity(const double *x1, const double *x2, size_t n_vars);

#endif /* ADAPTIVE_MODEL_UPDATER_H */
