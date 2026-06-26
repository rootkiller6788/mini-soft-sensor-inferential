#ifndef PLS_ONLINE_H
#define PLS_ONLINE_H
#include "pls_model.h"
#include "pls_statistics.h"
#include "matrix_ops.h"
#include <stddef.h>
#include <time.h>
#ifdef __cplusplus
extern "C" {
#endif

/* =================================================================
 * pls_online.h — Online PLS Prediction and Process Monitoring
 *
 * Runtime pipeline for deploying PLS models in real-time industrial
 * soft-sensing and multivariate statistical process control (MSPC).
 *
 * Knowledge Coverage:
 *   L3 - Engineering Structures: Online prediction pipeline,
 *        real-time data preprocessing, fault detection loop
 *   L7 - Industrial Applications: Soft sensor for quality estimation,
 *        MSPC with PLS, alarm generation and management
 *
 * Online Monitoring Pipeline:
 *   1. Acquire new measurement x_new (p variables)
 *   2. Preprocess using training parameters (center and scale)
 *   3. Predict y_hat using Beta coefficients
 *   4. Compute T2 and SPE diagnostics
 *   5. Check against 95% and 99% control limits
 *   6. Generate alarms if limits exceeded
 *
 * References:
 *   Wise & Gallagher, J. Process Control, 6(6):329-348, 1996.
 *   Kourti & MacGregor, Chemometrics Intell. Lab. Syst., 28:3-21, 1995.
 *   Qin, S.J., J. Chemometrics, 17:480-502, 2003.
 * ================================================================= */

/* ---- Alarm State Definitions ---- */

typedef enum {
    ALARM_NONE     = 0,
    ALARM_WARNING  = 1,
    ALARM_CRITICAL = 2,
    ALARM_BOTH     = 3
} AlarmState;

/* ---- Online Prediction Result ---- */

typedef struct {
    Vector   *y_pred;
    Vector   *t_scores;
    double    T2;
    double    T2_limit_95;
    double    T2_limit_99;
    double    SPE;
    double    SPE_limit_95;
    double    SPE_limit_99;
    double    DModX;
    double    DModX_crit;
    AlarmState alarm;
    time_t    timestamp;
} OnlinePrediction;

/* ---- Online Monitor State ---- */

typedef struct {
    PLSModel      *model;
    Vector        *score_variances;
    double         T2_limit_95;
    double         T2_limit_99;
    double         SPE_theta1;
    double         SPE_theta2;
    double         SPE_theta3;
    double         SPE_limit_95;
    double         SPE_limit_99;
    double         DModX_crit;
    size_t         sample_count;
    size_t         alarm_count;
    int            verbose;
} OnlineMonitor;

/* ---- Lifecycle ---- */

int  online_monitor_init(const PLSModel *model, OnlineMonitor *monitor,
                         double alpha_95, double alpha_99);
void online_monitor_free(OnlineMonitor *monitor);

/*
 * Process a single new sample through the complete prediction pipeline.
 * Returns 0 on success, -1 on error.
 */
int  online_predict_sample(OnlineMonitor *monitor,
                           const Vector *x_new, OnlinePrediction *pred);

/*
 * Process a batch of new samples.
 */
int  online_predict_batch(OnlineMonitor *monitor,
                          const Matrix *X_new, OnlinePrediction *preds);

void online_prediction_print(const OnlinePrediction *pred, size_t q_vars);
const char* alarm_state_string(AlarmState state);

OnlinePrediction* online_prediction_alloc(size_t q_vars, size_t a_lvs);
void              online_prediction_free(OnlinePrediction *pred);
void              online_monitor_reset(OnlineMonitor *monitor);

/*
 * Get monitor summary statistics.
 */
void online_monitor_summary(const OnlineMonitor *monitor,
                            size_t *total, size_t *alarms, double *alarm_rate);

/*
 * Compute variable contributions to T2 for fault diagnosis.
 * For each variable j: contrib_j = sum_a (t_a * w_{ja} * x_j / var(t_a))
 * Returns new Vector (p x 1). Caller must free.
 */
Vector* online_T2_contributions(const OnlineMonitor *monitor,
                                const Vector *x_processed,
                                const Vector *t_scores);

/*
 * Compute variable contributions to SPE for fault diagnosis.
 * contrib_j = (x_j - x_hat_j)^2  where x_hat = t * P^T
 * Returns new Vector (p x 1). Caller must free.
 */
Vector* online_SPE_contributions(const OnlineMonitor *monitor,
                                 const Vector *x_processed,
                                 const Vector *t_scores);

/*
 * Compute y_hat from x_processed using Beta and b0.
 * Convenience function used internally by the prediction pipeline.
 * y_pred must be pre-allocated (q_vars x 1).
 */
void online_compute_yhat(const PLSModel *model,
                         const Vector *x_processed, Vector *y_pred);

/*
 * Apply preprocessing to a single sample using stored parameters.
 * x_processed must be pre-allocated (p_vars x 1).
 */
void online_preprocess(const PLSModel *model,
                       const Vector *x_raw, Vector *x_processed);

#ifdef __cplusplus
}
#endif
#endif /* PLS_ONLINE_H */
