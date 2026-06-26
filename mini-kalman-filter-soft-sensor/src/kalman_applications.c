/**
 * @file kalman_applications.c
 * @brief Kalman Filter Industrial Applications — Soft Sensor Implementations
 *
 * L6 Canonical Problems: DC motor state estimation, CSTR temperature,
 *   GPS/INS navigation
 * L7 Industrial Applications: Quality estimator for DCS/historian deployment,
 *   sensor fault detection, OPC-style data snapshots
 *
 * References:
 *   Myke King (2016) "Process Control: A Practical Approach"
 *   Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control"
 *   Grewal & Andrews (2014) "Kalman Filtering: Theory and Practice with MATLAB"
 *   OSIsoft PI SDK documentation
 *   Honeywell Profit SensorPro documentation
 */
#include "kalman_applications.h"
#include "kalman_matrix_ops.h"
#include <math.h>
#include <string.h>
#include <time.h>

/* ===================================================================
 * DC Motor Estimator
 *
 * Model: 3-state augmented [angle, velocity, current]
 *
 * Continuous: d/dt[theta] = omega
 *             d/dt[omega] = -b/J * omega + Kt/J * i
 *             d/dt[i] = -R/L * i - Kb/L * omega + 1/L * u
 *
 * Discrete: x[k] = F * x[k-1] + B * u[k-1] (Euler forward)
 *
 * Measurement: H = [1, 0, 0] (measure angle only)
 * =================================================================== */

void dc_motor_estimator_init(DCMotorEstimator *est,
                              double resistance, double inductance,
                              double Kb, double Kt,
                              double inertia, double damping,
                              double dt,
                              double process_noise_q,
                              double measure_noise_r)
{
    if (!est) return;
    memset(est, 0, sizeof(*est));

    est->resistance = resistance;
    est->inductance = inductance;
    est->back_emf_const = Kb;
    est->torque_const = Kt;
    est->inertia = inertia;
    est->damping = damping;
    est->dt = dt;

    uint8_t n = DC_MOTOR_STATE_DIM, m = DC_MOTOR_MEAS_DIM;

    /* State transition matrix F (2x2): [angle, velocity] */
    /* F = [[1, dt], [0, 1 - b*dt/J]] */
    double F_data[4] = {1.0, dt, 0.0, 1.0 - damping * dt / inertia};
    memcpy(est->model.F, F_data, sizeof(F_data));

    /* B = [[0], [Kt*dt/J]] */
    est->model.B[0] = 0.0;
    est->model.B[1] = Kt * dt / inertia;

    /* H = [[1, 0]] */
    est->model.H[0] = 1.0;
    est->model.H[1] = 0.0;

    /* Q = process_noise_q * I_2 */
    est->model.Q[0] = process_noise_q;
    est->model.Q[1] = 0.0;
    est->model.Q[2] = 0.0;
    est->model.Q[3] = process_noise_q;

    /* R = measure_noise_r */
    est->model.R[0] = measure_noise_r;

    est->model.n = n;
    est->model.m = m;
    est->model.p = DC_MOTOR_INPUT_DIM;
    est->model.time_varying = 0;
    est->model.is_linear = 1;

    double x0[2] = {0.0, 0.0};
    double P0[4] = {1.0, 0.0, 0.0, 1.0};
    kf_init(&est->kf, &est->model, x0, P0, n, m);
    est->initialized = 1;
}

void dc_motor_estimator_step(DCMotorEstimator *est,
                              double voltage, double angle_meas)
{
    if (!est || !est->initialized) return;
    double u[1] = {voltage};
    double z[1] = {angle_meas};
    kf_step(&est->kf, &est->model, z, u);
    est->angle_est = est->kf.x[0];
    est->velocity_est = est->kf.x[1];
    /* Current estimated from torque relation: i = J/Kt * (domega/dt_est) */
    /* Simplified: use back EMF equation i_est = (u - Kb*omega)/R */
    if (est->resistance > MAT_EPSILON) {
        est->current_est = (voltage - est->back_emf_const * est->velocity_est)
                           / est->resistance;
    }
}

double dc_motor_get_velocity(const DCMotorEstimator *est)
{
    return est ? est->velocity_est : 0.0;
}

double dc_motor_get_torque(const DCMotorEstimator *est)
{
    if (!est) return 0.0;
    return est->torque_const * est->current_est;
}

/* ===================================================================
 * CSTR Estimator (EKF)
 *
 * State: x = [C_A, T] — concentration and temperature
 * Input: u = [Tc] — jacket temperature
 * Measurement: z = [T_meas] — temperature measurement
 *
 * dC_A/dt = (q/V)*(Ca0 - C_A) - k0*exp(-E_R/T)*C_A
 * dT/dt = (q/V)*(T0 - T) + (-dH_R)*k0*exp(-E_R/T)*C_A
 *         + (UA/(rho*Cp*V))*(Tc - T)
 *
 * Discrete via Euler forward with dt.
 * =================================================================== */

/* CSTR nonlinear state transition f(x, u) -> x_next */
static void cstr_process_fn(const double *x, const double *u,
                             uint8_t n, uint8_t p, double *x_next)
{
    if (!x || !x_next) return;
    /* Internal function — the CSTREstimator model is passed via global/static */
    /* For standalone use, we need the model reference stored in the estimator.
     * This function pointer is set up during init with closure via global. */
    (void)n; (void)p; (void)u;
    /* Placeholder: actual implementation uses the estimator's stored parameters */
    x_next[0] = x[0];
    x_next[1] = x[1];
}

/* CSTR nonlinear measurement h(x) -> z_pred */
static void cstr_measure_fn(const double *x, uint8_t n,
                             uint8_t m, double *z_pred)
{
    if (!x || !z_pred) return;
    (void)n; (void)m;
    z_pred[0] = x[1]; /* measure temperature */
}

void cstr_estimator_init(CSTREstimator *est,
                          double Ca0, double T0, double q, double V,
                          double k0, double E_R, double dH_R,
                          double rhoCp, double UA,
                          double dt,
                          double Q_conc, double Q_temp,
                          double R_meas)
{
    if (!est) return;
    memset(est, 0, sizeof(*est));

    est->feed_conc = Ca0;
    est->feed_temp = T0;
    est->flow_rate = q;
    est->volume = V;
    est->pre_exp_factor = k0;
    est->activation_E_R = E_R;
    est->delta_H_R = dH_R;
    est->rho_Cp = rhoCp;
    est->UA = UA;
    est->dt = dt;

    /* Set up EKF model function pointers */
    est->ekf_model.f = cstr_process_fn;
    est->ekf_model.h = cstr_measure_fn;
    est->ekf_model.F_jacobian = NULL; /* User supplies Jacobian externally */
    est->ekf_model.H_jacobian = NULL;

    est->ekf_model.n = CSTR_STATE_DIM;
    est->ekf_model.m = CSTR_MEAS_DIM;
    est->ekf_model.p = 1;

    /* Q = diag(Q_conc, Q_temp) */
    est->ekf_model.Q[0] = Q_conc; est->ekf_model.Q[1] = 0.0;
    est->ekf_model.Q[2] = 0.0;   est->ekf_model.Q[3] = Q_temp;

    /* R = R_meas */
    est->ekf_model.R[0] = R_meas;

    double x0[2] = {Ca0, T0};
    double P0[4] = {0.01, 0.0, 0.0, 1.0};

    ekf_init(&est->ekf, &est->ekf_model, x0, P0,
             CSTR_STATE_DIM, CSTR_MEAS_DIM, 1);
    est->initialized = 1;
}

void cstr_estimator_step(CSTREstimator *est,
                          double Tc, double temp_meas)
{
    if (!est || !est->initialized) return;
    est->jacket_temp = Tc;
    double u[1] = {Tc};
    double z[1] = {temp_meas};
    ekf_step(&est->ekf, z, u);
    est->conc_est = est->ekf.kf.x[0];
    est->temp_est = est->ekf.kf.x[1];
}

double cstr_get_conversion(const CSTREstimator *est)
{
    if (!est) return 0.0;
    double Ca0 = est->feed_conc;
    if (Ca0 < MAT_EPSILON) return 0.0;
    return (Ca0 - est->conc_est) / Ca0;
}

/* ===================================================================
 * INS/GPS Navigation (9-state loosely-coupled)
 *
 * State: [p_x, p_y, p_z, v_x, v_y, v_z, b_ax, b_ay, b_az]
 *   where b_a = accelerometer bias
 *
 * Dynamics (constant velocity + bias random walk):
 *   p[k+1] = p[k] + v[k]*dt + 0.5*(a_meas[k]-b_a[k])*dt^2
 *   v[k+1] = v[k] + (a_meas[k]-b_a[k])*dt
 *   b_a[k+1] = b_a[k]
 *
 * GPS measurement: H = [I_3x3, 0_3x6]
 * =================================================================== */

void ins_gps_init(INSGPSNavigator *nav, double dt,
                  double accel_noise, double accel_bias_noise,
                  double gps_std, double init_pos_std, double init_vel_std)
{
    if (!nav) return;
    memset(nav, 0, sizeof(*nav));
    nav->dt = dt;
    nav->gravity = 9.81;
    nav->accel_noise_psd = accel_noise;
    nav->accel_bias_psd = accel_bias_noise;
    nav->gps_pos_std = gps_std;

    uint8_t n = INS_STATE_DIM, m = INS_MEAS_DIM;

    /* F matrix: 9 x 9 */
    /* Position: p += v*dt + 0.5*a*dt^2, with -0.5*dt^2 for bias */
    /* Velocity: v += a*dt, with -dt for bias */
    /* Bias: constant */
    memset(nav->model.F, 0, sizeof(nav->model.F));
    for (uint8_t i = 0; i < 3; i++) {
        nav->model.F[i * 9 + i] = 1.0;               /* p[i] -> p[i] */
        nav->model.F[i * 9 + 3 + i] = dt;            /* v[i] -> p[i+1] */
        nav->model.F[(3+i) * 9 + (3+i)] = 1.0;       /* v[i] -> v[i] */
        nav->model.F[i * 9 + 6 + i] = -0.5 * dt * dt;/* b_a[i] -> p[i] */
        nav->model.F[(3+i) * 9 + 6 + i] = -dt;       /* b_a[i] -> v[i] */
        nav->model.F[(6+i) * 9 + (6+i)] = 1.0;       /* b_a[i] -> b_a[i] */
    }

    /* H: measure position only */
    memset(nav->model.H, 0, sizeof(nav->model.H));
    for (uint8_t i = 0; i < 3; i++)
        nav->model.H[i * 9 + i] = 1.0;

    /* Q: process noise from IMU */
    memset(nav->model.Q, 0, sizeof(nav->model.Q));
    double q_acc = accel_noise * dt * dt;
    double q_bias = accel_bias_noise * dt;
    for (uint8_t i = 0; i < 3; i++) {
        nav->model.Q[(3+i) * 9 + (3+i)] = q_acc;      /* velocity noise */
        nav->model.Q[(6+i) * 9 + (6+i)] = q_bias;      /* bias noise */
    }

    /* R: GPS measurement noise */
    memset(nav->model.R, 0, sizeof(nav->model.R));
    for (uint8_t i = 0; i < 3; i++)
        nav->model.R[i * 3 + i] = gps_std * gps_std;

    nav->model.n = n; nav->model.m = m; nav->model.p = INS_INPUT_DIM;

    double x0[9] = {0};
    double P0[81] = {0};
    for (uint8_t i = 0; i < 3; i++) {
        P0[i * 9 + i] = init_pos_std * init_pos_std;
        P0[(3+i) * 9 + (3+i)] = init_vel_std * init_vel_std;
        P0[(6+i) * 9 + (6+i)] = accel_bias_noise;
    }

    kf_init(&nav->kf, &nav->model, x0, P0, n, m);
    nav->initialized = 1;
}

void ins_gps_predict_imu(INSGPSNavigator *nav,
                          double ax, double ay, double az)
{
    if (!nav || !nav->initialized) return;

    /* Compensate for bias in the prediction (added to control) */
    double u[3] = {ax - nav->accel_bias_x,
                   ay - nav->accel_bias_y,
                   az - nav->accel_bias_z - nav->gravity};

    kf_predict(&nav->kf, &nav->model, u);
    nav->imu_count++;
}

void ins_gps_update_gps(INSGPSNavigator *nav,
                         double gps_x, double gps_y, double gps_z)
{
    if (!nav || !nav->initialized) return;
    double z[3] = {gps_x, gps_y, gps_z};
    kf_update(&nav->kf, &nav->model, z);

    /* Extract navigation solution */
    nav->pos_x = nav->kf.x[0]; nav->pos_y = nav->kf.x[1]; nav->pos_z = nav->kf.x[2];
    nav->vel_x = nav->kf.x[3]; nav->vel_y = nav->kf.x[4]; nav->vel_z = nav->kf.x[5];
    nav->accel_bias_x = nav->kf.x[6];
    nav->accel_bias_y = nav->kf.x[7];
    nav->accel_bias_z = nav->kf.x[8];

    /* Circular Error Probable (CEP) estimate:
     * CEP ≈ 0.589 * (sigma_x + sigma_y) or from trace of position cov */
    double pos_trace = nav->kf.P[0] + nav->kf.P[10] + nav->kf.P[20];
    nav->cep = sqrt(pos_trace);

    nav->gps_count++;
    nav->gps_valid = 1;
}

void ins_gps_get_position(const INSGPSNavigator *nav,
                           double *px, double *py, double *pz)
{
    if (!nav) { if(px)*px=0; if(py)*py=0; if(pz)*pz=0; return; }
    if (px) *px = nav->pos_x;
    if (py) *py = nav->pos_y;
    if (pz) *pz = nav->pos_z;
}

void ins_gps_get_velocity(const INSGPSNavigator *nav,
                           double *vx, double *vy, double *vz)
{
    if (!nav) { if(vx)*vx=0; if(vy)*vy=0; if(vz)*vz=0; return; }
    if (vx) *vx = nav->vel_x;
    if (vy) *vy = nav->vel_y;
    if (vz) *vz = nav->vel_z;
}

/* ===================================================================
 * Industrial Quality Estimator
 *
 * Infers product quality from process measurements using an
 * adaptive Kalman filter. Models the relationship:
 *   quality[k] = beta[0]*x1[k] + ... + beta[n-1]*xn[k] + intercept
 *
 * The KF tracks time-varying regression coefficients (beta) to
 * handle process drift, catalyst deactivation, fouling, etc.
 * =================================================================== */

void quality_estimator_init(IndustrialQualityEstimator *est,
                             const double *beta, double intercept,
                             uint8_t n_coeffs,
                             double Q0, double R0,
                             double update_period)
{
    if (!est || !beta) return;
    memset(est, 0, sizeof(*est));

    /* Store regression parameters */
    memcpy(est->beta, beta, n_coeffs * sizeof(double));
    est->intercept = intercept;
    est->num_sensors = n_coeffs;

    /* State dimension = number of regression coefficients
     * Measurement dimension = 1 (quality is a scalar) */
    uint8_t n = n_coeffs;
    uint8_t m = 1;

    /* Initialize adaptive KF */
    double x0[KF_MAX_STATE_DIM];
    memcpy(x0, beta, n * sizeof(double));

    double P0_arr[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    memset(P0_arr, 0, sizeof(P0_arr));
    for (uint8_t i = 0; i < n; i++) P0_arr[i * n + i] = 0.01;

    double Q0_arr[KF_MAX_STATE_DIM * KF_MAX_STATE_DIM];
    memset(Q0_arr, 0, sizeof(Q0_arr));
    for (uint8_t i = 0; i < n; i++) Q0_arr[i * n + i] = Q0;

    double R0_arr[KF_MAX_MEAS_DIM * KF_MAX_MEAS_DIM];
    R0_arr[0] = R0;

    akf_init(&est->akf, x0, P0_arr, Q0_arr, R0_arr,
             n, m, 1, 1, AKF_DEFAULT_WINDOW);

    /* Set up measurement model H */
    est->model.n = n;
    est->model.m = m;
    est->model.p = 0;

    /* F = I (beta drift is modeled by process noise) */
    mat_identity(est->model.F, n);

    /* R = R0 */
    memset(est->model.R, 0, sizeof(est->model.R));
    est->model.R[0] = R0;

    /* Q dynamically updated by AKF */

    est->ewma_alpha = 0.2;
    est->config.update_period = update_period;
    est->initialized = 1;
}

void quality_estimator_update(IndustrialQualityEstimator *est,
                               const double *x_meas)
{
    if (!est || !est->initialized || !x_meas) return;
    uint8_t n = est->akf.n;

    /* Build measurement matrix H from current process measurements */
    /* H = x_meas' (1 x n) — linear regression model */
    for (uint8_t j = 0; j < n; j++)
        est->model.H[j] = x_meas[j];

    /* Predicted quality */
    double quality_pred_raw = est->intercept;
    for (uint8_t j = 0; j < n; j++)
        quality_pred_raw += est->akf.kf.x[j] * x_meas[j];

    /* Measurement: we use the quality prediction as a pseudo-measurement
     * In real application, lab samples provide actual quality values */
    double z[1] = {quality_pred_raw};

    akf_step(&est->akf, &est->model, z, NULL);

    /* Recompute quality with updated coefficients */
    est->quality_pred = est->intercept;
    for (uint8_t j = 0; j < n; j++)
        est->quality_pred += est->akf.kf.x[j] * x_meas[j];

    /* Quality variance from KF covariance: var = H * P * H' */
    double HPHt = 0.0;
    for (uint8_t i = 0; i < n; i++)
        for (uint8_t j = 0; j < n; j++)
            HPHt += x_meas[i] * est->akf.kf.P[i * n + j] * x_meas[j];
    est->quality_std = sqrt(HPHt + est->akf.R_est[0]);

    /* 95% confidence bounds */
    est->quality_min = est->quality_pred - 1.96 * est->quality_std;
    est->quality_max = est->quality_pred + 1.96 * est->quality_std;

    /* EWMA of prediction error */
    double error = (est->lab_valid) ? (est->quality_pred - est->lab_value) : 0.0;
    est->ewma_error = est->ewma_alpha * error
                      + (1.0 - est->ewma_alpha) * est->ewma_error;

    /* Sensor fault detection */
    quality_estimator_check_sensors(est);

    est->update_count++;
}

void quality_estimator_calibrate(IndustrialQualityEstimator *est,
                                  double lab_quality)
{
    if (!est || !est->initialized) return;
    est->lab_value = lab_quality;
    est->lab_valid = 1;

    /* Bias correction: adjust intercept to match lab value */
    double bias = lab_quality - est->quality_pred;
    est->intercept += 0.3 * bias;  /* Partial correction for stability */
}

void quality_estimator_get_prediction(const IndustrialQualityEstimator *est,
                                       double *quality,
                                       double *lower_ci,
                                       double *upper_ci)
{
    if (!est) {
        if (quality) *quality = 0.0;
        if (lower_ci) *lower_ci = 0.0;
        if (upper_ci) *upper_ci = 0.0;
        return;
    }
    if (quality) *quality = est->quality_pred;
    if (lower_ci) *lower_ci = est->quality_min;
    if (upper_ci) *upper_ci = est->quality_max;
}

uint8_t quality_estimator_check_sensors(IndustrialQualityEstimator *est)
{
    if (!est) return 0;
    uint8_t fault_mask = 0;
    uint8_t n = est->akf.n;

    /* Check each measurement channel using innovation statistics */
    for (uint8_t i = 0; i < n && i < 8; i++) {
        /* Simple threshold check on coefficient stability */
        double beta_change = fabs(est->akf.kf.x[i] - est->beta[i]);
        if (beta_change > 3.0 * sqrt(est->akf.kf.P[i * n + i])) {
            fault_mask |= (1 << i);
            est->sensor_fault[i] = 1;
        } else {
            est->sensor_fault[i] = 0;
        }
    }

    return fault_mask;
}

/* ===================================================================
 * Historian Snapshot — OPC-style data quality
 * =================================================================== */

void historian_create_snapshot(const IndustrialQualityEstimator *est,
                                HistorianSnapshot *snap, uint8_t tag_idx)
{
    if (!est || !snap) return;
    memset(snap, 0, sizeof(*snap));

    /* Timestamp */
    snap->timestamp_sec = (uint32_t)time(NULL);
    snap->timestamp_ms = 0;

    /* Quality */
    if (!est->initialized) {
        snap->quality = 0;   /* Bad */
    } else if (est->sensor_fault[tag_idx]) {
        snap->quality = 64;  /* Uncertain */
    } else {
        snap->quality = 192; /* Good */
    }

    /* Value */
    if (tag_idx == 0) {
        snap->value = est->quality_pred;
    } else if (tag_idx <= est->akf.n) {
        snap->value = est->akf.kf.x[tag_idx - 1];
    } else {
        snap->value = 0.0;
    }
}
