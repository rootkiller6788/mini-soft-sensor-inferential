/**
 * Example 3: Dynamic Data Reconciliation using Kalman Filter
 *
 * A simple tank level system with known dynamics:
 *   dx/dt = (F_in - F_out) / A
 *
 * Discretized (Euler, dt=1.0):
 *   x_{k+1} = x_k + (u_k - d_k) / A + w_k
 *
 * where:
 *   x_k = tank level [m]
 *   u_k = inlet flow [m^3/s] (known input)
 *   d_k = outlet flow [m^3/s] (constant disturbance)
 *   A   = tank cross-section area [m^2]
 *   w_k ~ N(0, Q) process noise
 *
 * Measurements:
 *   y_k = x_k + v_k,  v_k ~ N(0, R)
 *
 * Kalman filter reconciles noisy level measurements with the dynamic
 * model to provide filtered (and smoothed) level estimates.
 */

#include "dr_dynamic.h"
#include "dr_core.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    int k;
    int nx = 1, nu = 1, ny = 1, nw = 1;
    int N = 20;  /* number of time steps */
    double A = 2.0;   /* tank area [m^2] */
    double dt = 1.0;  /* time step [s] */
    double Q_val = 0.01;  /* process noise variance */
    double R_val = 0.09;  /* measurement noise variance (sigma=0.3) */

    printf("=============================================================\n");
    printf("  Example 3: Dynamic DR - Tank Level Kalman Filter\n");
    printf("=============================================================\n\n");

    /* Create dynamic model */
    dr_dyn_model_t *model = dr_dyn_model_create(nx, nu, ny, nw);
    if (!model) { printf("Failed to create model\n"); return 1; }

    /* State transition: F = [1] (simple integrator with input) */
    double F[1] = {1.0};
    dr_dyn_model_set_F(model, F);

    /* Input matrix: B = [dt/A] */
    model->B[0] = dt / A;

    /* Measurement matrix: H = [1] */
    double H[1] = {1.0};
    dr_dyn_model_set_H(model, H);

    /* Noise covariances */
    double Q_arr[1] = {Q_val};
    double R_arr[1] = {R_val};
    dr_dyn_model_set_noise(model, Q_arr, R_arr);

    /* True state simulation */
    double x_true[N+1];
    double u_hist[N];
    double y_meas[N];
    double inlet_flow = 0.5;  /* constant inlet [m^3/s] */
    double outlet_flow = 0.3; /* constant outlet [m^3/s] */

    x_true[0] = 2.0;  /* initial level [m] */
    printf("Simulating tank dynamics (dt=%.1fs, A=%.1f m^2):\n", dt, A);
    printf("  Inlet=%.2f m^3/s, Outlet=%.2f m^3/s\n\n", inlet_flow, outlet_flow);

    /* Generate true state and noisy measurements */
    for (k = 0; k < N; k++) {
        u_hist[k] = inlet_flow;
        /* Process noise */
        double w = 0.0;  /* no process noise in simulation */
        x_true[k+1] = x_true[k] + (inlet_flow - outlet_flow) * dt / A + w;
        /* Measurement with noise */
        double v = 0.3 * ((double)rand() / RAND_MAX - 0.5) * 2.0;
        y_meas[k] = x_true[k] + v;
    }

    /* Initialize Kalman filter */
    dr_kf_state_t kf_hist[N];
    double x0[1] = {x_true[0] + 0.2};  /* initial guess (biased) */
    double P0[1] = {1.0};              /* initial uncertainty */
    dr_kf_init(&kf_hist[0], nx, ny, x0, P0);

    printf("Step  TrueState  Measured   Filtered   Residual\n");
    printf("-----------------------------------------------------\n");

    /* Run filter */
    for (k = 0; k < N; k++) {
        /* Predict */
        dr_kf_predict(&kf_hist[k], model, &u_hist[k]);
        /* Update with measurement */
        dr_kf_update(&kf_hist[k], model, &y_meas[k]);

        printf("%4d  %9.3f  %9.3f  %9.3f  %9.3f\n",
               k, x_true[k], y_meas[k], kf_hist[k].x_hat[0],
               kf_hist[k].x_hat[0] - x_true[k]);

        /* Copy to next state for prediction chain */
        if (k < N - 1) {
            /* Copy current kf state as initial for next step */
            double x_copy[1] = {kf_hist[k].x_hat[0]};
            double P_copy[1] = {kf_hist[k].P[0]};
            dr_kf_init(&kf_hist[k+1], nx, ny, x_copy, P_copy);
        }
    }

    /* RTS smoother */
    printf("\nRTS Smoother results:\n");
    printf("Step  TrueState  Filtered   Smoothed\n");
    printf("-----------------------------------------\n");

    double *x_smooth = (double *)malloc((size_t)N * (size_t)nx * sizeof(double));
    double *P_smooth = (double *)malloc((size_t)N * (size_t)nx * (size_t)nx * sizeof(double));
    if (x_smooth && P_smooth) {
        dr_kf_smooth_rts(kf_hist, model, N, x_smooth, P_smooth);
        for (k = 0; k < N; k++) {
            printf("%4d  %9.3f  %9.3f  %9.3f\n",
                   k, x_true[k], kf_hist[k].x_hat[0], x_smooth[k]);
        }
    }

    /* Compute RMSE */
    double rmse_filtered = 0.0, rmse_smoothed = 0.0;
    for (k = 0; k < N; k++) {
        double err_f = kf_hist[k].x_hat[0] - x_true[k];
        rmse_filtered += err_f * err_f;
        if (x_smooth) {
            double err_s = x_smooth[k] - x_true[k];
            rmse_smoothed += err_s * err_s;
        }
    }
    rmse_filtered = sqrt(rmse_filtered / N);
    rmse_smoothed = sqrt(rmse_smoothed / N);

    printf("\nRMSE: Filtered=%.4f m, Smoothed=%.4f m\n",
           rmse_filtered, rmse_smoothed);

    /* Steady-state Kalman gain */
    double K_ss[1], P_ss[1];
    dr_kf_steady_gain(model, K_ss, P_ss, 100, 1e-6);
    printf("Steady-state Kalman gain: %.4f\n", K_ss[0]);
    printf("Steady-state error covariance: %.4f\n", P_ss[0]);

    /* Cleanup */
    free(x_smooth); free(P_smooth);
    for (k = 0; k < N; k++) dr_kf_free(&kf_hist[k]);
    dr_dyn_model_free(model);

    printf("\nExample 3 completed successfully.\n");
    return 0;
}
