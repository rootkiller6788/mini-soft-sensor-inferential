/**
 * @example example_cstr_reactor.c
 * @brief CSTR Temperature Estimation using Extended Kalman Filter
 * L6: Canonical problem — infer concentration from temperature measurements
 * Reference: Seborg, Edgar, Mellichamp (2016) "Process Dynamics and Control"
 */
#include "kalman_applications.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== CSTR Reactor State Estimation (EKF) ===\n");
    printf("Inferring reactant concentration from temperature.\n\n");

    CSTREstimator est;
    cstr_estimator_init(&est,
        1.0,     /* Ca0: feed concentration (mol/L) */
        350.0,   /* T0: feed temperature (K) */
        100.0,   /* q: flow rate (L/min) */
        100.0,   /* V: volume (L) */
        7.2e10,  /* k0: pre-exponential factor (1/min) */
        8750.0,  /* E_R: activation energy / R (K) */
        200.0,   /* -dH_R: heat of reaction (K.L/mol) */
        1000.0,  /* rho*Cp: density * heat capacity */
        700.0,   /* UA: heat transfer (J/(min.K)) */
        0.1,     /* dt: sampling period (min) */
        1e-6,    /* Q_conc: process noise for concentration */
        0.01,    /* Q_temp: process noise for temperature */
        0.25);   /* R_meas: measurement noise variance */

    printf("CSTR Parameters:\n");
    printf("  Ca0=%.1f mol/L, T0=%.0f K, V=%.0f L, q=%.0f L/min\n",
           est.feed_conc, est.feed_temp, est.volume, est.flow_rate);
    printf("  E/R=%.0f K, -dH_R=%.0f, UA=%.0f J/(min.K)\n\n",
           est.activation_E_R, est.delta_H_R, est.UA);

    printf("%6s %10s %12s %12s %12s\n",
           "Step", "Tc(K)", "T_meas(K)", "T_est(K)", "Conversion");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i <= 40; i++) {
        double Tc = 320.0 + 5.0 * sin((double)i * 0.2);
        double true_temp = 380.0 + 3.0 * sin((double)i * 0.15);
        double noisy_temp = true_temp + 0.5 * ((double)((i*7)%5) - 2.0);

        cstr_estimator_step(&est, Tc, noisy_temp);

        if (i % 4 == 0) {
            double conv = cstr_get_conversion(&est);
            printf("%6d %10.1f %12.2f %12.2f %11.3f\n",
                   i, Tc, noisy_temp, est.temp_est, conv);
        }
    }

    printf("\nFinal Estimates:\n");
    printf("  Concentration: %.4f mol/L\n", est.conc_est);
    printf("  Temperature:   %.2f K\n", est.temp_est);
    printf("  Conversion:    %.2f%%\n", cstr_get_conversion(&est) * 100.0);
    printf("  EKF steps:     %u\n", est.ekf.kf.step_count);

    return 0;
}
