/**
 * @example example_dc_motor.c
 * @brief DC Motor State Estimation using Kalman Filter
 * L6: Canonical problem — estimate velocity from noisy angle measurements
 */
#include "kalman_applications.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== DC Motor Kalman Filter Estimation ===\n");
    printf("Estimating angular velocity from noisy angle measurements.\n\n");

    DCMotorEstimator est;
    dc_motor_estimator_init(&est,
        1.0, 0.001, 0.01, 0.01, 0.001, 0.0001, 0.001, 0.0001, 0.01);

    printf("Motor parameters:\n");
    printf("  R = %.2f Ohm, L = %.4f H\n", est.resistance, est.inductance);
    printf("  Kb = %.3f, Kt = %.3f, J = %.4f kg.m^2\n",
           est.back_emf_const, est.torque_const, est.inertia);
    printf("  dt = %.4f s\n\n", est.dt);

    printf("%8s %12s %12s %12s %12s\n",
           "Step", "Voltage(V)", "Angle(rad)", "VelEst(rad/s)", "TorqueEst(Nm)");
    printf("--------------------------------------------------------------\n");

    /* Simulate motor spinning at increasing speed */
    for (int i = 0; i <= 50; i++) {
        double voltage = 12.0;
        double true_angle = 0.5 * (double)i * est.dt;
        double noisy_angle = true_angle + 0.01 * sin((double)i * 0.3);

        dc_motor_estimator_step(&est, voltage, noisy_angle);

        if (i % 5 == 0) {
            double vel = dc_motor_get_velocity(&est);
            double torque = dc_motor_get_torque(&est);
            printf("%8d %12.2f %12.4f %12.4f %12.6f\n",
                   i, voltage, noisy_angle, vel, torque);
        }
    }

    printf("\nFinal estimate:\n");
    printf("  Angle:    %.4f rad\n", est.angle_est);
    printf("  Velocity: %.4f rad/s\n", est.velocity_est);
    printf("  Torque:   %.6f N.m\n", dc_motor_get_torque(&est));
    printf("  Steps:    %u\n", est.kf.step_count);

    return 0;
}
