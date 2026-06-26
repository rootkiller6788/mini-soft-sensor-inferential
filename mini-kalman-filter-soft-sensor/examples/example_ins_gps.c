/**
 * @example example_ins_gps.c
 * @brief GPS/INS Navigation using Kalman Filter
 * L6: Canonical problem — loosely-coupled INS/GPS integration
 * Reference: Grewal & Andrews (2014) "Kalman Filtering: Theory and Practice"
 */
#include "kalman_applications.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== INS/GPS Navigation Filter ===\n");
    printf("9-state loosely-coupled integration.\n\n");

    INSGPSNavigator nav;
    ins_gps_init(&nav,
        0.01,           /* dt: 100 Hz IMU */
        1e-4,           /* accel noise PSD */
        1e-8,           /* accel bias random walk */
        5.0,            /* GPS std (m) */
        10.0,           /* initial pos std */
        1.0);           /* initial vel std */

    printf("Sensor Specs:\n");
    printf("  IMU: 100 Hz, accel noise sigma = %.4f m/s^2\n",
           sqrt(nav.accel_noise_psd));
    printf("  GPS: 1 Hz, position sigma = %.1f m\n\n", nav.gps_pos_std);

    printf("%8s %12s %12s %12s %10s\n",
           "Time(s)", "PosX(m)", "PosY(m)", "PosZ(m)", "CEP(m)");
    printf("----------------------------------------------------------\n");

    double true_vx = 10.0, true_vy = 5.0, true_vz = 0.0;
    double pos_x = 0, pos_y = 0, pos_z = 100.0;

    for (int t = 0; t <= 1000; t++) {
        /* IMU update at 100 Hz */
        double ax = 0.0 + 0.01 * sin((double)t * 0.1);
        double ay = 0.0 + 0.01 * cos((double)t * 0.1);
        double az = 9.81;
        ins_gps_predict_imu(&nav, ax, ay, az);

        /* GPS update at 1 Hz */
        if (t % 100 == 0) {
            pos_x += true_vx * 1.0;
            pos_y += true_vy * 1.0;
            double gps_x = pos_x + 3.0 * ((double)(t%7) - 3.0);
            double gps_y = pos_y + 3.0 * ((double)((t*3)%7) - 3.0);
            double gps_z = pos_z + 3.0 * ((double)((t*5)%7) - 3.0);

            ins_gps_update_gps(&nav, gps_x, gps_y, gps_z);

            double px, py, pz;
            ins_gps_get_position(&nav, &px, &py, &pz);

            printf("%8.1f %12.2f %12.2f %12.2f %10.2f\n",
                   (double)t * 0.01, px, py, pz, nav.cep);
        }
    }

    printf("\nFinal Navigation Solution:\n");
    printf("  Position: (%.2f, %.2f, %.2f) m\n",
           nav.pos_x, nav.pos_y, nav.pos_z);
    printf("  Velocity: (%.2f, %.2f, %.2f) m/s\n",
           nav.vel_x, nav.vel_y, nav.vel_z);
    printf("  Accel Bias: (%.6f, %.6f, %.6f) m/s^2\n",
           nav.accel_bias_x, nav.accel_bias_y, nav.accel_bias_z);
    printf("  CEP: %.2f m\n", nav.cep);
    printf("  IMU count: %u, GPS count: %u\n",
           nav.imu_count, nav.gps_count);

    return 0;
}
