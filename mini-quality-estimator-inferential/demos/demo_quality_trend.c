/**
 * @file demo_quality_trend.c
 * @brief Demo: Generate CSV output of quality estimation trends for visualization.
 *
 * Outputs CSV data that can be plotted with gnuplot, matplotlib, or Excel
 * showing the relationship between process variables and inferred quality.
 */

#include "quality_estimator_types.h"
#include "bias_correction.h"
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void)
{
    printf("timestamp,T_top,Reflux,dP,y_model,bias,y_corrected,lower_95,upper_95\n");

    /* Simulate a sinusoidal perturbation in tray temperature with bias drift */
    bias_additive_t bc;
    bias_additive_init(&bc, 0.15, -20.0, 20.0, 5.0);

    double intercept = 98.5;
    double coeff[] = {-0.12, 0.05, 0.03};

    for (int minute = 0; minute < 480; minute++) {
        double t_hours = minute / 60.0;

        /* Process variables with sinusoidal variation */
        double T_top  = 80.0 + 5.0 * sin(2.0 * M_PI * t_hours / 8.0);
        double Reflux = 42.0 + 3.0 * cos(2.0 * M_PI * t_hours / 6.0);
        double DP     = 10.0 + 2.0 * sin(2.0 * M_PI * t_hours / 12.0);

        /* Model prediction */
        double y_model = intercept + coeff[0] * T_top + coeff[1] * Reflux + coeff[2] * DP;

        /* Bias-corrected estimate */
        double y_corrected = bias_additive_correct(&bc, y_model);

        /* Simulate lab sample every 240 minutes */
        if (minute > 0 && minute % 240 == 0) {
            double y_lab = y_model + 1.5 + 0.5 * sin(t_hours / 4.0);
            bias_additive_update(&bc, y_lab, y_model);
            y_corrected = bias_additive_correct(&bc, y_model);
        }

        /* Confidence bounds (±2 sigma) */
        double sigma = 0.5;
        double lower = y_corrected - 2.0 * sigma;
        double upper = y_corrected + 2.0 * sigma;

        printf("%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
               t_hours, T_top, Reflux, DP, y_model, bc.bias,
               y_corrected, lower, upper);
    }
    return 0;
}
