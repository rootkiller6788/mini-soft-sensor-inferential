/**
 * @file example_polymer_melt_index.c
 * @brief L6-L7 Example: Polymer Melt Index inference.
 *
 * Melt Index (MI) is a critical quality parameter in polymer production.
 * It is measured by lab rheometer every 2-4 hours. Inferential estimation
 * uses reactor temperature, pressure, and catalyst feed rate to predict
 * MI continuously, enabling real-time quality control.
 *
 * Reference: Ohshima, Tanigaki (2000) "Quality control of polymer production
 *            processes" — J. Process Control, 10(2-3).
 */

#include "quality_estimator_types.h"
#include "bias_correction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void)
{
    printf("=== Example: Polymer Melt Index Inference ===\n\n");

    /* Polymer MI model: log(MI) is approximately linear in reactor conditions.
     * MI = exp(beta_0 + beta_1*T + beta_2*P + beta_3*C2 + beta_4*H2)
     *
     * For this example, we use a linearized model around nominal conditions:
     * MI = 2.5 + 0.08*(T-80) - 0.02*(P-30) + 0.15*(H2/C2 - 0.1)
     *
     * where: T = reactor temp (°C), P = reactor pressure (bar),
     *        H2/C2 = hydrogen-to-ethylene ratio
     */

    /* Configure the estimator */
    qest_config_t cfg;
    qest_config_init(&cfg, "MeltIndex", "QI_MI", "g/10min", 3);
    cfg.model_type = QMODEL_DATA_DRIVEN;
    cfg.bias_strategy = BIAS_ADDITIVE;
    cfg.fast_sample_period = 30.0;
    cfg.lab_sample_period  = 7200.0;  /* Lab every 2 hours */
    cfg.bias_filter_gain   = 0.20;
    cfg.outlier_sigma      = 3.5;

    quality_estimator_t *qest = qest_alloc();
    qest_configure(qest, &cfg);

    double coeff[] = {0.08, -0.02, 0.15};
    qest_set_linear_model(qest, 2.5, coeff, 3, 0.92, 0.08);

    printf("Estimator: %s\n", cfg.estimator_name);
    printf("Quality: MI [%s]\n", cfg.units);
    printf("Bias strategy: Additive (EWMA α=0.20)\n\n");

    /* Simulate 12 hours of operation */
    printf("Hour |  T(C) | P(bar) | H2/C2 | MI_pred | MI_lab | MI_corrected\n");
    printf("-----|-------|--------|-------|---------|--------|-------------\n");

    /* Process data: hour-by-hour averages */
    double T_data[]   = {80, 81, 82, 83, 82, 81, 80, 80, 81, 82, 83, 82};
    double P_data[]   = {30, 30, 31, 31, 30, 30, 29, 30, 30, 31, 31, 30};
    double H2C2_data[] = {0.10, 0.11, 0.12, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12, 0.11};

    /* Lab samples at hours 3, 7, 11 */
    double lab_MI[] = {2.75, 2.82, 2.95};

    int lab_idx = 0;
    for (int h = 0; h < 12; h++) {
        double inputs[] = {T_data[h], P_data[h], H2C2_data[h]};
        qest_set_inputs(qest, inputs, 3);

        const quality_estimate_t *est = qest_step(qest);

        /* Lab sample at hours 3, 7, 11 */
        double y_lab = -1;
        if ((h == 3 || h == 7 || h == 11) && lab_idx < 3) {
            y_lab = lab_MI[lab_idx];
            lab_sample_t lab;
            memset(&lab, 0, sizeof(lab));
            lab.measured_value = y_lab;
            lab.lab_stddev = 0.03;
            lab.quality_flag = LAB_QUALITY_GOOD;
            qest_process_lab(qest, &lab);
            lab_idx++;
        }

        printf("  %2d | %5.0f |  %5.0f | %5.2f |  %6.3f | %6.3f |   %6.3f\n",
               h, T_data[h], P_data[h], H2C2_data[h],
               est->predicted_value,
               y_lab > 0 ? y_lab : est->bias_corrected_value,
               est->bias_corrected_value);
    }

    qest_performance_t perf;
    qest_get_performance(qest, &perf);
    printf("\nPerformance: RMSE=%.4f, MAE=%.4f, BiasUpdates=%lld\n",
           perf.rmse, perf.mae, (long long)perf.n_bias_updates);

    qest_free(qest);
    printf("\n=== Example complete ===\n");
    return 0;
}
