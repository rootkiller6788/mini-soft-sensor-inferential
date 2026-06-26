/**
 * @file bench_quality_performance.c
 * @brief Performance benchmark for quality estimation algorithms.
 *
 * Measures throughput of Kalman filter, RLS, and PLS prediction cycles
 * under realistic industrial loads.
 */

#include "quality_estimator_types.h"
#include "kalman_quality.h"
#include "quality_recursive_ls.h"
#include "bias_correction.h"
#include <stdio.h>
#include <time.h>

int main(void)
{
    printf("=== Quality Estimator Performance Benchmark ===\n\n");
    clock_t start, end;
    double elapsed;
    int n_iter = 100000;

    /* Benchmark 1: Kalman filter step */
    {
        kalman_filter_t kf;
        kf_alloc(&kf, 2, 4, 1);
        double A[] = {1.0,0,0,1.0}, C[] = {1.0,1.0};
        double Q[] = {0.1,0,0,0.01}, R[] = {1.0};
        double x0[] = {100.0, 0.0}, P0[] = {10.0,0,0,1.0};
        kf_set_matrices(&kf, A, NULL, C);
        kf_set_noise(&kf, Q, R);
        kf_set_initial(&kf, x0, P0);

        double u[] = {1200.0, 3.5, 105.0, 5.5};
        double y[] = {150.0};

        start = clock();
        for (int i = 0; i < n_iter; i++) {
            kf_step(&kf, u, y, NULL);
        }
        end = clock();
        elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Kalman filter (2-state): %d steps in %.3f s = %.0f steps/s\n",
               n_iter, elapsed, n_iter / elapsed);
        kf_free(&kf);
    }

    /* Benchmark 2: RLS estimator */
    {
        rls_estimator_t rls;
        rls_alloc(&rls, 5, 0.98, 100.0);
        double phi[] = {0.5, 0.3, 0.1, 0.7, 0.2};

        start = clock();
        for (int i = 0; i < n_iter; i++) {
            rls_update(&rls, phi, 10.0 + i * 0.001);
        }
        end = clock();
        elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("RLS (5-param):        %d steps in %.3f s = %.0f steps/s\n",
               n_iter, elapsed, n_iter / elapsed);
        rls_free(&rls);
    }

    /* Benchmark 3: Additive bias correction */
    {
        bias_additive_t bc;
        bias_additive_init(&bc, 0.15, -50.0, 50.0, 10.0);

        start = clock();
        for (int i = 0; i < n_iter; i++) {
            bias_additive_update(&bc, 100.0 + (i % 10) * 0.1, 100.0);
            bias_additive_correct(&bc, 100.0);
        }
        end = clock();
        elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Bias correction:      %d steps in %.3f s = %.0f steps/s\n",
               n_iter, elapsed, n_iter / elapsed);
    }

    /* Benchmark 4: Full estimator cycle */
    {
        quality_estimator_t *qest = qest_alloc();
        qest_config_t cfg;
        qest_config_init(&cfg, "Bench", "Q", "%%", 4);
        qest_configure(qest, &cfg);
        double coeff[] = {0.5, -0.3, 0.1, 0.7};
        qest_set_linear_model(qest, 100.0, coeff, 4, 0.95, 0.2);

        double inputs[] = {80.0, 42.0, 10.0, 5.0};
        qest_set_inputs(qest, inputs, 4);

        start = clock();
        for (int i = 0; i < n_iter / 10; i++) {
            qest_step(qest);
        }
        end = clock();
        elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Full estimator cycle: %d steps in %.3f s = %.0f steps/s\n",
               n_iter / 10, elapsed, (n_iter / 10) / elapsed);
        qest_free(qest);
    }

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
