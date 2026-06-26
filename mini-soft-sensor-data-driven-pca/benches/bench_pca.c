/**
 * bench_pca.c — Performance benchmarks for PCA operations
 * Measures: matrix multiplication, covariance, Jacobi eigen, NIPALS
 */
#include "pca_core.h"
#include "pca_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double time_diff_sec(clock_t start, clock_t end)
{
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main(void)
{
    size_t M = 20, N = 200;
    pca_matrix *X, *cov;
    pca_model *model;
    clock_t t0, t1;

    printf("=== PCA Benchmarks (M=%zu, N=%zu) ===\n\n", M, N);

    /* Generate random data */
    X = pca_matrix_alloc(N, M);
    {
        size_t i, j;
        for (i = 0; i < N; i++)
            for (j = 0; j < M; j++)
                X->data[i * M + j] = (double)rand() / RAND_MAX * 10.0;
    }

    /* Benchmark: covariance */
    t0 = clock();
    double mean[M];
    pca_center_columns(X, mean);
    cov = pca_compute_covariance(X);
    t1 = clock();
    printf("Covariance (N=%zu, M=%zu): %.4f sec\n", N, M, time_diff_sec(t0, t1));
    pca_matrix_free(cov);

    /* Benchmark: Jacobi eigen */
    {
        pca_matrix *X2 = pca_matrix_alloc(N, M);
        size_t i, j;
        for (i = 0; i < N; i++)
            for (j = 0; j < M; j++)
                X2->data[i * M + j] = (double)rand() / RAND_MAX * 10.0;
        pca_center_columns(X2, mean);
        pca_matrix *cov2 = pca_compute_covariance(X2);
        t0 = clock();
        double eig[M];
        pca_matrix *V = pca_matrix_alloc(M, M);
        pca_jacobi_eigen(cov2, eig, V, 50, 1e-10);
        t1 = clock();
        printf("Jacobi eigen (M=%zu): %.4f sec\n", M, time_diff_sec(t0, t1));
        pca_matrix_free(V); pca_matrix_free(cov2); pca_matrix_free(X2);
    }

    /* Benchmark: full PCA */
    {
        pca_matrix *X3 = pca_matrix_alloc(N, M);
        size_t i, j;
        for (i = 0; i < N; i++)
            for (j = 0; j < M; j++)
                X3->data[i * M + j] = (double)rand() / RAND_MAX * 10.0;
        model = pca_model_alloc(M);
        t0 = clock();
        pca_fit_full(X3, model, 50, 1e-10);
        t1 = clock();
        printf("Full PCA fit (N=%zu, M=%zu): %.4f sec\n", N, M, time_diff_sec(t0, t1));
        printf("  Eigenvalues (top 5): ");
        {
            size_t k;
            for (k = 0; k < 5 && k < M; k++)
                printf("%.3f ", model->eigenvalues[k]);
            printf("\n");
        }
        printf("  Cum. variance (3 PCs): %.1f%%\n", model->cum_var[2]*100);
        pca_model_free(model);
        pca_matrix_free(X3);
    }

    pca_matrix_free(X);
    printf("\n=== Benchmarks complete ===\n");
    return 0;
}
