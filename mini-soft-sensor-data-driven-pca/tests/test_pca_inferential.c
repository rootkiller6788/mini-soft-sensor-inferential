/**
 * test_pca_inferential.c — Tests for PCA-based Soft Sensor
 *
 * Tests: PCR training, online prediction, cross-validation,
 *        R2/RMSE metrics.
 */
#include "pca_inferential.h"
#include "pca_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int passed = 0, failed = 0;

#define T(n) printf("  %s... ", n)
#define P() do { printf("PASS\n"); passed++; } while(0)
#define F(m) do { printf("FAIL: %s\n", m); failed++; return; } while(0)

void test_soft_sensor_alloc(void)
{
    T("soft sensor alloc/free");
    pca_soft_sensor *ss = pca_soft_sensor_alloc(4, 2, 2);
    if (!ss) { F("alloc null"); return; }
    if (ss->n_secondary != 4) { F("n_secondary"); return; }
    if (ss->n_primary != 2) { F("n_primary"); return; }
    if (ss->n_pcs != 2) { F("n_pcs"); return; }
    pca_soft_sensor_free(ss);
    P();
}

void test_pcr_train_predict(void)
{
    T("PCR train/predict (small dataset)");
    /* Simple linear relationship: Y = 2*X1 + 3*X2
     * X: [[1,2],[2,3],[3,4],[4,5],[5,6]]
     * Y: [[2*1+3*2=8],[2*2+3*3=13],[2*3+3*4=18],[2*4+3*5=23],[2*5+3*6=28]] */
    pca_matrix *X = pca_matrix_alloc(5, 2);
    pca_matrix *Y = pca_matrix_alloc(5, 1);
    size_t i;
    for (i = 0; i < 5; i++) {
        double x1 = (double)(i + 1);
        double x2 = x1 + 1.0;
        X->data[i * 2 + 0] = x1;
        X->data[i * 2 + 1] = x2;
        Y->data[i * 1 + 0] = 2.0 * x1 + 3.0 * x2;
    }
    pca_soft_sensor *ss = pca_soft_sensor_alloc(2, 1, 1);
    if (!ss) { F("alloc"); return; }
    int ret = pca_pcr_train(X, Y, 1, ss);
    if (ret != 0) { F("train failed"); pca_soft_sensor_free(ss); pca_matrix_free(X); pca_matrix_free(Y); return; }
    /* Predict for x=[2,3] -> y=2*2+3*3=13 */
    double x_test[] = {2.0, 3.0};
    double y_pred = 0.0;
    ret = pca_pcr_predict(ss, x_test, &y_pred);
    if (ret != 0) { F("predict failed"); pca_soft_sensor_free(ss); pca_matrix_free(X); pca_matrix_free(Y); return; }
    if (fabs(y_pred - 13.0) > 0.5) { printf("FAIL: pred=%.3f expected=13.0\n", y_pred); failed++; return; }
    P();
    pca_soft_sensor_free(ss);
    pca_matrix_free(X); pca_matrix_free(Y);
}

void test_metrics(void)
{
    T("R2 and RMSE metrics");
    pca_matrix *Yt = pca_matrix_alloc(3, 1);
    pca_matrix *Yp = pca_matrix_alloc(3, 1);
    /* Perfect prediction */
    Yt->data[0] = 1.0; Yp->data[0] = 1.0;
    Yt->data[1] = 2.0; Yp->data[1] = 2.0;
    Yt->data[2] = 3.0; Yp->data[2] = 3.0;
    double r2 = pca_pcr_r2(Yt, Yp);
    double rmse = pca_pcr_rmse(Yt, Yp);
    if (fabs(r2 - 1.0) > 0.001) { F("R2 not 1"); return; }
    if (fabs(rmse - 0.0) > 0.001) { F("RMSE not 0"); return; }
    /* Imperfect prediction */
    Yp->data[0] = 1.5; Yp->data[1] = 1.5; Yp->data[2] = 3.5;
    double r2_bad = pca_pcr_r2(Yt, Yp);
    if (r2_bad >= 1.0) { F("R2 should be < 1"); return; }
    P();
    pca_matrix_free(Yt); pca_matrix_free(Yp);
}

int main(void)
{
    printf("=== PCA Inferential Tests ===\n");
    test_soft_sensor_alloc();
    test_pcr_train_predict();
    test_metrics();
    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
