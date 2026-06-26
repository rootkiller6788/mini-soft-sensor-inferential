/**
 * test_pca_monitoring.c — Tests for PCA-based Process Monitoring
 *
 * Tests: T2 statistic, SPE statistic, contribution plots,
 *        combined index, fault detection result.
 */
#include "pca_monitoring.h"
#include "pca_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int passed = 0, failed = 0;

#define T(name) printf("  %s... ", name)
#define P() do { printf("PASS\n"); passed++; } while(0)
#define F(m) do { printf("FAIL: %s\n", m); failed++; return; } while(0)
#define CEQ(a,b,m) if((a)!=(b)){printf("FAIL: %s\n",m);failed++;return;}

void test_t2_statistic(void)
{
    T("T2 statistic");
    pca_matrix *load = pca_matrix_alloc(3, 3);
    double eig[] = {3.0, 1.0, 0.2};
    double x[] = {1.0, 0.5, 0.1};
    /* Set loadings = identity for simple test */
    pca_matrix_set(load, 0, 0, 1); pca_matrix_set(load, 1, 1, 1);
    pca_matrix_set(load, 2, 2, 1);
    /* t1=1.0, t2=0.5, t3=0.1
     * T2 = 1^2/3 + 0.5^2/1 = 0.333 + 0.25 = 0.5833 */
    double t2 = pca_t2_statistic(x, load, eig, 2);
    if (fabs(t2 - 0.583333) > 0.001) { F("T2 value"); return; }
    P();
    pca_matrix_free(load);
}

void test_spe_statistic(void)
{
    T("SPE statistic");
    pca_matrix *load = pca_matrix_alloc(3, 3);
    double x[] = {1.0, 0.5, 0.3};
    pca_matrix_set(load, 0, 0, 1); pca_matrix_set(load, 1, 1, 1);
    pca_matrix_set(load, 2, 2, 1);
    /* With identity loadings and 2 PCs: x_hat = [1,0.5,0], e=[0,0,0.3], SPE=0.09 */
    double spe = pca_spe_statistic(x, load, 2, 3);
    if (fabs(spe - 0.09) > 0.001) { F("SPE value"); return; }
    P();
    pca_matrix_free(load);
}

void test_t2_threshold(void)
{
    T("T2 threshold");
    /* With 2 PCs, 100 samples, alpha=0.05 */
    double lim = pca_t2_threshold_f(2, 100, 0.05);
    if (lim <= 0.0) { F("T2 threshold <= 0"); return; }
    double lim_chi2 = pca_t2_threshold_chi2(2, 0.05);
    if (lim_chi2 <= 0.0) { F("chi2 threshold"); return; }
    P();
}

void test_contributions(void)
{
    T("contribution plots");
    pca_matrix *load = pca_matrix_alloc(3, 3);
    double eig[] = {3.0, 1.0, 0.2};
    double x[] = {1.0, 0.5, 0.1};
    double contribs_t2[3], contribs_spe[3];
    pca_matrix_set(load, 0, 0, 1); pca_matrix_set(load, 1, 1, 1);
    pca_matrix_set(load, 2, 2, 1);
    pca_t2_contributions(x, load, eig, 2, 3, contribs_t2);
    pca_spe_contributions(x, load, 2, 3, contribs_spe);
    if (contribs_spe[2] < 0.0) { F("SPE contrib negative"); return; }
    P();
    pca_matrix_free(load);
}

void test_fault_result(void)
{
    T("fault detection result");
    pca_matrix *load = pca_matrix_alloc(3, 3);
    double eig[] = {3.0, 1.0, 0.2};
    double x[] = {1.0, 0.5, 0.1};
    pca_matrix_set(load, 0, 0, 1); pca_matrix_set(load, 1, 1, 1);
    pca_matrix_set(load, 2, 2, 1);
    pca_fault_result *r = pca_monitor_observation(x, load, eig, 2, 3, 100, 0.05);
    if (!r) { F("null result"); return; }
    CEQ(r->n_vars, 3, "n_vars");
    pca_fault_result_print(r);
    pca_fault_result_free(r);
    P();
    pca_matrix_free(load);
}

int main(void)
{
    printf("=== PCA Monitoring Tests ===\n");
    test_t2_statistic();
    test_spe_statistic();
    test_t2_threshold();
    test_contributions();
    test_fault_result();
    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
