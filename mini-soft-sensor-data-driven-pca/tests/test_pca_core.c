/**
 * test_pca_core.c — Tests for PCA Core Matrix Operations
 *
 * Tests: matrix alloc/free, get/set, copy, centering, scaling,
 *        covariance computation, variance explained, Kaiser rule,
 *        cumulative variance rule.
 */
#include "pca_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %s... ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define CHECK_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (%zu vs %zu)\n", msg, (size_t)(a), (size_t)(b)); tests_failed++; return; } } while(0)
#define CHECK_DOUBLE(a, b, tol, msg) do { if (fabs((a)-(b)) > (tol)) { printf("FAIL: %s (%.6f vs %.6f)\n", msg, a, b); tests_failed++; return; } } while(0)

static void test_matrix_alloc_free(void)
{
    TEST("matrix alloc/free");
    pca_matrix *m = pca_matrix_alloc(3, 4);
    CHECK(m != NULL, "alloc failed");
    CHECK_EQ(m->rows, 3, "rows");
    CHECK_EQ(m->cols, 4, "cols");
    pca_matrix_free(m);
    PASS();
}

static void test_matrix_get_set(void)
{
    TEST("matrix get/set");
    pca_matrix *m = pca_matrix_alloc(2, 3);
    CHECK(m != NULL, "alloc");
    pca_matrix_set(m, 0, 0, 1.5);
    pca_matrix_set(m, 1, 2, -3.0);
    CHECK_DOUBLE(pca_matrix_get(m, 0, 0), 1.5, 1e-12, "get(0,0)");
    CHECK_DOUBLE(pca_matrix_get(m, 1, 2), -3.0, 1e-12, "get(1,2)");
    CHECK_DOUBLE(pca_matrix_get(m, 0, 1), 0.0, 1e-12, "zero init");
    pca_matrix_free(m);
    PASS();
}

static void test_matrix_copy(void)
{
    TEST("matrix copy");
    pca_matrix *a = pca_matrix_alloc(2, 2);
    pca_matrix_set(a, 0, 0, 1.0);
    pca_matrix_set(a, 0, 1, 2.0);
    pca_matrix_set(a, 1, 0, 3.0);
    pca_matrix_set(a, 1, 1, 4.0);
    pca_matrix *b = pca_matrix_copy(a);
    CHECK(b != NULL, "copy failed");
    CHECK_DOUBLE(b->data[0], 1.0, 1e-12, "copy[0]");
    CHECK_DOUBLE(b->data[3], 4.0, 1e-12, "copy[3]");
    pca_matrix_set(a, 1, 1, 99.0);
    CHECK_DOUBLE(b->data[3], 4.0, 1e-12, "deep copy independence");
    pca_matrix_free(a); pca_matrix_free(b);
    PASS();
}

static void test_center_columns(void)
{
    TEST("center columns");
    pca_matrix *X = pca_matrix_alloc(3, 2);
    double mean[2];
    /* X = [[1,4],[2,5],[3,6]] */
    pca_matrix_set(X, 0, 0, 1); pca_matrix_set(X, 0, 1, 4);
    pca_matrix_set(X, 1, 0, 2); pca_matrix_set(X, 1, 1, 5);
    pca_matrix_set(X, 2, 0, 3); pca_matrix_set(X, 2, 1, 6);
    pca_center_columns(X, mean);
    CHECK_DOUBLE(mean[0], 2.0, 1e-12, "mean[0]=2");
    CHECK_DOUBLE(mean[1], 5.0, 1e-12, "mean[1]=5");
    CHECK_DOUBLE(X->data[0], -1.0, 1e-12, "centered[0]");
    CHECK_DOUBLE(X->data[5], 1.0, 1e-12, "centered[5]");
    pca_matrix_free(X);
    PASS();
}

static void test_covariance(void)
{
    TEST("covariance matrix");
    pca_matrix *X = pca_matrix_alloc(4, 2);
    double mean[2];
    /* Simple 2-variable dataset */
    pca_matrix_set(X, 0, 0, 1); pca_matrix_set(X, 0, 1, 2);
    pca_matrix_set(X, 1, 0, 2); pca_matrix_set(X, 1, 1, 3);
    pca_matrix_set(X, 2, 0, 3); pca_matrix_set(X, 2, 1, 4);
    pca_matrix_set(X, 3, 0, 4); pca_matrix_set(X, 3, 1, 5);
    pca_center_columns(X, mean);
    pca_matrix *cov = pca_compute_covariance(X);
    CHECK(cov != NULL, "cov null");
    /* X=[1,2;2,3;3,4;4,5] -> centered: [-1.5,-1.5;-0.5,-0.5;0.5,0.5;1.5,1.5]
     * Cov = [[1.6667, 1.6667],[1.6667, 1.6667]] (approx)
     * Actually: sum(x^2)=5, sum(y^2)=5, sum(xy)=5; cov=5/3=1.6667 */
    CHECK_DOUBLE(cov->data[0], 1.666667, 0.001, "cov[0,0]");
    CHECK_DOUBLE(cov->data[1], cov->data[2], 1e-12, "symmetry");
    CHECK_DOUBLE(cov->data[3], 1.666667, 0.001, "cov[1,1]");
    pca_matrix_free(cov); pca_matrix_free(X);
    PASS();
}

static void test_variance_explained(void)
{
    TEST("variance explained");
    double eig[] = {3.0, 1.0, 0.5, 0.5};
    double ve[4], cv[4];
    pca_compute_variance_explained(eig, 4, ve, cv);
    CHECK_DOUBLE(ve[0], 0.6, 1e-12, "var_expl[0]=3/5");
    CHECK_DOUBLE(ve[1], 0.2, 1e-12, "var_expl[1]=1/5");
    CHECK_DOUBLE(cv[1], 0.8, 1e-12, "cum_var[1]=0.8");
    CHECK_DOUBLE(cv[3], 1.0, 1e-12, "cum_var[3]=1.0");
    PASS();
}

static void test_kaiser_rule(void)
{
    TEST("Kaiser rule");
    /* For correlation PCA, eigenvalues sum to M=5. avg=1. Keep >1 */
    double eig1[] = {2.5, 1.5, 0.5, 0.3, 0.2};
    size_t k1 = pca_kaiser_rule(eig1, 5);
    CHECK_EQ(k1, 2, "kaiser: 2 PCs");
    /* All equal -> keep none? Actually all equal to 1, none > avg=1 */
    double eig2[] = {1.0, 1.0, 1.0, 1.0};
    size_t k2 = pca_kaiser_rule(eig2, 4);
    CHECK_EQ(k2, 1, "kaiser: at least 1 PC");
    PASS();
}

static void test_cumvar_rule(void)
{
    TEST("cumulative variance rule");
    double cv[] = {0.5, 0.75, 0.90, 0.98, 1.0};
    CHECK_EQ(pca_cumvar_rule(cv, 5, 0.85), 3, "85%% -> 3 PCs");
    CHECK_EQ(pca_cumvar_rule(cv, 5, 0.95), 4, "95%% -> 4 PCs");
    CHECK_EQ(pca_cumvar_rule(cv, 5, 1.00), 5, "100%% -> 5 PCs");
    PASS();
}

static void test_model_lifecycle(void)
{
    TEST("model alloc/free");
    pca_model *m = pca_model_alloc(5);
    CHECK(m != NULL, "alloc");
    CHECK_EQ(m->n_vars, 5, "n_vars");
    CHECK(m->eigenvalues != NULL, "eig");
    CHECK(m->loadings != NULL, "loadings");
    pca_model_free(m);
    PASS();
}

int main(void)
{
    printf("=== PCA Core Tests ===\n");
    test_matrix_alloc_free();
    test_matrix_get_set();
    test_matrix_copy();
    test_center_columns();
    test_covariance();
    test_variance_explained();
    test_kaiser_rule();
    test_cumvar_rule();
    test_model_lifecycle();
    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
