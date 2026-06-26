#include "matrix_ops.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-10

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)
#define CHECK(cond) do { if (!(cond)) { FAIL(#cond); return; } } while(0)
#define CHECK_CLOSE(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("FAIL: |%g - %g| > %g\n", (a), (b), (eps)); return; \
    } \
} while(0)

/* Test matrix lifecycle */
static void test_matrix_alloc_free(void) {
    TEST("matrix alloc/free");
    Matrix *A = matrix_alloc(3, 4);
    CHECK(A != NULL);
    CHECK(A->rows == 3);
    CHECK(A->cols == 4);
    CHECK(A->owns_data == 1);
    /* Should be zero-initialized */
    for (size_t i = 0; i < 3 * 4; i++)
        CHECK(A->data[i] == 0.0);
    matrix_free(A);
    PASS();
}

/* Test matrix from array */
static void test_matrix_from_array(void) {
    TEST("matrix from array");
    double data[] = {1, 2, 3, 4, 5, 6};
    Matrix *A = matrix_from_array(2, 3, data);
    CHECK(A != NULL);
    CHECK(A->data[0] == 1.0);
    CHECK(A->data[3] == 4.0);
    CHECK(A->data[5] == 6.0);
    matrix_free(A);
    PASS();
}

/* Test matrix copy */
static void test_matrix_copy(void) {
    TEST("matrix copy");
    double data[] = {1, 2, 3, 4};
    Matrix *A = matrix_from_array(2, 2, data);
    Matrix *B = matrix_copy(A);
    CHECK(B != NULL);
    CHECK(B->data[0] == 1.0);
    CHECK(B->data[3] == 4.0);
    /* Modify A, B should be independent */
    A->data[0] = 99.0;
    CHECK(B->data[0] == 1.0);
    matrix_free(A);
    matrix_free(B);
    PASS();
}

/* Test matrix fill and eye */
static void test_matrix_fill_eye(void) {
    TEST("matrix fill/eye");
    Matrix *A = matrix_alloc(3, 3);
    matrix_fill(A, 5.0);
    for (size_t i = 0; i < 9; i++) CHECK(A->data[i] == 5.0);
    matrix_eye(A);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (i == j)
                CHECK(A->data[i * 3 + j] == 1.0);
            else
                CHECK(A->data[i * 3 + j] == 0.0);
        }
    }
    matrix_free(A);
    PASS();
}

/* Test vector dot product */
static void test_vector_dot(void) {
    TEST("vector dot product");
    double d1[] = {1, 2, 3}, d2[] = {4, 5, 6};
    Vector *v1 = vector_from_array(3, d1);
    Vector *v2 = vector_from_array(3, d2);
    double dot = vector_dot(v1, v2);
    CHECK_CLOSE(dot, 1*4 + 2*5 + 3*6, EPS);
    vector_free(v1); vector_free(v2);
    PASS();
}

/* Test vector norm */
static void test_vector_norm(void) {
    TEST("vector norm L2");
    double d[] = {3, 4};
    Vector *v = vector_from_array(2, d);
    double nrm = vector_norm_l2(v);
    CHECK_CLOSE(nrm, 5.0, EPS);
    vector_free(v);
    PASS();
}

/* Test matrix multiply */
static void test_matrix_multiply(void) {
    TEST("matrix multiply");
    double da[] = {1, 2, 3, 4, 5, 6};
    double db[] = {7, 8, 9, 10, 11, 12};
    Matrix *A = matrix_from_array(2, 3, da);
    Matrix *B = matrix_from_array(3, 2, db);
    Matrix *C = matrix_multiply(A, B);
    CHECK(C != NULL);
    CHECK(C->rows == 2 && C->cols == 2);
    /* C[0][0] = 1*7 + 2*9 + 3*11 = 58 */
    CHECK_CLOSE(C->data[0], 58.0, EPS);
    /* C[0][1] = 1*8 + 2*10 + 3*12 = 64 */
    CHECK_CLOSE(C->data[1], 64.0, EPS);
    /* C[1][0] = 4*7 + 5*9 + 6*11 = 139 */
    CHECK_CLOSE(C->data[2], 139.0, EPS);
    /* C[1][1] = 4*8 + 5*10 + 6*12 = 154 */
    CHECK_CLOSE(C->data[3], 154.0, EPS);
    matrix_free(A); matrix_free(B); matrix_free(C);
    PASS();
}

/* Test transpose */
static void test_matrix_transpose(void) {
    TEST("matrix transpose");
    double da[] = {1, 2, 3, 4, 5, 6};
    Matrix *A = matrix_from_array(2, 3, da);
    Matrix *At = matrix_transpose(A);
    CHECK(At != NULL);
    CHECK(At->rows == 3 && At->cols == 2);
    CHECK_CLOSE(At->data[0], 1.0, EPS);
    CHECK_CLOSE(At->data[1], 4.0, EPS);
    CHECK_CLOSE(At->data[5], 6.0, EPS);
    matrix_free(A); matrix_free(At);
    PASS();
}

/* Test outer product */
static void test_outer_product(void) {
    TEST("outer product");
    double d1[] = {1, 2}, d2[] = {3, 4, 5};
    Vector *u = vector_from_array(2, d1);
    Vector *v = vector_from_array(3, d2);
    Matrix *C = vector_outer_product(u, v);
    CHECK(C != NULL);
    CHECK(C->rows == 2 && C->cols == 3);
    CHECK_CLOSE(C->data[0], 3.0, EPS);   /* 1*3 */
    CHECK_CLOSE(C->data[1], 4.0, EPS);   /* 1*4 */
    CHECK_CLOSE(C->data[4], 8.0, EPS);   /* 2*4 */
    vector_free(u); vector_free(v); matrix_free(C);
    PASS();
}

/* Test Cholesky decomposition */
static void test_cholesky(void) {
    TEST("Cholesky decomposition");
    /* SPD matrix: [[4, 2], [2, 3]] */
    double da[] = {4, 2, 2, 3};
    Matrix *A = matrix_from_array(2, 2, da);
    int ok = matrix_cholesky_decomp(A);
    CHECK(ok == 0);
    /* L should be [[2, 0], [1, sqrt(2)]] */
    CHECK_CLOSE(A->data[0], 2.0, EPS);
    CHECK_CLOSE(A->data[1], 0.0, EPS);
    CHECK_CLOSE(A->data[2], 1.0, EPS);
    CHECK_CLOSE(A->data[3], sqrt(2.0), EPS);
    matrix_free(A);
    PASS();
}

/* Test mean and variance */
static void test_mean_variance(void) {
    TEST("vector mean/variance");
    double d[] = {1, 2, 3, 4, 5};
    Vector *v = vector_from_array(5, d);
    CHECK_CLOSE(vector_mean(v), 3.0, EPS);
    CHECK_CLOSE(vector_variance(v), 2.5, EPS);
    CHECK_CLOSE(vector_stddev(v), sqrt(2.5), EPS);
    vector_free(v);
    PASS();
}

/* Test column means */
static void test_column_means(void) {
    TEST("column means");
    double d[] = {1, 2, 3, 4, 5, 6};
    Matrix *A = matrix_from_array(2, 3, d);
    Vector *means = matrix_column_means(A);
    CHECK(means != NULL);
    CHECK(means->len == 3);
    CHECK_CLOSE(means->data[0], 2.5, EPS);
    CHECK_CLOSE(means->data[1], 3.5, EPS);
    CHECK_CLOSE(means->data[2], 4.5, EPS);
    matrix_free(A); vector_free(means);
    PASS();
}

/* Test center columns */
static void test_center_columns(void) {
    TEST("center columns");
    double d[] = {1, 3, 2, 4};
    Matrix *A = matrix_from_array(2, 2, d);
    double md[] = {2.0, 3.0};
    Vector *means = vector_from_array(2, md);
    matrix_center_columns(A, means);
    CHECK_CLOSE(A->data[0], -1.0, EPS);
    CHECK_CLOSE(A->data[1], 0.0, EPS);
    CHECK_CLOSE(A->data[2], 0.0, EPS);
    CHECK_CLOSE(A->data[3], 1.0, EPS);
    matrix_free(A); vector_free(means);
    PASS();
}

/* Test Cholesky on non-SPD matrix */
static void test_cholesky_non_spd(void) {
    TEST("Cholesky non-SPD");
    double d[] = {-1, 0, 0, 1};
    Matrix *A = matrix_from_array(2, 2, d);
    int ok = matrix_cholesky_decomp(A);
    CHECK(ok == -1);
    matrix_free(A);
    PASS();
}

/* Test matrix view */
static void test_matrix_view(void) {
    TEST("matrix view");
    double d[] = {1, 2, 3, 4, 5, 6};
    Matrix *A = matrix_from_array(2, 3, d);
    Matrix *V = matrix_view(A, 0, 1, 2, 2);
    CHECK(V != NULL);
    CHECK(V->owns_data == 0);
    CHECK_CLOSE(V->data[0], 2.0, EPS);
    CHECK_CLOSE(V->data[2], 4.0, EPS);
    matrix_free(V); matrix_free(A);
    PASS();
}

int main(void) {
    printf("=== Matrix Operations Tests ===\n");
    test_matrix_alloc_free();
    test_matrix_from_array();
    test_matrix_copy();
    test_matrix_fill_eye();
    test_vector_dot();
    test_vector_norm();
    test_matrix_multiply();
    test_matrix_transpose();
    test_outer_product();
    test_cholesky();
    test_mean_variance();
    test_column_means();
    test_center_columns();
    test_cholesky_non_spd();
    test_matrix_view();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
