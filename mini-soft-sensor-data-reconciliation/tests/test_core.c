#include "dr_core.h"
#include "dr_matrix.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

static int tests_run = 0, tests_passed = 0;
#define TEST(n) do { tests_run++; printf("  TEST: %s ... ", n); } while(0)
#define PASS() do { tests_passed++; printf("PASSED\n"); } while(0)
#define FAIL(m) do { printf("FAILED: %s\n", m); } while(0)
#define ASSERT_EQ(a,b,tol) do { \
    if (fabs((a)-(b)) > (tol)) { FAIL("assertion"); \
        printf("    expected %.6f, got %.6f\n", (double)(b), (double)(a)); return; } \
} while(0)

static void test_problem_create_free(void) {
    TEST("problem_create_free");
    dr_problem_t *p = dr_problem_create(5, 3);
    assert(p != NULL);
    assert(p->nvar == 5 && p->ncon == 3);
    dr_problem_free(p);
    dr_problem_free(NULL);
    PASS();
}

static void test_measurement_set(void) {
    TEST("measurement_set");
    dr_problem_t *p = dr_problem_create(3, 1);
    assert(p);
    assert(dr_set_measurement(p, 0, 10.0, 0.5, 101) == DR_OK);
    assert(p->measurements[0].value == 10.0);
    assert(p->measurements[0].is_present == 1);
    assert(dr_set_measurement(p, -1, 1.0, 0.1, 1) == DR_ERR_DIM_MISMATCH);
    assert(dr_set_measurement(p, 1, 1.0, 0.0, 1) == DR_ERR_NOT_SPD);
    dr_problem_free(p);
    PASS();
}

static void test_simple_reconciliation(void) {
    TEST("simple_reconciliation");
    dr_problem_t *p = dr_problem_create(3, 1);
    assert(p);
    dr_set_measurement(p, 0, 10.2, 0.5, 1);
    dr_set_measurement(p, 1,  5.1, 0.3, 2);
    dr_set_measurement(p, 2, 15.0, 0.4, 3);
    double coeffs[3] = {1.0, 1.0, -1.0};
    dr_set_constraint(p, 0, coeffs, 0.0, DR_CONSTRAINT_MASS);
    dr_result_t *r = dr_result_create(3, 1);
    assert(r);
    int status = dr_solve(p, r, DR_SOLVER_LAGRANGE);
    if (status != DR_OK) { FAIL("solver failed"); dr_result_free(r); dr_problem_free(p); return; }
    double res = r->x_reconciled[0] + r->x_reconciled[1] - r->x_reconciled[2];
    ASSERT_EQ(res, 0.0, 1e-10);
    assert(r->objective >= 0.0);
    dr_result_free(r);
    dr_problem_free(p);
    PASS();
}

static void test_lagrange_vs_qr(void) {
    TEST("lagrange_vs_qr");
    dr_problem_t *p = dr_problem_create(3, 1);
    assert(p);
    dr_set_measurement(p, 0, 10.2, 0.5, 1);
    dr_set_measurement(p, 1,  5.1, 0.3, 2);
    dr_set_measurement(p, 2, 15.0, 0.4, 3);
    double coeffs[3] = {1.0, 1.0, -1.0};
    dr_set_constraint(p, 0, coeffs, 0.0, DR_CONSTRAINT_MASS);
    dr_result_t *r1 = dr_result_create(3, 1);
    dr_result_t *r2 = dr_result_create(3, 1);
    assert(r1 && r2);
    assert(dr_solve(p, r1, DR_SOLVER_LAGRANGE) == DR_OK);
    assert(dr_solve(p, r2, DR_SOLVER_QR_ORTHOG) == DR_OK);
    for (int i = 0; i < 3; i++) ASSERT_EQ(r1->x_reconciled[i], r2->x_reconciled[i], 1e-6);
    dr_result_free(r2); dr_result_free(r1); dr_problem_free(p);
    PASS();
}

static void test_global_test(void) {
    TEST("global_test");
    dr_problem_t *p = dr_problem_create(3, 1);
    assert(p);
    dr_set_measurement(p, 0, 10.0, 0.5, 1);
    dr_set_measurement(p, 1,  5.0, 0.3, 2);
    dr_set_measurement(p, 2, 15.0, 0.4, 3);
    double coeffs[3] = {1.0, 1.0, -1.0};
    dr_set_constraint(p, 0, coeffs, 0.0, DR_CONSTRAINT_MASS);
    double stat; int df;
    assert(dr_global_test(p, NULL, &stat, &df) == DR_OK);
    assert(df == 1);
    ASSERT_EQ(stat, 0.0, 0.1);
    dr_problem_free(p);
    PASS();
}

static void test_V_matrix(void) {
    TEST("V_matrix");
    dr_problem_t *p = dr_problem_create(3, 1);
    assert(p);
    dr_set_measurement(p, 0, 10.0, 2.0, 1);
    dr_set_measurement(p, 1,  5.0, 1.0, 2);
    dr_set_measurement(p, 2, 15.0, 3.0, 3);
    double coeffs[3] = {1.0, 1.0, -1.0};
    dr_set_constraint(p, 0, coeffs, 0.0, DR_CONSTRAINT_MASS);
    double V[1];
    assert(dr_compute_V_matrix(p, V) == DR_OK);
    ASSERT_EQ(V[0], 14.0, 0.01);
    dr_problem_free(p);
    PASS();
}

static void test_matrix_ops(void) {
    TEST("matrix_ops");
    dr_matrix_t *A = dr_mat_alloc(2, 3);
    dr_matrix_t *B = dr_mat_alloc(3, 2);
    dr_matrix_t *C = dr_mat_alloc(2, 2);
    assert(A && B && C);
    A->data[0]=1; A->data[1]=2; A->data[2]=3;
    A->data[3]=4; A->data[4]=5; A->data[5]=6;
    B->data[0]=1; B->data[1]=0; B->data[2]=0;
    B->data[3]=1; B->data[4]=1; B->data[5]=0;
    assert(dr_mat_gemm(1.0, A, B, 0.0, C) == 0);
    ASSERT_EQ(C->data[0], 4.0, 1e-10);
    ASSERT_EQ(C->data[3], 5.0, 1e-10);
    dr_mat_free(C); dr_mat_free(B); dr_mat_free(A);
    PASS();
}

int main(void) {
    printf("=== Data Reconciliation Core Tests ===\n\n");
    test_problem_create_free();
    test_measurement_set();
    test_simple_reconciliation();
    test_lagrange_vs_qr();
    test_global_test();
    test_V_matrix();
    test_matrix_ops();
    printf("\n=== %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
