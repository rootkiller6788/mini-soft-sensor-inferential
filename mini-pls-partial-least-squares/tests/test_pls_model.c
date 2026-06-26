#include "pls_model.h"
#include "pls_nipals.h"
#include "pls_statistics.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define EPS 1e-8
static int tests_run = 0, tests_passed = 0;

#define TEST(n) do { tests_run++; printf("  TEST %s... ", n); } while(0)
#define PASS()  do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(m) do { printf("FAIL: %s\n", m); return; } while(0)
#define CHECK(c) do { if(!(c)) { FAIL(#c); return; } } while(0)
#define CLOSE(a,b) do { if(fabs((a)-(b))>EPS){printf("FAIL: |%g-%g|>%g\n",(a),(b),EPS);return;}}while(0)

/* Test PLS model allocation */
static void test_model_alloc(void) {
    TEST("PLSModel alloc");
    PLSModel *m = pls_model_alloc(10, 5, 2, 3);
    CHECK(m != NULL);
    CHECK(m->n_samples == 10);
    CHECK(m->p_vars == 5);
    CHECK(m->q_vars == 2);
    CHECK(m->a_lvs == 3);
    CHECK(m->W != NULL);
    CHECK(m->P != NULL);
    CHECK(m->T != NULL);
    CHECK(m->Beta != NULL);
    pls_model_free(m);
    PASS();
}

/* Test PLS model copy */
static void test_model_copy(void) {
    TEST("PLSModel copy");
    PLSModel *m = pls_model_alloc(5, 3, 1, 2);
    CHECK(m != NULL);
    m->center_x = 1;
    m->R2X_cum = 0.85;
    PLSModel *cp = pls_model_copy(m);
    CHECK(cp != NULL);
    CHECK(cp->center_x == 1);
    CLOSE(cp->R2X_cum, 0.85);
    pls_model_free(m);
    pls_model_free(cp);
    PASS();
}

/* Test Beta computation for simple case */
static void test_beta_computation(void) {
    TEST("Beta computation");
    /* Build a simple PLS model with known W, P, Q */
    PLSModel *m = pls_model_alloc(5, 3, 1, 2);
    CHECK(m != NULL);

    /* Set W = identity-ish, P = W, Q = [1, 0]^T */
    /* W: 3x2, P: 3x2, Q: 1x2 */
    double w_data[] = {1, 0, 0, 1, 0, 0};  /* 3x2 */
    double p_data[] = {1, 0, 0, 1, 0, 0};  /* 3x2 */
    double q_data[] __attribute__((unused)) = {2, 3};
    for (size_t i = 0; i < 6; i++) {
        m->W->data[i] = w_data[i];
        m->P->data[i] = p_data[i];
    }
    m->Q->data[0] = 2.0;
    m->Q->data[1] = 3.0;

    int ok = pls_model_compute_beta(m);
    CHECK(ok == 0);

    /* Beta should be [2, 3, 0]^T since P^T*W = I, Beta = W*I*Q^T */
    /* Beta = W * Q^T, so Beta[j] = W[j,0]*2 + W[j,1]*3 */
    CLOSE(m->Beta->data[0], 2.0);  /* 1*2 + 0*3 */
    CLOSE(m->Beta->data[1], 3.0);  /* 0*2 + 1*3 */
    CLOSE(m->Beta->data[2], 0.0);  /* 0*2 + 0*3 */

    pls_model_free(m);
    PASS();
}

/* Test single prediction */
static void test_predict_single(void) {
    TEST("PLS predict single");
    PLSModel *m = pls_model_alloc(1, 2, 1, 1);
    CHECK(m != NULL);

    /* Set Beta = [1, 2]^T, b0 = [0] */
    m->Beta->data[0] = 1.0;
    m->Beta->data[1] = 2.0;
    m->b0->data[0] = 0.0;

    double xd[] = {3.0, 4.0};
    Vector *x = vector_from_array(2, xd);
    Vector *y = vector_alloc(1);

    int ok = pls_model_predict_single(m, x, y);
    CHECK(ok == 0);
    CLOSE(y->data[0], 3.0*1.0 + 4.0*2.0);  /* 11 */

    vector_free(x); vector_free(y);
    pls_model_free(m);
    PASS();
}

/* Test prediction with centering */
static void test_predict_with_centering(void) {
    TEST("predict with centering");
    PLSModel *m = pls_model_alloc(1, 2, 1, 1);
    CHECK(m != NULL);
    m->Beta->data[0] = 1.0;
    m->Beta->data[1] = 1.0;
    m->b0->data[0] = 5.0;
    m->center_x = 1;
    m->scale_x = 0;
    m->X_mean->data[0] = 10.0;
    m->X_mean->data[1] = 20.0;

    double xd[] = {15.0, 25.0};
    Vector *x = vector_from_array(2, xd);
    Vector *y = vector_alloc(1);
    int ok = pls_model_predict_single(m, x, y);
    CHECK(ok == 0);
    /* x_processed = [15-10, 25-20] = [5, 5] */
    /* y = 5*1 + 5*1 + 5 = 15 */
    CLOSE(y->data[0], 15.0);

    vector_free(x); vector_free(y);
    pls_model_free(m);
    PASS();
}

/* Test Hotelling T2 */
static void test_T2_statistic(void) {
    TEST("Hotelling T2");
    double td[] = {2.0, 1.0};
    double vd[] = {4.0, 1.0};
    Vector *t = vector_from_array(2, td);
    Vector *v = vector_from_array(2, vd);
    double T2 = compute_T2_single(t, v);
    /* T2 = 2^2/4 + 1^2/1 = 1 + 1 = 2 */
    CLOSE(T2, 2.0);
    vector_free(t); vector_free(v);
    PASS();
}

/* Test SPE */
static void test_SPE(void) {
    TEST("SPE statistic");
    double rd[] = {0.5, -0.3, 0.1};
    Vector *r = vector_from_array(3, rd);
    double spe = compute_SPE_single(r);
    CLOSE(spe, 0.25 + 0.09 + 0.01);
    vector_free(r);
    PASS();
}

/* Test PRESS */
static void test_PRESS(void) {
    TEST("PRESS computation");
    double yd[] = {1, 2, 3, 4};
    double ypd[] = {1.1, 1.9, 3.2, 3.8};
    Matrix *Yt = matrix_from_array(2, 2, yd);
    Matrix *Yp = matrix_from_array(2, 2, ypd);
    double press = compute_PRESS(Yt, Yp);
    /* (1-1.1)^2 + (2-1.9)^2 + (3-3.2)^2 + (4-3.8)^2 = 0.01+0.01+0.04+0.04 = 0.10 */
    CLOSE(press, 0.10);
    matrix_free(Yt); matrix_free(Yp);
    PASS();
}

/* Test VIP */
static void test_VIP(void) {
    TEST("VIP computation");
    PLSModel *m = pls_model_alloc(3, 3, 1, 2);
    CHECK(m != NULL);
    /* Set up W and B_inner, T */
    m->W->data[0] = 1.0; m->W->data[1] = 0.0; m->W->data[2] = 0.0;  /* col 0 */
    m->W->data[3] = 0.0; m->W->data[4] = 0.5; m->W->data[5] = 0.5;  /* col 1 */
    /* B_inner diag = [2, 1] */
    m->B_inner->data[0] = 2.0;
    m->B_inner->data[4] = 1.0;
    /* T: n=3, a=2, t_a sums */
    m->T->data[0] = 1.0; m->T->data[1] = 2.0; m->T->data[2] = 3.0;  /* col 0 */
    m->T->data[3] = 1.0; m->T->data[4] = 1.0; m->T->data[5] = 1.0;  /* col 1 */

    Vector *vip = compute_VIP(m);
    CHECK(vip != NULL);
    CHECK(vip->len == 3);
    /* VIP should all be valid numbers */
    for (size_t j = 0; j < 3; j++)
        CHECK(isfinite(vip->data[j]));
    vector_free(vip);
    pls_model_free(m);
    PASS();
}

int main(void) {
    printf("=== PLS Model Tests ===\n");
    test_model_alloc();
    test_model_copy();
    test_beta_computation();
    test_predict_single();
    test_predict_with_centering();
    test_T2_statistic();
    test_SPE();
    test_PRESS();
    test_VIP();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
