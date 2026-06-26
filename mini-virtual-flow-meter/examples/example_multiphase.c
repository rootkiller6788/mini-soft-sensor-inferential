#include <stdio.h>
#include <math.h>

/* Forward declarations */
extern int multiphase_pattern_beggs_brill(double lambda_L, double Froude_mix);
extern double multiphase_holdup_beggs_brill(double lambda_L, double Froude_mix,
                                              int flow_pattern);
extern double multiphase_noslip_holdup(double Q_liq, double Q_gas);
extern double lockhart_martinelli_X(double quality, double rho_liq,
                                     double rho_gas, double mu_liq, double mu_gas);
extern double chisholm_phi_liquid_sq(double X, double C);
extern double multiphase_flow_from_dp(double diameter, double dp_meas,
                                       double length, double quality,
                                       double rho_liq, double rho_gas,
                                       double mu_liq, double mu_gas,
                                       int max_iter, double tol);

int main(void) {
    printf("===========================================\n");
    printf("  Example 5: Multi-Phase Virtual Flow Meter\n");
    printf("===========================================\n\n");

    /* Oil-gas two-phase flow in a 0.1m horizontal pipe */
    double D = 0.1;
    double Q_liq = 0.005;   /* 5 L/s liquid */
    double Q_gas = 0.010;   /* 10 L/s gas (at line conditions) */
    double rho_liq = 850.0; /* Oil density */
    double rho_gas = 50.0;  /* Gas density at line pressure */
    double mu_liq = 5e-3;   /* Oil viscosity */
    double mu_gas = 1.5e-5; /* Gas viscosity */

    /* Flow pattern prediction */
    double lambda_L = multiphase_noslip_holdup(Q_liq, Q_gas);
    double v_mix = (Q_liq + Q_gas) / (3.14159 * D * D / 4.0);
    double Fr = v_mix * v_mix / (9.80665 * D);
    int pattern = multiphase_pattern_beggs_brill(lambda_L, Fr);

    const char *pattern_names[] = {"Transition", "Segregated", "Intermittent", "Distributed"};
    printf("Flow conditions:\n");
    printf("  Q_liquid = %.3f L/s, Q_gas = %.3f L/s\n", Q_liq*1000, Q_gas*1000);
    printf("  v_mix = %.2f m/s, Fr = %.3f, lambda_L = %.3f\n", v_mix, Fr, lambda_L);
    printf("  Flow pattern: %s\n\n", pattern_names[pattern]);

    /* Liquid holdup */
    double HL = multiphase_holdup_beggs_brill(lambda_L, Fr, pattern);
    printf("  No-slip holdup: %.3f\n", lambda_L);
    printf("  Actual holdup:  %.3f (liquid occupies %.0f%% of pipe)\n\n", HL, HL*100);

    /* Lockhart-Martinelli two-phase pressure drop */
    double quality = (Q_gas * rho_gas) / (Q_gas * rho_gas + Q_liq * rho_liq);
    double X = lockhart_martinelli_X(quality, rho_liq, rho_gas, mu_liq, mu_gas);
    double phi_l_sq = chisholm_phi_liquid_sq(X, 20.0);

    printf("Two-phase parameters:\n");
    printf("  Mass quality: %.4f\n", quality);
    printf("  Lockhart-Martinelli X: %.3f\n", X);
    printf("  phi_L^2 (Chisholm C=20): %.3f\n\n", phi_l_sq);

    /* Virtual flow metering from pressure drop */
    double dp_meas = 50000.0;  /* 50 kPa over 100 m */
    double G = multiphase_flow_from_dp(D, dp_meas, 100.0, quality,
                                        rho_liq, rho_gas, mu_liq, mu_gas,
                                        50, 1e-6);
    printf("VFM from pressure drop:\n");
    printf("  Measured dP = %.0f Pa over 100 m\n", dp_meas);
    printf("  Estimated mass flux G = %.2f kg/(m2.s)\n", G);
    if (G > 0) {
        double rho_mix = quality * rho_gas + (1-quality) * rho_liq;
        double area = 3.14159 * D * D / 4.0;
        double Q_total = G * area / rho_mix;
        printf("  Estimated total flow = %.3f L/s (input = %.3f L/s)\n",
               Q_total * 1000, (Q_liq + Q_gas) * 1000);
    }

    printf("\nMulti-phase VFM uses pressure drop + phase fraction estimates\n");
    printf("to infer total flow rate in oil-gas-water well production.\n");
    return 0;
}