#include <stdio.h>
#include <math.h>
#include "virtual_flow_meter.h"
#include "flow_models.h"
#include "fluid_properties.h"

int main(void) {
    printf("============================================\n");
    printf("  Example 2: Pump Curve Virtual Flow Meter\n");
    printf("============================================\n\n");

    /* Centrifugal pump: rated 100 L/s at 30 m head, 1500 rpm */
    pump_curve_params_t pump;
    pump_curve_init(&pump, 0.10, 30.0, 36.0, 1500.0);

    printf("Pump specifications:\n");
    printf("  Rated flow:   %.1f L/s\n", pump.rated_flow * 1000.0);
    printf("  Rated head:   %.1f m\n", pump.rated_head);
    printf("  Shutoff head: %.1f m\n", pump.shutoff_head);
    printf("  Rated speed:  %.0f rpm\n", pump.rated_speed);
    printf("  Curve: H(Q) = %.1f - %.3f*Q - %.3f*Q^2\n\n",
           pump.shutoff_head, pump.curve_coeff_a, pump.curve_coeff_b);

    printf("Virtual flow estimation from measured head:\n");
    printf("  Speed [rpm] | Head [m] | Est. Flow [L/s] | Method\n");
    printf("  ------------+----------+-----------------+--------\n");

    double speeds[] = {1500, 1500, 1500, 1200, 1000};
    double heads[]  = {30.0, 20.0, 10.0, 20.0, 15.0};
    int i;
    for (i = 0; i < 5; i++) {
        double Q = pump_curve_flow_estimate(&pump, heads[i], speeds[i]);
        printf("  %10.0f | %8.1f | %15.2f | %s\n",
               speeds[i], heads[i], Q*1000.0,
               Q < 0 ? "no solution" : "affinity laws");
    }

    printf("\nThis VFM uses: differential pressure transmitter (head)\n");
    printf("             + tachometer (speed) -> estimated flow.\n");
    printf("No physical flow meter required.\n");
    return 0;
}