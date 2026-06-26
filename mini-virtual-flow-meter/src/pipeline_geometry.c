/**
 * @file pipeline_geometry.c
 * @brief Pipeline geometry and hydraulics implementation
 *
 * Implements pipe segments, fittings, minor losses, and total head loss.
 *
 * @module mini-virtual-flow-meter
 */

#include "flow_models.h"
#include "pipeline_geometry.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define G_STANDARD 9.80665
#define PI         3.14159265358979323846

/* ==========================================================================
 * L2: Pipe Geometry
 * ========================================================================== */

int pipe_segment_init(pipe_segment_t *seg, double id, double length,
                       double roughness, double dz)
{
    if (!seg) return -1;
    if (id <= 0.0 || length < 0.0) return -1;
    if (roughness < 0.0) roughness = 0.0;

    seg->internal_diameter   = id;
    seg->wall_thickness      = id * 0.05;  /* Default schedule 40: t ~ 0.05*D */
    seg->absolute_roughness  = roughness;
    seg->length              = length;
    seg->elevation_change    = dz;
    seg->cross_section_area  = pipe_cross_section_area(id);

    return 0;
}

double pipe_cross_section_area(double diameter)
{
    if (diameter <= 0.0) return 0.0;
    return PI * diameter * diameter / 4.0;
}

double pipe_hydraulic_diameter(double area, double wetted_perim)
{
    /*
     * Hydraulic diameter for non-circular ducts:
     * Dh = 4 * A / P_wetted
     *
     * For a circular pipe running full: P_wetted = pi*D, A = pi*D^2/4
     * => Dh = 4*(pi*D^2/4) / (pi*D) = D
     *
     * For a rectangular duct a x b: Dh = 2ab/(a+b)
     * For an annulus Do x Di: Dh = Do - Di
     * For open channel flow: Dh = 4A/P_wetted (same formula)
     */

    if (wetted_perim <= 0.0) return 0.0;
    return 4.0 * area / wetted_perim;
}

/* ==========================================================================
 * L3: Fitting Loss Coefficients
 * ========================================================================== */

int pipe_fitting_init(pipe_fitting_t *fitting, pipe_fitting_type_t type,
                       double size)
{
    /*
     * Initialize a pipe fitting with standard K-factor from Crane TP-410.
     *
     * K-factors are for fully turbulent flow (Re > 1e5).
     * At lower Reynolds numbers, K-factors are larger.
     *
     * Loss coefficient K is defined by:
     *   hL = K * v^2 / (2*g)
     */

    if (!fitting) return -1;
    (void)size;  /* Size-dependent K-factors not implemented here */

    fitting->type = (int)type;

    /* Standard K-factors for fully-turbulent flow (Crane TP-410) */
    switch (type) {
    case FITTING_ELBOW_90:
        fitting->k_factor = 0.75;
        fitting->equiv_length_ratio = 30.0;
        break;
    case FITTING_ELBOW_45:
        fitting->k_factor = 0.35;
        fitting->equiv_length_ratio = 16.0;
        break;
    case FITTING_ELBOW_90_LR:
        fitting->k_factor = 0.45;
        fitting->equiv_length_ratio = 20.0;
        break;
    case FITTING_TEE_THROUGH:
        fitting->k_factor = 0.40;
        fitting->equiv_length_ratio = 20.0;
        break;
    case FITTING_TEE_BRANCH:
        fitting->k_factor = 1.00;
        fitting->equiv_length_ratio = 60.0;
        break;
    case FITTING_RETURN_BEND:
        fitting->k_factor = 2.20;
        fitting->equiv_length_ratio = 75.0;
        break;
    case FITTING_GATE_VALVE:
        fitting->k_factor = 0.17;
        fitting->equiv_length_ratio = 8.0;
        break;
    case FITTING_GLOBE_VALVE:
        fitting->k_factor = 6.00;
        fitting->equiv_length_ratio = 300.0;
        break;
    case FITTING_BALL_VALVE:
        fitting->k_factor = 0.05;
        fitting->equiv_length_ratio = 3.0;
        break;
    case FITTING_BUTTERFLY:
        fitting->k_factor = 0.30;
        fitting->equiv_length_ratio = 15.0;
        break;
    case FITTING_CHECK_SWING:
        fitting->k_factor = 2.00;
        fitting->equiv_length_ratio = 100.0;
        break;
    case FITTING_CHECK_LIFT:
        fitting->k_factor = 5.00;
        fitting->equiv_length_ratio = 250.0;
        break;
    case FITTING_SUDDEN_EXP:
        fitting->k_factor = 0.50;  /* Default, depends on area ratio */
        fitting->equiv_length_ratio = 0.0;
        break;
    case FITTING_SUDDEN_CONT:
        fitting->k_factor = 0.25;  /* Default, depends on area ratio */
        fitting->equiv_length_ratio = 0.0;
        break;
    case FITTING_PIPE_ENTRANCE:
        fitting->k_factor = 0.50;  /* Square-edged entrance */
        fitting->equiv_length_ratio = 0.0;
        break;
    case FITTING_PIPE_EXIT:
        fitting->k_factor = 1.00;  /* Exit loss = one velocity head */
        fitting->equiv_length_ratio = 0.0;
        break;
    default:
        fitting->k_factor = 0.0;
        fitting->equiv_length_ratio = 0.0;
        return -1;
    }

    /* Copy description */
    const char *descriptions[] = {
        "90-deg elbow (standard radius)",
        "45-deg elbow",
        "90-deg long-radius elbow",
        "Tee - flow through run",
        "Tee - flow through branch",
        "180-deg return bend",
        "Gate valve (fully open)",
        "Globe valve (fully open)",
        "Ball valve (fully open)",
        "Butterfly valve (fully open)",
        "Swing check valve",
        "Lift check valve",
        "Sudden expansion",
        "Sudden contraction",
        "Pipe entrance",
        "Pipe exit"
    };
    size_t n_desc = sizeof(descriptions) / sizeof(descriptions[0]);
    if ((size_t)type < n_desc) {
        strncpy(fitting->description, descriptions[type], 63);
        fitting->description[63] = '\0';
    } else {
        fitting->description[0] = '\0';
    }

    return 0;
}

double pipe_minor_head_loss(const pipe_fitting_t *fitting, double velocity)
{
    /*
     * Minor (local) head loss for a single fitting:
     * hL_minor = K * v^2 / (2*g)
     *
     * The K-factor represents the number of velocity heads lost in
     * the fitting. For example, K=1 means the fitting dissipates
     * energy equal to one velocity head.
     */

    if (!fitting) return 0.0;
    double v2 = velocity * velocity;
    return fitting->k_factor * v2 / (2.0 * G_STANDARD);
}

double pipe_total_minor_loss(const pipe_fitting_t *fittings,
                              int n_fittings, double velocity)
{
    if (!fittings || n_fittings <= 0) return 0.0;

    double total = 0.0;
    int i;
    for (i = 0; i < n_fittings; i++) {
        total += pipe_minor_head_loss(&fittings[i], velocity);
    }
    return total;
}

/* ==========================================================================
 * L4: Pipeline Total Head Loss
 * ========================================================================== */

int pipeline_config_init(pipeline_config_t *pipeline,
                          const pipe_segment_t *segments, int n_segments,
                          const pipe_fitting_t *fittings, int n_fittings)
{
    if (!pipeline || !segments || n_segments <= 0) return -1;

    pipeline->segments     = (pipe_segment_t *)malloc(
        (size_t)n_segments * sizeof(pipe_segment_t));
    if (!pipeline->segments) return -1;

    memcpy(pipeline->segments, segments,
           (size_t)n_segments * sizeof(pipe_segment_t));
    pipeline->num_segments = n_segments;

    if (fittings && n_fittings > 0) {
        pipeline->fittings = (pipe_fitting_t *)malloc(
            (size_t)n_fittings * sizeof(pipe_fitting_t));
        if (!pipeline->fittings) {
            free(pipeline->segments);
            pipeline->segments = NULL;
            return -1;
        }
        memcpy(pipeline->fittings, fittings,
               (size_t)n_fittings * sizeof(pipe_fitting_t));
        pipeline->num_fittings = n_fittings;
    } else {
        pipeline->fittings     = NULL;
        pipeline->num_fittings = 0;
    }

    /* Compute summary statistics */
    pipeline->total_length    = 0.0;
    pipeline->total_elevation = 0.0;
    pipeline->min_diameter    = segments[0].internal_diameter;

    double sum_roughness_len = 0.0;
    int i;
    for (i = 0; i < n_segments; i++) {
        pipeline->total_length    += segments[i].length;
        pipeline->total_elevation += segments[i].elevation_change;

        if (segments[i].internal_diameter < pipeline->min_diameter) {
            pipeline->min_diameter = segments[i].internal_diameter;
        }

        sum_roughness_len += segments[i].absolute_roughness
                           * segments[i].length;
    }

    if (pipeline->total_length > 0.0) {
        pipeline->avg_roughness = sum_roughness_len
                                / pipeline->total_length;
    } else {
        pipeline->avg_roughness = 0.0;
    }

    return 0;
}

double pipeline_total_head_loss(const pipeline_config_t *pipeline,
                                 double flow_rate, double friction,
                                 double *major_loss, double *minor_loss)
{
    /*
     * Total head loss across a pipeline system.
     *
     * For each pipe segment i:
     *   hL_major_i = f * (L_i/D_i) * v_i^2 / (2*g)
     *
     * For each fitting j:
     *   hL_minor_j = K_j * v_j^2 / (2*g)
     *
     * Total: hL = sum(major) + sum(minor) + elevation
     *
     * Assumptions:
     *   - Uniform friction factor f across all segments (simplified)
     *   - Incompressible flow (velocity depends on local diameter)
     *   - Fittings at the local velocity of their segment
     */

    if (!pipeline || flow_rate < 0.0) return 0.0;

    double total_major = 0.0;
    double total_minor = 0.0;

    int i;
    for (i = 0; i < pipeline->num_segments; i++) {
        const pipe_segment_t *seg = &pipeline->segments[i];
        double area = seg->cross_section_area;
        if (area <= 0.0) continue;

        double velocity = flow_rate / area;
        total_major += darcy_weisbach_head_loss(
            friction, seg->length, seg->internal_diameter, velocity);
    }

    /* Minor losses - assume velocity at the first segment */
    if (pipeline->num_segments > 0 && pipeline->num_fittings > 0) {
        double v0 = flow_rate / pipeline->segments[0].cross_section_area;
        total_minor = pipe_total_minor_loss(
            pipeline->fittings, pipeline->num_fittings, v0);
    }

    if (major_loss) *major_loss = total_major;
    if (minor_loss) *minor_loss = total_minor;

    return total_major + total_minor + pipeline->total_elevation;
}

double pipeline_equivalent_length(const pipeline_config_t *pipeline,
                                   double friction)
{
    /*
     * Equivalent length: length of straight pipe that gives the same
     * head loss as the actual pipeline with fittings.
     *
     * Le_total = sum(L_i) + sum(K_j * D / f)
     *
     * where the second term converts each fitting's K-factor to an
     * equivalent length of pipe.
     *
     * This is useful for simplifying pipeline analysis and for
     * comparing the relative importance of major vs. minor losses.
     */

    if (!pipeline || friction <= 0.0) return 0.0;

    double Le = pipeline->total_length;

    int i;
    for (i = 0; i < pipeline->num_fittings; i++) {
        /* For each fitting: Le_fitting = K * D / f
         * Using the diameter of the first segment as reference.
         */
        double D = (pipeline->num_segments > 0)
            ? pipeline->segments[0].internal_diameter : 0.1;
        Le += pipeline->fittings[i].k_factor * D / friction;
    }

    return Le;
}

void pipeline_config_free(pipeline_config_t *pipeline)
{
    if (!pipeline) return;
    free(pipeline->segments);
    free(pipeline->fittings);
    pipeline->segments = NULL;
    pipeline->fittings = NULL;
    pipeline->num_segments = 0;
    pipeline->num_fittings = 0;
}
