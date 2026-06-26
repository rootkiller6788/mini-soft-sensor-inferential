/**
 * @file test_vfm.c
 * @brief Test suite for mini-virtual-flow-meter
 *
 * assert-based tests covering core VFM APIs and physics models.
 * @module mini-virtual-flow-meter
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include "virtual_flow_meter.h"
#include "flow_models.h"
#include "fluid_properties.h"
#include "vfm_state_estimation.h"
#include "vfm_uncertainty.h"
#include "pipeline_geometry.h"

#define EPS 1e-9
#define EPS_LARGE 1e-5

static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) do { tests_run++; printf("  %3d: %-50s", tests_run, name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(m) do { tests_failed++; printf("FAIL: %s\n", m); return; } while(0)
#define CHECK(cond, m) if(!(cond)){FAIL(m);}
#define CLOSE(a,b,eps,m) if(fabs((a)-(b))>(eps)){printf("FAIL: %s (%.6e != %.6e)\n",m,(double)(a),(double)(b));tests_failed++;return;}

/* -- L1/L2: Core API -- */
static void t_config_init(void) { TEST("config_init"); vfm_config_t c; CHECK(vfm_config_init(&c)==0,"init"); CHECK(c.pipe_diameter>0,"diam"); PASS(); }
static void t_config_validate(void) { TEST("config_validate"); vfm_config_t c; vfm_config_init(&c); CHECK(vfm_config_validate(&c)==0,"valid"); c.pipe_diameter=-1; CHECK(vfm_config_validate(&c)!=0,"neg"); PASS(); }
static void t_state_res(void) { TEST("state+result"); vfm_state_t s; vfm_result_t r; CHECK(vfm_state_init(&s)==0,"state"); CHECK(vfm_result_init(&r)==0,"result"); PASS(); }
static void t_sensor(void) { TEST("sensor ops"); vfm_state_t s; vfm_config_t c; vfm_sensor_t sn; vfm_state_init(&s); vfm_config_init(&c); int id; CHECK(vfm_sensor_register(&s,&c,"PT-101",&id)==0,"reg"); CHECK(id==0,"id"); CHECK(vfm_sensor_update(&sn,100000,100)==0,"upd"); CHECK(sn.valid==1,"valid"); CHECK(vfm_sensor_update(&sn,NAN,1)!=0,"nan"); vfm_sensor_mark_fault(&sn); CHECK(sn.valid==0,"fault"); PASS(); }

/* -- L3/L4: Flow Models -- */
static void t_reynolds(void) { TEST("Reynolds"); double Re=flow_reynolds_number(1.0,0.1,1e-6); CLOSE(Re,100000,EPS_LARGE,"Re"); CHECK(flow_regime_classify(1000)==VFM_REGIME_LAMINAR,"lam"); CHECK(flow_regime_classify(10000)==VFM_REGIME_TURBULENT,"turb"); PASS(); }
static void t_orifice(void) { TEST("orifice"); orifice_params_t p; CHECK(orifice_params_init(&p,0.1,0.05)==0,"init"); CLOSE(p.beta_ratio,0.5,EPS,"beta"); double Cd=orifice_discharge_coeff_iso5167(0.5,1e5); CHECK(Cd>0.55&&Cd<0.85,"Cd"); p.discharge_coeff=Cd; double Q=orifice_vol_flow(&p,10000,1000,1); CHECK(Q>0,"Q>0"); PASS(); }
static void t_venturi(void) { TEST("venturi"); venturi_params_t p; CHECK(venturi_params_init(&p,0.15,0.075)==0,"init"); double Q=venturi_mass_flow(&p,10000,1000,1); CHECK(Q>0,"Q>0"); PASS(); }
static void t_bernoulli(void) { TEST("Bernoulli"); double a=3.14159*0.05*0.05; double Q=bernoulli_flow_rate(200000,100000,a,1000,0,0); CHECK(Q>0,"Q>0"); PASS(); }
static void t_darcy(void) { TEST("Darcy"); double f=darcy_friction_laminar(1000); CLOSE(f,0.064,EPS_LARGE,"f_lam"); double ft=darcy_friction_colebrook(1e5,0.00045,50,1e-8); CHECK(ft>0.01&&ft<0.05,"f_turb"); double h=darcy_weisbach_head_loss(ft,100,0.1,2); CHECK(h>0,"hL>0"); double Q=darcy_inverse_flow(0.1,4.5e-5,100,h,1e-6,50,1e-6); CHECK(Q>0,"Q_inv>0"); PASS(); }
static void t_pump(void) { TEST("pump curve"); pump_curve_params_t p; CHECK(pump_curve_init(&p,0.01,30,36,1500)==0,"init"); double Q=pump_curve_flow_estimate(&p,30,1500); CLOSE(Q,0.01,EPS_LARGE,"rated"); PASS(); }
static void t_choke(void) { TEST("choke valve"); double Q=choke_valve_liquid_flow(10,100000,1000,1); CHECK(Q>0,"liq>0"); double Qg=choke_valve_gas_flow(10,200000,190000,300,0.65,1.4,0.7,1); CHECK(Qg>0,"gas>0"); PASS(); }

/* -- L5: Fluid Properties -- */
static void t_gas(void) { TEST("gas props"); double rho=gas_density_ideal(101325,293.15,0.0289644); CHECK(rho>1.1&&rho<1.3,"air"); double Z=gas_compressibility_dak(0.1,1.5,1e-8,50); CHECK(Z>0.95&&Z<1.05,"Z~1"); double Ppc,Tpc; gas_pseudocritical_sutton(0.65,&Ppc,&Tpc); CHECK(Ppc>1e6,"Ppc"); PASS(); }
static void t_viscosity(void) { TEST("viscosity"); double mu=gas_viscosity_sutherland(300,1.716e-5,273.15,110.4); CHECK(mu>1.7e-5&&mu<2.0e-5,"air"); double mw=water_viscosity_iapws(293.15); CHECK(mw>8e-4&&mw<1.2e-3,"water"); PASS(); }
static void t_liquid(void) { TEST("liquid props"); double rw=water_density(293.15); CHECK(rw>995&&rw<1000,"water"); double ro=oil_density_from_api(35); CHECK(ro>800&&ro<900,"oil"); double Pb=oil_bubble_point_standing(50,0.65,35,350); CHECK(Pb>1e6,"Pb"); double rm=mixture_density_homogeneous(0.2,10,1000); CHECK(rm>800&&rm<1000,"mix rho"); double mm=mixture_viscosity_mcadams(0.5,1e-5,1e-3); CHECK(mm>0,"mix mu"); PASS(); }

/* -- L6: State Estimation -- */
static void t_kalman(void) { TEST("Kalman"); vfm_kalman_t kf; CHECK(vfm_kalman_init(&kf,0.001,0.01)==0,"init"); int i; double f,u; for(i=0;i<20;i++){vfm_kalman_predict(&kf);vfm_kalman_update(&kf,0.01+0.0001*i);} vfm_kalman_get_estimate(&kf,&f,&u); CHECK(f>0,"flow>0"); CHECK(u>0,"uncert>0"); double b; vfm_kalman_get_bias(&kf,&b); double Rn=vfm_kalman_adapt_noise(&kf,0.05); CHECK(Rn>0,"Rn>0"); PASS(); }
static void t_rls(void) { TEST("RLS"); vfm_rls_t r; double t0[]={0.6,0.01,0}; CHECK(vfm_rls_init(&r,3,0.98,t0)==0,"init"); int i; for(i=0;i<50;i++){double dp=5000+100*i; double phi[]={sqrt(dp),1,0}; vfm_rls_update(&r,phi,0.6*sqrt(dp)+0.01);} double th[3]; vfm_rls_get_params(&r,th); CLOSE(th[0],0.6,0.05,"Cd"); CLOSE(th[1],0.01,0.05,"bias"); PASS(); }
static void t_mhe(void) { TEST("MHE"); vfm_mhe_t m; CHECK(vfm_mhe_init(&m,10,1,5)==0,"init"); double e; int i; for(i=0;i<15;i++) vfm_mhe_step(&m,0.01+0.0001*sin(i*0.5),0.01,&e); CHECK(fabs(e-0.01)<0.005,"track"); vfm_mhe_free(&m); PASS(); }
static void t_fusion(void) { TEST("fusion"); double es[]={0.01,0.011,0.009},vr[]={0.0001,0.0004,0.0009},fv,fr; CHECK(vfm_sensor_fusion_wls(es,vr,3,&fv,&fr)==0,"fusion"); CHECK(fv>0.009&&fv<0.011,"range"); double c2=vfm_consistency_chi2(es,vr,3,fv); CHECK(c2>=0,"chi2>=0"); PASS(); }
static void t_cusum(void) { TEST("CUSUM"); double sh=0,sl=0; int i,a=0; for(i=0;i<15;i++){a=vfm_cusum_drift_detect(10.8,10,0.5,5,&sh,&sl); if(a)break;} CHECK(a==1,"drift detected"); PASS(); }

/* -- L7: Uncertainty -- */
static void t_budget(void) { TEST("unc budget"); vfm_uncertainty_budget_t b; CHECK(vfm_uncertainty_budget_init(&b,0.95)==0,"init"); vfm_uncertainty_add_component(&b,"DP",100,1.5e-5,10,VFM_UNCERT_TYPE_B,VFM_DIST_NORMAL); vfm_uncertainty_add_component(&b,"rho",5,-2,100,VFM_UNCERT_TYPE_B,VFM_DIST_RECTANGULAR); double uc=vfm_uncertainty_combine(&b); CHECK(uc>0,"uc>0"); double U=vfm_uncertainty_expand(&b); CHECK(U>uc,"U>uc"); PASS(); }
static void t_taylor(void) { TEST("Taylor"); double s[]={1,2,-0.5},u[]={0.01,0.02,0.005}; double uc=vfm_taylor_combined_uncertainty(s,u,3); CHECK(uc>0,"uc>0"); PASS(); }
static void t_orif_unc(void) { TEST("orif unc"); double ru=vfm_orifice_relative_uncertainty(0.01,0.61,0.005,10000,50,1000,2,0.5,0.001,0.1,5e-5); CHECK(ru>0&&ru<0.1,"rel unc"); PASS(); }

/* -- L8: Pipeline -- */
static void t_pipeline(void) { TEST("pipeline"); pipe_segment_t s; CHECK(pipe_segment_init(&s,0.1,50,4.5e-5,5)==0,"seg"); pipe_fitting_t f; CHECK(pipe_fitting_init(&f,FITTING_ELBOW_90,0.1)==0,"fit"); CHECK(f.k_factor>0,"K>0"); pipeline_config_t p; CHECK(pipeline_config_init(&p,&s,1,&f,1)==0,"pipe"); double maj,min,h=pipeline_total_head_loss(&p,0.01,0.02,&maj,&min); CHECK(h>0,"hL>0"); double Le=pipeline_equivalent_length(&p,0.02); CHECK(Le>=p.total_length,"Le"); pipeline_config_free(&p); PASS(); }

/* -- Integration -- */
static void t_integration(void) { TEST("integration"); vfm_config_t c; vfm_state_t s; vfm_result_t r; vfm_sensor_t sn[3]; vfm_fluid_t fl; vfm_config_init(&c); vfm_state_init(&s); vfm_result_init(&r); fl.density=998.2; fl.viscosity_dynamic=1.002e-3; fl.viscosity_kinematic=1.004e-6; snprintf(sn[0].tag,32,"PDT-101"); sn[0].value=50000; sn[0].uncertainty=100; sn[0].valid=1; snprintf(sn[1].tag,32,"TT-101"); sn[1].value=293.15; sn[1].uncertainty=0.1; sn[1].valid=1; snprintf(sn[2].tag,32,"DP-201"); sn[2].value=10000; sn[2].uncertainty=50; sn[2].valid=1; int id; vfm_sensor_register(&s,&c,"PDT-101",&id); vfm_sensor_register(&s,&c,"TT-101",&id); vfm_sensor_register(&s,&c,"DP-201",&id); int rc=vfm_estimate(&r,&s,&c,sn,3,&fl); CHECK(rc==0,"estimate ok"); CHECK(r.flow_rate>0,"flow>0"); CHECK(r.reynolds_number>0,"Re>0"); CHECK(r.expanded_uncertainty>0,"unc>0"); PASS(); }

int main(void) {
    printf("\n==========================================================\n");
    printf("  mini-virtual-flow-meter Test Suite\n");
    printf("  Module: 14. mini-soft-sensor-inferential\n");
    printf("==========================================================\n\n");
    t_config_init(); t_config_validate(); t_state_res(); t_sensor();
    t_reynolds(); t_orifice(); t_venturi(); t_bernoulli(); t_darcy(); t_pump(); t_choke();
    t_gas(); t_viscosity(); t_liquid();
    t_kalman(); t_rls(); t_mhe(); t_fusion(); t_cusum();
    t_budget(); t_taylor(); t_orif_unc();
    t_pipeline();
    t_integration();
    printf("\n  RESULTS: %d/%d PASSED, %d FAILED\n", tests_passed, tests_run, tests_failed);
    printf("==========================================================\n\n");
    return (tests_failed > 0) ? 1 : 0;
}