# Polymer-family registry.
from .base import (Simulation_Config, Material_Config,
                   LinearElastic_Config, J2Plasticity_Config,
                   DruckerPrager_Config, Prony_Config, Friction_Config,
                   RateDependent_Config, gsell_jonas_table, natural_dt)



class PolymerFamily:

    def __init__(self, key, label, config_factory,
                 checks=None, sampling=None, description=""):
        self.key = key                                 # short tag stored on Material_Config.family
        self.label = label                             # Name
        self.config_factory = config_factory           # callable for Simulation_Config
        self.checks = list(checks) if checks else []   # verifier checks that apply
        self.sampling = sampling                       # parameter-sampling config 
        self.description = description

    def build_config(self):
        cfg = self.config_factory()
        cfg.material.family = self.key
        return cfg


# Verifier checks applicable to a pure-hyperelastic elastomer 
_ELASTOMER_CHECKS = (
    "quasi_static",      
    "hourglass",         
    "energy_total",      
    "artificial_energy", # ALLMW (mass scaling) / ALLPW (contact penalty) contamination
    "d1_validity",       # Specifically for Mooney-Rivlin
    "force_magnitude",   
    "strain_level",      
    "friction_physics", 
    "steady_state",      # RF/SCOF plateau over the second half of the scratch
    "settling",          # kinetic energy decayed before trusting the residual profile
    "recovery",          # residual ~ 0 (pure hyperelastic)
)

# Viscoelastic elastomer: the "recovery" VERDICT is excluded (the groove is in delayed viscoelastic recovery: neither ~0 nor permanent), but "profile"
# reports the residual depth / pile-up MEASUREMENT without a verdict.
_ELASTOMER_VE_CHECKS = tuple(c for c in _ELASTOMER_CHECKS if c != "recovery") + ("profile",)

# Verifier checks applicable to a dissipative (plastic) family.
_SEMICRYSTALLINE_CHECKS = (
    "quasi_static",      
    "hourglass",         
    "energy_total",      
    "artificial_energy", # ALLMW (mass scaling) / ALLPW (contact penalty) contamination
    "force_magnitude",   
    "strain_level",      
    "friction_physics",  
    "steady_state",      # RF/SCOF plateau over the second half of the scratch
    "settling",          # kinetic energy decayed before trusting the residual profile
    "recovery",          # residual groove expected (plastic)
)

_GLASSY_CHECKS = (
    "quasi_static",      
    "hourglass",         
    "energy_total",      # ETOTAL drift (now includes viscous dissipation ALLVD)
    "artificial_energy", # ALLMW (mass scaling) / ALLPW (contact penalty) contamination
    "force_magnitude",   
    "strain_level",      
    "friction_physics",  
    "steady_state",      # RF/SCOF plateau over the second half of the scratch
    "settling",          # kinetic energy decayed before trusting the residual profile
    "recovery",          # residual groove expected (dissipative)
)

# Configurations of polymer families using polymer_defaut and then adding the wanted models
"""
Currently available models :
- Linear Elasticity, Mooney-Rivlin, Arruda-Boyce, Yeoh, Ogden
- Von Mises, Drucker-Prager
- Prony Series
"""

def _elastomer_mr_config():
    # HyperElastic Elastomer
    cfg = Simulation_Config.polymer_default()                                                   # HyperElastic Model based on polymer_default
    cfg.solver.target_time_increment = 15.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x)    # 15 : results from the target time study (-5 for safety)
    return cfg

def _semicrystalline_config():
    # Rigid semicrystalline (HDPE)
    # Rate: Eyring slope S ~ 2.5 MPa/decade (yield ~ 26-33 MPa), used in Cowper-Symonds fit for scratch rates from  1/s to 1e3/s
    
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=0.95e-9,                                                                                      # 930kg/m3 for soft, 950kg/m3 for rigid
        hyperelastic=LinearElastic_Config(E=1000.0, nu=0.42),                                             # (200,0.4) for soft, (1000,0.42) for rigid
        plasticity=J2Plasticity_Config(
            yield_table=gsell_jonas_table(sigma_y0=28.0, h=0.22, Q=5.0, b=8.0, eps_max=3.0, n_points=60), # for rigid
          # yield_table=gsell_jonas_table(sigma_y0=10.0, h=0.20, Q=6.0, b=6.0, eps_max=3.0, n_points=60), # for soft 
          # yield_table=((28.0, 0.0), (30.0, 0.2), (40.0, 1.0), (60.0, 1.9)) without gsell_jonas
            rate_dependent=None),
            # rate_dependent=RateDependent_Config.from_eyring(sigma_y0=28.0, S_per_decade=2.5)),          # Rate: Eyring slope S ~ 2.5 MPa/decade (yield ~ 26-33 MPa), used in Cowper-Symonds fit for scratch rates from  1/s to 1e3/s
        friction=Friction_Config.briscoe(tau0=1.5, alpha=0.15),                                           # Plausible value for tau0 and alpha, to be determined
        #friction=Friction_Config(mu=0.3),
        family="semicrystalline_j2",
    )
    cfg.solver.target_time_increment = 30.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 30 : results from the target time study (-10 for safety)
    return cfg

def _semicrystalline_dp_config():
    # Rigid semicrystalline (HDPE) with pressure-dependent Drucker-Prager.
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=0.95e-9,                                                                                      # same as semicrystalline_j2 (rigid HDPE)
        hyperelastic=LinearElastic_Config(E=1000.0, nu=0.42),
        plasticity=DruckerPrager_Config(
            friction_angle=12.0, flow_stress_ratio=1.0, dilation_angle=5.0,                              # PLACEHOLDERS, to be calibrated
            yield_table=gsell_jonas_table(sigma_y0=28.0, h=0.22, Q=5.0, b=8.0, eps_max=3.0, n_points=60), # same table as semicrystalline_j2
            rate_dependent=None),
            # rate_dependent=RateDependent_Config.from_eyring(sigma_y0=28.0, S_per_decade=2.5)), # usual semicrystalline Eyring slope
        friction=Friction_Config.briscoe(tau0=1.5, alpha=0.15),                                           # Plausible value for tau0 and alpha, to be determined
        #friction=Friction_Config(mu=0.3),
        family="semicrystalline_dp",
    )
    cfg.solver.target_time_increment = 30.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 30 : same s as semicrystalline_j2 (same rho/E/mesh -> same natural dt)
    return cfg

def _elastomer_ve_config():
    # Arruda-Boyce + Prony viscoelasticity . The hyperelastic
    # constants define the INSTANTANEOUS response; long-term modulus is
    # (1 - sum g_i) times it. tau_i are LAB-time values chosen so the Deborah
    # numbers De = tau*v/(2a) span [0.1, 10] at v = 20 mm/s, a ~ 0.12 mm
    # (contact time 2a/v ~ 1.2e-2 s). If the simulation compresses time,
    # set solver.time_scale_factor: the material builder divides every tau by
    # it (see Solver_Config) so De is preserved.
    cfg = Simulation_Config.polymer_default()               # Careful to have AB in polymer-default
    # cfg.scratch.recovery_time=cfg.scratch.scratch_time      # Recovery time has to be longer for Visco-Elastic families (5x usually, 1x for now to decrease simualtion time)
    cfg.solver.n_field_frames_recovery=50                   # Increasing the number of frames for recovery
    cfg.material.viscoelastic = Prony_Config(
        prony_table=((0.25, 0.0, 1.5e-3),
                     (0.35, 0.0, 6.0e-3),
                     (0.20, 0.0, 2.5e-2)))   # sum g = 0.80 
    cfg.material.family = "elastomer_ve"
    cfg.solver.target_time_increment = 15.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 15 : Same as MR
    return cfg

def _glassy_config():
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=1.18e-9,                                            # 1180 kg/m3 
        hyperelastic=LinearElastic_Config(E=2400.0, nu=0.38),  
        plasticity=DruckerPrager_Config(
            friction_angle=25.0, flow_stress_ratio=0.85, dilation_angle=10.0,
            yield_table=((60.0, 0.0), (70.0, 0.1), (80.0, 0.4)),
            rate_dependent=None),
            # rate_dependent=RateDependent_Config.from_eyring(sigma_y0=60.0, S_per_decade=5.0)),   # usual glassy Eyring slope
        viscoelastic=None,                                                                       # Viscoelastic cannot be combined with plasticity
                     # Prony_Config(prony_table=((0.2, 0.0, 0.1), (0.1, 0.0, 0.001))),
        friction=Friction_Config.briscoe(tau0=3.0, alpha=0.2),                                           # Plausible value for tau0 and alpha, to be determined
        #friction=Friction_Config(mu=0.3),
        family="glassy_dp",
    )
    # cfg.solver.mass_scale = 500        # MS convergence study: < 5% only for MS <= 500.
    cfg.solver.target_time_increment = 15.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 15 : Same as PC
    return cfg

def _glassy_pc_config():
    # Polycarbonate: E = 2300 MPa, nu = 0.37, rho = 1200 kg/m3
    # DP friction angle beta = 15 deg 
    # K = 1.0  
    # dilation psi = 8 deg (small plastic dilatancy).
    # Usual true yield peak ~70 MPa (quasi-static)
    # Softening ~12 MPa over eps_s ~ 0.05, orientation hardening h = 0.35 (cf Mulliken-Boyce 2006 / van Breemen EGP data trends )
    # Rate: Eyring S ~ 4.5 MPa/decade below the beta-transition.
    # NB: recalibrate before quantitative comparison.
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=1.20e-9,
        hyperelastic=LinearElastic_Config(E=2300.0, nu=0.37),
        #plasticity=None,
        plasticity=DruckerPrager_Config(
            friction_angle=27.13, flow_stress_ratio=0.85, dilation_angle=0.0,
            yield_table=gsell_jonas_table(sigma_y0=70.0, h=0.35,
                                          soft_drop=12.0, eps_soft=0.05,
                                          eps_max=2.5, n_points=60),
            rate_dependent=None),
            # rate_dependent=RateDependent_Config.from_eyring(sigma_y0=70.0, S_per_decade=4.5)),
        friction=Friction_Config.briscoe(tau0=38.32, alpha=0.06),                                           # Plausible value for tau0 and alpha, to be determined
        #friction=Friction_Config(mu=0.3),
        family="glassy_pc",
    )
    cfg.solver.target_time_increment = 15.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 15 : results from the target time study (-5 for safety)
    # cfg.solver.mass_scale = 500
    return cfg


def _glassy_pmma_config():
    # PMMA: E = 3100 MPa, nu = 0.35, rho = 1190 kg/m3
    # DP beta = 20 deg 
    # K = 1.0;
    # psi = 10 deg. 
    # Usual true yield peak ~105 MPa (quasi-static)
    # Strong softening ~25 MPa over eps_s ~ 0.06, hardening h = 0.45.
    # Rate: Eyring S ~ 9 MPa/decade (PMMA is very rate-sensitive, expect beta-transition active at scratch rates).
    # NB: recalibrate before quantitative comparison.
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=1.19e-9,
        hyperelastic=LinearElastic_Config(E=3100.0, nu=0.35),
        plasticity=DruckerPrager_Config(
            friction_angle=20.0, flow_stress_ratio=1.0, dilation_angle=10.0,
            yield_table=gsell_jonas_table(sigma_y0=105.0, h=0.45,
                                          soft_drop=25.0, eps_soft=0.06,
                                          eps_max=2.0, n_points=60),
            rate_dependent=None),
            #rate_dependent=RateDependent_Config.from_eyring(sigma_y0=105.0, S_per_decade=9.0)),
        friction=Friction_Config.briscoe(tau0=4.0, alpha=0.2),                                           # Plausible value for tau0 and alpha, to be determined
        #friction=Friction_Config(mu=0.3),
        family="glassy_pmma",
    )
    cfg.solver.target_time_increment = 30.0 * natural_dt(cfg.material, cfg.mesh.fine_size_x) # 30 : results from the target time study (-10 for safety)
    # cfg.solver.mass_scale = 500
    return cfg

ELASTOMER_MR = PolymerFamily(
    key="elastomer_mr",
    label="Unfilled elastomer (Mooney-Rivlin)",
    config_factory=_elastomer_mr_config,
    checks=_ELASTOMER_CHECKS,
    sampling=None,
    description=("Pure hyperelastic Mooney-Rivlin elastomer; quasi-incompressible, "
                 "full groove recovery expected (no plasticity / damage)."),
)

SEMICRYSTALLINE_J2 = PolymerFamily(
    key="semicrystalline_j2",
    label="Soft semicrystalline (linear elastic + J2 plasticity)",
    config_factory=_semicrystalline_config,
    checks=_SEMICRYSTALLINE_CHECKS,
    sampling=None,
    description=("Linear-elastic base + isochoric J2 plasticity; "
                 "permanent groove + pile-up expected. (J2 is plastically incompressible)"),
)

SEMICRYSTALLINE_DP = PolymerFamily(
    key="semicrystalline_dp",
    label="Soft semicrystalline (linear elastic + Drucker-Prager plasticity)",
    config_factory=_semicrystalline_dp_config,
    checks=_SEMICRYSTALLINE_CHECKS,
    sampling=None,
    description=("Linear-elastic base + pressure-dependent Drucker-Prager "
                 "plasticity; same calibration as semicrystalline_j2 with a "
                 "placeholder beta=12 deg -- isolates pressure sensitivity."),
)

GLASSY_DP = PolymerFamily(
    key="glassy_dp",
    label="Glassy amorphous thermoplastic (linear elastic + Drucker-Prager)",
    config_factory=_glassy_config,
    checks=_GLASSY_CHECKS,
    sampling=None,
    description=("Linear-elastic base + pressure-dependent Drucker-Prager "
                 "plasticity; permanent groove expected."),
)

ELASTOMER_VE = PolymerFamily(
    key="elastomer_ve",
    label="Viscoelastic elastomer (Arruda-Boyce + Prony)",
    config_factory=_elastomer_ve_config,
    checks=_ELASTOMER_VE_CHECKS,
    sampling=None,
    description=("Instantaneous Arruda-Boyce + 3-term Prony series; "
                 "rate-dependent friction hysteresis and delayed groove "
                 "recovery expected (recovery check excluded by design)."),
)

GLASSY_PC = PolymerFamily(
    key="glassy_pc",
    label="Polycarbonate (elastic + Drucker-Prager, softening + G'Sell hardening, rate-dependent)",
    config_factory=_glassy_pc_config,
    checks=_GLASSY_CHECKS,
    sampling=None,
    description=("Literature based PC: beta=15 deg, intrinsic softening, "
                 "exponential orientation hardening, Cowper-Symonds rate "
                 "dependence fitted on an Eyring slope of 4.5 MPa/decade."),
)

GLASSY_PMMA = PolymerFamily(
    key="glassy_pmma",
    label="PMMA (elastic + Drucker-Prager, softening + G'Sell hardening, rate-dependent)",
    config_factory=_glassy_pmma_config,
    checks=_GLASSY_CHECKS,
    sampling=None,
    description=("Literature based PMMA: beta=20 deg, strong intrinsic "
                 "softening, h=0.45 orientation hardening, Cowper-Symonds "
                 "rate dependence fitted on 9 MPa/decade."),
)


# Registry of all implemented families.
FAMILIES = {
    ELASTOMER_MR.key: ELASTOMER_MR,
    ELASTOMER_VE.key: ELASTOMER_VE,
    SEMICRYSTALLINE_J2.key: SEMICRYSTALLINE_J2,
    SEMICRYSTALLINE_DP.key: SEMICRYSTALLINE_DP,
    GLASSY_DP.key: GLASSY_DP,
    GLASSY_PC.key: GLASSY_PC,
    GLASSY_PMMA.key: GLASSY_PMMA,
}


def get_family(key):
    if key not in FAMILIES:
        raise ValueError(
            "Unknown polymer family '%s'. Available: %s"
            % (key, ", ".join(sorted(FAMILIES)))
        )
    return FAMILIES[key]