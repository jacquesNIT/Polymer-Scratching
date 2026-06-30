# Polymer-family registry.
from .base import (Simulation_Config, Material_Config,
                   LinearElastic_Config, J2Plasticity_Config,
                   DruckerPrager_Config, Prony_Config, Friction_Config)



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
    "d1_validity",       # Specifically for Mooney-Rivlin
    "force_magnitude",   
    "strain_level",      
    "friction_physics", 
    "recovery",          # residual ~ 0 (pure hyperelastic)
)

# Verifier checks applicable to a dissipative (plastic) family.
_SEMICRYSTALLINE_CHECKS = (
    "quasi_static",      
    "hourglass",         
    "energy_total",      
    "force_magnitude",   
    "strain_level",      
    "friction_physics",  
    "recovery",          # residual groove expected (plastic)
)

_GLASSY_CHECKS = (
    "quasi_static",      
    "hourglass",         
    "energy_total",      # ETOTAL drift (now includes viscous dissipation ALLVD)
    "force_magnitude",   
    "strain_level",      
    "friction_physics",  
    "recovery",          # residual groove expected (dissipative)
)

# Configurations of polymer families using polymer_defaut and then adding the wanted models

def _semicrystalline_config():
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=0.93e-9,                                            # 930kg/m3 for soft, 950kg/m3 for rigid
        hyperelastic=LinearElastic_Config(E=200.0, nu=0.40),    # (200,0.4) for soft, (1000,0.42) for rigid
        plasticity=J2Plasticity_Config(
            yield_table=((28.0, 0.0), (30.0, 0.2), (40.0, 1.0), (60.0, 1.9))),           # For rigid, Coherent paramters for the study, need to adjust later
            # yield_table=((10.0, 0.0), (14.0, 0.2), (18.0, 0.6))),                      # For soft
        friction=Friction_Config(mu=0.3),
        family="semicrystalline_j2",
    )
    return cfg

def _glassy_config():
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=1.18e-9,                                            # 1180 kg/m3 
        hyperelastic=LinearElastic_Config(E=2400.0, nu=0.38),  
        plasticity=DruckerPrager_Config(
            friction_angle=25.0, flow_stress_ratio=0.85, dilation_angle=10.0,
            yield_table=((60.0, 0.0), (70.0, 0.1), (80.0, 0.4))),
        viscoelastic=None,          # Viscoelastic cannot be combined with plasticity
                     # Prony_Config(prony_table=((0.2, 0.0, 0.1), (0.1, 0.0, 0.001))),
        friction=Friction_Config(mu=0.3),
        family="glassy_dp",
    )
    cfg.solver.mass_scale = 5000       # E much higher than elastomers, need to compensate with more mass scaling (to be decided)
    return cfg

ELASTOMER_MR = PolymerFamily(
    key="elastomer_mr",
    label="Unfilled elastomer (Mooney-Rivlin)",
    config_factory=Simulation_Config.polymer_default,
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

GLASSY_DP = PolymerFamily(
    key="glassy_dp",
    label="Glassy amorphous thermoplastic (elastic + Drucker-Prager or elastic + Prony)",
    config_factory=_glassy_config,
    checks=_GLASSY_CHECKS,
    sampling=None,
    description=("Linear-elastic base + pressure-dependent Drucker-Prager "
                 "plasticity + Prony viscoelasticity; permanent groove expected. "),
)



# Registry of all implemented families.
FAMILIES = {
    ELASTOMER_MR.key: ELASTOMER_MR,
    SEMICRYSTALLINE_J2.key: SEMICRYSTALLINE_J2,
    GLASSY_DP.key: GLASSY_DP,
}


def get_family(key):
    if key not in FAMILIES:
        raise ValueError(
            "Unknown polymer family '%s'. Available: %s"
            % (key, ", ".join(sorted(FAMILIES)))
        )
    return FAMILIES[key]