# Polymer-family registry.
#
# A PolymerFamily bundles, for one row of the polymer-families reference:
#   - a factory that returns a fully-built Simulation_Config,
#   - the verifier checks that apply to that family,
#   - (later) sampling ranges and solver/mesh overrides.
#
# Pure configuration: NO Abaqus imports here, so this module is importable
# from both the Abaqus kernel and plain CPython (sampler / verifier).
#
# Phase 0: a single family (Mooney-Rivlin elastomer) that reproduces the
# current Simulation_Config.polymer_default() exactly. Adding a family later
# = add a PolymerFamily instance and register it in FAMILIES; nothing in the
# orchestration changes.

from .base import (Simulation_Config, Material_Config,
                   LinearElastic_Config, J2Plasticity_Config, Friction_Config)


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
    "recovery",          # residual groove EXPECTED (plastic)
)

def _semicrystalline_config():
    # Reuses the geometry, scratch and outputs of polymer_default. 
    # Only the material changes 
    cfg = Simulation_Config.polymer_default()
    cfg.material = Material_Config(
        rho=0.93e-9,                                            # ~930 kg/m3
        hyperelastic=LinearElastic_Config(E=200.0, nu=0.40),    # base-elasticity slot
        plasticity=J2Plasticity_Config(
            yield_table=((10.0, 0.0), (14.0, 0.2), (18.0, 0.6))),
        friction=Friction_Config(mu=0.3),
        family="semicrystalline_j2",
    )
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


# Registry of all implemented families.
FAMILIES = {
    ELASTOMER_MR.key: ELASTOMER_MR,
    SEMICRYSTALLINE_J2.key: SEMICRYSTALLINE_J2,
}


def get_family(key):
    if key not in FAMILIES:
        raise ValueError(
            "Unknown polymer family '%s'. Available: %s"
            % (key, ", ".join(sorted(FAMILIES)))
        )
    return FAMILIES[key]