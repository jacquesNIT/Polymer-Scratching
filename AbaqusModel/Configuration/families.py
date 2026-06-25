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

from .base import Simulation_Config


class PolymerFamily:

    def __init__(self, key, label, config_factory,
                 checks=None, sampling=None, description=""):
        self.key = key                      # short tag stored on Material_Config.family
        self.label = label                  # human-readable name
        self.config_factory = config_factory  # callable -> Simulation_Config
        self.checks = list(checks) if checks else []   # verifier checks that apply
        self.sampling = sampling            # parameter-sampling config (Phase 1+)
        self.description = description

    def build_config(self):
        # Build the Simulation_Config for this family and stamp the family tag
        # onto the material so the extractor / verifier can dispatch on it.
        cfg = self.config_factory()
        cfg.material.family = self.key
        return cfg


# Verifier checks applicable to a pure-hyperelastic elastomer (names match the
# check_* functions in results_verifier). Declared here so that, from Phase 1,
# the verifier can run exactly the relevant subset per family.
_ELASTOMER_CHECKS = (
    "quasi_static",      # ALLKE/ALLIE
    "hourglass",         # ALLAE/ALLIE
    "energy_total",      # ETOTAL drift
    "d1_validity",       # K/mu window (MR-specific)
    "force_magnitude",   # RF2 vs Hertz (MR-specific)
    "strain_level",      # Tabor / MR validity (MR-specific)
    "friction_physics",  # SCOF >= mu
    "recovery",          # residual ~ 0 (pure hyperelastic)
)


ELASTOMER_MR = PolymerFamily(
    key="elastomer_mr",
    label="Unfilled elastomer (Mooney-Rivlin)",
    config_factory=Simulation_Config.polymer_default,
    checks=_ELASTOMER_CHECKS,
    sampling=None,
    description=("Pure hyperelastic Mooney-Rivlin elastomer; quasi-incompressible, "
                 "full groove recovery expected (no plasticity / damage)."),
)


# Registry of all known families.
FAMILIES = {
    ELASTOMER_MR.key: ELASTOMER_MR,
}


def get_family(key):
    # Return the PolymerFamily registered under 'key', or raise with the list
    # of available families.
    if key not in FAMILIES:
        raise ValueError(
            "Unknown polymer family '%s'. Available: %s"
            % (key, ", ".join(sorted(FAMILIES)))
        )
    return FAMILIES[key]