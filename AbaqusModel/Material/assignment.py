# Substrate material creation and assignment for polymer scratch simulation.

from ScratchSimulation.AbaqusModel.abaqus_env import *

class SubstrateMaterialAssignment:

    def __init__(self, model, part, cfg):
                 
        self.model = model
        self.part = part
        self.cfg = cfg
        self.mat_cfg = cfg.material
        self.names = cfg.naming
        self.mat = None

    def apply(self):
        self.create_material()
        self.assign_section()
        self.update_friction()
        return self

    _HYPERELASTIC_BUILDERS = {"mooney_rivlin": "_mooney_rivlin", "elastic": "_linear_elastic", "arruda_boyce": "_arruda_boyce"}
    _VISCOELASTIC_BUILDERS = {"none": "_skip", "prony": "_prony"}
    _PLASTICITY_BUILDERS   = {"none": "_skip", "mises": "_j2_plasticity", "drucker_prager": "_drucker_prager"}
    _DAMAGE_BUILDERS       = {"none": "_skip"}

    #  Base-elasticity MODELS that are hyperelastic (mutually exclusive with plasticity)
    _HYPERELASTIC_MODELS = ("mooney_rivlin", "neo_hooke", "yeoh", "ogden", "arruda_boyce")


    def create_material(self):
        # Build the Abaqus material from Material_Config.

        # Remove old material if re-running in a loop
        if self.names.material_name in self.model.materials.keys():
            del self.model.materials[self.names.material_name]

        self.mat = self.model.Material(name=self.names.material_name)
        mc = self.mat_cfg
        self._validate_material(mc)

        self.mat.Density(table=((mc.rho,),))

        # 2-5. Constitutive blocks, dispatched by their MODEL string
        self._apply_block(mc.hyperelastic, self._HYPERELASTIC_BUILDERS, "hyperelastic")
        self._apply_block(mc.viscoelastic, self._VISCOELASTIC_BUILDERS, "viscoelastic")
        self._apply_block(mc.plasticity,   self._PLASTICITY_BUILDERS,   "plasticity")
        self._apply_block(mc.damage,       self._DAMAGE_BUILDERS,       "damage")

        return self.mat
    
    def _validate_material(self, mc):
        # Abaqus forbids combining a hyperelastic base with metal plasticity.
        # Linear elasticity + plasticity is the valid plastic combo.
        base = mc.hyperelastic.MODEL
        plast = mc.plasticity.MODEL
        visco = mc.viscoelastic.MODEL
        if base in self._HYPERELASTIC_MODELS and plast != "none":
            raise ValueError(
                "Invalid material: hyperelastic base '%s' cannot be combined with plasticity '%s'"% (base, plast))
 
        # Visco-elasticity is forbidden with any plasticity option.
        if visco != "none" and plast != "none":
            raise ValueError(
                "Invalid material: viscoelastic model '%s' cannot be combined with plasticity '%s'" % (visco, plast))

    def _apply_block(self, sub_cfg, registry, label):
        # Look up the builder for this sub-model's MODEL string and run it.
        builder_name = registry.get(sub_cfg.MODEL)
        if builder_name is None:
            raise ValueError("Unknown %s model: '%s'" % (label, sub_cfg.MODEL))
        getattr(self, builder_name)(sub_cfg)

    #  Builders
    def _skip(self, sub_cfg):
        # No-op builder for MODEL == "none".
        pass

    #  Base elasticity
    def _linear_elastic(self, e):
        self.mat.Elastic(table=((e.E, e.nu),))

    #  Hyperelastic models
    def _mooney_rivlin(self, h):
        self.mat.Hyperelastic(materialType=ISOTROPIC, type=MOONEY_RIVLIN, testData=OFF, table=((h.C10, h.C01, h.D1),))

    def _arruda_boyce(self, h):
        self.mat.Hyperelastic(materialType=ISOTROPIC, type=ARRUDA_BOYCE, testData=OFF, table=((h.mu, h.lambda_m, h.D),))

    #  Viscoelastic models
    def _prony(self, v):
        tsf = getattr(self.cfg.solver, "time_scale_factor", 1.0)
        tsf = 1.0 if tsf is None else float(tsf)
        if tsf <= 0.0:
            raise ValueError(
                "solver.time_scale_factor must be positive, got %r" % (tsf,))
        table = v.prony_table
        if tsf != 1.0:
            table = tuple(tuple(row[:-1]) + (row[-1] / tsf,) for row in table)
            print("[viscoelastic] Prony tau / time_scale_factor=%g -> tau_max=%.4g s"
                  % (tsf, max(r[-1] for r in table)))
        self.mat.Viscoelastic(domain=TIME, time=PRONY, table=table)


    #  Plasticity models
    def _j2_plasticity(self, p):
        self.mat.Plastic(table=p.yield_table)

    def _drucker_prager(self, p):
        dp = self.mat.DruckerPrager(
            table=((p.friction_angle, p.flow_stress_ratio, p.dilation_angle),))
        dp.DruckerPragerHardening(table=p.yield_table)

    #  Damage models

    #  Section assignment
    def assign_section(self):
        self.model.HomogeneousSolidSection(material=self.names.material_name,
                                           name=self.names.section_name,
                                           thickness=None)
        self.part.SectionAssignment(offset=0.0,offsetField="",
                                    offsetType=MIDDLE_SURFACE,
                                    region=self.part.sets[self.names.substrate_set],
                                    sectionName=self.names.section_name,
                                    thicknessAssignment=FROM_SECTION)


    #  Friction
    def update_friction(self):
        # Push the Friction_Config onto the contact property tangential behavior.
        #
        # Three supported cases:
        #   constant Coulomb              -> table = ((mu,),)
        #   pressure-dependent (Briscoe)  -> table = ((mu, p), ...)  + pressureDependency=ON
        #   slip-rate-dependent           -> table = ((mu, v), ...)  + slipRateDependency=ON
        #
        # Column order follows the Abaqus *FRICTION data line:
        #     mu, slip rate, contact pressure, temperature, field variables
        # The dependency flags and the table are set in ONE setValues() call so
        # Abaqus never sees an inconsistent (flags, table width) pair.
        #
        # OLD (Briscoe was configured in families.py but unreachable here):
        #   if f.pressure_dependent:
        #       raise NotImplementedError
        #   else:
        #       ...setValues(table=((f.mu,),))

        f = self.mat_cfg.friction
        tb = self.model.interactionProperties[self.names.contact_property].tangentialBehavior

        p_dep = bool(getattr(f, "pressure_dependent", False))
        r_dep = bool(getattr(f, "slip_rate_dependent", False))

        if not (p_dep or r_dep):
            tb.setValues(slipRateDependency=OFF, pressureDependency=OFF,
                         table=((f.mu,),))
            print("[friction] constant Coulomb: mu = %.4f" % f.mu)
            return

        table = getattr(f, "mu_table", None)
        if not table:
            raise ValueError(
                "Friction_Config declares pressure_dependent=%s / "
                "slip_rate_dependent=%s but mu_table is empty" % (p_dep, r_dep))

        n_col = 1 + int(r_dep) + int(p_dep)
        for row in table:
            if len(row) != n_col:
                raise ValueError(
                    "mu_table row %s has %d columns, expected %d "
                    "(mu%s%s)" % (row, len(row), n_col,
                                  ", slip_rate" if r_dep else "",
                                  ", pressure" if p_dep else ""))

        # Abaqus requires the independent variable(s) to be strictly ascending.
        if p_dep:
            p_col = n_col - 1
            p_vals = [row[p_col] for row in table]
            if any(p_vals[i + 1] <= p_vals[i] for i in range(len(p_vals) - 1)):
                raise ValueError(
                    "mu_table pressure column must be strictly ascending, got %s"
                    % (p_vals,))

        tb.setValues(
            slipRateDependency=ON if r_dep else OFF,
            pressureDependency=ON if p_dep else OFF,
            table=tuple(tuple(float(c) for c in row) for row in table),
        )

        mus = [row[0] for row in table]
        print("[friction] tabulated: pressure_dep=%s slip_rate_dep=%s "
              "rows=%d mu in [%.4f, %.4f]"
              % (p_dep, r_dep, len(table), min(mus), max(mus)))
        if p_dep:
            p_col = n_col - 1
            print("[friction] pressure range [%.3f, %.3f] MPa"
                  % (table[0][p_col], table[-1][p_col]))
