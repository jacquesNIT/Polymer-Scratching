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
    
    _HYPERELASTIC_BUILDERS = {"mooney_rivlin": "_mooney_rivlin", "elastic": "_linear_elastic","arruda_boyce": "_arruda_boyce", "yeoh": "_yeoh", "ogden": "_ogden"}
    _VISCOELASTIC_BUILDERS = {"none": "_skip", "prony": "_prony"}
    _PLASTICITY_BUILDERS   = {"none": "_skip", "mises": "_j2_plasticity", "drucker_prager": "_drucker_prager"}
    _DAMAGE_BUILDERS       = {"none": "_skip"}

    #  Base-elasticity models that are hyperelastic (mutually exclusive with plasticity)
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
        # Abaqus forbids combining a (true) hyperelastic base with metal plasticity.
        # A linear-elastic base ("elastic") + plasticity is the valid plastic combo.
        base = mc.hyperelastic.MODEL
        plast = mc.plasticity.MODEL
        visco = mc.viscoelastic.MODEL
        if base in self._HYPERELASTIC_MODELS and plast != "none":
            raise ValueError(
                "Invalid material: hyperelastic base '%s' cannot be combined with "
                "plasticity '%s' (mutually exclusive families in Abaqus). "
                "Use a linear-elastic base for plastic families." % (base, plast))
 
        # Abaqus/Explicit: *VISCOELASTIC is forbidden with any plasticity option.
        # Runtime error: "THE LINEAR VISCOELASTIC MODEL MAY NOT BE USED WITH
        #                 ANY OF THE PLASTICITY OPTIONS"
        if visco != "none" and plast != "none":
            raise ValueError(
                "Invalid material: viscoelastic model '%s' cannot be combined with "
                "plasticity '%s' in Abaqus/Explicit (*VISCOELASTIC and *PLASTIC / "
                "*DRUCKER PRAGER are mutually exclusive). "
                "Remove viscoelasticity for plastic families, or use a purely "
                "viscoelastic model without plasticity." % (visco, plast))

    
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

    def _yeoh(self, h):
        self.mat.Hyperelastic(materialType=ISOTROPIC, type=YEOH, testData=OFF, table=((h.C10, h.C20, h.C30, h.D1, 0.0, 0.0),))

    def _ogden(self, h):
        row = []
        for m, a in zip(h.mu, h.alpha):
            row.extend([m, a])
        row.append(h.D1)
        row.extend([0.0] * (h.N - 1))
        self.mat.Hyperelastic(materialType=ISOTROPIC, type=OGDEN, testData=OFF, n=h.N, table=(tuple(row),))


    #  Viscoelastic models
    def _prony(self, v):
        # Prony_Config stores real relaxation times. 
        # Divide tau by the same factor as simulated loading : solver.time_scale_factor (= t_lab / t_sim)
        scale = float(getattr(self.cfg.solver, "time_scale_factor", 1.0) or 1.0)
        table = tuple((g, k, tau / scale) for (g, k, tau) in v.prony_table)
        if scale != 1.0:
            print("Prony: relaxation times divided by time_scale_factor=%g "
                  "(lab tau %s -> sim tau %s)"
                  % (scale, [r[2] for r in v.prony_table], [r[2] for r in table]))
        self.mat.Viscoelastic(domain=TIME, time=PRONY, table=table)

    #  Plasticity models
    def _j2_plasticity(self, p):
        pl = self.mat.Plastic(table=p.yield_table)
        self._attach_rate_dependence(pl, p, "plastic")

    def _drucker_prager(self, p):
        dp = self.mat.DruckerPrager(
            table=((p.friction_angle, p.flow_stress_ratio, p.dilation_angle),))
        dph = dp.DruckerPragerHardening(table=p.yield_table)
        self._attach_rate_dependence(dp, p, "druckerPrager", fallback=dph)

    def _attach_rate_dependence(self, parent, p, label, fallback=None):
        # Cowped-Symonds: R = 1 + (eps_rate / D)**(1/n)
        rd = getattr(p, "rate_dependent", None)
        if rd is None:
            return
        table = ((rd.D, rd.n),)
        try:
            parent.RateDependent(type=POWER_LAW, table=table)
            return
        except Exception as exc:
            err = str(exc)
        if fallback is not None:
            try:
                fallback.RateDependent(type=POWER_LAW, table=table)
                return
            except Exception as exc:
                err = "%s / %s" % (err, exc)
        print("Warning: could not attach *RATE DEPENDENT to %s (%s). "
              "The material will run RATE-INDEPENDENT." % (label, err))

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

        f = self.mat_cfg.friction
        tb = self.model.interactionProperties[self.names.contact_property].tangentialBehavior

        if getattr(f, "mu_table", None):
            # Tabular Coulomb friction mu(slip rate, contact pressure).
            # Column order enforced by Friction_Config:
            #   (mu[, slip_rate][, pressure]) matching the ON flags below.
            tb.setValues(
                formulation=PENALTY,
                slipRateDependency=ON if f.slip_rate_dependent else OFF,
                pressureDependency=ON if f.pressure_dependent else OFF,
                temperatureDependency=OFF,
                dependencies=0,
                table=f.mu_table,
                fraction=f.elastic_slip_fraction,
            )
        else:
            tb.setValues(table=((f.mu,),))
