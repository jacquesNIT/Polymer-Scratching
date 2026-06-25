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
    
    _HYPERELASTIC_BUILDERS = {"mooney_rivlin": "_mooney_rivlin"}
    _VISCOELASTIC_BUILDERS = {"none": "_skip"}
    _PLASTICITY_BUILDERS   = {"none": "_skip"}
    _DAMAGE_BUILDERS       = {"none": "_skip"}

    def create_material(self):
        # Build the Abaqus material from Material_Config.

        # Remove old material if re-running in a loop
        if self.names.material_name in self.model.materials.keys():
            del self.model.materials[self.names.material_name]

        self.mat = self.model.Material(name=self.names.material_name)
        mc = self.mat_cfg

        # 2-5. Constitutive blocks, dispatched by their MODEL string
        self._apply_block(mc.hyperelastic, self._HYPERELASTIC_BUILDERS, "hyperelastic")
        self._apply_block(mc.viscoelastic, self._VISCOELASTIC_BUILDERS, "viscoelastic")
        self._apply_block(mc.plasticity,   self._PLASTICITY_BUILDERS,   "plasticity")
        self._apply_block(mc.damage,       self._DAMAGE_BUILDERS,       "damage")

        return self.mat
    
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



    #  Hyperelastic models
    def _mooney_rivlin(self, h):
        self.mat.Hyperelastic(materialType=ISOTROPIC, type=MOONEY_RIVLIN, testData=OFF, table=((h.C10, h.C01, h.D1),))

    #  Viscoelastic models

    #  Plasticity models

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

        if f.pressure_dependent:
            raise NotImplementedError
        else:
            # Constant Coulomb friction
            self.model.interactionProperties["IntProp-1"].tangentialBehavior.setValues(table=((f.mu,),))
