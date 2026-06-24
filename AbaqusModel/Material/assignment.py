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

    def create_material(self):
        # Build the Abaqus material from Material_Config.

        # Remove old material if re-running in a loop
        if self.names.material_name in self.model.materials.keys():
            del self.model.materials[self.names.material_name]

        self.mat = self.model.Material(name=self.names.material_name)
        mc = self.mat_cfg

        # 1. Density 
        self.mat.Density(table=((mc.rho,),))

        # 2. Hyperelastic (MN)
        h = mc.hyperelastic
        if h.MODEL == "mooney_rivlin":
            self._mooney_rivlin(h)
        else:
            raise ValueError("Unknown hyperelastic model: %s" % h.MODEL)

        # 3. Viscoelastic 
        v = mc.viscoelastic
        if v.MODEL == "none":
            pass
        else:
            raise ValueError("Unknown viscoelastic model: %s" % v.MODEL)

        # 4. Plasticity 
        p = mc.plasticity
        if p.MODEL == "none":
            pass
        else:
            raise ValueError("Unknown plasticity model: %s" % p.MODEL)

        # 5. Damage 
        d = mc.damage
        if d.MODEL == "none":
            pass
        else:
            raise ValueError("Unknown damage model: %s" % d.MODEL)

        return self.mat


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
