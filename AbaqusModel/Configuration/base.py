# Useful classes for the simulation 

import numpy as np

# 1. Indenter (Rockwell)
class Indenter_Config:
    # Creates the indenter configuration according to the specified values
    # As of now, only the Rockwell indenter

    ROCKWELL = "rockwell"

    def __init__( self, indenter_type="rockwell", tip_radius=0.2, cone_angle=60, rigid=True ):          
    # [mm] [degrees] 

        self.indenter_type = indenter_type
        self.tip_radius = tip_radius
        self.cone_angle = cone_angle

    def Rockwell_coords(self):
    # Returns a dictionnary with the Indenter coordinates

        # Indenter parameters 
        R = self.tip_radius
        theta = self.cone_angle
        rad = np.pi / 180.0

        # Indenter coordinates generation
        xc1 = 0.0
        yc1 = 0.0
        xc2 = R * np.cos(-theta * rad)
        yc2 = R + R * np.sin(-theta * rad)
        xc3 = R * np.cos((-theta - (90.0 - theta) / 2.0) * rad)
        yc3 = R + R * np.sin((-theta - (90.0 - theta) / 2.0) * rad)
        xl1 = xc2
        yl1 = yc2
        xl2 = xl1 + 0.5 * np.cos((90.0 - theta) * rad)
        yl2 = yl1 + 0.5 * np.sin((90.0 - theta) * rad)

        return dict( xc1=xc1, yc1=yc1, xc2=xc2, yc2=yc2, xc3=xc3, yc3=yc3, xl1=xl1, yl1=yl1, xl2=xl2, yl2=yl2)
     
# 2. Substrate
class Substrate_Config:
    # Dimensions and partitions of the substrate block

    def __init__(self,
                 xs1=0.0, ys1=0.0, zs1=0.0,             # Substrate box  (origin at xs1, ys1, zs1)
                 xs2=0.6, ys2=0.5, zs2=3.0,             # Width, height  and depth of the box [mm] (z is the scratch direction)
                 dpo_x=0.24, dpo_y=0.18, dpo_z=0.25 ):  # Partition offsets (from edges of refined zone) 

        self.xs1 = xs1
        self.ys1 = ys1
        self.zs1 = zs1
        self.xs2 = xs2
        self.ys2 = ys2
        self.zs2 = zs2
        self.dpo_x = dpo_x
        self.dpo_y = dpo_y
        self.dpo_z = dpo_z

# 3. Mesh
class Mesh_Config:
    # Mesh size and Element control

    def __init__(self,
                 fine_size_x=0.020, fine_size_y=0.020, fine_size_z=0.020,      # Fine mesh sizes in the refined contact zone (actual values to be determined after mesh convergence)
                 coarse_size_0=0.05, coarse_size_1=0.15, coarse_size_2=0.30,   # Coarse mesh (transition away from contact zone)
                 hourglass_control="RELAX STIFFNESS",                          # 'ENHANCED' for important deformations ('DEFAULT' otherwise)
                 distortion_control="DEFAULT",                                 # 'DEFAULT' for important deformations ('OFF' otherwise)
                 max_degradation=0.9,                                          # Best value for polymers ?
                 element_deletion=False,                                       # 'False' to capture the recovery phenomenon
                 second_order_accuracy=False):                                 # 'True' for complex models (AB, DP) (False otherwise) (increases simulation time)

        self.fine_size_x = fine_size_x
        self.fine_size_y = fine_size_y
        self.fine_size_z = fine_size_z
        self.coarse_size_0 = coarse_size_0
        self.coarse_size_1 = coarse_size_1
        self.coarse_size_2 = coarse_size_2
        self.hourglass_control = hourglass_control
        self.distortion_control = distortion_control
        self.max_degradation = max_degradation
        self.element_deletion = element_deletion
        self.second_order_accuracy = second_order_accuracy

# 4a. Linear Elastic Model (for glassy / semicrystalline bases)
class LinearElastic_Config:

    MODEL = "elastic"

    def __init__(self, E=200.0, nu=0.40):
        self.E = E
        self.nu = nu

    def params(self):
        return {"E": self.E, "nu": self.nu}


# 4b. Hyper-elastic Models (Mooney-Rivlin)
class HE_Model_Config:
    # For now, only the Mooney-Rivlin model
    #  W = C10 * (I1_bar - 3) + C01 * (I2_bar - 3) + (1/D1) * (J_el - 1)^2
    # Must still use a sampling method
   
    MODEL = "mooney_rivlin"

    def __init__(self, C10=1.0, C01=0.1, D1=0.018): # First and second parameter [MPa], Compressibility parameter [1/MPa]
        # change to MPa
        self.C10 = C10   
        self.C01 = C01   
        self.D1 = D1   

    def params(self):
        return {"C10": self.C10, "C01": self.C01, "D1": self.D1}

# 5. Visco-elastic Models (empty)
class VE_Model_Config:
    MODEL = "none"

    def params(self):
        return {}

# 6. Plasticity Models (empty)
class P_Model_Config:
    MODEL = "none"

    def params(self):
        return {}

# 6b. J2 / von Mises plasticity (isochoric, pressure-independent)
class J2Plasticity_Config:
    # yield_table: ((yield_stress [MPa], plastic_strain [-]), ...), first point = initial yield.
    MODEL = "mises"

    def __init__(self, yield_table=((10.0, 0.0), (14.0, 0.2), (18.0, 0.6))):
        self.yield_table = tuple(tuple(pt) for pt in yield_table)

    def params(self):
        # Expose the initial yield stress for the CSV / verifier; the full
        # hardening table is used only by the material assignment.
        return {"sigma_y0": self.yield_table[0][0]}

# 7. Scratching (Progressive and Constant)
class Scratch_Config:

    PROGRESSIVE = "progressive"
    CONSTANT = "constant"

    DISPLACEMENT = "displacement"
    FORCE = "force"

    def __init__(self,
                 depth_mode="constant",
                 control_mode="displacement",
                 scratch_length=2.0, 
                 scratch_force=20e-3,                                                                           # [N] for force driven scratch (>0)
                 scratch_depth=-40e-3,                                                                          # [mm] for dispalcement driven scratch (<0)
                 scratch_time=0.01, indentation_time=0.001, unload_time=0.0001, recovery_time=1.0,              # [s] To be studied
                 recovery_lift=0.05,                                                                            # [mm] clearance above surface during recovery
                 n_field_frames=20, n_field_frames_recovery=50, n_history_points=100 ):                         # Number of frames / field outputs for each step
        
        if depth_mode not in (self.PROGRESSIVE, self.CONSTANT):
            raise ValueError("depth_mode must be 'progressive' or 'constant', got '%s'" % depth_mode)
        
        if control_mode not in (self.DISPLACEMENT, self.FORCE):
            raise ValueError("control_mode must be 'displacement' or 'force', got '%s'" % control_mode)
              
        if control_mode == self.DISPLACEMENT and recovery_lift <= 0.0 and recovery_time > 0.0:
            raise ValueError("recovery_lift must be positive to ensure indenter separation during recovery")
            
        self.depth_mode = depth_mode
        self.control_mode = control_mode
        self.scratch_length = scratch_length
        self.scratch_force = scratch_force
        self.scratch_depth = scratch_depth
        self.scratch_time = scratch_time
        self.indentation_time = indentation_time
        self.unload_time = unload_time
        self.recovery_time = recovery_time
        self.recovery_lift = recovery_lift
        self.n_field_frames = n_field_frames
        self.n_field_frames_recovery = n_field_frames_recovery
        self.n_history_points = n_history_points


    # Functions to gather information about the Scratching for other files
    @property
    def has_recovery_step(self): # True if there is a post-unload recovery step.
        return self.recovery_time > 0.0

    @property
    def t_indent_end(self): # End of indentation phase [s]. Returns 0 in progressive mode.
        if self.depth_mode == self.CONSTANT:
            return self.indentation_time
        return 0.0

    @property
    def t_scratch_end(self): # End of scratching phase [s].
        return self.t_indent_end + self.scratch_time

    @property
    def t_unload_end(self): # End of unloading phase [s].
        return self.t_scratch_end + self.unload_time

    @property
    def t_recovery_end(self): # End of recovery phase [s]. Equals t_unload_end if no recovery.
        return self.t_unload_end + self.recovery_time

    @property
    def total_time(self): # Total simulation time including all phases [s].
        return self.t_recovery_end
    
    @property
    def field_interval_indentation(self): # Field output interval during indentation [s]. Constant mode only.
        if self.depth_mode == self.CONSTANT:
            return self.indentation_time / max(self.n_field_frames // 4, 1)
        return None

    @property
    def field_interval_scratch(self): # Field output interval during scratch [s].
        return self.scratch_time / self.n_field_frames

    @property
    def field_interval_unload(self): # Field output interval during unloading [s].
        return self.unload_time / self.n_field_frames

    @property
    def field_interval_recovery(self): # Field output interval during recovery [s].
        if self.has_recovery_step:
            return self.recovery_time / self.n_field_frames_recovery
        return None

    @property
    def history_interval(self): # History output interval during scratch [s].
        return self.scratch_time / self.n_history_points
    
    @property
    def is_force_controlled(self): 
        return self.control_mode == self.FORCE

    @property
    def uses_single_amplitude(self): 
        return (not self.is_force_controlled
                and self.depth_mode == self.PROGRESSIVE
                and not self.has_recovery_step)

    #  Amplitude tables for Abaqus 
    def depth_amplitude(self):
        # Amplitude table for the depth (u2) displacement BC according to the scratching type
       
        t1 = self.t_indent_end
        t2 = self.t_scratch_end
        t3 = self.t_unload_end

        if self.has_recovery_step:
            t4 = self.t_recovery_end
            lift_value = self.recovery_lift / self.scratch_depth  # negative number
        
        if self.depth_mode == self.PROGRESSIVE:
            if not self.has_recovery_step:
                return ((0.0, 0.0),(t2,  1.0),(t3,  0.0))
            else:
                return ((0.0,  0.0),(t2,   1.0),(t3,   lift_value),(t4,   lift_value))
        
        else:  
            if not self.has_recovery_step:
                return ((0.0, 0.0),(t1,  1.0),(t2,  1.0),(t3,  0.0))
            else:
                return ((0.0,  0.0),(t1,   1.0),(t2,   1.0),(t3,   lift_value),(t4,   lift_value))

    def length_amplitude(self):
        # Amplitude table for the length (u3) displacement BC according to the scratching type
       
        t1 = self.t_indent_end
        t2 = self.t_scratch_end
        t3 = self.t_unload_end

        if self.depth_mode == self.PROGRESSIVE:
            if not self.has_recovery_step:
                return ((0.0, 0.0),(t2,  1.0),(t3,  0.0))
            else:
                t4 = self.t_recovery_end
                return ((0.0,  0.0),(t2,   1.0),(t3,   1.0),(t4,   1.0))
        else:  
            if not self.has_recovery_step:
                return ((0.0, 0.0),(t1,  0.0),(t2,  1.0),(t3,  1.0))
            else:
                t4 = self.t_recovery_end
                return ((0.0,  0.0),(t1,   0.0),(t2,   1.0),(t3,   1.0),(t4,   1.0))
            
    def force_amplitude(self):
        # Amplitude table for the force (cf2), goes back to 0 at unload

        t1 = self.t_indent_end
        t2 = self.t_scratch_end
        t3 = self.t_unload_end

        if self.depth_mode == self.PROGRESSIVE:
            if not self.has_recovery_step:
                return ((0.0, 0.0), (t2, 1.0), (t3, 0.0))
            else:
                t4 = self.t_recovery_end
                return ((0.0, 0.0), (t2, 1.0), (t3, 0.0), (t4, 0.0))
        else:
            if not self.has_recovery_step:
                return ((0.0, 0.0), (t1, 1.0), (t2, 1.0), (t3, 0.0))
            else:
                t4 = self.t_recovery_end
                return ((0.0, 0.0), (t1, 1.0), (t2, 1.0), (t3, 0.0), (t4, 0.0))

# 8. Damage Models (empty)
class Damage_Config:
    MODEL = "none"

    def params(self):
        return {}

# 9. Friction Models (Pressure independent)
class Friction_Config:
    # For now, pressure independent

    def __init__(self, mu=0.3, formulation="penalty", elastic_slip_fraction=0.005, pressure_dependent=False):

        self.mu = mu
        self.formulation = formulation
        self.elastic_slip_fraction = elastic_slip_fraction
        self.pressure_dependent = pressure_dependent

# 10. Material specification
class Material_Config:
    # Complete definition of the desired models for the material behavior

    def __init__(self,
                 rho=1.2e-9,
                 hyperelastic=None,
                 viscoelastic=None,
                 plasticity=None,
                 damage=None,
                 friction=None,
                 family="elastomer_mr"):

        self.rho = rho
        self.hyperelastic = hyperelastic or HE_Model_Config()
        self.viscoelastic = viscoelastic or VE_Model_Config()
        self.plasticity = plasticity or P_Model_Config()
        self.damage = damage or Damage_Config()
        self.friction = friction or Friction_Config()
        self.family = family

    def to_dict(self):

        d = {"rho": self.rho}
        d.update(self.hyperelastic.params())
        d.update(self.viscoelastic.params())
        d.update(self.plasticity.params())
        d.update(self.damage.params())
        d["mu_friction"] = self.friction.mu

        return d

# 11. Solver 
class Solver_Config:

    def __init__(self,
                 mass_scale=1000,
                 target_time_increment=0.0,
                 use_ALE=True,
                 num_cpus=20,
                 linear_bulk_viscosity=0.06, quad_bulk_viscosity=1.2, # Default Abaqus values
                 ale_frequency=20, ale_mesh_sweeps=1, ale_smoothing_priority="GRADED", ale_smoothing_algorithm="GEOMETRY_ENHANCED"): # ALE parameters
    
        self.mass_scale = mass_scale
        self.target_time_increment = target_time_increment
        self.use_ALE = use_ALE
        self.num_cpus = num_cpus
        self.num_domains = num_cpus
        self.linear_bulk_viscosity = linear_bulk_viscosity
        self.quad_bulk_viscosity = quad_bulk_viscosity
        self.ale_frequency = ale_frequency
        self.ale_mesh_sweeps = ale_mesh_sweeps
        self.ale_smoothing_priority = ale_smoothing_priority
        self.ale_smoothing_algorithm = ale_smoothing_algorithm

# 12. Outputs
class Output_Config:

    def __init__(self,
                 field_variables=None,
                 contact_force_variables=None,
                 history_force_variables=None,
                 history_energy_substrate=None,
                 history_energy_whole=None):

        

        self.field_variables = field_variables or ("S", "MISES", "TRIAX",                           # Stress Distributions
                                                   "PRESS",                                         # Pressure distribution
                                                   "LE", "NE", "PE", "PEEQ",                        # Deformation distributions
                                                   "U", "COORD",                                    # Displacement Distributions
                                                   "SDV", "SDEG", "STATUS", "CSTRESS")              # State and damage of the mesh
        self.contact_force_variables = contact_force_variables or ("CFORCE",)
        self.history_force_variables = history_force_variables or ("RF1", "RF2", "RF3")             # Reaction forces
        self.history_energy_substrate = history_energy_substrate or ("ALLKE", "ALLIE", "ALLAE")     # Substrate energy values 
        self.history_energy_whole = history_energy_whole or ("ALLKE", "ALLIE", "ALLVD", "ALLFD",    # Whole model energy values
                                                             "ALLWK", "ALLPW", "ALLCW", "ALLMW", "ETOTAL")

# 13. Naming conventions
class Naming_Config:

    def __init__(self,
                 model_name="Model-1",
                 indenter_name="RockwellIndenter",
                 substrate_name="Substrate",
                 master_surface="m_Surf-1",
                 slave_surface="s_Surf-1",
                 contact_region_nodes="contactRegionNodes"):

        # Model 
        self.model_name = model_name

        # Parts / instances / sets 
        self.indenter_name = indenter_name
        self.substrate_name = substrate_name
        self.indenter_set = indenter_name + "Set"
        self.indenter_instance = indenter_name + "Inst"
        self.substrate_set = substrate_name + "Set"
        self.substrate_instance = substrate_name + "Inst"
        self.refined_set = "RefinedArea"
        self.inertia_name = "IndenterInertia"

        # Surfaces / contact-node set 
        self.master_surface = master_surface
        self.slave_surface = slave_surface
        self.contact_region_nodes = contact_region_nodes

        # Contact (property + interaction)
        self.contact_property = "IntProp-1"
        self.contact_interaction = "Int-1"

        # Boundary-condition sets & BCs
        self.fixed_set = "FIXEDBCSET"
        self.symmetry_set = "XsymmetryBCSet"
        self.fixed_bc = "Fixed_constraint"
        self.symmetry_bc = "x_axis_symmetry"
        self.indenter_constraint_bc = "IndenterConstraint"

        # Loading: amplitudes & displacement BCs 
        self.amp_single = "Amp-1"
        self.amp_depth = "Amp-Depth"
        self.amp_length = "Amp-Length"
        self.bc_scratch = "IndenterScratching"
        self.bc_depth = "IndenterDepth"
        self.bc_travel = "IndenterTravel"

        # Output requests 
        self.out_reaction = "ReactionForces"
        self.out_indenter_disp = "IndenterDisp"
        self.out_energy_substrate = "Energy"
        self.out_energy_whole = "EnergyBalance"
        self.out_field = "FieldOutput"
        self.out_contact = "ContactForce"

        # Material / section 
        self.material_name = "SubstrateMaterial"
        self.section_name = "SubstrateSection"

        # ALE adaptive meshing 
        self.ale_control = "Ada-1"
        self.ale_domain_set = "ALE_Domain"

        # Steps 
        self.step_indent = "IndentationStep"
        self.step_scratch = "ScratchStep"
        self.step_unload = "UnloadStep"
        self.step_recovery = "RecoveryStep"

# 14. Simulation
class Simulation_Config:

    def __init__(self,
                 indenter=None,
                 substrate=None,
                 mesh=None,
                 material=None,
                 solver=None,
                 scratch=None,
                 output=None,
                 naming=None,
                 job_name="ScratchTest",
                 sheet_size=10):
        
        self.indenter = indenter or Indenter_Config()
        self.substrate = substrate or Substrate_Config()
        self.mesh = mesh or Mesh_Config()
        self.material = material or Material_Config()
        self.solver = solver or Solver_Config()
        self.scratch = scratch or Scratch_Config()
        self.output = output or Output_Config()
        self.naming = naming or Naming_Config()
        self.job_name = job_name
        self.sheet_size = sheet_size

    @staticmethod
    def polymer_default():
        # Typical polymer scratch test configuration.
        return Simulation_Config(
            indenter=Indenter_Config(),
            substrate=Substrate_Config(),
            mesh=Mesh_Config(
                fine_size_x=0.030,       
                fine_size_y=0.030,
                fine_size_z=0.030,    
                coarse_size_0=0.06,     # *2
                coarse_size_1=0.12,     # *2
                coarse_size_2=0.24,     # *2
                hourglass_control="RELAX STIFFNESS",      # RELAX STIFFNESS Might be innacurate but only one usable for now
                distortion_control="DEFAULT",
                max_degradation=0.9,
                element_deletion=False,
                second_order_accuracy=False,
            ),
            material=Material_Config(
                rho=1.2e-9,
                hyperelastic=HE_Model_Config(C10=1.0, C01=0.1, D1=1.8e-2),
                viscoelastic=None,
                plasticity=None,
                damage=None,
                friction=Friction_Config(),
                family="elastomer_mr",
            ),
            solver=Solver_Config(
                mass_scale=1000,    
                target_time_increment=0.0,
                use_ALE=False,
                num_cpus=36,
                linear_bulk_viscosity=0.06,
                quad_bulk_viscosity=1.2,
                ale_frequency=20,
                ale_mesh_sweeps=1,
                ale_smoothing_priority="GRADED",
                ale_smoothing_algorithm="GEOMETRY_ENHANCED",
            ),
            scratch=Scratch_Config(
                depth_mode=Scratch_Config.PROGRESSIVE,
                control_mode=Scratch_Config.DISPLACEMENT,
                scratch_length=2.0,
                scratch_force=20e-3,
                scratch_depth=-40e-3,
                scratch_time=0.1,
                indentation_time=0.01,
                unload_time=0.01,
                recovery_time=0.0005,
                recovery_lift=0.05,
                n_field_frames=40,
                n_field_frames_recovery=10,
                n_history_points=100,
            ),
            output=Output_Config(),
            naming=Naming_Config(),
            job_name="PolymerScratch",
            sheet_size=10,
        )