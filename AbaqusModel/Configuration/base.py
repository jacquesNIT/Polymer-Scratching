# Useful classes for the simulation 

import numpy as np

# 1. Indenter (Rockwell sphere-cone / pyramid)
class Indenter_Config:
    # Creates the indenter configuration according to the specified values

    ROCKWELL = "rockwell"
    PYRAMID = "pyramid"          

    def __init__( self, indenter_type="rockwell", tip_radius=0.2, cone_angle=60, rigid=True,
                  n_faces=4, face_angle=None, base_apothem=0.2, orientation="face",
                  extrude_depth=None, mesh_size=None, mesh_min_size=None, tip_bias=False ):
    # [mm] [degrees]

        self.indenter_type = indenter_type
        self.tip_radius = tip_radius        # Rockwell only [mm]
        self.cone_angle = cone_angle        # Rockwell: HALF-apex angle from the axis [deg]
        self.rigid = rigid

        # pyramid-only parameters 
        self.n_faces = int(n_faces)                 # 3 (Berkovich-like) or 4 (Vickers-like)
        self.face_angle = cone_angle if face_angle is None else face_angle
                                                    # HALF-apex angle, axis -> FACE plane [deg]
        self.base_apothem = base_apothem            # axis -> face distance at the base [mm]
        self.orientation = orientation              # "face" = a face leads +z, "edge" = an edge leads
        self.extrude_depth = extrude_depth          # [mm], None -> automatic
        self.mesh_size = mesh_size                  # [mm], None -> 0.5 * substrate fine size
        self.mesh_min_size = mesh_min_size          # [mm], only used when tip_bias is True
        self.tip_bias = tip_bias                    # bias the rigid seeds towards the apex

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

    #  Pyramidal indenter (PYRAMID_INDENTER_PATCH)
    def Pyramid_coords(self):
        n = int(self.n_faces)
        if n < 3:
            raise ValueError("n_faces must be >= 3 (got %s)" % n)

        rad = np.pi / 180.0
        theta = float(self.face_angle) * rad          # half-apex angle, axis -> face
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("face_angle must lie strictly between 0 and 90 degrees")

        a0 = float(self.base_apothem)                 # apothem of the base polygon
        R0 = a0 / np.cos(np.pi / n)                   # circumradius of the base polygon
        H = a0 / np.tan(theta)                        # apex height above the base plane

        # Azimuth of the first face normal: 0 -> a face leads the scratch (+z)
        phi0 = 0.0 if str(self.orientation).lower().startswith("f") else np.pi / n

        face_azim = [phi0 + 2.0 * np.pi * k / n for k in range(n)]
        vert_azim = [phi0 + (2.0 * k + 1.0) * np.pi / n for k in range(n)]
        vertices = [(R0 * np.sin(p), R0 * np.cos(p)) for p in vert_azim]

        return dict(n=n, theta=theta, theta_deg=float(self.face_angle), a0=a0, R0=R0,
                    H=H, phi0=phi0, face_azim=face_azim, vert_azim=vert_azim,
                    vertices=vertices)

    def pyramid_face_points(self, y_apex, z_apex, x_apex=0.0, h_frac=0.5):
        # One probe point per lateral face, in GLOBAL coordinates, for
        # ind_inst.faces.findAt(). h_frac is the relative height above the apex.
        pc = self.Pyramid_coords()
        h = float(h_frac) * pc["H"]
        r = h * np.tan(pc["theta"])                   # axis -> face distance at height h
        return [(x_apex + r * np.sin(p), y_apex + h, z_apex + r * np.cos(p))
                for p in pc["face_azim"]]

    def pyramid_edge_points(self, s=0.35):
        # One probe point per lateral (corner) edge, in PART LOCAL coordinates.
        pc = self.Pyramid_coords()
        return [((1.0 - s) * vx, (1.0 - s) * vy, s * pc["H"])
                for (vx, vy) in pc["vertices"]]

    def pyramid_equivalent_cone_angle(self):
        # Half-apex angle of the CONE with the same projected contact area vs depth:
        #   pi * a_eq^2 = n * (h tan(theta))^2 * tan(pi/n)
        # 4 faces / theta=60 deg -> 62.90 deg ; 3 faces / theta=60 deg -> 65.81 deg.
        pc = self.Pyramid_coords()
        k = np.sqrt(pc["n"] * np.tan(np.pi / pc["n"]) / np.pi)
        return float(np.degrees(np.arctan(k * np.tan(pc["theta"]))))
     
# 2. Substrate
class Substrate_Config:
    # Dimensions and partitions of the substrate block

    def __init__(self,
                 xs1=0.0, ys1=0.0, zs1=0.0,             # Substrate box  (origin at xs1, ys1, zs1)
                 xs2=0.6, ys2=0.5, zs2=3.0,             # Width, height  and depth of the box [mm] (z is the scratch direction)
                 dpo_x=0.25, dpo_y=0.15, dpo_z=0.25 ):  # Partition offsets (from edges of refined zone) 

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
                 second_order_accuracy=False,                                  # 'True' for complex models (AB, DP) (False otherwise) (increases simulation time)
                 length_ratio=0.1):                                            # For distortion control, between 0 and 1

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
        self.length_ratio = length_ratio

# 4a. Linear Elastic Model (for glassy / semicrystalline bases)
class LinearElastic_Config:

    MODEL = "elastic"

    def __init__(self, E=200.0, nu=0.40):
        self.E = E
        self.nu = nu

    def params(self):
        return {"E": self.E, "nu": self.nu}

# 4b. Hyper-elastic Model (Mooney-Rivlin)
class HE_Model_Config:
    #  W = C10 * (I1_bar - 3) + C01 * (I2_bar - 3) + (1/D1) * (J_el - 1)^2
    MODEL = "mooney_rivlin"

    def __init__(self, C10=1.0, C01=0.1, D1=0.018): # First and second parameter [MPa], Compressibility parameter [1/MPa]
        # change to MPa
        self.C10 = C10   
        self.C01 = C01   
        self.D1 = D1   

    def params(self):
        return {"C10": self.C10, "C01": self.C01, "D1": self.D1}
    
# 4c. Arruda-Boyce (eight-chain) hyper-elastic Model
class AB_Model_Config:
    #  W = mu * sum_{i=1..5} C_i / lambda_m^(2i-2) * (I1_bar^i - 3^i) + (1/D) * ((J_el^2 - 1)/2 - ln(J_el))
    # Abaqus direct-coefficient table order: (mu, lambda_m, D).
    MODEL = "arruda_boyce"

    def __init__(self, mu=2.0, lambda_m=2.5, D=0.018):
        self.mu = mu               # initial shear modulus [MPa]
        self.lambda_m = lambda_m   # locking stretch [-]
        self.D = D                 # compressibility [1/MPa], D = 2/K0

    def params(self):
        return {"mu_AB": self.mu, "lambda_m": self.lambda_m, "D_AB": self.D}

# 4d. Yeoh (reduced 3rd-order polynomial, I1-only) hyper-elastic model
class Yeoh_Model_Config:
    #  W = sum_{i=1..3} Ci0 * (I1_bar - 3)^i + sum_{i=1..3} (1/Di) * (J_el - 1)^(2i)
    # Abaqus table order: (C10, C20, C30, D1, D2, D3).
    # I1-only: cheap and stable up to large strains; C20 < 0 reproduces the
    # mid-strain softening of filled rubbers, C30 > 0 the final upturn.
    # Initial shear modulus mu_0 = 2*C10 (C20/C30 do not contribute at I1=3).
    MODEL = "yeoh"

    def __init__(self, C10=1.1, C20=-0.055, C30=0.0055, D1=0.0165):
        self.C10 = C10   # [MPa]
        self.C20 = C20   # [MPa]  (usually < 0)
        self.C30 = C30   # [MPa]  (usually > 0, upturn)
        self.D1 = D1     # [1/MPa], D1 = 2/K0 ; D2 = D3 = 0 (single volumetric term)

    def params(self):
        return {"C10_Y": self.C10, "C20_Y": self.C20,
                "C30_Y": self.C30, "D1_Y": self.D1}

# 4e. Ogden (principal-stretch based) hyper-elastic model
class Ogden_Model_Config:
    #  W = sum_{i=1..N} 2*mu_i/alpha_i^2 * (lam1_bar^alpha_i + lam2_bar^alpha_i + lam3_bar^alpha_i - 3) + sum_{i=1..N} (1/Di) * (J_el - 1)^(2i)
    # Abaqus N=2 table order: (mu1, alpha1, mu2, alpha2, D1, D2).
    # In the Abaqus convention the initial shear modulus is mu_0 = sum(mu_i)
    # regardless of the alpha_i. Principal-stretch formulation: the only one
    # of the four that is NOT a pure I1/I2 function -- discriminates in
    # non-equibiaxial states like the scratch bow-wave.
    MODEL = "ogden"

    def __init__(self, mu=(1.87, 0.33), alpha=(1.6, 5.5), D1=0.0165):
        mu = tuple(float(m) for m in mu)
        alpha = tuple(float(a) for a in alpha)
        if len(mu) != len(alpha):
            raise ValueError("Ogden: mu and alpha must have the same length "
                             "(got %d and %d)" % (len(mu), len(alpha)))
        if not (1 <= len(mu) <= 6):
            raise ValueError("Ogden order N must be in [1, 6], got %d" % len(mu))
        if any(a == 0.0 for a in alpha):
            raise ValueError("Ogden: alpha_i must be non-zero")
        self.mu = mu          # [MPa] per-term moduli, mu_0 = sum(mu)
        self.alpha = alpha    # [-]   per-term exponents
        self.D1 = D1          # [1/MPa], D1 = 2/K0 ; D2..DN = 0
        self.N = len(mu)

    def params(self):
        d = {"ogden_N": self.N, "D1_O": self.D1}
        for i, (m, a) in enumerate(zip(self.mu, self.alpha), start=1):
            d["mu%d_O" % i] = m
            d["alpha%d_O" % i] = a
        return d


# 5. Visco-elastic Models (empty)
class VE_Model_Config:
    MODEL = "none"

    def params(self):
        return {}
    
# 5b. Prony-series linear viscoelasticity
class Prony_Config:
    # prony_table: ((g_i, k_i, tau_i), ...) — normalized shear/bulk moduli and
    # relaxation times [s]. sum(g_i) < 1 for stability; k_i often 0 (shear only).
    MODEL = "prony"

    def __init__(self, prony_table=((0.2, 0.0, 0.1), (0.1, 0.0, 0.001))):
        self.prony_table = tuple(tuple(row) for row in prony_table)

    def params(self):
        taus = [row[2] for row in self.prony_table]
        return {"prony_terms": len(self.prony_table),
                "tau_max": max(taus) if taus else 0.0}

# 6. Plasticity Models (empty)
class P_Model_Config:
    MODEL = "none"

    def params(self):
        return {}

# 6b. Von Mises plasticity (isochoric, pressure-independent)
class J2Plasticity_Config:
    """
    Usually used for metals.
    Plasticity is driven by distortion energy only.
    """
    MODEL = "mises"

    def __init__(self, 
                 yield_table=((10.0, 0.0), (14.0, 0.2), (18.0, 0.6)),        # (yield_stress [MPa], plastic_strain [-])
                 rate_dependent=None):                                       # RateDependent_Config or None
        self.yield_table = tuple(tuple(pt) for pt in yield_table)
        self.rate_dependent = rate_dependent

    def params(self):
        # Expose the initial yield stress for the CSV / verifier; the full hardening table is used only by the material assignment.
        d = {"sigma_y0": self.yield_table[0][0]}
        if self.rate_dependent is not None:
            d.update(self.rate_dependent.params())
        return d
    
# 6c. Drucker-Prager pressure-dependent plasticity (glassy / thermoset bases)
class DruckerPrager_Config:
    """
    Makes the Simulation Pressure Dependent (linearly), better for polymers.
    """
    MODEL = "drucker_prager"

    def __init__(self, 
                 friction_angle=25.0,                                   # [deg] friction angle
                 flow_stress_ratio=0.85,                                # [-] flow_stress_ratio (usually between 0.8 and 1.0)
                 dilation_angle=10.0,                                   # [deg] Control the change of volume when exposed to shearing
                 yield_table=((60.0, 0.0), (70.0, 0.1), (80.0, 0.4)),
                 rate_dependent=None):                          
        self.friction_angle = friction_angle            
        self.flow_stress_ratio = flow_stress_ratio      
        self.dilation_angle = dilation_angle           
        self.yield_table = tuple(tuple(pt) for pt in yield_table)
        self.rate_dependent = rate_dependent

    def params(self):
        d = {"sigma_y0": self.yield_table[0][0],
             "friction_angle": self.friction_angle,
             "dilation_angle": self.dilation_angle}
        if self.rate_dependent is not None:
            d.update(self.rate_dependent.params())
        return d

# 6d. Rate dependence of the yield surface (Cowper-Symonds overstress power law)
class RateDependent_Config:
    """
    *RATE DEPENDENT, TYPE=POWER LAW -- valid with *PLASTIC and *DRUCKER PRAGER
    HARDENING in Abaqus/Explicit (unlike *VISCOELASTIC, which is forbidden
    with plasticity). The dynamic yield ratio is

        R(eps_rate) = sigma_dyn / sigma_stat = 1 + (eps_rate / D)**(1/n)

    Polymers are usually characterised by an EYRING line instead:
        sigma_y(eps_rate) = sigma_y0 + S * log10(eps_rate / rate_qs)
    with S [MPa/decade] from the literature (indicative, compression, RT:
    HDPE ~ 2-3, PC ~ 4-5 below the beta-transition, PMMA ~ 8-11 MPa/decade).
    Use from_eyring() to convert (sigma_y0, S) into a Cowper-Symonds pair
    fitted over a rate window -- the fit is exact at both window ends and
    within ~2% inside it, but MUST NOT be extrapolated far outside.
    """
    MODEL = "cowper_symonds"

    def __init__(self, D=1.0e6, n=10.0, fit_window=None, S_per_decade=None):
        if D <= 0.0 or n <= 0.0:
            raise ValueError("Cowper-Symonds parameters must be positive (D=%s, n=%s)" % (D, n))
        self.D = float(D)                     # reference rate [1/s]
        self.n = float(n)                     # exponent [-]
        self.fit_window = fit_window          # (rate_lo, rate_hi) of the Eyring fit, for the verifier
        self.S_per_decade = S_per_decade      # traceability [MPa/decade]

    @classmethod
    def from_eyring(cls, sigma_y0, S_per_decade, rate_qs=1e-3, rate_lo=1.0, rate_hi=1e3):
        """
        Two-point closed-form fit of the Cowper-Symonds law on the Eyring line
        at rate_lo and rate_hi (quasi-static calibration rate rate_qs, where
        the yield table itself was measured).
        """
        if not (rate_qs < rate_lo < rate_hi):
            raise ValueError("Need rate_qs < rate_lo < rate_hi")
        R1 = 1.0 + (S_per_decade / float(sigma_y0)) * np.log10(rate_lo / rate_qs)
        R2 = 1.0 + (S_per_decade / float(sigma_y0)) * np.log10(rate_hi / rate_qs)
        if not (R2 > R1 > 1.0):
            raise ValueError("Eyring fit requires R2 > R1 > 1 (got %.3f, %.3f)" % (R1, R2))
        inv_n = np.log((R2 - 1.0) / (R1 - 1.0)) / np.log(rate_hi / rate_lo)
        n = 1.0 / inv_n
        D = rate_lo / (R1 - 1.0) ** n
        return cls(D=D, n=n, fit_window=(rate_lo, rate_hi), S_per_decade=S_per_decade)

    def ratio(self, eps_rate):
        """Dynamic/static yield ratio R at a given plastic strain rate [1/s]."""
        if eps_rate <= 0.0:
            return 1.0
        return 1.0 + (eps_rate / self.D) ** (1.0 / self.n)

    def params(self):
        return {"cs_D": self.D, "cs_n": self.n}

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
                 n_field_frames=20, n_field_frames_recovery=50, n_history_points=100,                           # Number of frames / field outputs for each step
                 amplitude_smoothing=0.25,                                                                    # [-] SMOOTH fraction of the tabular amplitudes (0-0.5, None = solver default).
                 depth_hold_frac=0.05 ):   # [-] PROGRESSIVE: plateau plat au sommet, fraction de scratch_time (garantit la profondeur nominale malgre le lissage du pic ; cf depth_amplitude()).
                                                                                                                # Rounds the velocity discontinuities at the amplitude kinks (t1/t2/t3).
                                                                                                                 
        if depth_mode not in (self.PROGRESSIVE, self.CONSTANT):
            raise ValueError("depth_mode must be 'progressive' or 'constant', got '%s'" % depth_mode)
        
        if control_mode not in (self.DISPLACEMENT, self.FORCE):
            raise ValueError("control_mode must be 'displacement' or 'force', got '%s'" % control_mode)
              
        if control_mode == self.DISPLACEMENT and recovery_lift <= 0.0 and recovery_time > 0.0:
            raise ValueError("recovery_lift must be positive to ensure indenter separation during recovery")
        
        if control_mode == self.FORCE and scratch_force <= 0.0:
            raise ValueError("scratch_force must be positive for force-controlled scratch, got %s" % scratch_force)

        if amplitude_smoothing is not None and not (0.0 <= amplitude_smoothing <= 0.5):
            raise ValueError("amplitude_smoothing must be in [0, 0.5] or None, got %s" % amplitude_smoothing)

        if not (0.0 <= depth_hold_frac < 0.5):
            raise ValueError("depth_hold_frac must be in [0, 0.5), got %s" % depth_hold_frac)


            
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
        self.amplitude_smoothing = amplitude_smoothing
        self.depth_hold_frac = depth_hold_frac


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
    def depth_hold(self): # [s] Plateau tenu a la profondeur pic avant decharge (PROGRESSIVE). Fraction de scratch_time.
        return self.depth_hold_frac * self.scratch_time

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
            # --- PLATEAU FIX (sous-tir par arrondi du pic interieur) -------------
            # Original (pic (t2,1.0) interieur -> lisse VERS LE BAS, profondeur nominale
            # jamais atteinte ; manque ~ smooth_window/scratch_time -> fausse dependance
            # de RF2 au scratch_time) :
            #     if not self.has_recovery_step:
            #         return ((0.0, 0.0),(t2,  1.0),(t3,  0.0))
            #     else:
            #         return ((0.0,  0.0),(t2,   1.0),(t3,   lift_value),(t4,   lift_value))
            # Fix : sommet plat de largeur depth_hold (scale avec scratch_time). L'interieur
            # plat = depth_hold*(1 - 2*smooth) > 0 est atteint exactement pour tout smooth<0.5,
            # donc profondeur nominale garantie et profil normalise invariant. t2/t3 inchanges.
            t2h = t2 - self.depth_hold
            if not self.has_recovery_step:
                return ((0.0, 0.0),(t2h,  1.0),(t2,  1.0),(t3,  0.0))
            else:
                return ((0.0,  0.0),(t2h,  1.0),(t2,  1.0),(t3,  lift_value),(t4,  lift_value))
        
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

# 9. Friction Models 
class Friction_Config:
    """
    Uses either a constant Coulomb mu or a mu_table from Briscoe

    mu_table follows the Abaqus friction column order: (mm/s, MPa)
        (mu[, slip_rate][, contact_pressure])
    """

    def __init__(self, mu=0.3, formulation="penalty", elastic_slip_fraction=0.005,
                 pressure_dependent=False, slip_rate_dependent=False, mu_table=None, briscoe_params=None):

        self.mu = mu
        self.formulation = formulation
        self.elastic_slip_fraction = elastic_slip_fraction
        self.pressure_dependent = pressure_dependent
        self.slip_rate_dependent = slip_rate_dependent
        self.mu_table = tuple(tuple(r) for r in mu_table) if mu_table else None
        self.briscoe_params = dict(briscoe_params) if briscoe_params else None

        if self.mu_table:
            expected = 1 + int(bool(slip_rate_dependent)) + int(bool(pressure_dependent))
            for row in self.mu_table:
                if len(row) != expected:
                    raise ValueError(
                        "mu_table rows must have %d columns (mu%s%s), got %d"
                        % (expected,
                           ", slip_rate" if slip_rate_dependent else "",
                           ", pressure" if pressure_dependent else "",
                           len(row)))

    @classmethod
    def briscoe(cls, 
                tau0=2.0,                       # [MPa] Briscoe's adhesive shear stress (between 0.5 and 5 depending on the polymer)
                alpha=0.2,                      # [-] Pressure coefficient, mu asymptote at high pressure
                p_min=1.0, p_max=600.0,         # [Mpa] Pressure Bounds for the table, covers most polymers
                n_points=12,                    # Number of points in the table (log sampled)
                mu_cap=0.6,                     # Ceiling to avoid mu divergence
                elastic_slip_fraction=0.005):
        """
        Pressure-dependent Coulomb table from the Briscoe interfacial shear model: 
            mu(p) = tau0/p + alpha

        The apparent friction decreases with contact pressure toward the asymptote alpha.
        NB : Scratch pressures reach ~2-3*sigma_y = 50-100 MPa for semicrystallines and 150-300 MPa for glassy polymers. 
        """
        p = np.logspace(np.log10(p_min), np.log10(p_max), n_points)
        rows = tuple((float(round(min(tau0 / pv + alpha, mu_cap), 4)), float(round(pv, 3)))
                     for pv in p)
        return cls(mu=alpha, pressure_dependent=True, mu_table=rows,
                   elastic_slip_fraction=elastic_slip_fraction,
                   briscoe_params={"tau0": tau0, "alpha": alpha,
                                   "p_min": p_min, "p_max": p_max,
                                   "n_points": n_points, "mu_cap": mu_cap})

    def set_briscoe_alpha(self, alpha):
        if not self.briscoe_params:
            raise ValueError(
                "set_briscoe_alpha needs a Friction_Config built by "
                "Friction_Config.briscoe(); this one carries no Briscoe "
                "parameters (pressure_dependent=%s)" % self.pressure_dependent)
        params = dict(self.briscoe_params)
        params["alpha"] = float(alpha)
        rebuilt = Friction_Config.briscoe(
            elastic_slip_fraction=self.elastic_slip_fraction, **params)
        self.mu = rebuilt.mu
        self.mu_table = rebuilt.mu_table
        self.pressure_dependent = True
        self.briscoe_params = rebuilt.briscoe_params
        return self

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
        d["he_model"] = self.hyperelastic.MODEL
        d["mu_friction"] = self.friction.mu
        d["mu_pressure_dep"] = 1.0 if getattr(self.friction, "pressure_dependent", False) else 0.0
        d["mu_rate_dep"] = 1.0 if getattr(self.friction, "slip_rate_dependent", False) else 0.0

        return d

# 11. Solver 
class Solver_Config:

    def __init__(self,
                 mass_scale=500,
                 target_time_increment=0,
                 use_ALE=False,
                 num_cpus=6,
                 time_scale_factor=1.0,
                 linear_bulk_viscosity=0.06, quad_bulk_viscosity=1.2, # Default Abaqus values
                # ale_frequency=20, ale_mesh_sweeps=1,   # OLD defaults 
                 ale_frequency=650, ale_mesh_sweeps=3, ale_smoothing_priority="GRADED", ale_smoothing_algorithm="GEOMETRY_ENHANCED", 
                 ale_curvature_refinement=1,      # > 1 concentrates nodes on the groove shoulder (1 = uniform)
                 ale_domain="refined",            # "refined" | "contact" | "full" (legacy) -- see _setup_ale
                 ale_in_passive_steps=False):     # ALE during unload / recovery -- see _setup_ale
    
        if time_scale_factor < 1.0:
            raise ValueError("time_scale_factor must be >= 1 (lab time / simulated time)")
        self.mass_scale = mass_scale
        self.target_time_increment = target_time_increment
        self.use_ALE = use_ALE
        self.num_cpus = num_cpus
        self.num_domains = num_cpus
        self.time_scale_factor = float(time_scale_factor)
        self.linear_bulk_viscosity = linear_bulk_viscosity
        self.quad_bulk_viscosity = quad_bulk_viscosity
        if ale_domain not in ("refined", "contact", "full"):
            raise ValueError("ale_domain must be 'refined', 'contact' or 'full', got '%s'" % ale_domain)
        self.ale_frequency = ale_frequency
        self.ale_mesh_sweeps = ale_mesh_sweeps
        self.ale_smoothing_priority = ale_smoothing_priority
        self.ale_smoothing_algorithm = ale_smoothing_algorithm
        self.ale_curvature_refinement = ale_curvature_refinement
        self.ale_domain = ale_domain
        self.ale_in_passive_steps = ale_in_passive_steps

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
                                                             "ALLCD", "ALLSE",
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
        self.amp_force = "Amp-Force"
        self.bc_scratch = "IndenterScratching"
        self.bc_depth = "IndenterDepth"
        self.bc_travel = "IndenterTravel"
        self.bc_force = "IndenterForce"

        # Output requests 
        self.out_reaction = "ReactionForces"
        self.out_indenter_disp = "IndenterDisp"
        self.out_energy_substrate = "Energy"
        self.out_energy_whole = "EnergyBalance"
        self.out_field = "FieldOutput"
        self.out_contact = "ContactForce"
        self.out_contact_pair = "ContactPairForce"

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
                fine_size_x=0.015,       
                fine_size_y=0.015,
                fine_size_z=0.015,    
                coarse_size_0=0.02,     # Unused
                coarse_size_1=0.028,     # 0.07*4 
                coarse_size_2=0.056,     # 0.07*8
                hourglass_control="ENHANCED",      # RELAX STIFFNESS with ALE / ENHANCED without ALE 
                distortion_control="DEFAULT",
                max_degradation=0.9,
                element_deletion=False,
                second_order_accuracy=False,
            ),
            material=Material_Config(
                rho=1.2e-9,
                #hyperelastic=HE_Model_Config(C10=1.0, C01=0.1, D1=1.8e-2),
                hyperelastic=AB_Model_Config(mu=2.0, lambda_m=2.5, D=1.8e-2),
                viscoelastic=None,
                plasticity=None,
                damage=None,
                friction=Friction_Config(),
                family="elastomer_mr",
            ),
            solver=Solver_Config(
                mass_scale=500,    
                target_time_increment=0.0,
                use_ALE=False,                     
                num_cpus=6,                        # "submit.sh CPU value is prioritized"
                linear_bulk_viscosity=0.06,
                quad_bulk_viscosity=1.2,
                ale_frequency=650,                # C_remesh ~ 0.1 (glassy) to ~0.6 (elastomer); see ale_remesh_courant()
                ale_mesh_sweeps=3,                # absorbs the larger distortion between two (now rarer) remeshings
                ale_smoothing_priority="GRADED",
                ale_smoothing_algorithm="GEOMETRY_ENHANCED",
                ale_curvature_refinement=1,       # try 2-3 to sharpen the groove shoulder (pile-up resolution)
                ale_domain="refined",             # ALE restricted to the refined contact cell
                ale_in_passive_steps=False,       # no advection during unload / recovery
            ),
            scratch=Scratch_Config(
                depth_mode=Scratch_Config.PROGRESSIVE,
                control_mode=Scratch_Config.DISPLACEMENT,
                scratch_length=2.0,
                scratch_force=40e-3,
                scratch_depth=-40e-3,
                scratch_time=0.05,
                indentation_time=0.01,
                unload_time=0.02,
                recovery_time=0.02,
                recovery_lift=0.05,
                n_field_frames=20,
                n_field_frames_recovery=10,
                n_history_points=100,
            ),
            output=Output_Config(),
            naming=Naming_Config(),
            job_name="PolymerScratch",
            sheet_size=10,
        )
    
# 15. Module-level generators (helpers that build configurations)
def _ab_mu0_correction(lambda_m):
    # Initial-shear-modulus correction of the Arruda-Boyce eight-chain model (mu_0 = mu * corr; ~1.11 at lambda_m = 2.5, -> 1 as lambda_m -> inf).
    # Used for models comparaison
    if not lambda_m or lambda_m <= 0.0:
        return 1.0
    l2 = float(lambda_m) ** 2
    return (1.0 + 3.0 / (5.0 * l2) + 99.0 / (175.0 * l2 ** 2)
            + 513.0 / (875.0 * l2 ** 3) + 42039.0 / (67375.0 * l2 ** 4))

def elastic_moduli(material):
    # Small-strain bulk/shear moduli (K0, G0) [MPa] of the BASE elasticity of
    # a Material_Config, dispatched on hyperelastic.MODEL. Pure numpy
    # (Abaqus-free): usable by the CPython samplers and the Abaqus kernel.
    # Viscoelastic (Prony) families: the hyperelastic constants are the
    # INSTANTANEOUS moduli, which are exactly what the stable-increment
    # estimate must use in Explicit.
    he = material.hyperelastic
    m = he.MODEL
    if m == "elastic":
        K = he.E / (3.0 * (1.0 - 2.0 * he.nu))
        G = he.E / (2.0 * (1.0 + he.nu))
    elif m == "mooney_rivlin":
        G = 2.0 * (he.C10 + he.C01)
        K = 2.0 / he.D1
    elif m == "arruda_boyce":
        G = he.mu * _ab_mu0_correction(he.lambda_m)
        K = 2.0 / he.D
    elif m == "yeoh":
        G = 2.0 * he.C10
        K = 2.0 / he.D1
    elif m == "ogden":
        G = float(sum(he.mu))
        K = 2.0 / he.D1
    else:
        raise ValueError("elastic_moduli: unsupported base elasticity '%s'" % m)
    return K, G

def natural_dt(material, L_min):
    """
    Estimate of the smallest stable time increment [s]: dt_nat = L_min / c_d,   c_d = sqrt(M / rho),   M = K0 + 4*G0/3
    NB : mesh-based estimate: the increment actually reported in the .sta/.msg can be lower (element distortion, contact penalty stiffness).
    L_min - Smallest element characteristic size (fine_size_..)
    c_d - Dilatation wave speed
    """
    K, G = elastic_moduli(material)
    M = K + 4.0 * G / 3.0
    c_d = np.sqrt(M / material.rho)
    return float(L_min) / c_d

def ale_remesh_courant(cfg):
    # Remeshing Courant number of an ALE run:
    #     C = (indenter travel between two remeshings) / (fine element size)
    #       = ale_frequency * scratch_length * dt / (scratch_time * L_min)
    # Because dt ~ L_min / c_d, C is MESH-INDEPENDENT: a single ale_frequency
    # keeps a consistent advection error across a mesh study.
    #   C -> 0 : maximum numerical diffusion of the advected state variables
    #            (PEEQ, stresses) -- biases RF2 low on dissipative families.
    #   C -> 1 : advection becomes exact, but the mesh must survive a full
    #            element of distortion between two remeshings.
    # Target ~ 0.2-0.5. Returns None when the estimate does not apply.
    try:
        L_min = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)
        dt_nat = natural_dt(cfg.material, L_min)
    except (ValueError, AttributeError, TypeError):
        return None
    target_dt = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
    if target_dt > 0.0:
        dt = target_dt                     # variable mass scaling drives dt to the target
    else:
        s = float(getattr(cfg.solver, "mass_scale", 1.0) or 1.0)
        dt = dt_nat * np.sqrt(max(s, 1.0))
    f = float(getattr(cfg.solver, "ale_frequency", 0) or 0)
    if f <= 0.0 or cfg.scratch.scratch_time <= 0.0:
        return None
    travel = cfg.scratch.scratch_length * dt * f / float(cfg.scratch.scratch_time)
    return float(travel / L_min)

# G'Sell-Jonas dense hardening-table generator (used for the yield tables)
def gsell_jonas_table(sigma_y0, h, Q=0.0, b=0.0, soft_drop=0.0, eps_soft=0.05,
                      eps_max=3.0, n_points=60):
    # Dense (sigma_y, eps_p) table for *PLASTIC / *DRUCKER PRAGER HARDENING:
    # sigma(eps_p) = [sigma_y0 - soft_drop*(1 - exp(-eps_p/eps_soft)) + Q*(1 - exp(-b*eps_p))] * exp(h*eps_p^2)

    # * exp(h*eps_p^2) : G'Sell-Jonas orientation hardening (the term Bucaille et al. identified as controlling pile-up and scratch resistance)
    # * soft_drop/eps_soft : intrinsic post-yield softening of glassy polymers (NB: softening makes the response mesh-dependent through localisation)
    # * Q/b : Voce-type initial hardening (semicrystallines).

    # A dense table up to eps_p ~ 3 avoids the perfectly-plastic plateau that Abaqus extrapolates beyond the last point.

    if sigma_y0 <= 0.0:
        raise ValueError("sigma_y0 must be positive")
    n_lo = max(n_points // 3, 8)
    eps = np.unique(np.concatenate([
        np.linspace(0.0, min(0.25, eps_max), n_lo),
        np.linspace(min(0.25, eps_max), eps_max, n_points - n_lo + 1),
    ]))
    sig = (sigma_y0
           - soft_drop * (1.0 - np.exp(-eps / max(eps_soft, 1e-9)))
           + Q * (1.0 - np.exp(-b * eps))) * np.exp(h * eps ** 2)
    if np.any(sig <= 0.0):
        raise ValueError("gsell_jonas_table produced non-positive stresses; check soft_drop")
    return tuple((float(round(sv, 4)), float(round(ev, 6))) for sv, ev in zip(sig, eps))

def matched_hyperelastic_set(mu0=2.2, K_mu=55.0,
                             models=("mooney_rivlin", "arruda_boyce", "yeoh", "ogden"),
                             mr_ratio=0.1, ab_lambda_m=2.5,
                             yeoh_c20_ratio=-0.05, yeoh_c30_ratio=0.005,
                             ogden_alphas=(1.6, 5.5), ogden_weights=(0.85, 0.15)):
    
    #Hyperelastic model set calibrated to the same small-strain response (identical mu_0 and K_0 = K_mu * mu_0 for every model)
    #Scratch comparison isolates the model form: I2-dependence (MR), locking (AB), higher-order I1 (Yeoh), principal-stretch formulation (Ogden)
    
    K0 = K_mu * mu0
    D = 2.0 / K0
    out = []
    for m in models:
        if m == "mooney_rivlin":
            C10 = mu0 / (2.0 * (1.0 + mr_ratio))
            out.append(("MR", HE_Model_Config(C10=round(C10, 6),
                                              C01=round(mr_ratio * C10, 6), D1=D)))
        elif m == "arruda_boyce":
            mu_ab = mu0 / _ab_mu0_correction(ab_lambda_m)
            out.append(("AB", AB_Model_Config(mu=round(mu_ab, 6),
                                              lambda_m=ab_lambda_m, D=D)))
        elif m == "yeoh":
            C10 = mu0 / 2.0
            out.append(("Yeoh", Yeoh_Model_Config(C10=round(C10, 6),
                                                  C20=round(yeoh_c20_ratio * C10, 6),
                                                  C30=round(yeoh_c30_ratio * C10, 6),
                                                  D1=D)))
        elif m == "ogden":
            wsum = float(sum(ogden_weights))
            mus = tuple(round(mu0 * w / wsum, 6) for w in ogden_weights)
            out.append(("Ogden%d" % len(mus),
                        Ogden_Model_Config(mu=mus, alpha=ogden_alphas, D1=D)))
        else:
            raise ValueError("Unknown model '%s' for matched_hyperelastic_set" % m)
    # plain list of pairs keeps insertion order on Py2 (Abaqus kernel) and Py3
    return out