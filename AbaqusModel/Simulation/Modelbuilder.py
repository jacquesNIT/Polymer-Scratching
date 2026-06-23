# Scratch test model builder for polymer simulation.
#Orchestrates geometry creation, assembly, step definition, boundary conditions, contact modelling, and output requests.

from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

from ScratchSimulation.AbaqusModel.Geometry.indenter import create_indenter
from ScratchSimulation.AbaqusModel.Geometry.substrate import create_substrate, mesh_substrate
from ScratchSimulation.AbaqusModel.Geometry.partition import  partition_top_face

def build_scratch_model(cfg):
    # Build a complete scratch-test model (geometry + steps + BCs + contact + outputs).

    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

    model = mdb.models["Model-1"]
    sub = cfg.substrate
    names = cfg.naming
    scratch = cfg.scratch
    solver = cfg.solver

    #  1. Geometry
    substrate_part = create_substrate(model, cfg)
    mesh_substrate(substrate_part, cfg)
    indenter_part = create_indenter(model, cfg)
    # partition_top_face(substrate_part, cfg)

    #  2. Assembly
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(dependent=ON, name=names.indenter_instance, part=indenter_part)
    asm.Instance(dependent=ON, name=names.substrate_instance, part=substrate_part)

    ind_inst = asm.instances[names.indenter_instance]
    sub_inst = asm.instances[names.substrate_instance]

    # Position indenter: tip at top surface of substrate, at z = dpo_z
    asm.translate(instanceList=(names.indenter_instance,), vector=(0.0, sub.ys2, 0.0))
    asm.translate(instanceList=(names.indenter_instance,), vector=(0.0, 0.0, sub.dpo_z))

    #  3. Steps
    steps = _create_steps(model, cfg)

    #  4. Boundary conditions (substrate)
    _apply_boundary_conditions(model, asm, ind_inst, sub_inst, cfg, steps["first"])

    #  5. Loading (displacement-controlled indenter via amplitudes)
    _apply_loading(model, ind_inst, cfg, steps["first"])

    #  6. Output requests
    _setup_output_requests(model, ind_inst, sub_inst, cfg, steps)

    #  7. Contact
    _setup_contact(model, asm, ind_inst, sub_inst, cfg, steps["first"])

    #  8. ALE adaptive meshing
    if solver.use_ALE:
        _setup_ale(model, asm, sub_inst, cfg, steps)

    return model, substrate_part


#  Step creation
def _create_steps(model, cfg):
    # Create all analysis steps based on the scratch configuration.
    # Returns a dict with step names keyed by role (first, indent, scratch, unload, recovery, all_active, all)
   
    scratch = cfg.scratch
    solver = cfg.solver

    # Mass scaling tuple (shared by all active steps)
    use_variable = solver.target_time_increment > 0.0
    ms_tuple = (
        SEMI_AUTOMATIC,
        MODEL,
        THROUGHOUT_STEP if use_variable else AT_BEGINNING,
        0.0 if use_variable else solver.mass_scale,
        solver.target_time_increment,
        BELOW_MIN if use_variable else None,
        0, 10, 0.0, 0.0, 0, None,
    )

    steps = {
        "indent": None,
        "scratch": None,
        "unload": None,
        "recovery": None,
        "all_active": [],
        "all": [],
    }
    previous = "Initial"

    # Indentation step (constant depth mode only) 
    if scratch.depth_mode == scratch.CONSTANT:
        name = "IndentationStep"
        model.ExplicitDynamicsStep(
            improvedDtMethod=ON,
            massScaling=(ms_tuple,),
            name=name,
            previous=previous,
            timePeriod=scratch.indentation_time,
            nlgeom=ON,
            linearBulkViscosity=solver.linear_bulk_viscosity,
            quadBulkViscosity=solver.quad_bulk_viscosity,
        )
        steps["indent"] = name
        steps["all_active"].append(name)
        steps["all"].append(name)
        previous = name

    # Scratch step (always) 
    name = "ScratchStep"
    model.ExplicitDynamicsStep(
        improvedDtMethod=ON,
        massScaling=(ms_tuple,),
        name=name,
        previous=previous,
        timePeriod=scratch.scratch_time,
        nlgeom=ON,
        linearBulkViscosity=solver.linear_bulk_viscosity,
        quadBulkViscosity=solver.quad_bulk_viscosity,
    )
    steps["scratch"] = name
    steps["all_active"].append(name)
    steps["all"].append(name)
    previous = name

    # Unload step (always) 
    name = "UnloadStep"
    model.ExplicitDynamicsStep(
        improvedDtMethod=ON,
        name=name,
        previous=previous,
        timePeriod=scratch.unload_time,
    )
    steps["unload"] = name
    steps["all"].append(name)
    previous = name

    # Recovery step (optional) 
    if scratch.has_recovery_step:
        name = "RecoveryStep"
        model.ExplicitDynamicsStep(
            improvedDtMethod=ON,
            name=name,
            previous=previous,
            timePeriod=scratch.recovery_time,
        )
        steps["recovery"] = name
        steps["all"].append(name)

    # First step (for BC creation)
    steps["first"] = steps["all"][0]

    return steps



#  Loading
def _apply_loading(model, ind_inst, cfg, first_step):
    # Create amplitude tables and displacement BCs on the indenter.

    scratch = cfg.scratch
    names = cfg.naming
    region = ind_inst.sets[names.indenter_set]

    if scratch.uses_single_amplitude:
        # Progressive without recovery: depth and length share one amplitude
        model.TabularAmplitude(
            data=scratch.depth_amplitude(),
            name="Amp-1",
            smooth=SOLVER_DEFAULT,
            timeSpan=TOTAL,
        )
        model.DisplacementBC(
            amplitude="Amp-1",
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name="IndenterScratching",
            region=region,
            u1=UNSET,
            u2=scratch.scratch_depth,
            u3=scratch.scratch_length,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
        )
    else:
        # Two separate amplitudes (constant mode, or progressive with recovery)
        model.TabularAmplitude(
            data=scratch.depth_amplitude(),
            name="Amp-Depth",
            smooth=SOLVER_DEFAULT,
            timeSpan=TOTAL,
        )
        model.TabularAmplitude(
            data=scratch.length_amplitude(),
            name="Amp-Length",
            smooth=SOLVER_DEFAULT,
            timeSpan=TOTAL,
        )
        model.DisplacementBC(
            amplitude="Amp-Depth",
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name="IndenterDepth",
            region=region,
            u1=UNSET,
            u2=scratch.scratch_depth,
            u3=UNSET,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
        )
        model.DisplacementBC(
            amplitude="Amp-Length",
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name="IndenterTravel",
            region=region,
            u1=UNSET,
            u2=UNSET,
            u3=scratch.scratch_length,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
        )

    # Lock transverse translation and all rotations
    model.DisplacementBC(
        amplitude=UNSET,
        createStepName=first_step,
        distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
        name="IndenterConstraint",
        region=region,
        u1=SET, u2=UNSET, u3=UNSET,
        ur1=SET, ur2=SET, ur3=SET,
    )



#  Boundary conditions
def _apply_boundary_conditions(model, asm, ind_inst, sub_inst, cfg, first_step):
    sub = cfg.substrate

    # Fixed bottom face (y = ys1)
    fixed_coords = [
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),
    ]
    asm.Set(
        faces=sub_inst.faces.findAt(*[(c,) for c in fixed_coords]),
        name="FIXEDBCSET",
    )
    model.EncastreBC(
        createStepName=first_step, localCsys=None,
        name="Fixed_constraint",
        region=asm.sets["FIXEDBCSET"],
    )

    # Symmetry on x = 0
    sym_coords = [
        (sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z / 2.0),
        (sub.xs1, sub.ys1 + sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1, sub.ys2 - sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z / 2.0),
    ]
    asm.Set(
        faces=sub_inst.faces.findAt(*[(c,) for c in sym_coords]),
        name="XsymmetryBCSet",
    )
    model.XsymmBC(
        createStepName=first_step, localCsys=None,
        name="x_axis_symmetry",
        region=asm.sets["XsymmetryBCSet"],
    )



#  Output requests
def _setup_output_requests(model, ind_inst, sub_inst, cfg, steps):
    names = cfg.naming
    scratch = cfg.scratch
    out = cfg.output

    # Remove Abaqus defaults
    for key in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[key]
    for key in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[key]

    # The first active step (indent or scratch) gets all outputs
    first_active = steps["all_active"][0]

    # History outputs (forces + energies during active steps) 
    model.HistoryOutputRequest(
        createStepName=first_active, name="ReactionForces",
        rebar=EXCLUDE,
        region=ind_inst.sets[names.indenter_set],
        sectionPoints=DEFAULT,
        timeInterval=scratch.history_interval,
        variables=out.history_force_variables,
    )

    model.HistoryOutputRequest(
        createStepName=first_active, name="IndenterDisp",
        rebar=EXCLUDE,
        region=ind_inst.sets[names.indenter_set],
        sectionPoints=DEFAULT,
        timeInterval=scratch.history_interval,
        variables=getattr(out, "history_disp_variables", ("U1", "U2", "U3")),
    )

    # Substrate-only energies (ALLKE, ALLIE, ALLAE) -> quasi-static & hourglass
    # checks. The rigid driver must NOT enter these, hence region=substrate.
    model.HistoryOutputRequest(
        createStepName=first_active, name="Energy",
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.history_interval,
        variables=out.history_energy_substrate,
    )

    # Whole-model energy balance (all components + ETOTAL), no region argument.
    # The driver's kinetic energy legitimately appears here as a ~constant
    # baseline; the balance must share this scope to be reconstructable.
    # (Abaqus also silently writes zeros for ETOTAL if requested on a set.)
    model.HistoryOutputRequest(
        createStepName=first_active, name="EnergyBalance",
        timeInterval=scratch.history_interval,
        variables=out.history_energy_whole,
    )

    # Field outputs 
    model.FieldOutputRequest(
        createStepName=first_active, name="FieldOutput",
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.field_interval_scratch,
        variables=out.field_variables,
    )

    model.FieldOutputRequest(
        createStepName=first_active, name="ContactForce",
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.field_interval_scratch,
        variables=out.contact_force_variables,
    )

    # Adjust output frequency per step 

    # Indentation step (if exists): fewer field frames
    if steps["indent"] is not None:
        model.fieldOutputRequests["FieldOutput"].setValuesInStep(
            stepName=steps["indent"],
            timeInterval=scratch.field_interval_indentation,
        )

    # Unload step: adjusted frequency, deactivate history
    model.fieldOutputRequests["FieldOutput"].setValuesInStep(
        stepName=steps["unload"],
        timeInterval=scratch.field_interval_unload,
    )
    model.fieldOutputRequests["ContactForce"].deactivate(steps["unload"])
    model.historyOutputRequests["Energy"].deactivate(steps["unload"])
    model.historyOutputRequests["EnergyBalance"].deactivate(steps["unload"])
    model.historyOutputRequests["ReactionForces"].deactivate(steps["unload"])
    model.historyOutputRequests["IndenterDisp"].deactivate(steps["unload"])

    # Recovery step (if exists): coarser field output, no history/contact
    if steps["recovery"] is not None:
        model.fieldOutputRequests["FieldOutput"].setValuesInStep(
            stepName=steps["recovery"],
            timeInterval=scratch.field_interval_recovery,
        )
        model.fieldOutputRequests["ContactForce"].deactivate(steps["recovery"])
        model.historyOutputRequests["Energy"].deactivate(steps["recovery"])
        model.historyOutputRequests["EnergyBalance"].deactivate(steps["recovery"])
        model.historyOutputRequests["ReactionForces"].deactivate(steps["recovery"])
        model.historyOutputRequests["IndenterDisp"].deactivate(steps["recovery"])



#  Contact
def _setup_contact(model, asm, ind_inst, sub_inst, cfg, first_step):
    sub = cfg.substrate
    names = cfg.naming
    fric = cfg.material.friction

    # Contact property 
    model.ContactProperty("IntProp-1")
    model.interactionProperties["IntProp-1"].TangentialBehavior(
        formulation=PENALTY,
        table=((0.0,),),       # friction updated by SubstrateMaterialAssignment
        fraction=fric.elastic_slip_fraction,
    )
    model.interactionProperties["IntProp-1"].NormalBehavior(
        allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD,
    )

    # Surfaces 
    rc = cfg.indenter.Rockwell_coords()

    asm.Surface(
        name=names.master_surface,
        side1Faces=ind_inst.faces.findAt(
            ((sub.xs1, sub.ys2, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + rc["xl2"], sub.ys2 + rc["yl2"], sub.zs1 + sub.dpo_z),),
        ),
    )

    asm.Surface(
        name=names.slave_surface,
        side1Faces=sub_inst.faces.findAt(
            ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2, (sub.zs2 + sub.zs1) / 2.0),),
        ),
    )

    # General contact 
    model.ContactExp(createStepName="Initial", name="Int-1")
    model.interactions["Int-1"].includedPairs.setValuesInStep(
        addPairs=((
            asm.surfaces[names.master_surface],
            asm.surfaces[names.slave_surface],
        ),),
        stepName="Initial",
        useAllstar=OFF,
    )
    model.interactions["Int-1"].contactPropertyAssignments.appendInStep(
        assignments=((GLOBAL, SELF, "IntProp-1"),),
        stepName="Initial",
    )
    model.interactions["Int-1"].smoothingAssignments.appendInStep(
        assignments=((asm.surfaces[names.slave_surface], REVOLUTION),),
        stepName=first_step,
    )

    # Node set for post-processing
    asm.Set(
        name=names.contact_region_nodes,
        nodes=asm.allSurfaces[names.slave_surface].nodes,
    )



#  ALE adaptive meshing
def _setup_ale(model, asm, sub_inst, cfg, steps):
    sub = cfg.substrate
    solver = cfg.solver
    zmid = (sub.zs1 + sub.zs2) / 2.0

    smoothing_priority = GRADED if solver.ale_smoothing_priority == "GRADED" else UNIFORM
    smoothing_algorithm = (
        GEOMETRY_ENHANCED if solver.ale_smoothing_algorithm == "GEOMETRY_ENHANCED"
        else VOLUMETRIC
    )

    model.AdaptiveMeshControl(
        name="Ada-1",
        smoothingPriority=smoothing_priority,
        smoothingAlgorithm=smoothing_algorithm,
        curvatureRefinement=1,
        volumetricSmoothingWeight=1,
        laplacianSmoothingWeight=0,
        equipotentialSmoothingWeight=0,
    )

    ale_cell_coords = [
        (sub.xs1, sub.ys2, zmid),
        (sub.xs1, sub.ys1, zmid),
        (sub.xs2, sub.ys1, zmid),
        (sub.xs1, sub.ys1, sub.zs1),
        (sub.xs2, sub.ys1, sub.zs1),
        (sub.xs1, sub.ys1, sub.zs2),
        (sub.xs2, sub.ys1, sub.zs2),
    ]
    asm.Set(
        name="ALE_Domain",
        cells=sub_inst.cells.findAt(*[(c,) for c in ale_cell_coords]),
    )

    # Active steps: full ALE frequency
    for step_name in steps["all_active"]:
        model.steps[step_name].AdaptiveMeshDomain(
            controls="Ada-1",
            meshSweeps=solver.ale_mesh_sweeps,
            frequency=solver.ale_frequency,
            region=asm.sets["ALE_Domain"],
        )

    # Unload step: lower frequency (less deformation happening)
    model.steps[steps["unload"]].AdaptiveMeshDomain(
        controls="Ada-1",
        meshSweeps=1,
        frequency=400,
        initialMeshSweeps=1,
        region=asm.sets["ALE_Domain"],
    )

    # Recovery step: minimal ALE (indenter is lifted, substrate relaxes)
    if steps["recovery"] is not None:
        model.steps[steps["recovery"]].AdaptiveMeshDomain(
            controls="Ada-1",
            meshSweeps=1,
            frequency=1000,
            initialMeshSweeps=1,
            region=asm.sets["ALE_Domain"],
        )