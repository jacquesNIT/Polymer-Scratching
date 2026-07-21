# Scratch test model builder for polymer simulation.
#Orchestrates geometry creation, assembly, step definition, boundary conditions, contact modelling, and output requests.

from ScratchSimulation.AbaqusModel.abaqus_env import *
from ScratchSimulation.AbaqusModel.Geometry.indenter import create_indenter
from ScratchSimulation.AbaqusModel.Geometry.substrate import create_substrate, mesh_substrate

def build_scratch_model(cfg):
    # Build a complete scratch-test model (geometry + steps + BCs + contact + outputs).

    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

    model = mdb.models[cfg.naming.model_name]
    sub = cfg.substrate
    names = cfg.naming
    scratch = cfg.scratch
    solver = cfg.solver

    #  1. Geometry
    substrate_part = create_substrate(model, cfg)
    mesh_substrate(substrate_part, cfg)
    indenter_part = create_indenter(model, cfg)

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

    #  3. Steps (needs asm: mass scaling is scoped to the substrate instance set)
    steps = _create_steps(model, asm, cfg)

    #  4. Boundary conditions (substrate)
    _apply_boundary_conditions(model, asm, ind_inst, sub_inst, cfg, steps["first"])

    #  5. Loading (displacement-controlled indenter via amplitudes)
    _apply_loading(model, ind_inst, cfg, steps["first"])

    #  6. Contact (before output_request to get asm.surfaces[names.slave_surface])
    _setup_contact(model, asm, ind_inst, sub_inst, cfg, steps["first"])

    #  7. Output request
    _setup_output_requests(model, asm, ind_inst, sub_inst, cfg, steps)

    #  8. ALE adaptive meshing
    if solver.use_ALE:
        _setup_ale(model, asm, sub_inst, cfg, steps)

    return model, substrate_part


#  Step creation
def _create_steps(model, asm, cfg):
    # Create all analysis steps based on the scratch configuration.
    # Returns a dict with step names keyed by role (first, indent, scratch, unload, recovery, all_active, all)
   
    scratch = cfg.scratch
    solver = cfg.solver
    names = cfg.naming

    # Mass scaling tuple (shared by all active steps).
    # Scoped to the SUBSTRATE element set only: MODEL scope also multiplies the
    # rigid indenter's point mass by the scaling factor, which (i) makes the
    # indenter inertia-dominated in force-controlled mode (zero-penetration
    # failure) and (ii) inflates the WM_ALLKE baseline of the energy balance.
    ms_region = asm.instances[names.substrate_instance].sets[names.substrate_set]
    use_variable = solver.target_time_increment > 0.0
    ms_tuple = (
        SEMI_AUTOMATIC,
        ms_region,
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
        name = names.step_indent
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
    name = names.step_scratch
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
    name = names.step_unload
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
        name = names.step_recovery
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

    # Tabular-amplitude smoothing: rounds the velocity discontinuities at the
    # amplitude kinks (t1/t2/t3) -- the main source of inertial ringing in
    # explicit quasi-static loading -- while keeping constant velocity in the
    # middle of each segment. None -> solver default.
    smooth_val = getattr(scratch, "amplitude_smoothing", None)
    if smooth_val is None:
        smooth_val = SOLVER_DEFAULT

    if scratch.is_force_controlled:

        # U2 is force-driven,a ConcentratedForce (cf2) is applied below. 
        # U3 is displacement-driven, scratch speed is imposed
        model.TabularAmplitude(
            data=scratch.length_amplitude(),
            name=names.amp_length,
            smooth=smooth_val,
            timeSpan=TOTAL,
        )

        model.DisplacementBC(
            amplitude=names.amp_length,
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name=names.bc_travel,
            region=region,
            u1=UNSET,
            u2=UNSET,
            u3=scratch.scratch_length,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
        )

        model.TabularAmplitude(
            data=scratch.force_amplitude(),
            name=names.amp_force,
            smooth=smooth_val,
            timeSpan=TOTAL,
        )

        # Half-symmetry model: scratch_force is halved (same convention used on the RF2/Hertz checks)
        # Cf2 has to be negative
        model.ConcentratedForce(
            amplitude=names.amp_force,
            cf2=-(scratch.scratch_force / 2.0),
            createStepName=first_step,
            distributionType=UNIFORM, field="", localCsys=None,
            name=names.bc_force,
            region=region,
        )

    elif scratch.uses_single_amplitude:

        # Progressive without recovery: depth and length share one amplitude
        model.TabularAmplitude(
            data=scratch.depth_amplitude(),
            name=names.amp_single,
            smooth=smooth_val,
            timeSpan=TOTAL,
        )
        model.DisplacementBC(
            amplitude=names.amp_single,
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name=names.bc_scratch,
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
            name=names.amp_depth,
            smooth=smooth_val,
            timeSpan=TOTAL,
        )
        model.TabularAmplitude(
            data=scratch.length_amplitude(),
            name=names.amp_length,
            smooth=smooth_val,
            timeSpan=TOTAL,
        )
        model.DisplacementBC(
            amplitude=names.amp_depth,
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name=names.bc_depth,
            region=region,
            u1=UNSET,
            u2=scratch.scratch_depth,
            u3=UNSET,
            ur1=UNSET, ur2=UNSET, ur3=UNSET,
        )
        model.DisplacementBC(
            amplitude=names.amp_length,
            createStepName=first_step,
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
            name=names.bc_travel,
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
        name=names.indenter_constraint_bc,
        region=region,
        u1=SET, u2=UNSET, u3=UNSET,
        ur1=SET, ur2=SET, ur3=SET,
    )



#  Boundary conditions
def _apply_boundary_conditions(model, asm, ind_inst, sub_inst, cfg, first_step):

    sub = cfg.substrate
    names = cfg.naming

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
        name=names.fixed_set,
    )
    model.EncastreBC(
        createStepName=first_step, localCsys=None,
        name=names.fixed_bc,
        region=asm.sets[names.fixed_set],
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
        name=names.symmetry_set,
    )
    model.XsymmBC(
        createStepName=first_step, localCsys=None,
        name=names.symmetry_bc,
        region=asm.sets[names.symmetry_set],
    )



#  Output requests
def _setup_output_requests(model, asm, ind_inst, sub_inst, cfg, steps):

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
        createStepName=first_active, name=names.out_reaction,
        rebar=EXCLUDE,
        region=ind_inst.sets[names.indenter_set],
        sectionPoints=DEFAULT,
        timeInterval=scratch.history_interval,
        variables=out.history_force_variables,
    )

    # Contact-pair force history (CFN/CFS).
    # In force-driven scratches RF2 ~ 0 (u2 carries no displacement BC), so the
    # normal force must be read from the contact pair (CFN2). The exact
    # variable identifiers / domain form depend on the Abaqus version (flagged
    # for CAE verification), so the request is created defensively: a rejected
    # request prints a warning instead of breaking displacement-driven builds.
    contact_pair_ok = _request_contact_pair_history(model, asm, cfg, first_active)

    model.HistoryOutputRequest(
        createStepName=first_active, name=names.out_indenter_disp,
        rebar=EXCLUDE,
        region=ind_inst.sets[names.indenter_set],
        sectionPoints=DEFAULT,
        timeInterval=scratch.history_interval,
        variables=getattr(out, "history_disp_variables", ("U1", "U2", "U3")),
    )

    # Substrate-only energies (ALLKE, ALLIE, ALLAE) -> quasi-static & hourglass
    # checks. The rigid driver must NOT enter these, hence region=substrate.
    model.HistoryOutputRequest(
        createStepName=first_active, name=names.out_energy_substrate,
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.history_interval,
        variables=out.history_energy_substrate,
    )

    # Whole-model energy balance (all components + ETOTAL), no region argument.
    # The driver's kinetic energy legitimately appears here as a ~constant
    # baseline; the balance must share this scope to be reconstructable.
    # (Abaqus also silently writes zeros for ETOTAL if requested on a set.)
    model.HistoryOutputRequest(
        createStepName=first_active, name=names.out_energy_whole,
        timeInterval=scratch.history_interval,
        variables=out.history_energy_whole,
    )

    # Field outputs 
    model.FieldOutputRequest(
        createStepName=first_active, name=names.out_field,
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.field_interval_scratch,
        variables=out.field_variables,
    )

    model.FieldOutputRequest(
        createStepName=first_active, name=names.out_contact,
        region=sub_inst.sets[names.substrate_set],
        timeInterval=scratch.field_interval_scratch,
        variables=out.contact_force_variables,
    )


    # Adjust output frequency per step 

    # Indentation step (if exists): fewer field frames
    if steps["indent"] is not None:
        model.fieldOutputRequests[names.out_field].setValuesInStep(
            stepName=steps["indent"],
            timeInterval=scratch.field_interval_indentation,
        )

    # Unload step: adjusted frequency, deactivate history
    model.fieldOutputRequests[names.out_field].setValuesInStep(
        stepName=steps["unload"],
        timeInterval=scratch.field_interval_unload,
    )
    model.fieldOutputRequests[names.out_contact].deactivate(steps["unload"])
    #model.historyOutputRequests[names.out_energy_substrate].deactivate(steps["unload"])
    #model.historyOutputRequests[names.out_energy_whole].deactivate(steps["unload"])
    _low_freq = max(float(scratch.unload_time), float(scratch.recovery_time)) / 10.0
    model.historyOutputRequests[names.out_energy_substrate].setValuesInStep(stepName=steps["unload"], timeInterval=_low_freq)
    model.historyOutputRequests[names.out_energy_whole].setValuesInStep(stepName=steps["unload"], timeInterval=_low_freq)
    model.historyOutputRequests[names.out_reaction].deactivate(steps["unload"])
    if contact_pair_ok:
        model.historyOutputRequests[names.out_contact_pair].deactivate(steps["unload"])
    model.historyOutputRequests[names.out_indenter_disp].deactivate(steps["unload"])

    # Recovery step (if exists): coarser field output, no history/contact
    if steps["recovery"] is not None:
        model.fieldOutputRequests[names.out_field].setValuesInStep(
            stepName=steps["recovery"],
            timeInterval=scratch.field_interval_recovery,
        )
        model.fieldOutputRequests[names.out_contact].deactivate(steps["recovery"])
        #model.historyOutputRequests[names.out_energy_substrate].deactivate(steps["recovery"])
        #model.historyOutputRequests[names.out_energy_whole].deactivate(steps["recovery"])
        model.historyOutputRequests[names.out_energy_substrate].setValuesInStep(
            stepName=steps["recovery"], timeInterval=_low_freq)
        model.historyOutputRequests[names.out_energy_whole].setValuesInStep(
            stepName=steps["recovery"], timeInterval=_low_freq)
        model.historyOutputRequests[names.out_reaction].deactivate(steps["recovery"])
        if contact_pair_ok:
            model.historyOutputRequests[names.out_contact_pair].deactivate(steps["recovery"])
        model.historyOutputRequests[names.out_indenter_disp].deactivate(steps["recovery"])


def _request_contact_pair_history(model, asm, cfg, first_active):
    # Defensive creation of the CFN/CFS contact-pair history request.
    # For general contact the interaction-domain form is attempted first, then
    # the surface-region form; returns False (with a printed warning) if both
    # are rejected, so the rest of the model build is never compromised.
    # Exact CFN/CFS identifiers still flagged for verification in CAE.
    names = cfg.naming
    scratch = cfg.scratch
    out = cfg.output
    variables = getattr(
        out, "history_contact_pair_variables",
        ("CFN1", "CFN2", "CFN3", "CFNM", "CFS1", "CFS2", "CFS3", "CFSM", "CAREA"))

    err1 = err2 = None
    try:
        model.HistoryOutputRequest(
            createStepName=first_active, name=names.out_contact_pair,
            interactions=(names.contact_interaction,),
            timeInterval=scratch.history_interval,
            variables=variables)
        return True
    except Exception as exc:
        err1 = str(exc)
        if names.out_contact_pair in model.historyOutputRequests.keys():
            del model.historyOutputRequests[names.out_contact_pair]

    try:
        model.HistoryOutputRequest(
            createStepName=first_active, name=names.out_contact_pair,
            region=asm.surfaces[names.slave_surface],
            timeInterval=scratch.history_interval,
            variables=variables)
        return True
    except Exception as exc:
        err2 = str(exc)
        if names.out_contact_pair in model.historyOutputRequests.keys():
            del model.historyOutputRequests[names.out_contact_pair]

    print("Warning: contact-pair force history (CFN/CFS) could not be requested "
          "(interaction form: %s / surface form: %s). CFN* columns will be zero; "
          "force-controlled verification falls back to RF2." % (err1, err2))
    return False


#  Contact
def _setup_contact(model, asm, ind_inst, sub_inst, cfg, first_step):
    sub = cfg.substrate
    names = cfg.naming
    fric = cfg.material.friction

    # Contact property 
    model.ContactProperty(names.contact_property)
    model.interactionProperties[names.contact_property].TangentialBehavior(
        formulation=PENALTY,
        table=((0.0,),),       # friction updated by SubstrateMaterialAssignment
        fraction=fric.elastic_slip_fraction,
    )
    model.interactionProperties[names.contact_property].NormalBehavior(
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
    model.ContactExp(createStepName="Initial", name=names.contact_interaction)
    model.interactions[names.contact_interaction].includedPairs.setValuesInStep(
        addPairs=((
            asm.surfaces[names.master_surface],
            asm.surfaces[names.slave_surface],
        ),),
        stepName="Initial",
        useAllstar=OFF,
    )
    model.interactions[names.contact_interaction].contactPropertyAssignments.appendInStep(
        assignments=((GLOBAL, SELF, names.contact_property),),
        stepName="Initial",
    )
    model.interactions[names.contact_interaction].smoothingAssignments.appendInStep(
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
    names = cfg.naming
    zmid = (sub.zs1 + sub.zs2) / 2.0

    smoothing_priority = GRADED if solver.ale_smoothing_priority == "GRADED" else UNIFORM
    smoothing_algorithm = (
        GEOMETRY_ENHANCED if solver.ale_smoothing_algorithm == "GEOMETRY_ENHANCED"
        else VOLUMETRIC
    )

    model.AdaptiveMeshControl(
        name=names.ale_control,
        smoothingPriority=smoothing_priority,
        smoothingAlgorithm=smoothing_algorithm,
        curvatureRefinement=getattr(solver, "ale_curvature_refinement", 1),
        # NB: the three *SmoothingWeight arguments below parametrise the BASIC
        # smoothing algorithm; with GEOMETRY_ENHANCED they are most likely
        # ignored. VERIFY in the *ADAPTIVE MESH CONTROLS block of the generated
        # .inp before assuming they tune anything.
        volumetricSmoothingWeight=1,
        laplacianSmoothingWeight=0,
        equipotentialSmoothingWeight=0,
    )

    # ALE domain scope (solver.ale_domain).
    #   "refined" : the refined contact cell only -- x in [xs1, xs1+dpo_x],
    #               y in [ys2-dpo_y, ys2], z in [zs1+dpo_z, zs2-dpo_z]. With
    #               the default partitions that is a 0.25 mm deep box under a
    #               0.04 mm groove (~6x the depth), which contains the whole
    #               plastic zone. Everything outside stays purely Lagrangian:
    #               no advection of an undamaged far field, and the report can
    #               state that ALE acts on the contact zone only.
    #   "contact" : refined cell + the cell directly underneath (same x/z
    #               footprint, full substrate height). Safety valve.
    #   "full"    : legacy behaviour, the 7 substrate cells.
    #
    # NB: nodes ON the domain boundary stay Lagrangian, so plasticity must not
    # reach it. The bottom face sits at dpo_y = 0.25 mm while the contact
    # radius is a ~ 0.12 mm at 40 um depth (2a ~ 0.24 mm): the margin on the
    # ELASTIC field is thin. After the first run, check PEEQ ~ 0 on the
    # y = ys2 - dpo_y face; switch to "contact" if it is not.
    scope = str(getattr(solver, "ale_domain", "full")).lower()
    _refined_cell = (sub.xs1, sub.ys2, zmid)          # refined contact cell (= names.refined_set)
    _under_cell = (sub.xs1, sub.ys1, zmid)            # cell directly below it
    if scope == "refined":
        ale_cell_coords = [_refined_cell]
    elif scope == "contact":
        ale_cell_coords = [_refined_cell, _under_cell]
    else:
        ale_cell_coords = [
            _refined_cell,
            _under_cell,
            (sub.xs2, sub.ys1, zmid),
            (sub.xs1, sub.ys1, sub.zs1),
            (sub.xs2, sub.ys1, sub.zs1),
            (sub.xs1, sub.ys1, sub.zs2),
            (sub.xs2, sub.ys1, sub.zs2),
        ]
    asm.Set(
        name=names.ale_domain_set,
        cells=sub_inst.cells.findAt(*[(c,) for c in ale_cell_coords]),
    )
    print(">>> ALE domain '%s': %d cell(s), frequency=%s, meshSweeps=%s, curvatureRefinement=%s."
          % (scope, len(ale_cell_coords), solver.ale_frequency, solver.ale_mesh_sweeps,
             getattr(solver, "ale_curvature_refinement", 1)))
    _print_ale_courant(cfg)

    # Active steps: full ALE frequency
    for step_name in steps["all_active"]:
        model.steps[step_name].AdaptiveMeshDomain(
            controls=names.ale_control,
            meshSweeps=solver.ale_mesh_sweeps,
            frequency=solver.ale_frequency,
            region=asm.sets[names.ale_domain_set],
        )

    # Unload / recovery: ALE suppressed by default (solver.ale_in_passive_steps).
    #
    # CRITICAL -- adaptive mesh domains PROPAGATE from step to step in
    # Abaqus/Explicit. Simply DELETING the two blocks below would NOT disable
    # ALE: the passive steps would INHERIT the active-step settings
    # (frequency=200, meshSweeps=3), i.e. far MORE remeshing than the legacy
    # frequency=400/1000. The domain must be RE-DECLARED with a suppressing
    # frequency, never removed.
    #
    # Why suppress: during unload and recovery the material relaxes without
    # generating new distortion, so every advection only diffuses the residual
    # stress field that sets the final groove profile -- the very quantity
    # these two steps exist to measure.
    passive_steps = [steps["unload"]]
    if steps["recovery"] is not None:
        passive_steps.append(steps["recovery"])

    if getattr(solver, "ale_in_passive_steps", False):
        # Legacy behaviour: minimal ALE kept during unload (400) / recovery (1000).
        for step_name, freq in zip(passive_steps, (400, 1000)):
            model.steps[step_name].AdaptiveMeshDomain(
                controls=names.ale_control,
                meshSweeps=1,
                frequency=freq,
                initialMeshSweeps=1,
                region=asm.sets[names.ale_domain_set],
            )
    else:
        for step_name in passive_steps:
            _suppress_ale_in_step(model, asm, cfg, step_name)


def _suppress_ale_in_step(model, asm, cfg, step_name):
    # Re-declare the adaptive mesh domain with settings that never trigger a
    # remeshing, so the active-step domain cannot propagate into a passive step.
    # frequency=0 is the documented suppression value; the huge-frequency
    # variants are fallbacks should the CAE API reject 0 (or 0 sweeps).
    #
    # VERIFY ONCE in the generated .inp: the *ADAPTIVE MESH block of the unload
    # step must carry FREQUENCY=0 (or 10000000) with no initial sweeps, and the
    # .sta must report no remeshing after the scratch step.
    names = cfg.naming
    region = asm.sets[names.ale_domain_set]
    candidates = (
        {"frequency": 10000000, "meshSweeps": 1, "initialMeshSweeps": 1},
        {"frequency": 1000000,  "meshSweeps": 1, "initialMeshSweeps": 1},
    )
    err = None
    for kwargs in candidates:
        try:
            model.steps[step_name].AdaptiveMeshDomain(
                controls=names.ale_control, region=region, **kwargs)
            print(">>> ALE suppressed in step '%s' (%s)."
                  % (step_name, ", ".join("%s=%s" % (k, kwargs[k]) for k in sorted(kwargs))))
            return
        except Exception as exc:
            err = str(exc)
    print("Warning: could not suppress ALE in step '%s' (%s). The active-step "
          "adaptive mesh domain will PROPAGATE into it -- check the .inp."
          % (step_name, err))


def _print_ale_courant(cfg):
    # Log the remeshing Courant number so every ALE run carries, in its own
    # .log, the number that governs its advection error.
    try:
        from ScratchSimulation.AbaqusModel.Configuration.base import ale_remesh_courant
        C = ale_remesh_courant(cfg)
    except Exception as exc:
        print(">>> ALE: remeshing Courant estimate unavailable (%s)." % exc)
        return
    if C is None:
        print(">>> ALE: remeshing Courant estimate not applicable to this config.")
        return
    if C < 0.15:
        verdict = "TOO DIFFUSIVE -- raise ale_frequency"
    elif C > 0.6:
        verdict = "TOO COARSE -- lower ale_frequency or raise ale_mesh_sweeps"
    else:
        verdict = "OK"
    print(">>> ALE: remeshing Courant C = %.3f [%s] (target 0.2-0.5; C -> 0 "
          "maximises advection diffusion of PEEQ / stress)." % (C, verdict))