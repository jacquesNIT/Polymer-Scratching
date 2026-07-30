# -*- coding: utf-8 -*-
"""
Abaqus-kernel builders for the V&V benchmarks.

Three models:
  build_single_element_model()      level 0  -- one C3D8R, uniaxial / shear / relaxation
  build_indentation_model()         level 1a/1b -- Explicit elastic indentation
  build_indentation_model_standard() level 2 -- Abaqus/Standard quasi-static

Design constraints followed here:
  * the indentation models reuse create_substrate / mesh_substrate /
    create_indenter UNCHANGED, so a benchmark validates the production mesh
    generator, not a bespoke one;
  * loading uses SmoothStepAmplitude (C2-continuous, zero initial velocity
    AND acceleration) instead of a tabular amplitude with SMOOTH. The
    tabular+smooth combination used in production has a smoothing half-width
    of  smooth * min(scratch_time, unload_time), whose RELATIVE size changes
    when scratch_time is swept -- a confound a benchmark must not inherit;
  * no ALE anywhere: a reference case must not advect anything.

Items flagged VERIFY-IN-CAE are places where the exact API spelling depends on
the Abaqus release; each is wrapped so a rejection degrades gracefully and
prints instead of aborting the build.
"""

from ScratchSimulation.AbaqusModel.abaqus_env import *
from ScratchSimulation.AbaqusModel.Geometry.indenter import create_indenter
from ScratchSimulation.AbaqusModel.Geometry.substrate import create_substrate, mesh_substrate


# ==========================================================================
# Level 0 -- single element
# ==========================================================================

_ELEM_PART = "BenchCube"
_ELEM_SET = "BenchCubeSet"
_ELEM_INST = "BenchCubeInst"


def build_single_element_model(cfg, bench):
    """
    One C3D8R unit cube with the family's exact material card.

    Kinematics per mode (all with a single element, so the deformation
    gradient is uniform and the comparison with a material-point integration
    is exact):

      tension / compression : symmetry on x=0, y=0, z=0 ; u3 prescribed on
                              z=1. Lateral faces free -> uniaxial STRESS.
      shear                 : y=0 encastred, y=1 given u1 = gamma, u2 = u3 = 0
                              -> simple shear F = I + gamma e1 (x) e2.
      relaxation            : as tension, ramped over element_time then HELD.
    """
    model = mdb.models[cfg.naming.model_name]
    names = cfg.naming
    mode = bench.element_mode

    # ---- geometry: unit cube -------------------------------------------
    model.ConstrainedSketch(name="__cube__", sheetSize=10.0)
    sk = model.sketches["__cube__"]
    sk.rectangle(point1=(0.0, 0.0), point2=(1.0, 1.0))
    model.Part(dimensionality=THREE_D, name=_ELEM_PART, type=DEFORMABLE_BODY)
    part = model.parts[_ELEM_PART]
    part.BaseSolidExtrude(depth=1.0, sketch=sk)
    del model.sketches["__cube__"]

    part.Set(cells=part.cells.findAt(((0.5, 0.5, 0.5),)), name=_ELEM_SET)
    # The material assignment reuses SubstrateMaterialAssignment, which looks
    # for cfg.naming.substrate_set on the part -- give it the same handle.
    part.Set(cells=part.cells.findAt(((0.5, 0.5, 0.5),)), name=names.substrate_set)

    part.setMeshControls(elemShape=HEX, regions=part.cells, technique=STRUCTURED)
    part.setElementType(
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                            secondOrderAccuracy=OFF,
                            hourglassControl=ENHANCED,
                            distortionControl=OFF),),
        regions=(part.cells,))
    part.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(dependent=ON, name=_ELEM_INST, part=part)
    inst = asm.instances[_ELEM_INST]

    # ---- face sets ------------------------------------------------------
    def face(x, y, z):
        return inst.faces.findAt(((x, y, z),))

    asm.Set(faces=face(0.0, 0.5, 0.5), name="FX0")
    asm.Set(faces=face(0.5, 0.0, 0.5), name="FY0")
    asm.Set(faces=face(0.5, 0.5, 0.0), name="FZ0")
    asm.Set(faces=face(0.5, 1.0, 0.5), name="FY1")
    asm.Set(faces=face(0.5, 0.5, 1.0), name="FZ1")

    # ---- step -----------------------------------------------------------
    hold = 0.0
    if mode == "relaxation":
        taus = [float(r[2]) for r in getattr(cfg.material.viscoelastic,
                                             "prony_table", ((0.0, 0.0, 1.0),))]
        tsf = float(getattr(cfg.solver, "time_scale_factor", 1.0) or 1.0)
        hold = 40.0 * (max(taus) / tsf)
    total = float(bench.element_time) + hold

    # No mass scaling at all on a material-point test: the element is one cubic
    # millimetre, the cost is negligible, and any added mass would put inertia
    # between the card and the stress it is being checked against.
    model.ExplicitDynamicsStep(
        name="BenchStep", previous="Initial", timePeriod=total, nlgeom=ON,
        improvedDtMethod=ON,
        linearBulkViscosity=cfg.solver.linear_bulk_viscosity,
        quadBulkViscosity=cfg.solver.quad_bulk_viscosity)

    # ---- amplitude ------------------------------------------------------
    # Ramp to 1 over element_time, then HOLD -- SmoothStep gives zero velocity
    # and acceleration at both ends, which is what a material-point test needs.
    amp_data = ((0.0, 0.0), (float(bench.element_time), 1.0))
    model.SmoothStepAmplitude(name="BenchAmp", timeSpan=STEP, data=amp_data)

    # ---- boundary conditions -------------------------------------------
    strain = float(bench.element_strain)
    if mode in ("tension", "compression", "relaxation"):
        # Log strain is prescribed: u3 = exp(eps) - 1 on a unit cube.
        import math
        sgn = -1.0 if mode == "compression" else 1.0
        u3 = math.exp(sgn * abs(strain)) - 1.0
        model.XsymmBC(name="SymX", createStepName="Initial", region=asm.sets["FX0"])
        model.YsymmBC(name="SymY", createStepName="Initial", region=asm.sets["FY0"])
        model.ZsymmBC(name="SymZ", createStepName="Initial", region=asm.sets["FZ0"])
        model.DisplacementBC(name="Load", createStepName="BenchStep",
                             region=asm.sets["FZ1"], u3=u3,
                             amplitude="BenchAmp", distributionType=UNIFORM,
                             fieldName="", fixed=OFF, localCsys=None)
    elif mode == "shear":
        # Only u1 is prescribed on the top face. Leaving u2 and u3 free there
        # matters twice over: the element keeps free degrees of freedom (an
        # Explicit model whose every node is prescribed is fragile), and the
        # lateral faces stay traction-free so p = 0. A pressure-sensitive yield
        # surface is then probed at the r/q = 0 meridian, which is exactly the
        # point that separates K from beta -- tension and compression alone
        # only give their combination.
        model.EncastreBC(name="Bottom", createStepName="Initial",
                         region=asm.sets["FY0"], localCsys=None)
        model.DisplacementBC(name="Load", createStepName="BenchStep",
                             region=asm.sets["FY1"], u1=float(strain),
                             amplitude="BenchAmp", distributionType=UNIFORM,
                             fieldName="", fixed=OFF, localCsys=None)
    else:
        raise ValueError("Unknown single-element mode '%s'" % mode)

    # ---- output ---------------------------------------------------------
    for key in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[key]
    for key in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[key]

    n = 2000
    # The variable list must follow the CARD: requesting PEEQ on a purely
    # hyperelastic family, or SDV when no user subroutine is present, is a
    # rejected request rather than a silently empty column.
    fvars = ["S", "LE", "U", "EVOL"]
    if getattr(cfg.material.plasticity, "MODEL", "none") != "none":
        fvars += ["PE", "PEEQ"]
    evars = ["ALLKE", "ALLIE", "ALLSE", "ALLAE"]
    if getattr(cfg.material.plasticity, "MODEL", "none") != "none":
        evars.append("ALLPD")
    if getattr(cfg.material.viscoelastic, "MODEL", "none") != "none":
        evars.append("ALLCD")

    model.FieldOutputRequest(
        name="ElemField", createStepName="BenchStep",
        region=inst.sets[_ELEM_SET], numIntervals=n,
        variables=tuple(fvars))
    model.HistoryOutputRequest(
        name="ElemEnergy", createStepName="BenchStep", numIntervals=n,
        variables=tuple(evars))

    print(">>> single-element benchmark: mode=%s, strain=%.4g, T=%.4g s (hold %.4g s)"
          % (mode, strain, total, hold))
    return model, part


# ==========================================================================
# Level 1a / 1b -- Explicit elastic indentation (Hertz)
# ==========================================================================

def _indent_geometry(cfg, bench):
    """Common geometry/assembly for both indentation builders.

    The indenter is placed at MID-SPAN (z = zs2/2) rather than at the scratch
    start: an indentation has no travel direction, and mid-span maximises the
    distance to the z boundaries (the half-space assumption Hertz relies on).
    """
    model = mdb.models[cfg.naming.model_name]
    sub = cfg.substrate
    names = cfg.naming

    substrate_part = create_substrate(model, cfg)
    mesh_substrate(substrate_part, cfg)
    indenter_part = create_indenter(model, cfg)

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(dependent=ON, name=names.indenter_instance, part=indenter_part)
    asm.Instance(dependent=ON, name=names.substrate_instance, part=substrate_part)

    z_tip = 0.5 * (sub.zs1 + sub.zs2)
    asm.translate(instanceList=(names.indenter_instance,),
                  vector=(0.0, sub.ys2, 0.0))
    asm.translate(instanceList=(names.indenter_instance,),
                  vector=(0.0, 0.0, z_tip))
    return model, asm, substrate_part, indenter_part, z_tip


def _indent_bcs(model, asm, cfg, first_step, z_tip, u2, amp_name):
    sub = cfg.substrate
    names = cfg.naming
    sub_inst = asm.instances[names.substrate_instance]
    ind_inst = asm.instances[names.indenter_instance]

    fixed_coords = [
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),
    ]
    asm.Set(faces=sub_inst.faces.findAt(*[(c,) for c in fixed_coords]),
            name=names.fixed_set)
    model.EncastreBC(createStepName=first_step, localCsys=None,
                     name=names.fixed_bc, region=asm.sets[names.fixed_set])

    sym_coords = [
        (sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z / 2.0),
        (sub.xs1, sub.ys1 + sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1, sub.ys2 - sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
        (sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z / 2.0),
    ]
    asm.Set(faces=sub_inst.faces.findAt(*[(c,) for c in sym_coords]),
            name=names.symmetry_set)
    model.XsymmBC(createStepName=first_step, localCsys=None,
                  name=names.symmetry_bc, region=asm.sets[names.symmetry_set])

    region = ind_inst.sets[names.indenter_set]
    model.DisplacementBC(
        amplitude=amp_name, createStepName=first_step,
        distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None,
        name="BenchIndent", region=region,
        u1=0.0, u2=float(u2), u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)


def _indent_contact(model, asm, cfg, explicit=True):
    sub = cfg.substrate
    names = cfg.naming
    fric = cfg.material.friction
    ind_inst = asm.instances[names.indenter_instance]
    sub_inst = asm.instances[names.substrate_instance]
    rc = cfg.indenter.Rockwell_coords()
    z_tip = 0.5 * (sub.zs1 + sub.zs2)

    p_dep = bool(getattr(fric, "pressure_dependent", False))
    r_dep = bool(getattr(fric, "slip_rate_dependent", False))
    ncol = 1 + int(p_dep) + int(r_dep)

    model.ContactProperty(names.contact_property)
    model.interactionProperties[names.contact_property].TangentialBehavior(
        formulation=PENALTY,
        slipRateDependency=ON if r_dep else OFF,
        pressureDependency=ON if p_dep else OFF,
        table=(tuple([0.0] * ncol),),
        fraction=fric.elastic_slip_fraction)
    model.interactionProperties[names.contact_property].NormalBehavior(
        allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD)

    asm.Surface(name=names.master_surface,
                side1Faces=ind_inst.faces.findAt(
                    ((sub.xs1, sub.ys2, z_tip),),
                    ((sub.xs1 + rc["xl2"], sub.ys2 + rc["yl2"], z_tip),)))
    asm.Surface(name=names.slave_surface,
                side1Faces=sub_inst.faces.findAt(
                    ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2,
                      (sub.zs2 + sub.zs1) / 2.0),)))

    if explicit:
        model.ContactExp(createStepName="Initial", name=names.contact_interaction)
        inter = model.interactions[names.contact_interaction]
        inter.includedPairs.setValuesInStep(
            addPairs=((asm.surfaces[names.master_surface],
                       asm.surfaces[names.slave_surface]),),
            stepName="Initial", useAllstar=OFF)
        inter.contactPropertyAssignments.appendInStep(
            assignments=((GLOBAL, SELF, names.contact_property),),
            stepName="Initial")
    else:
        # VERIFY-IN-CAE: general contact in Abaqus/Standard. Falls back to a
        # surface-to-surface contact pair, which is the safer construct for a
        # single rigid indenter anyway.
        try:
            model.ContactStd(createStepName="Initial", name=names.contact_interaction)
            inter = model.interactions[names.contact_interaction]
            inter.includedPairs.setValuesInStep(
                addPairs=((asm.surfaces[names.master_surface],
                           asm.surfaces[names.slave_surface]),),
                stepName="Initial", useAllstar=OFF)
            inter.contactPropertyAssignments.appendInStep(
                assignments=((GLOBAL, SELF, names.contact_property),),
                stepName="Initial")
        except Exception as exc:
            print(">>> ContactStd rejected (%s); using a SurfaceToSurfaceContactStd "
                  "pair instead." % exc)
            model.SurfaceToSurfaceContactStd(
                name=names.contact_interaction, createStepName="Initial",
                master=asm.surfaces[names.master_surface],
                slave=asm.surfaces[names.slave_surface],
                sliding=FINITE, interactionProperty=names.contact_property,
                thickness=ON, adjustMethod=NONE,
                initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

    asm.Set(name=names.contact_region_nodes,
            nodes=asm.allSurfaces[names.slave_surface].nodes)


def _indent_outputs(model, asm, cfg, bench, step_name):
    names = cfg.naming
    ind_inst = asm.instances[names.indenter_instance]
    sub_inst = asm.instances[names.substrate_instance]

    for key in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[key]
    for key in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[key]

    nh = int(bench.n_history)
    nf = int(bench.n_field)

    # The whole point of the Hertz benchmark is a DENSE P(h) curve: one
    # monotonic ramp sweeps N_a = a/h continuously, so the error-vs-resolution
    # relation comes out of a single job.
    model.HistoryOutputRequest(
        name="BenchReaction", createStepName=step_name,
        region=ind_inst.sets[names.indenter_set], numIntervals=nh,
        variables=("RF1", "RF2", "RF3", "U1", "U2", "U3"))
    model.HistoryOutputRequest(
        name="BenchEnergySub", createStepName=step_name,
        region=sub_inst.sets[names.substrate_set], numIntervals=nh,
        variables=("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"))
    model.HistoryOutputRequest(
        name="BenchEnergyWhole", createStepName=step_name, numIntervals=nh,
        variables=("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLFD",
                   "ALLWK", "ALLPW", "ALLCW", "ALLMW", "ETOTAL"))

    # CAREA is the direct measurement of the contact area -- the quantity the
    # SCOF of a Briscoe law is entirely made of, and the one the production
    # extractor requests but never writes to CSV.
    try:
        model.HistoryOutputRequest(
            name="BenchContact", createStepName=step_name,
            interactions=(names.contact_interaction,), numIntervals=nh,
            variables=("CFN1", "CFN2", "CFN3", "CFNM",
                       "CFS1", "CFS2", "CFS3", "CFSM", "CAREA"))
    except Exception as exc:
        print(">>> contact-pair history (CFN/CAREA) rejected: %s. The contact "
              "radius will be measured from the CPRESS field instead." % exc)

    # CSTRESS at the field frames gives CPRESS on every slave node -> a_num
    # measured directly, and the p(r) profile compared with Hertz.
    model.FieldOutputRequest(
        name="BenchField", createStepName=step_name,
        region=sub_inst.sets[names.substrate_set], numIntervals=nf,
        variables=("S", "MISES", "PRESS", "LE", "PE", "PEEQ", "U", "COORD",
                   "CSTRESS", "CDISP"))


def build_indentation_model(cfg, bench):
    """Level 1a (s ~ 1) / 1b (production s) -- Abaqus/Explicit."""
    model, asm, substrate_part, _ind, z_tip = _indent_geometry(cfg, bench)
    names = cfg.naming
    solver = cfg.solver

    use_variable = float(getattr(solver, "target_time_increment", 0.0) or 0.0) > 0.0
    ms_region = asm.instances[names.substrate_instance].sets[names.substrate_set]
    ms_tuple = (
        SEMI_AUTOMATIC, ms_region,
        THROUGHOUT_STEP if use_variable else AT_BEGINNING,
        0.0 if use_variable else solver.mass_scale,
        solver.target_time_increment,
        BELOW_MIN if use_variable else None,
        0, 10, 0.0, 0.0, 0, None,
    )

    step = "BenchIndentStep"
    model.ExplicitDynamicsStep(
        name=step, previous="Initial", timePeriod=bench.total_time, nlgeom=ON,
        improvedDtMethod=ON, massScaling=(ms_tuple,),
        linearBulkViscosity=solver.linear_bulk_viscosity,
        quadBulkViscosity=solver.quad_bulk_viscosity)

    if bench.hold_time > 0.0:
        frac = bench.ramp_time / bench.total_time
        model.SmoothStepAmplitude(name="BenchAmp", timeSpan=STEP,
                                  data=((0.0, 0.0), (frac, 1.0)))
    else:
        model.SmoothStepAmplitude(name="BenchAmp", timeSpan=STEP,
                                  data=((0.0, 0.0), (1.0, 1.0)))

    _indent_bcs(model, asm, cfg, step, z_tip, -abs(bench.depth_max), "BenchAmp")
    _indent_contact(model, asm, cfg, explicit=True)
    _indent_outputs(model, asm, cfg, bench, step)

    print(">>> Hertz benchmark (EXPLICIT): depth=%.4g mm, T=%.4g s, h=%.4g mm, "
          "mass scaling %s"
          % (bench.depth_max, bench.total_time, cfg.mesh.fine_size_x,
             ("variable dt_target=%.3e s" % solver.target_time_increment)
             if use_variable else ("fixed f=%g" % solver.mass_scale)))
    return model, substrate_part


def build_indentation_model_standard(cfg, bench):
    """
    Level 2 -- Abaqus/Standard, quasi-static by construction.

    No mass scaling, no bulk viscosity, no time-increment choice: the gap
    between this RF2 and the Explicit RF2 at the same depth IS the
    quasi-staticity error of the Explicit protocol, measured rather than
    assumed.
    """
    model, asm, substrate_part, _ind, z_tip = _indent_geometry(cfg, bench)
    names = cfg.naming

    # mesh_substrate() assigns the EXPLICIT element library; retype for Standard.
    sub = cfg.substrate
    zmid = 0.5 * (sub.zs1 + sub.zs2)
    coords = [(sub.xs1, sub.ys1, sub.zs1), (sub.xs2, sub.ys1, sub.zs1),
              (sub.xs1, sub.ys1, sub.zs2), (sub.xs2, sub.ys1, sub.zs2),
              (sub.xs1, sub.ys1, zmid), (sub.xs1, sub.ys2, zmid),
              (sub.xs2, sub.ys1, zmid)]
    cells = substrate_part.cells.findAt(*[(c,) for c in coords])
    substrate_part.setElementType(
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                            hourglassControl=ENHANCED),
                   ElemType(elemCode=C3D6, elemLibrary=STANDARD),
                   ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
        regions=(cells,))
    substrate_part.generateMesh()
    asm.regenerate()

    step = "BenchStaticStep"
    model.StaticStep(name=step, previous="Initial", timePeriod=1.0,
                     nlgeom=ON, maxNumInc=10000,
                     initialInc=0.01, minInc=1e-9, maxInc=0.02,
                     # Contact chatter at first touch is the usual failure
                     # mode; adaptive stabilisation with a damping factor that
                     # the report then CHECKS against ALLSD/ALLIE.
                     stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
                     stabilizationMagnitude=2e-4,
                     adaptiveDampingRatio=0.05,
                     continueDampingFactors=False)

    model.SmoothStepAmplitude(name="BenchAmp", timeSpan=STEP,
                              data=((0.0, 0.0), (1.0, 1.0)))

    _indent_bcs(model, asm, cfg, step, z_tip, -abs(bench.depth_max), "BenchAmp")
    _indent_contact(model, asm, cfg, explicit=False)

    ind_inst = asm.instances[names.indenter_instance]
    sub_inst = asm.instances[names.substrate_instance]
    for key in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[key]
    for key in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[key]
    model.HistoryOutputRequest(
        name="BenchReaction", createStepName=step,
        region=ind_inst.sets[names.indenter_set], frequency=1,
        variables=("RF1", "RF2", "RF3", "U1", "U2", "U3"))
    model.HistoryOutputRequest(
        name="BenchEnergyWhole", createStepName=step, frequency=1,
        variables=("ALLIE", "ALLSE", "ALLPD", "ALLWK", "ALLSD", "ETOTAL"))
    model.FieldOutputRequest(
        name="BenchField", createStepName=step,
        region=sub_inst.sets[names.substrate_set],
        numIntervals=int(bench.n_field),
        variables=("S", "MISES", "PRESS", "LE", "PE", "PEEQ", "U", "COORD",
                   "CSTRESS", "CDISP"))

    print(">>> Quasi-static reference (STANDARD): depth=%.4g mm, h=%.4g mm, "
          "no mass scaling, no inertia." % (bench.depth_max, cfg.mesh.fine_size_x))
    return model, substrate_part