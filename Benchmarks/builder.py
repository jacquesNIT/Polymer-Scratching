# -*- coding: utf-8 -*-
"""
Abaqus construction for the benchmarks.

    build_single_element_model()        level 0
    build_indentation_model()           levels 1a / 1b (Explicit)
    build_indentation_model_standard()  level 2 (Standard)

NB: Reuses create_substrate / mesh_substrate / create_indenter
"""

import math
from ..AbaqusModel.abaqus_env import *
from ..AbaqusModel.Geometry.indenter import create_indenter
from ..AbaqusModel.Geometry.substrate import create_substrate, mesh_substrate

_ELEM_PART = "BenchCube"
_ELEM_SET = "BenchCubeSet"
_ELEM_INST = "BenchCubeInst"

def _ramp_amplitude(model, ramp_time, name="BenchAmp"):
    """Continuous ramp reaching full amplitude at ramp_time."""
    model.SmoothStepAmplitude(name=name, timeSpan=STEP, data=((0.0, 0.0), (float(ramp_time), 1.0)))
    return name

def _clear_output_requests(model):
    for key in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[key]
    for key in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[key]



# Level 0: single element

def build_single_element_model(cfg, bench):
    """
        One C3D8R element cube.
        Deformation gradient uniform, so the comparison with a material-point integration is exact.
    """
    model = mdb.models[cfg.naming.model_name]
    names, mode = cfg.naming, bench.element_mode

    model.ConstrainedSketch(name="__cube__", sheetSize=10.0)
    sk = model.sketches["__cube__"]
    sk.rectangle(point1=(0.0, 0.0), point2=(1.0, 1.0))
    model.Part(dimensionality=THREE_D, name=_ELEM_PART, type=DEFORMABLE_BODY)
    part = model.parts[_ELEM_PART]
    part.BaseSolidExtrude(depth=1.0, sketch=sk)
    del model.sketches["__cube__"]

    cell = part.cells.findAt(((0.5, 0.5, 0.5),))
    part.Set(cells=cell, name=_ELEM_SET)
    # The material assignment looks for cfg.naming.substrate_set.
    part.Set(cells=cell, name=names.substrate_set)

    part.setMeshControls(elemShape=HEX, regions=part.cells, technique=STRUCTURED)
    part.setElementType(
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                            secondOrderAccuracy=OFF,
                            hourglassControl=ENHANCED, distortionControl=OFF),),
        regions=(part.cells,))
    part.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(dependent=ON, name=_ELEM_INST, part=part)
    inst = asm.instances[_ELEM_INST]

    for nm, pt in (("FX0", (0.0, 0.5, 0.5)), ("FY0", (0.5, 0.0, 0.5)),
                   ("FZ0", (0.5, 0.5, 0.0)), ("FY1", (0.5, 1.0, 0.5)),
                   ("FZ1", (0.5, 0.5, 1.0))):
        asm.Set(faces=inst.faces.findAt((pt,)), name=nm)

    # Relaxation holds long enough for the slowest Prony term to decay.
    hold = 0.0
    if mode == "relaxation":
        taus = [float(r[2]) for r in getattr(cfg.material.viscoelastic,
                                             "prony_table", ((0.0, 0.0, 1.0),))]
        tsf = float(getattr(cfg.solver, "time_scale_factor", 1.0) or 1.0)
        hold = 40.0 * (max(taus) / tsf)
    total = float(bench.element_time) + hold

    # No mass scaling: added mass would sit between the card and the stress.
    model.ExplicitDynamicsStep(
        name="BenchStep", previous="Initial", timePeriod=total, nlgeom=ON,
        improvedDtMethod=ON,
        linearBulkViscosity=cfg.solver.linear_bulk_viscosity,
        quadBulkViscosity=cfg.solver.quad_bulk_viscosity)
    _ramp_amplitude(model, bench.element_time)

    strain = float(bench.element_strain)
    if mode in ("tension", "compression", "relaxation"):
        # Log strain prescribed: u3 = exp(eps) - 1 on a unit cube. Lateral
        # faces free -> uniaxial STRESS.
        sgn = -1.0 if mode == "compression" else 1.0
        model.XsymmBC(name="SymX", createStepName="Initial", region=asm.sets["FX0"])
        model.YsymmBC(name="SymY", createStepName="Initial", region=asm.sets["FY0"])
        model.ZsymmBC(name="SymZ", createStepName="Initial", region=asm.sets["FZ0"])
        model.DisplacementBC(
            name="Load", createStepName="BenchStep", region=asm.sets["FZ1"],
            u3=math.exp(sgn * abs(strain)) - 1.0, amplitude="BenchAmp",
            distributionType=UNIFORM, fieldName="", fixed=OFF, localCsys=None)
    elif mode == "shear":
        # Only u1 prescribed on top: the element keeps free DOFs and the
        # lateral faces stay traction-free, so p = 0 and the r/q = 0 meridian
        # is probed -- the point that separates K from beta.
        model.EncastreBC(name="Bottom", createStepName="Initial",
                         region=asm.sets["FY0"], localCsys=None)
        model.DisplacementBC(
            name="Load", createStepName="BenchStep", region=asm.sets["FY1"],
            u1=strain, amplitude="BenchAmp", distributionType=UNIFORM,
            fieldName="", fixed=OFF, localCsys=None)
    else:
        raise ValueError("Unknown single-element mode '%s'" % mode)

    _clear_output_requests(model)
    # Output list follows the card: PEEQ on a hyperelastic family or SDV with
    # no user subroutine are rejected requests, not empty columns.
    has_plastic = getattr(cfg.material.plasticity, "MODEL", "none") != "none"
    has_visco = getattr(cfg.material.viscoelastic, "MODEL", "none") != "none"
    fvars = ["S", "LE", "U", "EVOL"] + (["PE", "PEEQ"] if has_plastic else [])
    evars = ["ALLKE", "ALLIE", "ALLSE", "ALLAE"]
    if has_plastic:
        evars.append("ALLPD")       # plastic dissipation
    if has_visco:
        evars.append("ALLCD")       # creep/viscous dissipation

    model.FieldOutputRequest(name="ElemField", createStepName="BenchStep",
                             region=inst.sets[_ELEM_SET], numIntervals=2000,
                             variables=tuple(fvars))
    model.HistoryOutputRequest(name="ElemEnergy", createStepName="BenchStep",
                               numIntervals=2000, variables=tuple(evars))

    print(">>> single element: mode=%s, strain=%.4g, T=%.4g s (hold %.4g s)"
          % (mode, strain, total, hold))
    return model, part



# Shared indentation geometry

def _indent_geometry(cfg):
    """Substrate + indenter placed at mid-span."""
    model = mdb.models[cfg.naming.model_name]
    sub, names = cfg.substrate, cfg.naming

    substrate_part = create_substrate(model, cfg)
    mesh_substrate(substrate_part, cfg)
    indenter_part = create_indenter(model, cfg)

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(dependent=ON, name=names.indenter_instance, part=indenter_part)
    asm.Instance(dependent=ON, name=names.substrate_instance, part=substrate_part)

    z_tip = 0.5 * (sub.zs1 + sub.zs2)
    asm.translate(instanceList=(names.indenter_instance,),
                  vector=(0.0, sub.ys2, z_tip))
    return model, asm, substrate_part, z_tip


def _indent_bcs(model, asm, cfg, step, u2, amp="BenchAmp"):
    sub, names = cfg.substrate, cfg.naming
    sub_inst = asm.instances[names.substrate_instance]

    bottom = [(sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),
              (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
              (sub.xs1 + sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
              (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),
              (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),
              (sub.xs2 - sub.dpo_x / 2.0, sub.ys1, sub.zs1 + sub.dpo_z / 2.0)]
    asm.Set(faces=sub_inst.faces.findAt(*[(c,) for c in bottom]),
            name=names.fixed_set)
    model.EncastreBC(createStepName=step, localCsys=None, name=names.fixed_bc,
                     region=asm.sets[names.fixed_set])

    symm = [(sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z / 2.0),
            (sub.xs1, sub.ys1 + sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
            (sub.xs1, sub.ys2 - sub.dpo_y / 2.0, (sub.zs2 + sub.zs1) / 2.0),
            (sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z / 2.0)]
    asm.Set(faces=sub_inst.faces.findAt(*[(c,) for c in symm]),
            name=names.symmetry_set)
    model.XsymmBC(createStepName=step, localCsys=None, name=names.symmetry_bc,
                  region=asm.sets[names.symmetry_set])

    model.DisplacementBC(
        amplitude=amp, createStepName=step, distributionType=UNIFORM,
        fieldName="", fixed=OFF, localCsys=None, name="BenchIndent",
        region=asm.instances[names.indenter_instance].sets[names.indenter_set],
        u1=0.0, u2=float(u2), u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)


def _indent_contact(model, asm, cfg, explicit=True):
    sub, names, fric = cfg.substrate, cfg.naming, cfg.material.friction
    rc = cfg.indenter.Rockwell_coords()
    z_tip = 0.5 * (sub.zs1 + sub.zs2)

    ncol = 1 + int(bool(getattr(fric, "pressure_dependent", False))) \
             + int(bool(getattr(fric, "slip_rate_dependent", False)))
    model.ContactProperty(names.contact_property)
    prop = model.interactionProperties[names.contact_property]
    prop.TangentialBehavior(
        formulation=PENALTY,
        slipRateDependency=ON if getattr(fric, "slip_rate_dependent", False) else OFF,
        pressureDependency=ON if getattr(fric, "pressure_dependent", False) else OFF,
        table=(tuple([0.0] * ncol),), fraction=fric.elastic_slip_fraction)
    prop.NormalBehavior(allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
                        pressureOverclosure=HARD)

    asm.Surface(name=names.master_surface,
                side1Faces=asm.instances[names.indenter_instance].faces.findAt(
                    ((sub.xs1, sub.ys2, z_tip),),
                    ((sub.xs1 + rc["xl2"], sub.ys2 + rc["yl2"], z_tip),)))
    asm.Surface(name=names.slave_surface,
                side1Faces=asm.instances[names.substrate_instance].faces.findAt(
                    ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2,
                      (sub.zs2 + sub.zs1) / 2.0),)))
    pair = ((asm.surfaces[names.master_surface],
             asm.surfaces[names.slave_surface]),)

    if explicit:
        model.ContactExp(createStepName="Initial", name=names.contact_interaction)
        inter = model.interactions[names.contact_interaction]
        inter.includedPairs.setValuesInStep(addPairs=pair, stepName="Initial",
                                            useAllstar=OFF)
        inter.contactPropertyAssignments.appendInStep(
            assignments=((GLOBAL, SELF, names.contact_property),),
            stepName="Initial")
    else:
        # VERIFY-IN-CAE: the general-contact spelling varies by release, so a
        # surface-to-surface pair is used as fallback.
        try:
            model.ContactStd(createStepName="Initial",
                             name=names.contact_interaction)
            inter = model.interactions[names.contact_interaction]
            inter.includedPairs.setValuesInStep(addPairs=pair,
                                                stepName="Initial",
                                                useAllstar=OFF)
            inter.contactPropertyAssignments.appendInStep(
                assignments=((GLOBAL, SELF, names.contact_property),),
                stepName="Initial")
        except Exception as exc:
            print(">>> ContactStd rejected (%s); using a contact pair." % exc)
            model.SurfaceToSurfaceContactStd(
                name=names.contact_interaction, createStepName="Initial",
                master=asm.surfaces[names.master_surface],
                slave=asm.surfaces[names.slave_surface], sliding=FINITE,
                interactionProperty=names.contact_property, thickness=ON,
                adjustMethod=NONE, initialClearance=OMIT, datumAxis=None,
                clearanceRegion=None)

    asm.Set(name=names.contact_region_nodes,
            nodes=asm.allSurfaces[names.slave_surface].nodes)


def _indent_outputs(model, asm, cfg, bench, step, explicit=True):
    names = cfg.naming
    ind = asm.instances[names.indenter_instance]
    sub = asm.instances[names.substrate_instance]
    _clear_output_requests(model)
    nh, nf = int(bench.n_history), int(bench.n_field)
    freq = dict(numIntervals=nh) if explicit else dict(frequency=1)

    # RF2/U2 at the rigid reference point: the P(h) curve compared with Hertz.
    # A dense history matters because one monotonic ramp sweeps N_a = a/h.
    model.HistoryOutputRequest(
        name="BenchReaction", createStepName=step, region=ind.sets[names.indenter_set],
        variables=("RF1", "RF2", "RF3", "U1", "U2", "U3"), **freq)

    if explicit:
        # Substrate-scoped energies: ALLKE/ALLIE is the quasi-staticity check,
        # ALLAE the hourglass check, ALLIE itself is compared with (2/5)Ph.
        model.HistoryOutputRequest(
            name="BenchEnergySub", createStepName=step,
            region=sub.sets[names.substrate_set], numIntervals=nh,
            variables=("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"))
        # Whole-model energies: ALLPW is the contact penalty work, the series
        # compliance that makes the numerical force fall below Hertz.
        model.HistoryOutputRequest(
            name="BenchEnergyWhole", createStepName=step, numIntervals=nh,
            variables=("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLFD",
                       "ALLWK", "ALLPW", "ALLCW", "ALLMW", "ETOTAL"))
    else:
        # ALLSD tracks the stabilisation energy, which must stay small against
        # ALLIE for the implicit reference to be trustworthy.
        model.HistoryOutputRequest(
            name="BenchEnergyWhole", createStepName=step, frequency=1,
            variables=("ALLIE", "ALLSE", "ALLPD", "ALLWK", "ALLSD", "ETOTAL"))

    if explicit:
        # CAREA is the direct contact-area measurement; with a Briscoe law the
        # apparent friction is nothing but alpha + tau0*A_c/Fn.
        try:
            model.HistoryOutputRequest(
                name="BenchContact", createStepName=step,
                interactions=(names.contact_interaction,), numIntervals=nh,
                variables=("CFN1", "CFN2", "CFN3", "CFNM",
                           "CFS1", "CFS2", "CFS3", "CFSM", "CAREA"))
        except Exception as exc:
            print(">>> contact-pair history rejected (%s); the contact radius "
                  "will come from CPRESS instead." % exc)

    # CSTRESS on the field frames gives CPRESS per slave node -> a_num measured
    # directly and the p(r) profile compared with the Hertz ellipse.
    model.FieldOutputRequest(
        name="BenchField", createStepName=step,
        region=sub.sets[names.substrate_set], numIntervals=nf,
        variables=("S", "MISES", "PRESS", "LE", "PE", "PEEQ", "U", "COORD",
                   "CSTRESS", "CDISP"))



# Levels 1a / 1b -- Explicit indentation

def build_indentation_model(cfg, bench):
    model, asm, part, _z = _indent_geometry(cfg)
    names, solver = cfg.naming, cfg.solver
    step = "BenchIndentStep"

    variable = float(getattr(solver, "target_time_increment", 0.0) or 0.0) > 0.0
    # Mass scaling scoped to the substrate: MODEL scope would inflate the
    # rigid indenter point mass and contaminate the energy balance.
    ms = (SEMI_AUTOMATIC,
          asm.instances[names.substrate_instance].sets[names.substrate_set],
          THROUGHOUT_STEP if variable else AT_BEGINNING,
          0.0 if variable else solver.mass_scale,
          solver.target_time_increment,
          BELOW_MIN if variable else None, 0, 10, 0.0, 0.0, 0, None)

    model.ExplicitDynamicsStep(
        name=step, previous="Initial", timePeriod=bench.total_time, nlgeom=ON,
        improvedDtMethod=ON, massScaling=(ms,),
        linearBulkViscosity=solver.linear_bulk_viscosity,
        quadBulkViscosity=solver.quad_bulk_viscosity)

    _ramp_amplitude(model, bench.ramp_time)
    _indent_bcs(model, asm, cfg, step, -abs(bench.depth_max))
    _indent_contact(model, asm, cfg, explicit=True)
    _indent_outputs(model, asm, cfg, bench, step, explicit=True)

    print(">>> indentation (EXPLICIT): depth=%.4g mm, T=%.4g s, h=%.4g mm, %s"
          % (bench.depth_max, bench.total_time, cfg.mesh.fine_size_x,
             ("variable dt=%.3e s" % solver.target_time_increment) if variable
             else ("fixed f=%g" % solver.mass_scale)))
    return model, part



# Level 2 -- Abaqus/Standard

def build_indentation_model_standard(cfg, bench):
    model, asm, part, _z = _indent_geometry(cfg)
    sub = cfg.substrate
    zmid = 0.5 * (sub.zs1 + sub.zs2)

    # mesh_substrate() assigns the EXPLICIT element library; retype for Standard.
    coords = [(sub.xs1, sub.ys1, sub.zs1), (sub.xs2, sub.ys1, sub.zs1),
              (sub.xs1, sub.ys1, sub.zs2), (sub.xs2, sub.ys1, sub.zs2),
              (sub.xs1, sub.ys1, zmid), (sub.xs1, sub.ys2, zmid),
              (sub.xs2, sub.ys1, zmid)]
    part.setElementType(
        elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                            hourglassControl=ENHANCED),
                   ElemType(elemCode=C3D6, elemLibrary=STANDARD),
                   ElemType(elemCode=C3D4, elemLibrary=STANDARD)),
        regions=(part.cells.findAt(*[(c,) for c in coords]),))
    part.generateMesh()
    asm.regenerate()

    step, period = "BenchStaticStep", 1.0
    model.StaticStep(name=step, previous="Initial", timePeriod=period,
                     nlgeom=ON, maxNumInc=10000, initialInc=0.01, minInc=1e-9,
                     maxInc=0.02,
                     # Contact chatter at first touch is the usual failure
                     # mode; the report then checks ALLSD against ALLIE.
                     stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
                     stabilizationMagnitude=2e-4, adaptiveDampingRatio=0.05,
                     continueDampingFactors=False)

    # Tied to the step period: amplitude data is in step seconds.
    _ramp_amplitude(model, period)
    _indent_bcs(model, asm, cfg, step, -abs(bench.depth_max))
    _indent_contact(model, asm, cfg, explicit=False)
    _indent_outputs(model, asm, cfg, bench, step, explicit=False)

    print(">>> indentation (STANDARD): depth=%.4g mm, h=%.4g mm, no mass "
          "scaling, no inertia." % (bench.depth_max, cfg.mesh.fine_size_x))
    return model, part