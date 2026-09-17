# Rigid indenter geometry (Rockwell sphere-cone or pyramid) and its placement on the substrate.

from ScratchSimulation.AbaqusModel.abaqus_env import *

def create_indenter(model, cfg):           
    # Creation of the indenter Part according to the configuration
    
    ind = cfg.indenter

    if ind.indenter_type == ind.ROCKWELL:
        return create_rockwell(model, cfg)
    elif ind.indenter_type == ind.PYRAMID:
        return create_pyramid(model, cfg)
    else:
        raise ValueError("Unknown indenter type: %s" % ind.indenter_type)
    

#  Rockwell indenter
def create_rockwell(model, cfg):

    ind = cfg.indenter
    names = cfg.naming
    rc = ind.Rockwell_coords()

    # Sketch
    model.ConstrainedSketch(name="__profile__", sheetSize=cfg.sheet_size)
    sk = model.sketches["__profile__"]

    # Vertical construction line (revolution axis)
    sk.ConstructionLine(point1=(0.0, -5.0), point2=(0.0, 5.0))
    sk.FixedConstraint(entity=sk.geometry.findAt((0.0, 0.0)))

    # Horizontal construction line
    sk.ConstructionLine(point1=(0.0, 0.0), point2=(1.0, 0.0))
    sk.HorizontalConstraint(addUndoState=False, entity=sk.geometry.findAt((0.5, 0.0)))
    sk.FixedConstraint(entity=sk.geometry.findAt((1.0, 0.0)))

    # Arc for spherical tip
    sk.ArcByCenterEnds(
        center=(0.0, ind.tip_radius),
        point1=(rc["xc1"], rc["yc1"]),
        point2=(rc["xc2"], rc["yc2"]),
    )
    sk.CoincidentConstraint(
        entity1=sk.vertices.findAt((rc["xc1"], rc["yc1"])),
        entity2=sk.geometry.findAt((0.5, 0.0)),
    )
    sk.CoincidentConstraint(
        entity1=sk.vertices.findAt((rc["xc1"], rc["yc1"])),
        entity2=sk.geometry.findAt((0.0, 1.0)),
    )

    # Conical line tangent to tip
    sk.Line(point1=(rc["xl1"], rc["yl1"]), point2=(rc["xl2"], rc["yl2"]))
    sk.TangentConstraint(
        entity1=sk.geometry.findAt((rc["xl2"], rc["yl2"])),
        entity2=sk.geometry.findAt((rc["xc3"], rc["yc3"])),
    )
    sk.CoincidentConstraint(
        entity1=sk.vertices.findAt((rc["xl1"], rc["yl1"])),
        entity2=sk.vertices.findAt((rc["xc2"], rc["yc2"])),
    )

    # Revolve into analytic rigid surface
    sk.sketchOptions.setValues(constructionGeometry=ON)
    sk.assignCenterline(line=sk.geometry.findAt((0.0, 1.0)))

    model.Part(dimensionality=THREE_D, name=names.indenter_name, type=ANALYTIC_RIGID_SURFACE)
    model.parts[names.indenter_name].AnalyticRigidSurfRevolve(sketch=sk)
    del sk

    part = model.parts[names.indenter_name]

    # Reference point & inertia
    part.ReferencePoint(point=part.vertices.findAt((rc["xc1"], rc["yc1"], 0.0)))
    part.Set(name=names.indenter_set, referencePoints=(part.referencePoints[2],))

    part.engineeringFeatures.PointMassInertia(
        alpha=0.0, composite=0.0,
        i11=0.0, i22=0.0, i33=0.0, mass=1e-6,  # to be adjusted to mass scaling for force-driven tests
        name=names.inertia_name,
        region=part.sets[names.indenter_set],
    )

    return part

def create_pyramid(model, cfg):

    ind = cfg.indenter
    names = cfg.naming
    pc = ind.Pyramid_coords()
    H = pc["H"]                    # virtual apex height
    Hf = pc["H_frustum"]  # actual height of the truncated pyramid

    # Geometric sanity check: the pyramid must be tall enough for the prescribed depth
    depth = abs(float(getattr(cfg.scratch, "scratch_depth", 0.0) or 0.0))
    if depth > 0.0 and Hf <= 1.5 * depth:
        raise ValueError(
            "Pyramid too shallow: apex height H=%.4f mm for a scratch depth of %.4f mm. "
            "Increase Indenter_Config.base_apothem." % (H, depth))

    # Sketch: regular n-gon centred on the indenter axis
    model.ConstrainedSketch(name="__profile__", sheetSize=cfg.sheet_size)
    sk = model.sketches["__profile__"]
    verts = pc["vertices"]
    for i in range(pc["n"]):
        sk.Line(point1=verts[i], point2=verts[(i + 1) % pc["n"]])

    if ind.extrude_depth:
        depths = [float(ind.extrude_depth)]
    elif pc["a_tip"] > 0.0:
        depths = [Hf]
    else:
        # sharp apex requested: over-extrude and let Abaqus clip, then fall back
        depths = [max(1.0, 2.0 * H), 1.05 * H, H]

    part = None
    last_err = None
    for d in depths:
        if names.indenter_name in model.parts.keys():
            del model.parts[names.indenter_name]
        model.Part(dimensionality=THREE_D, name=names.indenter_name,
                   type=DISCRETE_RIGID_SURFACE)
        try:
            model.parts[names.indenter_name].BaseSolidExtrude(
                depth=d, draftAngle=-pc["theta_deg"], sketch=sk)
            part = model.parts[names.indenter_name]
            break
        except Exception as err:
            last_err = err
    if part is None:
        raise ValueError("Pyramid extrusion failed (draft angle %.3f deg): %s"
                         % (pc["theta_deg"], last_err))
    del sk

    # Solid -> rigid shell: dropping the cell keeps all the faces
    part.RemoveCells(cellList=part.cells[0:len(part.cells)])

    # Reference point at the lowest point of the indenter
    rp = part.ReferencePoint(point=(0.0, 0.0, Hf))
    part.Set(name=names.indenter_set,
             referencePoints=(part.referencePoints[rp.id],))

    _m = float(ind.rigid_mass)
    _i = ind.rigid_inertia
    if _i is None:
        _i = _m * float(ind.base_apothem) ** 2
    _i = float(_i)
    part.engineeringFeatures.PointMassInertia(
        alpha=0.0, composite=0.0,
        i11=_i, i22=_i, i33=_i, mass=_m,
        name=names.inertia_name,
        region=part.sets[names.indenter_set],
    )

    # Mesh: R3D4 / R3D3 rigid elements, no contribution to the stable increment.
    ms = ind.mesh_size
    if not ms:
        ms = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_z)
    part.seedPart(size=ms, deviationFactor=0.1, minSizeFactor=0.1)

    if ind.tip_bias:
        # Optional refinement towards the apex (check the bias direction in CAE).
        ms_min = ind.mesh_min_size or 0.25 * ms
        edge_pts = tuple([(p,) for p in ind.pyramid_edge_points(s=0.35)])
        part.seedEdgeByBias(
            biasMethod=SINGLE, constraint=FINER,
            end2Edges=part.edges.findAt(*edge_pts),
            maxSize=ms, minSize=ms_min,
        )

    part.generateMesh()

    return part


# Placement of the indenter instance
def place_indenter(asm, cfg):
    # Brings the tip/apex onto the substrate top surface at z = zs1 + dpo_z.

    ind = cfg.indenter
    names = cfg.naming
    sub = cfg.substrate

    if ind.indenter_type == ind.PYRAMID:
        Hf = ind.Pyramid_coords()["H_frustum"]
        asm.rotate(instanceList=(names.indenter_instance,),
                   axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0),
                   angle=90.0)
        asm.translate(instanceList=(names.indenter_instance,),
                      vector=(0.0, sub.ys2 + Hf, sub.zs1 + sub.dpo_z))
    else:
        asm.translate(instanceList=(names.indenter_instance,),
                      vector=(0.0, sub.ys2, 0.0))
        asm.translate(instanceList=(names.indenter_instance,),
                      vector=(0.0, 0.0, sub.dpo_z))