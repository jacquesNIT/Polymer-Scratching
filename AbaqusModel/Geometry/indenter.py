# Generation of the rigid Rockwell indenter geometry

from ScratchSimulation.AbaqusModel.abaqus_env import *

def create_indenter(model, cfg):           
    # Creation of the indenter Part according to the configuration
    
    ind = cfg.indenter

    if ind.indenter_type == ind.ROCKWELL:
        return create_rockwell(model, cfg)
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
        i11=0.0, i22=0.0, i33=0.0, mass=1e-3,     # mass changed from 1.0t to 1e-3t = 1.0kg 
        name=names.inertia_name,
        region=part.sets[names.indenter_set],
    )

    return part