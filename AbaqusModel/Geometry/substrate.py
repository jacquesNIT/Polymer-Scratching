# Substrate geometry creation, partitioning, and meshing.

from ScratchSimulation.AbaqusModel.abaqus_env import *

#  Geometry + partitions
def create_substrate(model, cfg):
    # Create the substrate part with all datum-plane partitions.

    sub = cfg.substrate
    names = cfg.naming

    # Sketch & extrude 
    model.ConstrainedSketch(name="__profile__", sheetSize=cfg.sheet_size)
    sk = model.sketches["__profile__"]
    sk.rectangle(point1=(sub.xs1, sub.ys1), point2=(sub.xs2, sub.ys2))
    model.Part(dimensionality=THREE_D, name=names.substrate_name, type=DEFORMABLE_BODY)
    model.parts[names.substrate_name].BaseSolidExtrude(depth=sub.zs2, sketch=sk)
    del sk

    part = model.parts[names.substrate_name]

    # ---- Datum planes ----
    part.DatumPlaneByPrincipalPlane(offset=sub.dpo_x, principalPlane=YZPLANE)
    part.DatumPlaneByPrincipalPlane(offset=sub.zs1 + sub.dpo_z, principalPlane=XYPLANE)
    part.DatumPlaneByPrincipalPlane(offset=sub.zs2 - sub.dpo_z, principalPlane=XYPLANE)
    part.DatumPlaneByPrincipalPlane(offset=sub.ys2 - sub.dpo_y, principalPlane=XZPLANE)

    # Cell partitions 
    # Partition along x-axis
    part.PartitionCellByDatumPlane(
        cells=part.cells.findAt(((sub.xs2, sub.ys2, sub.zs2),)),
        datumPlane=part.datums[2],
    )

    # Partitions along z-axis (front & back)
    part.PartitionCellByDatumPlane(
        cells=part.cells.findAt(
            ((sub.xs1, sub.ys1, sub.zs1),),
            ((sub.xs2, sub.ys1, sub.zs1),),
        ),
        datumPlane=part.datums[3],
    )
    part.PartitionCellByDatumPlane(
        cells=part.cells.findAt(
            ((sub.xs1, sub.ys1, sub.zs2),),
            ((sub.xs2, sub.ys1, sub.zs2),),
        ),
        datumPlane=part.datums[4],
    )

    # Partition along y-axis (top layer)
    part.PartitionCellByDatumPlane(
        cells=part.cells.findAt(
            ((sub.xs1, sub.ys1, (sub.zs2 + sub.zs1) / 2.0),),
        ),
        datumPlane=part.datums[5],
    )

    # Named sets 
    zmid = (sub.zs1 + sub.zs2) / 2.0
    part.Set(
        cells=part.cells.findAt(
            ((sub.xs1, sub.ys1, sub.zs1),),
            ((sub.xs2, sub.ys1, sub.zs1),),
            ((sub.xs1, sub.ys1, sub.zs2),),
            ((sub.xs2, sub.ys1, sub.zs2),),
            ((sub.xs1, sub.ys1, zmid),),
            ((sub.xs1, sub.ys2, zmid),),
            ((sub.xs2, sub.ys1, zmid),),
        ),
        name=names.substrate_set,
    )

    part.Set(
        cells=part.cells.findAt(((sub.xs1, sub.ys2, zmid),)),
        name=names.refined_set,
    )

    return part

#  Meshing
def mesh_substrate(part, cfg):

    sub = cfg.substrate
    msh = cfg.mesh
    names = cfg.naming

    zmid = (sub.zs1 + sub.zs2) / 2.0

    all_cell_coords = [
        (sub.xs1, sub.ys1, sub.zs1),
        (sub.xs2, sub.ys1, sub.zs1),
        (sub.xs1, sub.ys1, sub.zs2),
        (sub.xs2, sub.ys1, sub.zs2),
        (sub.xs1, sub.ys1, zmid),
        (sub.xs1, sub.ys2, zmid),
        (sub.xs2, sub.ys1, zmid),
    ]

    cell_region = part.cells.findAt(*[((c),) for c in all_cell_coords])

    # Mesh controls 
    part.setMeshControls(elemShape=HEX, regions=cell_region, technique=STRUCTURED)

    # ---- Element type ----
    if msh.hourglass_control == "ENHANCED":
        hg = ENHANCED
    elif msh.hourglass_control == "RELAX STIFFNESS":
        hg = RELAX_STIFFNESS
    else:
        hg = DEFAULT
    dc = DEFAULT

    part.setElementType(
        elemTypes=(
            ElemType(
                elemCode=C3D8R,
                elemLibrary=EXPLICIT,
                secondOrderAccuracy=ON if msh.second_order_accuracy else OFF,
                distortionControl=dc,
                hourglassControl=hg,
                elemDeletion=ON if msh.element_deletion else OFF,
                maxDegradation=msh.max_degradation,
            ),
            ElemType(elemCode=C3D6, elemLibrary=EXPLICIT),
            ElemType(elemCode=C3D4, elemLibrary=EXPLICIT),
        ),
        regions=(cell_region,),
    )

    # Edge seeds : refined zone 
    # Z-direction edges (scratch direction, refined zone)
    part.seedEdgeBySize(
        constraint=FINER, deviationFactor=0.1,
        edges=part.edges.findAt(
            ((sub.xs1, sub.ys2, zmid),),
            ((sub.xs1, sub.ys2 - sub.dpo_y, zmid),),
            ((sub.xs1 + sub.dpo_x, sub.ys2, zmid),),
            ((sub.xs1 + sub.dpo_x, sub.ys2 - sub.dpo_y, zmid),),
        ),
        size=msh.fine_size_z,
    )

    # X-direction edges (width, refined zone)
    part.seedEdgeBySize(
        constraint=FINER, deviationFactor=0.1,
        edges=part.edges.findAt(
            ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2 - sub.dpo_y, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2, sub.zs2 - sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x / 2.0, sub.ys2 - sub.dpo_y, sub.zs2 - sub.dpo_z),),
        ),
        size=msh.fine_size_x,
    )

    # Y-direction edges (depth, refined zone)
    part.seedEdgeBySize(
        constraint=FINER, deviationFactor=0.1,
        edges=part.edges.findAt(
            ((sub.xs1, sub.ys2 - sub.dpo_y / 2.0, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x, sub.ys2 - sub.dpo_y / 2.0, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x, sub.ys2 - sub.dpo_y / 2.0, sub.zs2 - sub.dpo_z),),          
        ),
        size=msh.fine_size_y,
    )

    # Edge seeds : biased transitions 
    # Y-direction transition (fine near surface -> coarse at bottom)
    part.seedEdgeByBias(
        biasMethod=SINGLE, constraint=FINER,
        end1Edges=part.edges.findAt(
            ((sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z),),
            ((sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z),),
            ((sub.xs2, (sub.ys1 + sub.ys2) / 2.0, sub.zs1),),
            ((sub.xs2, (sub.ys1 + sub.ys2) / 2.0, sub.zs2),),
            ((sub.xs1 + sub.dpo_x, (sub.ys1 + sub.ys2) / 2.0, sub.zs2),),
        ),
        end2Edges=part.edges.findAt(
            ((sub.xs2, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z),),
            ((sub.xs2, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x, (sub.ys1 + sub.ys2) / 2.0, sub.zs1 + sub.dpo_z),),
            ((sub.xs1 + sub.dpo_x, (sub.ys1 + sub.ys2) / 2.0, sub.zs2 - sub.dpo_z),),
            ((sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs1),),
            ((sub.xs1, (sub.ys1 + sub.ys2) / 2.0, sub.zs2),),
            ((sub.xs1 + sub.dpo_x, (sub.ys1 + sub.ys2) / 2.0, sub.zs1),),
        ),
        maxSize=msh.coarse_size_1,
        minSize=msh.fine_size_y,
    )

    # Z-direction transition (fine near centre -> coarse at ends)
    part.seedEdgeByBias(
        biasMethod=SINGLE, constraint=FINER,
        end1Edges=part.edges.findAt(
            ((sub.xs1 + sub.dpo_x, sub.ys2, sub.zs2 - sub.dpo_z / 2.0),),
            ((sub.xs1, sub.ys2, sub.zs2 - sub.dpo_z / 2.0),),
            ((sub.xs2, sub.ys2, sub.zs2 - sub.dpo_z / 2.0),),
            ((sub.xs1, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),),
            ((sub.xs1 + sub.dpo_x, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),),
            ((sub.xs2, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),),
        ),
        end2Edges=part.edges.findAt(
            ((sub.xs1 + sub.dpo_x, sub.ys2, sub.zs1 + sub.dpo_z / 2.0),),
            ((sub.xs1, sub.ys2, sub.zs1 + sub.dpo_z / 2.0),),
            ((sub.xs2, sub.ys2, sub.zs1 + sub.dpo_z / 2.0),),
            ((sub.xs1, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),),
            ((sub.xs1 + sub.dpo_x, sub.ys1, sub.zs2 - sub.dpo_z / 2.0),),
            ((sub.xs2, sub.ys1, sub.zs1 + sub.dpo_z / 2.0),),
        ),
        maxSize=msh.coarse_size_1,
        minSize=msh.fine_size_z,
    )

    # X-direction transition (fine near contact -> coarse far side)
    part.seedEdgeByBias(
        biasMethod=SINGLE, constraint=FINER,
        end1Edges=part.edges.findAt(
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys2, sub.zs1),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys2, sub.zs2),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys1, sub.zs1 + sub.dpo_z),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys1, sub.zs2 - sub.dpo_z),),
        ),
        end2Edges=part.edges.findAt(
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys2, sub.zs1 + sub.dpo_z),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys2, sub.zs2 - sub.dpo_z),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys1, sub.zs1),),
            ((sub.dpo_x + (sub.xs2 - sub.dpo_x) / 2.0, sub.ys1, sub.zs2),),
        ),
        maxSize=msh.coarse_size_2,
        minSize=msh.fine_size_x,
    )

    part.generateMesh()