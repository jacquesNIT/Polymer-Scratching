# partition patterns on the substrate top face to guide the mesher

""" Si implémentation dans le reste, penser à référencer cfg plutot que mdb.models"""
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


def partition_top_face(part, cfg):

    sub = cfg.substrate
    msh = cfg.mesh

    partition_area_length = sub.zs2 - 2 * sub.dpo_z

    top_face = part.faces.getByBoundingBox(
        sub.xs1 + sub.dpo_x, sub.ys2 - 1e-3, sub.zs1 + sub.dpo_z,
        sub.xs2, sub.ys2 + 1e-3, sub.zs2 - sub.dpo_z,
    )

    sk_transform = part.MakeSketchTransform(
        sketchPlane=top_face[0],
        sketchPlaneSide=SIDE1,
        sketchUpEdge=part.edges.findAt(
            ((sub.xs2 + (sub.xs1 + sub.dpo_x)) / 2.0, sub.ys2, sub.zs2 - sub.dpo_z),
        ),
        origin=(0.0, sub.ys2, 0.0),
        sketchOrientation=RIGHT,
    )

    s = mdb.models[part.modelName].ConstrainedSketch(
        name="__profile__", sheetSize=10, transform=sk_transform,
    )
    part.projectReferencesOntoSketch(
        filter=COPLANAR_EDGES,
        sketch=mdb.models["Model-1"].sketches["__profile__"],
    )

    n_partitions = int((partition_area_length / msh.fine_size_z) / 2)

    for n in range(n_partitions):
        x1 = sub.xs1 + sub.dpo_x
        x2 = x1 + 2 * msh.fine_size_x
        z1 = sub.zs1 + sub.dpo_z + n * 2 * msh.fine_size_z
        z2 = z1 + 2 * msh.fine_size_z
        if n % 2 == 0:
            _sketch_block(s, x1, x2, z1, z2)
        else:
            _sketch_block(s, x1, x2, z2, z1)

    part.PartitionCellBySketch(
        cells=part.cells.findAt(
            (((sub.xs2 + (sub.xs1 + sub.dpo_x)) / 2.0, sub.ys2, (sub.zs2 + sub.zs1) / 2.0),)
        ),
        sketch=s,
        sketchPlane=top_face[0],
        sketchUpEdge=part.edges.findAt(
            ((sub.xs2 + (sub.xs1 + sub.dpo_x)) / 2.0, sub.ys2, sub.zs2 - sub.dpo_z),
        ),
    )
    del s

def _sketch_block(s, x1, x2, z1, z2):

    x_mid = (x2 + x1) / 2.0
    z_mid = (z2 + z1) / 2.0

    s.rectangle(point1=(z1, x1), point2=(z2, x2))
    s.Line(point1=(z1, x2), point2=(z_mid, x_mid))
    s.Line(point1=(z_mid, x1), point2=(z_mid, x_mid))
    s.Line(point1=(z_mid, x_mid), point2=(z2, x_mid))
