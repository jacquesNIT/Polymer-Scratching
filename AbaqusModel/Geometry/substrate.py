# Substrate geometry creation, partitioning, and meshing.

from ScratchSimulation.AbaqusModel.abaqus_env import *

# helper 
def _zone_bounds(cfg):
    sub = cfg.substrate
    # Adjustable fractions for mesh refining
    fx = sub.xs2 * 0.6          # fine | C1   (x)
    cx = sub.xs2 * 0.8          # C1   | C2   (x)
    y_fine = sub.ys2 * 0.4      # fine | C1   (y, from surface)
    y_c1   = sub.ys2 * 0.2      # C1   | C2   (y)
    fz = sub.dpo_z               
    cz = sub.dpo_z * 0.50        
    return dict(fx=fx, cx=cx, y_fine=y_fine, y_c1=y_c1,
                zf1=fz, zf2=sub.zs2 - fz,
                zc_lo=fz - cz, zc_hi=(sub.zs2 - fz) + cz)


def create_substrate(model, cfg):
    sub = cfg.substrate
    names = cfg.naming
    zb = _zone_bounds(cfg)
    TOL = 1e-6

    model.ConstrainedSketch(name="__profile__", sheetSize=cfg.sheet_size)
    sk = model.sketches["__profile__"]
    sk.rectangle(point1=(sub.xs1, sub.ys1), point2=(sub.xs2, sub.ys2))
    model.Part(dimensionality=THREE_D, name=names.substrate_name, type=DEFORMABLE_BODY)
    model.parts[names.substrate_name].BaseSolidExtrude(depth=sub.zs2, sketch=sk)
    del sk
    part = model.parts[names.substrate_name]

    def _plane(principal, offset):
        return part.DatumPlaneByPrincipalPlane(principalPlane=principal, offset=offset).id

    def _cut(datum_id, cell_pick):
        part.PartitionCellByDatumPlane(datumPlane=part.datums[datum_id], cells=cell_pick)

    def _bbox(**kw):  
        d = dict(xMin=sub.xs1 - TOL, xMax=sub.xs2 + TOL,
                 yMin=sub.ys1 - TOL, yMax=sub.ys2 + TOL,
                 zMin=sub.zs1 - TOL, zMax=sub.zs2 + TOL)
        d.update(kw)
        return part.cells.getByBoundingBox(**d)

    _cut(_plane(YZPLANE, zb["fx"]), part.cells)
    _cut(_plane(YZPLANE, zb["cx"]), _bbox(xMin=zb["fx"] - TOL))
    _cut(_plane(XZPLANE, zb["y_fine"]), part.cells)
    _cut(_plane(XZPLANE, zb["y_c1"]),  _bbox(yMax=zb["y_fine"] + TOL))
    _cut(_plane(XYPLANE, zb["zf1"]), part.cells)
    _cut(_plane(XYPLANE, zb["zf2"]),   _bbox(zMin=zb["zf1"] - TOL))
    _cut(_plane(XYPLANE, zb["zc_lo"]), _bbox(zMax=zb["zf1"] + TOL))
    _cut(_plane(XYPLANE, zb["zc_hi"]), _bbox(zMin=zb["zf2"] - TOL))

    part.Set(cells=part.cells, name=names.substrate_set)
    fine = part.cells.getByBoundingBox(
        xMin=sub.xs1 - TOL, xMax=zb["fx"] + TOL,
        yMin=zb["y_fine"] - TOL, yMax=sub.ys2 + TOL,
        zMin=zb["zf1"] - TOL, zMax=zb["zf2"] + TOL)
    part.Set(cells=fine, name=names.refined_set)
    return part


def mesh_substrate(part, cfg):
    sub = cfg.substrate
    msh = cfg.mesh
    zb = _zone_bounds(cfg)
    TOL = 1e-6

    for _lbl, _f in (("x", msh.fine_size_x), ("y", msh.fine_size_y), ("z", msh.fine_size_z)):
        if not (_f < msh.coarse_size_1 < msh.coarse_size_2):
            raise ValueError(
                "mesh_substrate: il faut fine_size_%s (%g) < coarse_size_1 (%g) "
                "< coarse_size_2 (%g)." % (_lbl, _f, msh.coarse_size_1, msh.coarse_size_2))

    all_cells = part.cells
    part.setMeshControls(elemShape=HEX, regions=all_cells, technique=STRUCTURED)

    # Element controls  
    if msh.hourglass_control == "ENHANCED":
        hg = ENHANCED
    elif msh.hourglass_control == "RELAX STIFFNESS":
        hg = RELAX_STIFFNESS
    else:
        hg = DEFAULT

    _dc_raw = msh.distortion_control
    _dc = str(_dc_raw).strip().upper()
    if _dc in ("ON", "TRUE", "YES"):
        dc, _dc_label = ON, "ON"
    elif _dc in ("OFF", "FALSE", "NO"):
        dc, _dc_label = OFF, "OFF"
    elif _dc == "DEFAULT":
        dc, _dc_label = DEFAULT, "DEFAULT"
    else:
        raise ValueError("Bad mesh.distortion_control %r. Valid: ON/True, "
                         "OFF/False, DEFAULT." % (_dc_raw,))

    _lr = float(msh.length_ratio)
    if not (0.0 < _lr <= 1.0):
        raise ValueError("Bad mesh.length_ratio %r: need 0 < r <= 1." % (msh.length_ratio,))

    print(">>> Element controls: distortion_control=%r -> %s | length_ratio=%s | hourglass=%s"
          % (_dc_raw, _dc_label, _lr, msh.hourglass_control))

    part.setElementType(
        elemTypes=(
            ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                     secondOrderAccuracy=ON if msh.second_order_accuracy else OFF,
                     distortionControl=dc, lengthRatio=_lr, hourglassControl=hg,
                     elemDeletion=ON if msh.element_deletion else OFF,
                     maxDegradation=msh.max_degradation),
            ElemType(elemCode=C3D6, elemLibrary=EXPLICIT),
            ElemType(elemCode=C3D4, elemLibrary=EXPLICIT),
        ),
        regions=(all_cells,),
    )

    def _edge_axis(e):
        v = e.getVertices()
        a = part.vertices[v[0]].pointOn[0]
        b = part.vertices[v[1]].pointOn[0]
        d = [abs(b[k] - a[k]) for k in (0, 1, 2)]
        return d.index(max(d))

    def _seed(axis, lo, hi, size):
        pts = []
        for e in part.edges:
            if _edge_axis(e) != axis:
                continue
            p = e.pointOn[0]
            if (lo + TOL) < p[axis] < (hi - TOL):   
                pts.append((p,))
        if pts:
            part.seedEdgeBySize(edges=part.edges.findAt(*pts),
                                size=size, deviationFactor=0.1, constraint=FINER)

    # X
    _seed(0, sub.xs1,   zb["fx"],  msh.fine_size_x)
    _seed(0, zb["fx"],  zb["cx"],  msh.coarse_size_1)
    _seed(0, zb["cx"],  sub.xs2,   msh.coarse_size_2)
    # Y 
    _seed(1, zb["y_fine"], sub.ys2,     msh.fine_size_y)
    _seed(1, zb["y_c1"],   zb["y_fine"],msh.coarse_size_1)
    _seed(1, sub.ys1,      zb["y_c1"],  msh.coarse_size_2)
    # Z 
    _seed(2, zb["zf1"],    zb["zf2"],   msh.fine_size_z)
    _seed(2, zb["zc_lo"],  zb["zf1"],   msh.coarse_size_1)
    _seed(2, zb["zf2"],    zb["zc_hi"], msh.coarse_size_1)
    _seed(2, sub.zs1,      zb["zc_lo"], msh.coarse_size_2)
    _seed(2, zb["zc_hi"],  sub.zs2,     msh.coarse_size_2)

    part.generateMesh()