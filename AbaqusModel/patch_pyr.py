# -*- coding: utf-8 -*-
"""
patch_pyramid_tip_contact.py
============================

Follow-up to patch_pyramid_indenter.py. Fixes the substrate elements that get
"hooked" on the pyramidal indenter, and the general-contact deep-penetration
warning on edges (node set "InfoEdgeDeepPenetFirst").

What is changed and why
-----------------------
1. TRUNCATED TIP (the main fix).
   The sharp apex is a geometric singularity: the contact normal is undefined
   there, so slave nodes slip between the facets that meet at the point, end up
   INSIDE the rigid body, and -- because rigid-element surfaces are double-sided
   in Abaqus/Explicit general contact -- they are then pushed from the inside and
   stay trapped, dragged along by friction. The pyramid is now built as a
   frustum with a small flat tip of apothem `tip_flat` (default 5 um).
   Side effect: the extrusion depth becomes exactly H - h_tip, so the fragile
   "over-extrude and let Abaqus clip at the apex" trick disappears.

2. CLOSED MASTER SURFACE.
   The flat tip face is added to the master surface. Without it the truncated
   indenter has a hole at its lowest point -- the one place every slave node
   passes under.

3. REFERENCE POINT AT THE REAL TIP.
   The RP is moved from the virtual apex to the centre of the flat tip, and the
   assembly translation follows. `scratch_depth` therefore keeps its meaning:
   penetration of the lowest point of the indenter below the free surface.

4. COARSER RIGID FACETS.
   The default seed goes from 0.5x to 1.0x the substrate fine size. The faces of
   a pyramid are PLANAR, so -- unlike a sphere -- facet size introduces no
   geometric error whatsoever. A master surface finer than the slave only
   multiplies contact entities and edge-to-edge pairs, which is what the
   deep-penetration warning is about.

5. RIGID MASS AND ROTARY INERTIA.
   i11 = i22 = i33 = 0.0 is inherited from the analytic Rockwell tip, which has
   no element nodes. A discrete rigid body does, and Abaqus uses the rigid-body
   mass properties to build the contact penalties ("please make sure that these
   edges have sufficient mass"). Inertia now defaults to mass * base_apothem^2.

6. OPTIONAL FEATURE-EDGE CRITERION (opt-in, off by default).
   `feature_edge_criterion` lets you relax or disable edge-to-edge contact on
   the indenter surface. NOTE: the exact signature of surfaceFeatureAssignments
   could not be verified outside Abaqus, so the call is guarded and only prints
   a warning if it fails.

Requires patch_pyramid_indenter.py to have been applied first.

Usage
-----
  python patch_pyramid_tip_contact.py [--root PATH] [--dry-run]
  --root defaults to the directory holding this script (drop it in AbaqusModel/).
"""

import argparse
import ast
import io
import os
import sys

MARK1 = "PYRAMID_INDENTER_PATCH"
MARK = "PYRAMID_TIP_CONTACT_PATCH"

SKIP_DIRS = ("__pycache__", ".git", ".svn", ".idea", ".vscode", "backup", "backups")


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------
def read(path):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def nl_of(text):
    return "\r\n" if "\r\n" in text else "\n"


def comment_out(block, nl):
    return nl.join(ln if ln.strip() == "" else "# " + ln for ln in block.split(nl))


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError("anchor '%s': found %d occurrences (expected 1)" % (label, n))
    return text.replace(old, new, 1)


def locate(root, fname, subdir):
    for cand in (os.path.join(root, subdir, fname), os.path.join(root, fname)):
        if os.path.isfile(cand):
            return cand
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if fname in filenames:
            hits.append(os.path.join(dirpath, fname))
    if not hits:
        raise IOError("%s not found under %s" % (fname, os.path.abspath(root)))
    if len(hits) > 1:
        raise IOError("%s is ambiguous, %d copies: %s" % (fname, len(hits), ", ".join(hits)))
    return hits[0]


# --------------------------------------------------------------------------
#  base.py
# --------------------------------------------------------------------------
def patch_base(src):
    nl = nl_of(src)

    # --- signature -------------------------------------------------------
    a = '                  extrude_depth=None, mesh_size=None, mesh_min_size=None, tip_bias=False ):' + nl
    new = ('                  extrude_depth=None, mesh_size=None, mesh_min_size=None, tip_bias=False,' + nl +
           '                  tip_flat=0.005, rigid_mass=1e-6, rigid_inertia=None,' + nl +
           '                  feature_edge_criterion=None ):    # ' + MARK + nl)
    src = replace_once(src, a, new, "base/pyramid signature")

    # --- attributes ------------------------------------------------------
    a = '        self.tip_bias = tip_bias                    # bias the rigid seeds towards the apex' + nl
    new = (a +
           '        # --- tip / contact conditioning (' + MARK + ') ---' + nl +
           '        self.tip_flat = tip_flat                    # apothem of the flat tip [mm], 0 -> sharp apex' + nl +
           '        self.rigid_mass = rigid_mass                # point mass at the RP [tonne]' + nl +
           '        self.rigid_inertia = rigid_inertia          # None -> rigid_mass * base_apothem**2' + nl +
           '        self.feature_edge_criterion = feature_edge_criterion' + nl +
           '                                                    # None / "NONE" / "PERIMETER" / "ALL" / angle [deg]' + nl)
    src = replace_once(src, a, new, "base/pyramid attributes")

    # --- Pyramid_coords: frustum quantities ------------------------------
    a = ('        a0 = float(self.base_apothem)                 # apothem of the base polygon' + nl +
         '        R0 = a0 / np.cos(np.pi / n)                   # circumradius of the base polygon' + nl +
         '        H = a0 / np.tan(theta)                        # apex height above the base plane' + nl)
    new = (a + nl +
           '        # Truncated tip (' + MARK + '): the pyramid is built as a frustum, the' + nl +
           '        # virtual apex being cut h_tip above the flat. A sharp apex (tip_flat=0)' + nl +
           '        # has no defined contact normal and traps slave nodes inside the body.' + nl +
           '        a_tip = max(0.0, float(self.tip_flat or 0.0))  # apothem of the flat tip' + nl +
           '        if a_tip >= a0:' + nl +
           '            raise ValueError("tip_flat (%g) must be smaller than base_apothem (%g)"' + nl +
           '                             % (a_tip, a0))' + nl +
           '        h_tip = a_tip / np.tan(theta)                 # height of the removed apex cone' + nl +
           '        H_frustum = H - h_tip                         # actual extrusion depth' + nl +
           '        R_tip = a_tip / np.cos(np.pi / n)             # circumradius of the flat tip' + nl)
    src = replace_once(src, a, new, "base/Pyramid_coords frustum")

    a = ('        return dict(n=n, theta=theta, theta_deg=float(self.face_angle), a0=a0, R0=R0,' + nl +
         '                    H=H, phi0=phi0, face_azim=face_azim, vert_azim=vert_azim,' + nl +
         '                    vertices=vertices)' + nl)
    new = ('        return dict(n=n, theta=theta, theta_deg=float(self.face_angle), a0=a0, R0=R0,' + nl +
           '                    H=H, phi0=phi0, face_azim=face_azim, vert_azim=vert_azim,' + nl +
           '                    vertices=vertices,' + nl +
           '                    a_tip=a_tip, h_tip=h_tip, H_frustum=H_frustum, R_tip=R_tip)  # ' + MARK + nl)
    src = replace_once(src, a, new, "base/Pyramid_coords return")

    # --- pyramid_face_points ---------------------------------------------
    a = ('        pc = self.Pyramid_coords()' + nl +
         '        h = float(h_frac) * pc["H"]' + nl +
         '        r = h * np.tan(pc["theta"])                   # axis -> face distance at height h' + nl +
         '        return [(x_apex + r * np.sin(p), y_apex + h, z_apex + r * np.cos(p))' + nl +
         '                for p in pc["face_azim"]]' + nl)
    new = ('        # (' + MARK + ') heights are now measured from the FLAT TIP, not from' + nl +
           '        # the virtual apex: r = a_tip + h tan(theta). Identical to the previous' + nl +
           '        # formula when tip_flat = 0.' + nl +
           '        pc = self.Pyramid_coords()' + nl +
           '        h = float(h_frac) * pc["H_frustum"]' + nl +
           '        r = pc["a_tip"] + h * np.tan(pc["theta"])    # axis -> face distance at height h' + nl +
           '        return [(x_apex + r * np.sin(p), y_apex + h, z_apex + r * np.cos(p))' + nl +
           '                for p in pc["face_azim"]]' + nl +
           nl +
           '    def pyramid_tip_face_point(self, y_tip, z_tip, x_tip=0.0):' + nl +
           '        # (' + MARK + ') probe point on the FLAT TIP face, in GLOBAL coordinates.' + nl +
           '        # Returns [] for a sharp apex (there is no tip face to select).' + nl +
           '        pc = self.Pyramid_coords()' + nl +
           '        if pc["a_tip"] <= 0.0:' + nl +
           '            return []' + nl +
           '        return [(x_tip, y_tip, z_tip)]' + nl)
    src = replace_once(src, a, new, "base/pyramid_face_points")

    # --- pyramid_edge_points ---------------------------------------------
    a = ('        return [((1.0 - s) * vx, (1.0 - s) * vy, s * pc["H"])' + nl +
         '                for (vx, vy) in pc["vertices"]]' + nl)
    new = ('        # (' + MARK + ') the lateral edges now run from the base vertex to the' + nl +
           '        # TOP (frustum) vertex, so the radial scaling is no longer (1 - s).' + nl +
           '        f = 1.0 - s * (1.0 - pc["R_tip"] / pc["R0"])' + nl +
           '        return [(f * vx, f * vy, s * pc["H_frustum"])' + nl +
           '                for (vx, vy) in pc["vertices"]]' + nl)
    src = replace_once(src, a, new, "base/pyramid_edge_points")

    return src


# --------------------------------------------------------------------------
#  indenter.py
# --------------------------------------------------------------------------
def patch_indenter(src):
    nl = nl_of(src)

    # --- sanity check on depth: use the frustum height --------------------
    a = ('    pc = ind.Pyramid_coords()' + nl +
         '    H = pc["H"]' + nl)
    new = ('    pc = ind.Pyramid_coords()' + nl +
           '    H = pc["H"]                    # virtual apex height' + nl +
           '    Hf = pc["H_frustum"]           # actual height of the truncated pyramid (' + MARK + ')' + nl)
    src = replace_once(src, a, new, "ind/coords")

    a = ('    if depth > 0.0 and H <= 1.5 * depth:' + nl)
    new = ('    if depth > 0.0 and Hf <= 1.5 * depth:' + nl)
    src = replace_once(src, a, new, "ind/depth check")

    # --- extrusion: exact frustum depth, no over-extrusion ---------------
    a = ('    # Solid extrusion with a NEGATIVE draft angle: the section shrinks along +z' + nl +
         '    # and closes on a sharp apex at z = H = a0 / tan(theta).' + nl +
         '    if ind.extrude_depth:' + nl +
         '        depths = [float(ind.extrude_depth)]' + nl +
         '    else:' + nl +
         '        # over-extrude first (Abaqus clips at the apex), then fall back' + nl +
         '        depths = [max(1.0, 2.0 * H), 1.05 * H, H]' + nl +
         nl +
         '    part = None' + nl +
         '    last_err = None' + nl +
         '    for d in depths:' + nl)
    new = ('    # Solid extrusion with a NEGATIVE draft angle: the section shrinks along +z.' + nl +
           '    # (' + MARK + ') the depth is now exactly H_frustum, which leaves a flat tip of' + nl +
           '    # apothem a_tip instead of a singular apex -- and removes the previous' + nl +
           '    # over-extrusion, whose clipping behaviour at the apex was not guaranteed.' + nl +
           '    if ind.extrude_depth:' + nl +
           '        depths = [float(ind.extrude_depth)]' + nl +
           '    elif pc["a_tip"] > 0.0:' + nl +
           '        depths = [Hf]' + nl +
           '    else:' + nl +
           '        # sharp apex requested: over-extrude and let Abaqus clip, then fall back' + nl +
           '        depths = [max(1.0, 2.0 * H), 1.05 * H, H]' + nl +
           nl +
           '    part = None' + nl +
           '    last_err = None' + nl +
           '    for d in depths:' + nl)
    src = replace_once(src, a, new, "ind/extrusion depth")

    # --- RP + inertia -----------------------------------------------------
    a = ('    # Reference point at the APEX (same convention as the Rockwell tip)' + nl +
         '    rp = part.ReferencePoint(point=(0.0, 0.0, H))' + nl +
         '    part.Set(name=names.indenter_set,' + nl +
         '             referencePoints=(part.referencePoints[rp.id],))' + nl +
         nl +
         '    part.engineeringFeatures.PointMassInertia(' + nl +
         '        alpha=0.0, composite=0.0,' + nl +
         '        i11=0.0, i22=0.0, i33=0.0, mass=1e-6,   # same placeholder as the Rockwell tip' + nl +
         '        name=names.inertia_name,' + nl +
         '        region=part.sets[names.indenter_set],' + nl +
         '    )' + nl)
    new = ('    # Reference point at the LOWEST POINT of the indenter (' + MARK + ').' + nl +
           '    # For a truncated tip this is the centre of the flat, NOT the virtual apex,' + nl +
           '    # so that the prescribed scratch_depth stays the true penetration depth.' + nl +
           '    rp = part.ReferencePoint(point=(0.0, 0.0, Hf))' + nl +
           '    part.Set(name=names.indenter_set,' + nl +
           '             referencePoints=(part.referencePoints[rp.id],))' + nl +
           nl +
           '    # A DISCRETE rigid body has element nodes, unlike the analytic Rockwell tip:' + nl +
           '    # Abaqus builds the general-contact penalties from its mass properties, and' + nl +
           '    # a null rotary inertia is what triggers the "sufficient mass" warning on the' + nl +
           '    # feature edges. Kinematics are unaffected (displacement-controlled RP).' + nl +
           '    _m = float(ind.rigid_mass)' + nl +
           '    _i = ind.rigid_inertia' + nl +
           '    if _i is None:' + nl +
           '        _i = _m * float(ind.base_apothem) ** 2' + nl +
           '    _i = float(_i)' + nl +
           '    part.engineeringFeatures.PointMassInertia(' + nl +
           '        alpha=0.0, composite=0.0,' + nl +
           '        i11=_i, i22=_i, i33=_i, mass=_m,' + nl +
           '        name=names.inertia_name,' + nl +
           '        region=part.sets[names.indenter_set],' + nl +
           '    )' + nl)
    src = replace_once(src, a, new, "ind/RP and inertia")

    # --- seed size --------------------------------------------------------
    a = ('    # Mesh: R3D4 / R3D3 rigid elements, no contribution to the stable increment.' + nl +
         '    # Default: half the substrate fine size, so the master surface is finer' + nl +
         '    # than the slave and the faceting error stays below the contact tolerance.' + nl +
         '    ms = ind.mesh_size' + nl +
         '    if not ms:' + nl +
         '        ms = 0.5 * min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_z)' + nl)
    new = ('    # Mesh: R3D4 / R3D3 rigid elements, no contribution to the stable increment.' + nl +
           '    # (' + MARK + ') default is now 1.0x the substrate fine size, not 0.5x: the faces' + nl +
           '    # of a pyramid are PLANAR, so facet size introduces no geometric error at all' + nl +
           '    # (unlike a sphere). A master surface finer than the slave only multiplies the' + nl +
           '    # contact entities and the edge-to-edge pairs that trigger the deep-penetration' + nl +
           '    # warning.' + nl +
           '    ms = ind.mesh_size' + nl +
           '    if not ms:' + nl +
           '        ms = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_z)' + nl)
    src = replace_once(src, a, new, "ind/seed size")

    # --- placement --------------------------------------------------------
    a = ('        H = ind.Pyramid_coords()["H"]' + nl +
         '        asm.rotate(instanceList=(names.indenter_instance,),' + nl +
         '                   axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0),' + nl +
         '                   angle=90.0)' + nl +
         '        asm.translate(instanceList=(names.indenter_instance,),' + nl +
         '                      vector=(0.0, sub.ys2 + H, sub.zs1 + sub.dpo_z))' + nl)
    new = ('        # (' + MARK + ') H_frustum, so the FLAT TIP -- the lowest point of the' + nl +
           '        # indenter -- lands on the substrate top surface.' + nl +
           '        Hf = ind.Pyramid_coords()["H_frustum"]' + nl +
           '        asm.rotate(instanceList=(names.indenter_instance,),' + nl +
           '                   axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0),' + nl +
           '                   angle=90.0)' + nl +
           '        asm.translate(instanceList=(names.indenter_instance,),' + nl +
           '                      vector=(0.0, sub.ys2 + Hf, sub.zs1 + sub.dpo_z))' + nl)
    src = replace_once(src, a, new, "ind/placement")

    return src


# --------------------------------------------------------------------------
#  Modelbuilder.py
# --------------------------------------------------------------------------
def patch_modelbuilder(src):
    nl = nl_of(src)

    # --- master surface: add the flat tip face ---------------------------
    a = ('        _pyr_pts = cfg.indenter.pyramid_face_points(sub.ys2, sub.zs1 + sub.dpo_z)' + nl +
         '        asm.Surface(' + nl +
         '            name=names.master_surface,' + nl +
         '            side1Faces=ind_inst.faces.findAt(*tuple([(p,) for p in _pyr_pts])),' + nl +
         '        )' + nl)
    new = ('        # (' + MARK + ') the FLAT TIP face must be part of the master surface:' + nl +
           '        # without it the truncated indenter has a hole at its lowest point,' + nl +
           '        # exactly where every slave node passes underneath.' + nl +
           '        _pyr_pts = cfg.indenter.pyramid_face_points(sub.ys2, sub.zs1 + sub.dpo_z)' + nl +
           '        _pyr_pts = list(_pyr_pts) + list(' + nl +
           '            cfg.indenter.pyramid_tip_face_point(sub.ys2, sub.zs1 + sub.dpo_z))' + nl +
           '        asm.Surface(' + nl +
           '            name=names.master_surface,' + nl +
           '            side1Faces=ind_inst.faces.findAt(*tuple([(p,) for p in _pyr_pts])),' + nl +
           '        )' + nl)
    src = replace_once(src, a, new, "mb/master surface tip face")

    # --- optional feature-edge criterion ---------------------------------
    a = ('    model.interactions[names.contact_interaction].contactPropertyAssignments.appendInStep(' + nl +
         '        assignments=((GLOBAL, SELF, names.contact_property),),' + nl +
         '        stepName="Initial",' + nl +
         '    )' + nl)
    new = (a + nl +
           '    # (' + MARK + ') optional control of the feature edges used by edge-to-edge' + nl +
           '    # general contact. Opt-in via Indenter_Config.feature_edge_criterion; the' + nl +
           '    # call is guarded because its signature could not be verified offline.' + nl +
           '    _fec = getattr(cfg.indenter, "feature_edge_criterion", None)' + nl +
           '    if _fec is not None:' + nl +
           '        _crit = {"NONE": NONE, "PERIMETER": PERIMETER, "ALL": ALL}.get(' + nl +
           '            str(_fec).upper(), _fec)' + nl +
           '        try:' + nl +
           '            model.interactions[names.contact_interaction]' \
           '.surfaceFeatureAssignments.appendInStep(' + nl +
           '                assignments=((asm.surfaces[names.master_surface], _crit),),' + nl +
           '                stepName="Initial",' + nl +
           '            )' + nl +
           '        except Exception as _err:' + nl +
           '            print("WARNING: feature_edge_criterion ignored (%s)" % _err)' + nl)
    src = replace_once(src, a, new, "mb/feature edges")

    return src


# --------------------------------------------------------------------------
#  driver
# --------------------------------------------------------------------------
JOBS = [
    ("base.py", "Configuration", patch_base),
    ("indenter.py", "Geometry", patch_indenter),
    ("Modelbuilder.py", "Simulation", patch_modelbuilder),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=here)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print("root: %s\n" % root)

    rc = 0
    for fname, subdir, fn in JOBS:
        try:
            path = locate(root, fname, subdir)
        except IOError as err:
            print("[MISSING] %s" % err)
            rc = 1
            continue

        rel = os.path.relpath(path, root)
        src = read(path)

        if MARK1 not in src:
            print("[FAIL]    %s -- apply patch_pyramid_indenter.py first" % rel)
            rc = 1
            continue
        if MARK in src:
            print("[already] %s -- patch already applied" % rel)
            continue

        try:
            out = fn(src)
            ast.parse(out)
        except Exception as err:
            print("[FAIL]    %s -- %s" % (rel, err))
            rc = 1
            continue

        if args.dry_run:
            print("[dry-run] %s -- would grow by %d chars" % (rel, len(out) - len(src)))
        else:
            write(path, out)
            print("[ok]      %s -- patched (+%d chars)" % (rel, len(out) - len(src)))

    return rc


if __name__ == "__main__":
    sys.exit(main())