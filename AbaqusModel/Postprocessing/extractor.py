# Post-processing: extract forces, energies, and surface profiles from the ODB.
# Produces a single CSV per simulation 

from odbAccess import *
import numpy as np
import os
import re
from itertools import zip_longest
import csv
import time as time_module


def post_process(job_name, file_name, cfg):

    names = cfg.naming
    sub = cfg.substrate
    indenter = cfg.indenter
    scratch = cfg.scratch
    solver = cfg.solver
    mesh = cfg.mesh
    material_params = cfg.material.to_dict()

    odb_path = job_name + ".odb"
    odb = openOdb(path=odb_path, readOnly=True)

    output_folder = "SimDataOutputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, file_name + "_Results.csv")

    #  Surface node coordinates (undeformed)
    all_contact_nodes = odb.rootAssembly.surfaces[
        names.slave_surface.upper()
    ].nodes[0]
    unique_nodes = {node.label: node for node in all_contact_nodes}.values()

    undeformed = [    # Get the coordinates in the local substrate base
        (
            node.label,
            node.coordinates[0],
            node.coordinates[1] - sub.ys2,
            node.coordinates[2] - sub.dpo_z,
        )
        for node in unique_nodes
    ]
    undeformed_sorted = sorted(undeformed, key=lambda c: (c[3], -c[1], c[2]))

    #  Displacement field (last frame where "U" exists, searching steps in reverse)
    disp_field = None
    for step in reversed(list(odb.steps.values())):
        for frame in reversed(list(step.frames)):
            try:
                disp_field = frame.fieldOutputs["U"]
                break
            except KeyError:
                continue
        if disp_field is not None:
            break

    if disp_field is None:
        odb.close()
        raise ValueError(
            "No frames containing displacement data ('U') found in any ODB step. "
        )

    disp_subset = disp_field.getSubset(
        region=odb.rootAssembly.nodeSets[names.contact_region_nodes.upper()]
    )
    displacements = {
        v.nodeLabel: np.array(v.data) for v in disp_subset.values
    }

    deformed = []
    for label, x, y, z in undeformed_sorted:
        d = displacements.get(label, np.array([0.0, 0.0, 0.0]))
        deformed.append((label, x + d[0], y + d[1], z + d[2]))

    indenter_region = None
    substrate_region = None
    whole_model_region = None     # Needed for Etotal drift calculations
    contact_pair_region = None    # Contact-pair force history (CFN/CFS)
    history_step = None

    for sname in odb.steps.keys():
        for rk in odb.steps[sname].historyRegions.keys():
            hop = list(odb.steps[sname].historyRegions[rk].historyOutputs.keys())
            if indenter_region is None and any("RF" in k for k in hop):
                indenter_region = rk
                history_step = sname
            if whole_model_region is None and "ETOTAL" in hop:
                whole_model_region = rk
            if (substrate_region is None and "ALLIE" in hop and "ETOTAL" not in hop):
                substrate_region = rk
            if contact_pair_region is None and any(k.startswith("CFN") for k in hop):
                contact_pair_region = rk
        if (indenter_region is not None and substrate_region is not None
                and whole_model_region is not None):
            break

    if history_step is None:
        odb.close()
        raise ValueError("No history output with RF data found in any step.")

    # Fallbacks for older single-scope models.
    if substrate_region is None:
        substrate_region = whole_model_region
        print("Warning: no substrate-only energy region found, substrate energies fall back to the whole-model values ")
    if whole_model_region is None:
        whole_model_region = substrate_region
        print("Warning: no ETOTAL/whole-model energy region found, the balance will be reconstructed from the available components.")
    if contact_pair_region is None:
        print("Warning: no contact-pair force history region found (CFN*). CFN1-3/CFS1-3 columns will be written as zero. This is expected ")




    #  History data — forces (indenter region)
    time_arr, force_data = _get_history(odb, history_step, indenter_region)
    rf1 = force_data.get("RF1", np.zeros_like(time_arr))
    rf2 = force_data.get("RF2", np.zeros_like(time_arr))
    rf3 = force_data.get("RF3", np.zeros_like(time_arr))

    ind_u2 = force_data.get("U2", np.zeros_like(time_arr))   # indenter penetration trace

    #  History data — contact-pair force (CFN/CFS). Used in place of RF2 by results_verifier.py when control_mode == "force"
    if contact_pair_region is not None:
        _, contact_data = _get_history(odb, history_step, contact_pair_region)
        cfn1 = _align(contact_data.get("CFN1", np.zeros_like(time_arr)), len(time_arr))
        cfn2 = _align(contact_data.get("CFN2", np.zeros_like(time_arr)), len(time_arr))
        cfn3 = _align(contact_data.get("CFN3", np.zeros_like(time_arr)), len(time_arr))
        cfs1 = _align(contact_data.get("CFS1", np.zeros_like(time_arr)), len(time_arr))
        cfs2 = _align(contact_data.get("CFS2", np.zeros_like(time_arr)), len(time_arr))
        cfs3 = _align(contact_data.get("CFS3", np.zeros_like(time_arr)), len(time_arr))
    else:
        cfn1 = cfn2 = cfn3 = cfs1 = cfs2 = cfs3 = np.zeros_like(time_arr)

    #  History data — substrate energies (deformable body only)
    _, sub_data = _get_history(odb, history_step, substrate_region)
    z = np.zeros_like(time_arr)
    ke = sub_data.get("ALLKE", z.copy())     # substrate kinetic energy
    ie = sub_data.get("ALLIE", z.copy())     # substrate internal energy
    ae = sub_data.get("ALLAE", z.copy())     # substrate artificial (hourglass) energy

    #  History data — whole-model energy balance
    _, wm_data = _get_history(odb, history_step, whole_model_region)

    def _wm(name):
        if name not in wm_data and name != "ETOTAL":
            print("Warning: whole-model term %s absent." % name)
        return wm_data.get(name, z.copy())

    # ETOTAL = ALLIE + ALLVD + ALLFD + ALLKE - ALLWK - ALLPW - ALLCW - ALLMW
    wm_ke  = _wm("ALLKE")    # incl. rigid-driver KE (the ~constant baseline)
    wm_ie  = _wm("ALLIE")
    wm_vd  = _wm("ALLVD")    # viscous dissipation
    wm_fd  = _wm("ALLFD")    # frictional dissipation
    wm_wk  = _wm("ALLWK")    # external work (energy input)
    wm_pw  = _wm("ALLPW")    # contact penalty work
    wm_cw  = _wm("ALLCW")    # constraint penalty work
    wm_mw  = _wm("ALLMW")    # mass-scaling work
    etotal = wm_data.get("ETOTAL", None)
    if etotal is None:
        etotal = wm_ie + wm_vd + wm_fd + wm_ke - wm_wk - wm_pw - wm_cw - wm_mw


    #  Wallclock time from .sta file
    wallclock = _extract_wallclock(job_name)

    depth_mode = "constant" if scratch.depth_mode == scratch.CONSTANT else "progressive"

    #  Write CSV
    with open(output_path, "w") as f:
        writer = csv.writer(f)

        # Metadata header
        f.write("# Simulated using Abaqus — Aarhus University\n")
        f.write("# Made by Peter Thorhauge Moellmann(ft. Jacques Nithart)\n")
        ts = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime())
        f.write("# Simulation date and time: %s\n" % ts)
        f.write("# ----------------------------\n")
        f.write(
            "# Indenter type: %s with tip radius %smm and cone angle %s degrees\n"
            % (indenter.indenter_type, indenter.tip_radius, indenter.cone_angle)
        )
        mat_str = ", ".join(["%s=%s" % (k, v) for k, v in material_params.items()])
        f.write("# Material parameters: %s\n" % mat_str)
        f.write("# family = %s\n" % getattr(cfg.material, "family", "elastomer_mr"))
        f.write("# Simulation Parameters:depth_mode=%s, control_mode=%s, scratch_depth=%.6g, "
                "scratch_force=%.6g, scratch_time=%.6g, "
                "recovery_time=%.6g, mass_scale=%.6g, fine_size_x=%.6g\n"
                % (depth_mode, scratch.control_mode, abs(scratch.scratch_depth), scratch.scratch_force,
                scratch.scratch_time, scratch.recovery_time, solver.mass_scale, mesh.fine_size_x)
        )
        f.write("# WallclockTime=%.2f s\n" % wallclock)

        writer.writerow([
            "Time", "RF1", "RF2", "RF3",
            "CFN1", "CFN2", "CFN3", "CFS1", "CFS2", "CFS3",  # contact-pair force (force-driven mode)
            "ALLKE", "ALLIE", "ALLAE",                       # substrate (deformable body)
            "WM_ALLKE", "WM_ALLIE", "WM_ALLVD", "WM_ALLFD",  # whole-model balance terms
            "WM_ALLWK", "WM_ALLPW", "WM_ALLCW", "WM_ALLMW", "ETOTAL",
            "IndenterU2", "NodeLabel",
            "x_undeformed", "y_undeformed", "z_undeformed",
            "x_deformed", "y_deformed", "z_deformed",
        ])

        node_labels, xu, yu, zu, xd, yd, zd = [], [], [], [], [], [], []
        for (lbl, x0, y0, z0), (_, x1, y1, z1) in zip(undeformed_sorted, deformed):
            node_labels.append(lbl)
            xu.append(x0); yu.append(y0); zu.append(z0)
            xd.append(x1); yd.append(y1); zd.append(z1)

        rows = zip_longest(
            time_arr.reshape(-1), rf1, rf2, rf3,
            cfn1, cfn2, cfn3, cfs1, cfs2, cfs3,
            ke, ie, ae,
            wm_ke, wm_ie, wm_vd, wm_fd, wm_wk, wm_pw, wm_cw, wm_mw, etotal,
            ind_u2, node_labels, xu, yu, zu, xd, yd, zd,
            fillvalue="",
        )
        writer.writerows(rows)

    print("CSV results written: %s" % output_path)
    odb.close()



#  Helpers
def _align(arr, n):
    """
    Pad (with the last value) or truncate so arr has exactly n samples.
    Guards against a (rare) frame-count mismatch between the indenter and
    contact-pair history regions, even though both share the same
    timeInterval/step -- avoids a silent row misalignment in the CSV.
    """

    arr = np.asarray(arr, dtype=float)
    if arr.size == n:
        return arr
    if arr.size > n:
        return arr[:n]
    pad_value = arr[-1] if arr.size > 0 else 0.0
    return np.concatenate([arr, np.full(n - arr.size, pad_value)])

def _get_history(odb, step_name, region_name):
    """Extract time + history-output dict from a given history region."""
    hr = odb.steps[step_name].historyRegions[region_name]
    keys = list(hr.historyOutputs.keys())
    # Time is stored as the first column of every output — use the first key
    time_arr = np.array(hr.historyOutputs[keys[0]].data).T[0, :]
    data = {}
    for key in keys:
        out = np.array(hr.historyOutputs[key].data).T
        data[key] = out[1, :]
    return time_arr, data


def _extract_wallclock(job_name):
    """Parse wallclock time from the .sta status file."""
    sta = job_name + ".sta"
    if not os.path.exists(sta) or os.path.getsize(sta) == 0:
        print("Warning: %s missing or empty. Using wallclock=0.0." % sta)
        return 0.0

    with open(sta, "r") as f:
        content = f.read()

    match = re.search(r"WALLCLOCK\s*TIME\s*(?:.*\s*)?=\s*([\d\.]+)", content, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Fallback: last number on last non-empty line
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if lines:
        fb = re.search(r"([\d\.]+)\s*$", lines[-1])
        if fb:
            return float(fb.group(1))

    print("Warning: Could not parse wallclock time from %s. Using 0.0." % sta)
    return 0.0
