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

    _tsf = float(getattr(solver, "time_scale_factor", 1.0) or 1.0)
    if _tsf != 1.0 and material_params.get("tau_max"):
        material_params["tau_max"] = material_params["tau_max"] / _tsf

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

    # A slave-surface node absent from the displacement subset used to be
    # written out silently as "not displaced" (y = 0), biasing the residual
    # profile with no trace. The fallback stays; it is now counted.
    deformed = []
    n_missing_u = 0
    for label, x, y, z in undeformed_sorted:
        d = displacements.get(label)
        if d is None:
            d = np.array([0.0, 0.0, 0.0])
            n_missing_u += 1
        deformed.append((label, x + d[0], y + d[1], z + d[2]))
    if n_missing_u:
        print("Warning: %d of %d slave-surface nodes have no displacement data and "
              "are written as undeformed (y = 0); the residual profile is biased."
              % (n_missing_u, len(undeformed_sorted)))

    substrate_region = None
    whole_model_region = None     # Needed for Etotal drift calculations
    contact_pair_region = None    # Contact-pair force history (CFN/CFS)
    history_step = None

    # Indenter region detection, two priority levels:
    #   strong -- a 'Node ...' region on the indenter INSTANCE carrying RF keys
    #             (the reference point of the rigid indenter);
    #   weak   -- any region with RF-like keys (legacy fallback). The old
    #             substring test  any("RF" in k)  could latch onto a parasitic
    #             region (e.g. one created by the contact-pair request) and
    #             then silently write zero RF2/IndenterU2 columns.
    indenter_strong = None
    indenter_weak = None
    strong_step = weak_step = None
    indenter_hint = names.indenter_instance.upper()

    for sname in odb.steps.keys():
        for rk in odb.steps[sname].historyRegions.keys():
            hop = list(odb.steps[sname].historyRegions[rk].historyOutputs.keys())
            has_rf = any(_key_matches(k, n) for n in ("RF1", "RF2", "RF3") for k in hop)
            if has_rf:
                rku = rk.upper()
                if indenter_strong is None and rku.startswith("NODE") and indenter_hint in rku:
                    indenter_strong, strong_step = rk, sname
                if indenter_weak is None:
                    indenter_weak, weak_step = rk, sname
            if whole_model_region is None and any(_key_matches(k, "ETOTAL") for k in hop):
                whole_model_region = rk
            if (substrate_region is None
                    and any(_key_matches(k, "ALLIE") for k in hop)
                    and not any(_key_matches(k, "ETOTAL") for k in hop)):
                substrate_region = rk
            if contact_pair_region is None and any(
                    _key_matches(k, n) for n in ("CFN1", "CFN2", "CFN3", "CFNM") for k in hop):
                contact_pair_region = rk

    indenter_region = indenter_strong if indenter_strong is not None else indenter_weak
    history_step = strong_step if indenter_strong is not None else weak_step
    if indenter_strong is None and indenter_weak is not None:
        print("Warning: no 'Node <%s>' history region found; falling back to "
              "region '%s' for the indenter forces." % (indenter_hint, indenter_weak))

    if history_step is None:
        _dump_history_layout(odb)
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




    #  History data — per-region extraction, each with its OWN time axis.
    # The indenter/contact force histories are deactivated in the unload and
    # recovery steps, so their time axes STOP at the end of the scratch; the
    # energy histories stay alive (low frequency) through unload/recovery.
    # The MASTER time axis is therefore the LONGEST axis (in practice the
    # energy one). Previously time_arr came from the indenter region and the
    # energy series were _align-ed (truncated by SAMPLE COUNT) onto it: the
    # unload/recovery energy samples were silently dropped, so the settling
    # check of results_verifier could never see the recovery phase.
    t_ind, force_data = _get_history_multi(odb, indenter_region)
    if t_ind.size == 0:
        odb.close()
        raise ValueError("Indenter history region '%s' contains no data in any step." % indenter_region)

    t_sub, sub_data = _get_history_multi(odb, substrate_region)
    if whole_model_region == substrate_region:
        t_wm, wm_data = t_sub, sub_data
    else:
        t_wm, wm_data = _get_history_multi(odb, whole_model_region)
    if contact_pair_region is not None:
        t_cp, contact_data = _get_history_multi(odb, contact_pair_region)
    else:
        t_cp, contact_data = np.array([]), {}

    time_arr = t_ind
    for _t in (t_sub, t_wm, t_cp):
        if _t.size and _t[-1] > time_arr[-1]:
            time_arr = _t
    n_t = len(time_arr)

    z_ind = np.zeros_like(t_ind)
    rf1_raw = _pick(force_data, "RF1", z_ind)
    rf2_raw = _pick(force_data, "RF2", z_ind)
    rf3_raw = _pick(force_data, "RF3", z_ind)
    u2_raw  = _pick(force_data, "U2",  z_ind)   # indenter penetration trace

    # Self-diagnosis: a zero RF2 AND a zero U2 on a moving indenter is
    # impossible for the true RP region -- dump the ODB history layout into
    # the job log so the mismatch (wrong region / renamed keys) is visible.
    if (float(np.max(np.abs(rf2_raw))) < 1e-20
            and float(np.max(np.abs(u2_raw))) < 1e-20):
        print("Warning: RF2 and IndenterU2 both read as zero from region '%s'. "
              "This region is probably NOT the indenter reference point, or its "
              "output keys are not named 'RF2'/'U2'. History layout dump follows:"
              % indenter_region)
        _dump_history_layout(odb)

    # Resampling onto the master axis: linear interpolation inside each
    # region's coverage, BLANK (NaN -> '') beyond it, because the request was
    # deactivated there and no value must be fabricated for those rows.
    rf1 = _resample(t_ind, rf1_raw, time_arr)
    rf2 = _resample(t_ind, rf2_raw, time_arr)
    rf3 = _resample(t_ind, rf3_raw, time_arr)
    ind_u2 = _resample(t_ind, u2_raw, time_arr)

    #  History data — contact-pair force (CFN/CFS). Used in place of RF2 by results_verifier.py when control_mode == "force"
    z_cp = np.zeros_like(t_cp)
    cfn1 = _resample(t_cp, _pick(contact_data, "CFN1", z_cp), time_arr)
    cfn2 = _resample(t_cp, _pick(contact_data, "CFN2", z_cp), time_arr)
    cfn3 = _resample(t_cp, _pick(contact_data, "CFN3", z_cp), time_arr)
    cfs1 = _resample(t_cp, _pick(contact_data, "CFS1", z_cp), time_arr)
    cfs2 = _resample(t_cp, _pick(contact_data, "CFS2", z_cp), time_arr)
    cfs3 = _resample(t_cp, _pick(contact_data, "CFS3", z_cp), time_arr)
    # CAREA / CFNM / CFSM were already REQUESTED by
    # Modelbuilder._request_contact_pair_history but never extracted. With a
    # Briscoe law  SCOF = alpha + tau0 * A_c / Fn , so the contact area IS the
    # SCOF: without this column the mesh drift of the SCOF can only be inferred.
    cfnm = _resample(t_cp, _pick(contact_data, "CFNM", z_cp), time_arr)
    cfsm = _resample(t_cp, _pick(contact_data, "CFSM", z_cp), time_arr)
    carea = _resample(t_cp, _pick(contact_data, "CAREA", z_cp), time_arr)

    #  History data — substrate energies (deformable body only)
    # (t_sub / sub_data already extracted above; kept alive at low frequency
    # through unload/recovery, hence resampled -- NOT truncated -- onto the
    # master axis.)
    z_sub = np.zeros_like(t_sub)
    ke = _resample(t_sub, _pick(sub_data, "ALLKE", z_sub), time_arr)     # substrate kinetic energy
    ie = _resample(t_sub, _pick(sub_data, "ALLIE", z_sub), time_arr)     # substrate internal energy
    ae = _resample(t_sub, _pick(sub_data, "ALLAE", z_sub), time_arr)     # substrate artificial (hourglass) energy

    #  History data — whole-model energy balance (t_wm / wm_data above)
    def _wm(name, optional=False):
        series = _pick(wm_data, name, None)
        if series is None and name != "ETOTAL" and not optional:
            print("Warning: whole-model term %s absent." % name)
        return _resample(t_wm, series if series is not None else np.zeros_like(t_wm), time_arr)

    # ETOTAL = ALLIE + ALLVD + ALLFD + ALLKE - ALLWK - ALLPW - ALLCW - ALLMW
    wm_ke  = _wm("ALLKE")    # incl. rigid-driver KE (the ~constant baseline)
    wm_ie  = _wm("ALLIE")
    wm_vd  = _wm("ALLVD")    # viscous dissipation
    wm_fd  = _wm("ALLFD")    # frictional dissipation
    wm_cd = _wm("ALLCD", optional=True) # Viscoelastic dissipation
    wm_se = _wm("ALLSE", optional=True)  # Recoverable strain energy
    wm_wk  = _wm("ALLWK")    # external work (energy input)
    wm_pw  = _wm("ALLPW")    # contact penalty work
    wm_cw  = _wm("ALLCW")    # constraint penalty work
    wm_mw  = _wm("ALLMW")    # mass-scaling work
    etotal = _pick(wm_data, "ETOTAL", None)
    if etotal is None:
        etotal = wm_ie + wm_vd + wm_fd + wm_ke - wm_wk - wm_pw - wm_cw - wm_mw
    else:
        etotal = _resample(t_wm, etotal, time_arr)


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
        # --- original indenter header (PYRAMID_INDENTER_PATCH) ---
#         f.write(
#             "# Indenter type: %s with tip radius %smm and cone angle %s degrees\n"
#             % (indenter.indenter_type, indenter.tip_radius, indenter.cone_angle)
#         )
        if getattr(indenter, "indenter_type", "") == "pyramid":
            _pyr = indenter.Pyramid_coords()
            _eqa = indenter.pyramid_equivalent_cone_angle()
            # "tip radius 0.0mm" + the EQUIVALENT cone angle are written on purpose:
            # results_verifier._contact_radius(depth, R=0, alpha) then returns
            # depth*tan(alpha), i.e. the equal-projected-area cone radius.
            f.write(
                "# Indenter type: pyramid (%d faces, %s-forward), face semi-angle "
                "%.6g degrees, tip radius 0.0mm and cone angle %.4f degrees "
                "(equivalent cone)\n"
                % (_pyr["n"], indenter.orientation, float(indenter.face_angle), _eqa)
            )
            f.write("# pyramid_n_faces=%d\n" % _pyr["n"])
            f.write("# pyramid_face_angle=%.6g\n" % float(indenter.face_angle))
            f.write("# pyramid_base_apothem=%.6g\n" % _pyr["a0"])
            f.write("# pyramid_apex_height=%.6g\n" % _pyr["H"])
            f.write("# pyramid_equivalent_cone_angle=%.6g\n" % _eqa)
        else:
            f.write(
                "# Indenter type: %s with tip radius %smm and cone angle %s degrees\n"
                % (indenter.indenter_type, indenter.tip_radius, indenter.cone_angle)
            )
        mat_str = ", ".join(["%s=%s" % (k, v) for k, v in material_params.items()])
        f.write("# Material parameters: %s\n" % mat_str)
        f.write("# family = %s\n" % getattr(cfg.material, "family", "elastomer_mr"))
        f.write("# Simulation Parameters:depth_mode=%s, control_mode=%s, scratch_depth=%.6g, "
                "scratch_force=%.6g, scratch_length=%.6g, scratch_time=%.6g, "
                "indentation_time=%.6g, unload_time=%.6g, "
                "recovery_time=%.6g, mass_scale=%.6g, target_time_increment=%.6g, "
                "time_scale_factor=%.6g, fine_size_x=%.6g\n"
                % (depth_mode, scratch.control_mode, abs(scratch.scratch_depth), scratch.scratch_force,
                scratch.scratch_length, scratch.scratch_time,
                scratch.indentation_time, scratch.unload_time,
                scratch.recovery_time, solver.mass_scale,
                getattr(solver, "target_time_increment", 0.0),
                getattr(solver, "time_scale_factor", 1.0), mesh.fine_size_x)
        )
        f.write("# WallclockTime=%.2f s\n" % wallclock)

        # --- derived quantities -------------------------------------------
        # These govern the result but appear in NO config field, so two CSVs
        # written without them are not comparable. Written one per line as
        # "# key=value" so results_verifier.parse_results_csv picks them up.
        try:
            from ScratchSimulation.AbaqusModel.Verification.analytic import (
                mass_scaling_factor, amplitude_smoothing_window,
                contact_radius_rockwell, elements_per_contact_radius)
            _ms = mass_scaling_factor(cfg)
            f.write("# natural_dt=%.6e\n" % _ms["dt_nat"])
            f.write("# mass_factor_eff=%.6e\n" % _ms["f"])
            f.write("# dm_over_m=%.6e\n" % _ms["dm_over_m"])
            f.write("# dt_effective=%.6e\n" % _ms["dt_eff"])
            _depth = abs(float(scratch.scratch_depth))
            _a = contact_radius_rockwell(_depth, indenter.tip_radius,
                                         indenter.cone_angle)
            f.write("# contact_radius=%.6e\n" % _a)
            f.write("# N_a=%.6e\n" % elements_per_contact_radius(
                _a, min(mesh.fine_size_x, mesh.fine_size_z)))
            f.write("# elements_per_depth=%.6e\n" % (_depth / mesh.fine_size_y))
            _sm = amplitude_smoothing_window(cfg)
            if _sm.get("w") is not None:
                f.write("# smooth_window=%.6e\n" % _sm["w"])
                f.write("# smooth_window_rel=%.6e\n" % _sm["w_rel"])
        except Exception:
            f.write("# derived_quantities=unavailable\n")

        writer.writerow([
            "Time", "RF1", "RF2", "RF3",
            "CFN1", "CFN2", "CFN3", "CFS1", "CFS2", "CFS3",  # contact-pair force (force-driven mode)
            "CFNM", "CFSM", "CAREA",                     # contact magnitudes + contact AREA
            "ALLKE", "ALLIE", "ALLAE",                       # substrate (deformable body)
            "WM_ALLKE", "WM_ALLIE", "WM_ALLVD", "WM_ALLFD",  # whole-model balance terms
            "WM_ALLCD", "WM_ALLSE",                          
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
            cfnm, cfsm, carea,
            ke, ie, ae,
            wm_ke, wm_ie, wm_vd, wm_fd, wm_cd, wm_se,
            wm_wk, wm_pw, wm_cw, wm_mw, etotal,
            ind_u2, node_labels, xu, yu, zu, xd, yd, zd,
            fillvalue="",
        )
        writer.writerows([_cell(v) for v in row] for row in rows)

    print("CSV results written: %s" % output_path)
    odb.close()



#  Helpers
def _key_matches(key, name):
    """
    True when a history-output key IS `name` or contains it as a separate
    token. Handles verbose ODB keys such as 'RF2 at Node INST.1' or
    'Reaction force: RF2 PI: ... Node 1', which the exact dict lookup misses
    while a naive substring test over-matches.
    """
    if key == name:
        return True
    cleaned = key.upper().replace(":", " ").replace(",", " ").replace(";", " ")
    return name.upper() in cleaned.split()


def _pick(data, name, default=None):
    """
    Robust series lookup: exact key first, else the UNIQUE key matching by
    token (see _key_matches). Ambiguous (>1 candidate) returns default.
    """
    if name in data:
        return data[name]
    cands = [k for k in data if _key_matches(k, name)]
    if len(cands) == 1:
        return data[cands[0]]
    return default


def _dump_history_layout(odb):
    """
    Compact diagnostic dump of every history region and its output keys
    (with peak absolute value), printed into the job log. Zero-cost insurance:
    whenever the extraction reads suspicious zeros, the next .log tells the
    whole story without a manual odb_diag run.
    """
    try:
        for sname in odb.steps.keys():
            hrs = odb.steps[sname].historyRegions
            print("  [layout] Step '%s': %d history regions" % (sname, len(hrs.keys())))
            for rk in hrs.keys():
                keys = list(hrs[rk].historyOutputs.keys())
                peaks = []
                for k in keys[:12]:
                    try:
                        # [PATCH:abort-visibility] original :
                        # arr = np.array(hrs[rk].historyOutputs[k].data)
                        # peaks.append("%s(max|v|=%.3g)" % (k, float(np.max(np.abs(arr[:, 1]))) if arr.size else 0.0))
                        # np.array(None).size vaut 1 : l'ancien garde-fou
                        # `if arr.size` ne protegeait donc PAS le cas None.
                        _t, _v = _hist_pairs(hrs[rk].historyOutputs[k])
                        peaks.append("%s(n=%d, max|v|=%.3g)"
                                     % (k, _v.size,
                                        float(np.max(np.abs(_v))) if _v.size else 0.0))
                    except Exception:
                        peaks.append("%s(?)" % k)
                print("  [layout]   region '%s' -> %s%s"
                      % (rk, ", ".join(peaks), " ..." if len(keys) > 12 else ""))
    except Exception as e:
        print("  [layout] dump failed: %s" % e)


def _resample(t_src, v_src, t_dst):
    """
    Linear resampling of a history series onto the master time axis t_dst.

    Inside the source coverage: np.interp on (t_src, v_src). Beyond the last
    source sample: NaN, written as a BLANK CSV cell by _cell -- the output
    request was deactivated there (e.g. forces in unload/recovery), so no
    value is fabricated.

    Replaces _align in post_process: _align padded/truncated by SAMPLE COUNT,
    which silently dropped the low-frequency unload/recovery samples of the
    energy regions whenever the master axis came from the (shorter, force)
    indenter region -- the settling check then never saw the recovery phase.
    """
    t_dst = np.asarray(t_dst, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    v = np.asarray(v_src, dtype=float)
    if t_src.size == 0 or v.size == 0:
        return np.full(t_dst.shape, np.nan)
    m = min(t_src.size, v.size)
    t_src, v = t_src[:m], v[:m]
    out = np.interp(t_dst, t_src, v)
    tol = 1e-12 + 1e-9 * max(abs(float(t_src[-1])), 1.0)
    out[t_dst > t_src[-1] + tol] = np.nan
    return out


def _cell(v):
    """NaN -> '' so deactivated-phase samples become blank CSV cells."""
    if isinstance(v, float) and v != v:
        return ""
    return v


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

# [PATCH:abort-visibility] begin -- lecture robuste d'un historyOutput.
def _hist_pairs(hout, where=""):
    """
    (time, value) arrays of ONE historyOutput, robust to an ODB written by an
    aborted or truncated analysis.

    Abaqus returns ``.data = None`` for a history request that was declared
    but never written -- typically a step that opened and then aborted before
    the first output interval. ``np.array(None)`` is a 0-D object array, so
    the historical expression

        np.array(hout.data).T[0, :]

    raised "too many indices for array: array is 0-dimensional, but 2 were
    indexed". That IndexError short-circuited the ``if t.size == 0`` guard
    placed right after it and masked the real cause: the job aborted.

    Returns two EMPTY arrays instead, and names the offending
    step / region / key in the job log so the .sta no longer has to be read
    by hand.
    """
    raw = getattr(hout, "data", None)
    if raw is None:
        if where:
            print("Warning: history output '%s' carries no data (data is None: "
                  "the step opened but the analysis aborted before the first "
                  "output interval). Skipped." % where)
        return np.array([]), np.array([])
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        if where:
            print("Warning: history output '%s' has shape %s, expected (n, 2). "
                  "Skipped." % (where, arr.shape))
        return np.array([]), np.array([])
    return arr[:, 0], arr[:, 1]
# [PATCH:abort-visibility] end


def _get_history(odb, step_name, region_name):
    """Extract time + history-output dict from a given history region.
    (Legacy single-step helper, kept for compatibility; post_process now
    uses _get_history_multi.)"""
    hr = odb.steps[step_name].historyRegions[region_name]
    keys = list(hr.historyOutputs.keys())
    # Time is stored as the first column of every output — use the first key
    # [PATCH:abort-visibility] original :
    # time_arr = np.array(hr.historyOutputs[keys[0]].data).T[0, :]
    time_arr, _ = _hist_pairs(hr.historyOutputs[keys[0]],
                              "%s / %s / %s" % (step_name, region_name, keys[0]))
    data = {}
    for key in keys:
        # [PATCH:abort-visibility] original :
        # out = np.array(hr.historyOutputs[key].data).T
        # data[key] = out[1, :]
        _t, _v = _hist_pairs(hr.historyOutputs[key],
                             "%s / %s / %s" % (step_name, region_name, key))
        data[key] = _v
    return time_arr, data


def _get_history_multi(odb, region_name):
    """
    Extract time + history-output dict for a region, concatenated across ALL
    ODB steps, with time rebasing when a step stores step-time (axis restarts
    near 0) instead of total time. Duplicated boundary samples are dropped.

    Rationale: single-step extraction reads only the first step holding the
    region -- in constant depth_mode that is IndentationStep, so the scratch
    phase was silently missing from the CSV.
    """
    step_names = list(odb.steps.keys())

    # Union of output keys over the steps that contain the region (a request
    # deactivated in a step simply removes the region from that step).
    all_keys = []
    for sname in step_names:
        hrs = odb.steps[sname].historyRegions
        if region_name in hrs.keys():
            for k in hrs[region_name].historyOutputs.keys():
                if k not in all_keys:
                    all_keys.append(k)
    if not all_keys:
        return np.array([]), {}

    time_parts = []
    data_parts = {}
    t_end = 0.0

    for sname in step_names:
        hrs = odb.steps[sname].historyRegions
        if region_name not in hrs.keys():
            continue
        hr = hrs[region_name]
        keys = list(hr.historyOutputs.keys())
        if not keys:
            continue
        # [PATCH:abort-visibility] original :
        # t = np.array(hr.historyOutputs[keys[0]].data).T[0, :]
        t, _t_vals = _hist_pairs(hr.historyOutputs[keys[0]],
                                 "%s / %s / %s" % (sname, region_name, keys[0]))
        if t.size == 0:
            continue

        # Rebase when the step stores step-time (restarts near 0); keep as-is
        # when the axis already carries total time.
        offset = t_end if (time_parts and t[0] < 0.5 * t_end) else 0.0
        t = t + offset

        # Drop a duplicated boundary sample at the step junction.
        tol = 1e-12 + 1e-9 * max(abs(t_end), 1.0)
        start = 1 if (time_parts and t.size > 1 and abs(t[0] - t_end) <= tol) else 0
        n = t.size - start
        if n <= 0:
            continue

        time_parts.append(t[start:])
        for key in all_keys:
            if key in keys:
                # [PATCH:abort-visibility] original :
                # col = np.array(hr.historyOutputs[key].data).T[1, :][start:]
                # _align recadre en plus sur t.size : une cle dont la region a
                # ete tronquee a un echantillon de moins que la cle de temps
                # produisait un decalage silencieux de colonne.
                _tk, _vk = _hist_pairs(hr.historyOutputs[key],
                                       "%s / %s / %s" % (sname, region_name, key))
                col = _align(_vk, t.size)[start:] if _vk.size else np.zeros(n)
            else:
                col = np.zeros(n)
            data_parts.setdefault(key, []).append(col)

        t_end = t[-1]

    if not time_parts:
        return np.array([]), {}

    time_arr = np.concatenate(time_parts)
    data = dict((k, np.concatenate(v)) for k, v in data_parts.items())
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