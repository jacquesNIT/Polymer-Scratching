This file contains explanations for each file of the project.

ScratchSimulation

    AbaqusModel

        Configuration

            base.py
            It contains every class useful for the simulation. Parameters for scratch, indenter, substrate, mesh... can be changed here.
            The "Polymer_default" function contains the base parameters of the simulation (it overrides the ones defined in the classes).
            Pretty much every aspect of the simulation can be changed here.

            families.py
            It contains functions defining the various polymer families (PC, PMMA, semi-crystallines...).
            They all start from "Polymer_default"; the differences in material cards and models used are set here.

            sampling.py
            Screening / sweep design layer for the Drucker-Prager families.
            Defines the Factor class (bounds, lin/log scale, physical mapping), the campaign
            containers (SAMPLING_DP_UNIFIED and the legacy split campaigns) and the design generators morris_design() and sobol_design().

        Geometry

            substrate.py
            Builds the substrate (half-symmetry block).
            Partitions it into the wanted zones (fine / C1 / C2).
            Creates the sets and surfaces used later (substrate set, top contact surface, symmetry face).

            indenter.py
            Generates the rigid indenter geometry and dispatches on cfg.indenter.indenter_type.
            create_rockwell() sketches the sphere-cone profile (create_pyramid() does the same for a pyramid).
            place_indenter() moves the instance so the tip touches the surface at z = dpo_z, with a placement adapted to the indenter type.

        Material

            assignment.py
            Class SubstrateMaterialAssignment: turns Material_Config into a real Abaqus material and assigns it to the part.
            Density is written first, then four independent blocks are dispatched according to the model (hyperelastic / viscoelastic / plasticity / damage).
            _validate_material() forbids invalid combinations (hyperelastic + plasticity for example).
            update_friction() writes the contact property (Coulomb or Briscoe pressure-dependent).

        Postprocessing

            extractor.py
            Runs inside the Abaqus kernel right after the job. Opens the .odb and writes one CSV per simulation containing:
            a metadata header, the history time series (RF2/RF3, CFN/CSHEAR, all energies) and the deformed surface node coordinates.
            Helper functions handle history-region name matching, resampling on a common time base and wallclock extraction from the .sta file.

        Simulation

            Modelbuilder.py
            build_scratch_model(cfg) orchestrates the whole model: geometry (substrate + mesh + indenter), assembly and indenter placement,
            explicit steps with their mass scaling, boundary conditions, displacement-controlled loading through amplitudes, the contact pair and its friction
            property, ALE adaptive meshing, and finally the field/history output requests.

        utils

            job.py
            run_job_and_wait(): creates the job with the solver settings (precision, MPI, domain parallelisation, num_cpus from Solver_Config).
            It deletes any stale job or .lck left by a previous run, submits and waits.
            _check_job_status() raises an error if Abaqus returned anything other than "completed".

            cleanup.py
            cleanup_abaqus_junk(): removes the Abaqus scratch files (.abq, .com, .mdl, .pac, .prt, .sel, .stt, .rpy...) after a run.

        abaqus_env.py
        Centralizes the Abaqus imports (abaqus, abaqusConstants, part, material, step, interaction, mesh, odbAccess...),
        so the other modules only import from one place.

    designs

        Design CSVs read by the "design" study of run_parameter_study.py. Each one starts with a commented header
        (campaign, method, factors and their bounds, frozen parameters) followed by one row per run.

        glassy_pc_morris.csv
        Morris screening design for PC (90 runs, 8 factors: X, h, w, eps_c, beta, K, mu_eff, phi), written by generate_design.py.

        glassy_pmma_calib_sobol.csv
        First PMMA XT calibration design (38 runs, 4 factors: psi, soft_drop, tau_split, sigma_scale, at 20 um depth).
        Despite its name it is a factorial design, not a Sobol sequence. Kept because the first calibration sweep was run with it.

        glassy_pmma_calib.csv
        Second PMMA XT calibration design (31 runs). Built from what the first sweep showed: mu_eff becomes a factor,
        psi and sigma_scale windows are moved up, and soft_drop is held and only probed on two runs.

        glassy_pc_calib.csv
        PC calibration design (32 runs, 25 um depth). Same structure as the PMMA one, but with its own
        reference pressure (230 MPa) and windows centred on the PC baseline of families.py.

    Generators

        generate_design.py
        Generator of the screening / sweep designs.
        Reads the campaign of a family from sampling.py, draws a Morris trajectory design or a Sobol sequence and writes the design CSV.

        generate_pmma_calib.py
        Writes designs/glassy_pmma_calib.csv. It is a hand-built factorial split in blocks: BASE (best point of the first sweep,
        re-run so both sweeps share a common point), CORE (psi x sigma_scale x mu_eff full factorial), SOFT and SPLIT (a few runs
        checking that holding soft_drop and phi does not bias the rest). --blocks allows to generate only part of it.

        generate_pc_calib.py
        Same thing for PC, writes designs/glassy_pc_calib.csv. The header of the file explains why the PC design is not
        simply the PMMA one with other numbers (reference pressure, sigma_scale centring, lower psi, deeper scratch).

    Lab results

        Additionnal_Batch/     Extra laboratory tests ran before the main campaign
        Main_Batch/            Main experimental campaign, sorted by material, load and load rate
                               Contains .bcrf topographies CSVs, with the repeats of each condition.
        Polymer_references/    Photos od the polymer panels used to fix the material cards.

    Other

        old_graphs/            Old figures kept for reference, no longer produced by the current scripts.
        Studies/               Contains some of the studies and researches used during the project

    runs

        Isolated working directories created per job (runs/<study>_<family>_<tag>),
        holding the .odb/.sta of each run and the SimDataOutputs/ CSVs.

        Calib/                 Calibration sweeps (PMMA and PC), run from the calib designs.
        Dpeth_sweep/           Depth sweep, used to separate soft_drop from sigma_scale.
        Morris6/, Morris7/     Morris screening campaigns (6th and 7th iterations).
        Old/                   Earlier runs kept for reference.

    ScratchFeatures

        Shared constants and helpers of the topography pipeline, taken from the original ScratchFeatures code.
        results_values.py copies the parts it needs from here.

        constants.py
        Grid shape, domain size, scratch length and prescribed depth, plus the smoothing / denoising settings
        of the experimental topography pipeline.

        scratch_simulation_helpers.py
        Loads a simulation CSV (parameters, time, forces, energies, node coordinates), projects the node cloud
        onto a regular grid and maps the forces onto the scratch abscissa z.

        scratch_feature_extraction_helpers.py
        Extracts the tribological features from a cross-section of the gridded topography: residual depth,
        pile-up height and width, scratch width, groove and pile-up areas, frontal pile-up, volumes.
        Can also save a diagnostic figure for each feature.

    Screening Results

        Collected sweep tables, Morris analyses and the generated reports.

        batch_analysis/, batch_analysis2/, batch_analysis6/, batch_analysis6_1/, batch_analysis7/
        Outputs of results_values_batch.py for the successive campaigns: the tidy CSV and the exploratory figures.

        screening_dp/, screening_dp1/ ... screening_dp7/ 
        Outputs of dp_screening.py for each Drucker-Prager screening: SCREENING_REPORT.md, its figures and the Morris tables.

    verifiers

        bcrf_values.py
        Reads one .bcrf scan of a scratch and measures along the track: residual depth h_r, lateral pile-up h_p on both sides,
        groove width w0 and the penetration estimated from it, the terminal frontal mound and the groove / pile-up area balance
        on a few transverse sections (one of them at the deepest point).
        The scan is cleaned first (spike removal, form removal on the reference bands). --material picks the recovery law
        used for the penetration (pmma or pc), --rough is a preset for wavy surfaces. Writes a figure and, with --export, two CSVs.
        Usage: "python bcrf_values.py Test3_PMMAXT_10N.bcrf".

        bcrf_comparator.py
        Runs bcrf_values on every .bcrf of a folder and overlays the results: h_r / h_p along the track for all files,
        and the deepest transverse section of each file. Useful to check the repeatability of a test condition.

        csv_values.py
        Reads a CSV force export and plots along the track the normal and tangential forces, the SCOF,
        the in-situ depth (with detection of the capacitive sensor saturation) and SCOF / depth versus load.

        results_values.py
        Reference post-processing of one simulation CSV: the single source of truth for every QoI.
        Parses the file, rebuilds the time masks (active / loading window), extracts the normal and
        tangential forces at the end of the loading ramp, the SCOF averaged on the [10 %, 90 %] band,
        the residual depth h_r, the lateral pile-up h_p and the frontal pile-up h_fp at the deepest section,
        plus the energy diagnostics (KE/IE, AE/IE, ETOTAL drift, ALLPW, settling).
        Usage: "python results_values.py <folder_or_csv> [<out_csv>] [<z_mm>]".

        results_values_batch.py
        Same as results_values applied to a whole folder of CSVs.
        Walks a campaign folder recursively for *_Results.csv, calls extract_values() on each run in parallel (--jobs),
        catches per-run failures, flags suspicious runs, joins the design levels on the run id (--design) and writes
        one tidy CSV plus exploratory figures in --out-dir. It adds NO new physics: changing a formula in
        results_values.py changes it here too.

        sim_values.py
        The simulation counterpart of bcrf_values: gives the curves ALONG the scratch instead of single values
        (forces, SCOF, w0, h_r, h_p, groove and pile-up areas, commanded depth and lateral recovery factor).
        The node cloud is turned into a height map shaped like a .bcrf scan, then measured with the bcrf_values functions,
        so the lab and the simulation go through the same estimators. Options: --csv, --png, --sections, --smooth, --no-show...

        calib_fit.py
        Scores every run of a calibration sweep against the lab data. Both sides are compared at the same groove width w0
        (the simulation is depth-driven, the lab is load-driven, w0 is what they share). Three observables are scored:
        F_n (flow stress level), A_pile/A_groove (dilatancy) and SCOF (friction). It does not optimise: it ranks the runs,
        gives the effect of each factor and shows which factors the data can actually identify.

        sweep_calib_curves_pc1.csv, sweep_calib_curves_pmma1.csv, sweep_calib_curves_pcmma2.csv
        Along-track curves written by sim_values.py (--csv) for the calibration sweeps: PC, first PMMA and second PMMA.
        One row per (run, z sample). These are the --sim inputs of calib_fit.py.

        calib_pmma/, calib_fit_pmma2/, calib_fit_pc1/
        Outputs of calib_fit.py (rankings, factor effects, figures) for the first PMMA sweep, the second PMMA sweep and the PC sweep.

    Root

        run_parameter_study.py
        Driver called by submit.sh.
        'STUDIES' holds every possible campaign: single, mesh, mass_scale, target_dt, friction, material, design, models, depth, gsell_h.
        'Default + Selection' allows to choose the values of each study.
        For each case it builds the model, submits the job, post-processes and cleans up.

        launch_cluster_jobs.py
        Multi-job launcher for the cluster, based on 'run_parameter_study'. Each study is launched by creating a job in the 'JOB' section.
        '_OVERRIDE_ALIASES' lists every parameter that can be changed directly in this file, for each job.
        Each job must have its own tag.

        submit.sh
        Calls 'run_parameter_study' and allows to change the partition, CPUs, max time and memory.

        analytic.py
        Closed-form references for the NUMERICAL settings, Abaqus-free (numpy only) so it can be
        imported both by the kernel (extractor writes its outputs in the CSV header) and by the CPython
        scripts. Answers what no simulation output answers alone: the mass factor actually applied and
        the resulting stable increment, the contact radius and width, how many elements resolve the
        contact, and how much of the scratch step is consumed by amplitude smoothing.

        sweep_collector.py
        Aggregates a whole sweep into one tidy table: walks <results_dir> recursively, runs the
        results_values backend on each *_Results.csv, joins the design factor levels and writes
        sweep_<family>.csv. status is an INTEGRITY flag only (OK / FAIL): no run is ever excluded on a
        quality criterion, the energy diagnostics are written as plain numeric columns and reported
        downstream as indicators.

        morris_analysis.py
        Morris elementary-effects analysis of a collected sweep. Reads the tidy table plus the design
        (to recover delta, the active factors and the trajectory structure), computes mu, mu* and sigma
        per factor and per QoI with bootstrap confidence intervals, and ranks the factors. Retention
        uses a RELATIVE threshold (mu*_lo / mu*_max >= --retain-frac), no absolute noise floor.
        Default QoI: F_n, SCOF_mean, h_r, h_p.

        dp_screening.py
        Creates the screening report for the Drucker-Prager campaign: runs the whole chain
        (collection, elementary effects on every QoI, consolidated multi-QoI ranking) and writes
        SCREENING_REPORT.md with its figures. Collection is skipped if --table already points at a
        collected tidy CSV. Redundant QoI (H_MPa, Ft_half_N, pile_up_ratio) are deliberately dropped
        so the multiplicity correction is not inflated by duplicated signals.

        sweep.csv
        Example of a collected sweep table (one row per run): identification (id, family, campaign,
        method, traj, step, moved, sign, status), numerical settings, the extracted QoI, the energy
        diagnostics and quality flags, and the factor levels in unit (u_*), grid (g_*) and physical
        (p_*) coordinates. This is the file consumed by morris_analysis.py / dp_screening.py.