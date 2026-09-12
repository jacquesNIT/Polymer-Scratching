General Use

To be modified before use

    submit.sh
    "/home/au824386/bin/subabqpy_mine -p q36 -c 10 -m 100 -t 4-00:00:00 run_parameter_study"

ScratchSimulation

    AbaqusModel

        Configuration

            base.py
            It Contains every classes useful for the simulation. Parameters for scratch, indenter, substrate, mesh.. can be changed here.
            The "Polymer_default" function contains the base parameters of the simulation (overrides the ones defined in the classes). 
            Pretty mush every aspects of the simulation can be changed here.

            families.py
            It contains functions defining various kinds of polymer families (PC, PMMA, Semi-crystallines..).
            Based on "Polymer_default", differencies in material cards and used models are found here.

            sampling.py
            Screening / sweep design layer for the Drucker-Prager families.
            Defines the Factor class (bounds, lin/log scale, physical mapping), the campaign
            containers (SAMPLING_DP_UNIFIED and the legacy split campaigns) and the design generators morris_design() and sobol_design(). 

        Geometry

            substrate.py
            Builds the substrate (half-symmetry block).
            Applies a partitioning into wanted zones (fine / C1 / C2). 
            Creates the sets and surfaces used later (substrate set, top contact surface, symmetry face).

            indenter.py
            Generates the rigid indenter geometry and dispatches on cfg.indenter.indenter_type.
            create_rockwell() sketches the sphere-cone profile (same for create_pyramid())
            place_indenter() moves the instance so the tip touches the surface at z = dpo_z, with the placement adapted to the indenter type.

        Material

            assignment.py
            Class SubstrateMaterialAssignment: turns Material_Config into a real Abaqus material and assigns it to the part. 
            Density is written, then four independent blocks are dispatched by theiraccording to the model (hyperelastic / viscoelastic / plasticity / damage).
            _validate_material() forbids the invalid combinations ( hyperelastic + plasticity for example).
            update_friction() writes the contact property (Coulomb or Briscoe pressure-dependent).

        Postprocessing

            extractor.py
            Runs inside the Abaqus kernel right after the job. Opens the .odb and writes one csv per simulation containing: 
            metadata header, history time series (RF2/RF3, CFN/CSHEAR, all energies), deformed surface node coordinates. 
            Helper functions handle history-region name matching, resampling on a common time base and wallclock extraction from the .sta file.

        Simulation

            Modelbuilder.py
            build_scratch_model(cfg) orchestrates the whole model: geometry (substrate + mesh + indenter), assembly and indenter placement, 
            creation of the explicit steps with their mass scaling, boundary conditions, the displacement-controlled loading through amplitudes, the contact pair and its friction
            property, ALE adaptive meshing, and finally the field/history output requests.


        abaqus_env.py
        Used to centralize Abaqus imports (abaqus, abaqusConstants, part, material, step, interaction, mesh, odbAccess...). 

    utils

        job.py
        run_job_and_wait(): creates the job with the solver settings (precision, MPI, domain parallelisation, num_cpus from Solver_Config).
        It deletes any stale job or .lck left by a previous run, submits and waits.
        _check_job_status() raises an error if Abaqus returned anything else than completed

        cleanup.py
        cleanup_abaqus_junk(): removes the Abaqus scratch files (.abq, .com, .mdl, .pac, .prt,
        .sel, .stt, .rpy...) after a run.

    Root 

        run_parameter_study.py
        Driver called by submit.sh. 
        'STUDIES' holds every possible campaign: single, mesh, mass_scale, target_dt, friction, material, design, models, depth, gsell_h. 
        'Default + Selection' allows to choose the values of each studies.
        For each case it builds the model, submits the job, post-processes and cleans up.

        launch_cluster_jobs.py
        Multi-job launcher for the cluster based on 'run_parameter_study'. Each study can be done by creating a job in the 'JOB' section.
        '_OVERRIDE_ALIASES' contains a list of all parameters that can be changed directly in this file, for each job. 
        Each job must have a respective tag.

        submit.sh
        Calls 'run_parameter_study' and allows to change partition, CPUs, max time and memory.

        analytic.py
        Closed-form references for the NUMERICAL settings, Abaqus-free (numpy only) so it can be
        imported both by the kernel (extractor writes its outputs in the CSV header) and by the CPython
        scripts. Answers what no simulation output answers alone: the mass factor actually applied and
        the resulting stable increment, the contact radius and width, how many elements resolve the
        contact, and how much of the scratch step is consumed by amplitude smoothing.

        generate_design.py
        Generator of the screening / sweep designs. 
        Reads the campaign of a family from sampling.py, draws a Morris trajectory design or a Sobol sequenceand writes the design csv.
        That csv is the input consumed by the "design" study of run_parameter_study.py.

        results_values.py
        Reference post-processing of one simulation CSV: the single source of truth for every QoI.
        Parses the file, rebuilds the time masks (active / loading window), extracts the normal and
        tangential forces (RF2/RF3 of the half model), the SCOF averaged on the [10 %, 90 %] band,
        the residual depth h_r, the lateral pile-up h_p and the frontal pile-up h_fp on a transverse
        section at z, plus the energy diagnostics (KE/IE, AE/IE, ETOTAL drift). Usage:
        "python results_values.py <folder_or_csv> [<out_csv>] [<z_mm>]".

        result_values_batch.py
        Same as 'results_values' applied on a folder of csv's.
        Batch companion of results_values.py: walks a campaign folder recursively for *_Results.csv,
        calls extract_values() on each run in parallel (--jobs), catches per-run failures, joins the
        design levels on the run id (--design) and writes one tidy CSV plus exploratory figures in
        --out-dir. It adds NO new physics: changing a formula in results_values.py changes it here too.

        scof.py
        Plots and exports the SCOF ALONG the scratch instead of its band average. It imports the
        estimators from results_values.py (masks, RF2/CFN2 choice, band bounds) rather than copying
        them, keeps the point-by-point series and maps it onto the groove abscissa z. Options: --csv,
        --png, --smooth <mm>, --no-show, --no-plot. Averaging the curve over the band gives back
        SCOF_mean bit for bit.

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
        Creates the Screening report for the Drucker-Prager campaign: runs the whole chain
        (collection, elementary effects on every QoI, consolidated multi-QoI ranking) and writes
        SCREENING_REPORT.md with its figures. Collection is skipped if --table already points at a
        collected tidy CSV. Redundant QoI (H_MPa, Ft_half_N, pile_up_ratio) are deliberately dropped
        so the multiplicity correction is not inflated by duplicated signals.

        sweep.csv
        Example of a collected sweep table (one row per run): identification (id, family, campaign,
        method, traj, step, moved, sign, status), numerical settings, the extracted QoI, the energy
        diagnostics and quality flags, and the factor levels in unit (u_*), grid (g_*) and physical
        (p_*) coordinates. This is the file consumed by morris_analysis.py / dp_screening.py.


    Data folders

        designs          
        Design CSVs written by generate_design.py and read by the "design" study.

        runs             
        Isolated working directories created per job (runs/<study>_<family>_<tag>),
                          holding the .odb/.sta of the run and the SimDataOutputs/ CSVs.
        Screening Results/  Collected sweep tables, Morris analyses and the generated reports.
        Lab results/      Raw experimental data from the Rtec tribometer (.bcrf topographies, force CSVs).
        Lab_verifier/     Scripts used to read and analyse the lab files (bcrf_values.py and companions).
        Polymer_references/  Datasheets and literature used to fix the material cards.
        ScratchFeatures/  Shared constants and helpers of the topography pipeline (grid shape, domain size).
        Other/            Scratch space for one-off scripts and temporary material.