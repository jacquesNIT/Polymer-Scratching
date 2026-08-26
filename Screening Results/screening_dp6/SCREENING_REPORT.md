# Morris screening -- unified Drucker-Prager campaign

| | |
|---|---|
| Campaign | `CDP_drucker_prager_unified` |
| Host family | `glassy_pc` |
| Factors | 8 : `X`, `h`, `w`, `eps_c`, `beta`, `K`, `mu_eff`, `phi` |
| Trajectories r | 10 |
| Step Delta | 0.666667 |
| Planned / usable runs | 90 / 80 |
| Complete trajectories | 6 / 10 |
| Retention rule | mu*_lo / mu*_max >= 0.20 (relative threshold) |
| QoI analysed | 4 |
| Multiplicity correction | alpha/4 over QoI (bootstrap percentile 1.250%) |

> **10 unusable run(s).** Each missing point destroys two elementary effects. Quality is NOT a cause: no run is excluded on an energy criterion.
>
> - **10 never produced (no file)**: 00013, 00016, 00017, 00028, 00029, 00037, 00065, 00067, 00068, 00069

## 1. Consolidated ranking

Rule applied: a factor is **retained if it exceeds the threshold for at least one QoI**. `mu*` is normalised per QoI (1 = dominant factor for that QoI), which makes the columns comparable across QoI of different units.

This rule is a **union of 4 tests per factor**. Without correction, the risk of wrongly retaining a null factor would reach 19%; the bootstrap threshold is therefore corrected to alpha/4.

| Factor | mu* max | mu* mean | best rank | mean rank | sigma/mu* max | rel_lo max | deciding QoI | Verdict |
|---|---|---|---|---|---|---|---|---|
| `X` | 1 | 0.79 | 1 | 1.25 | 0.524 | 0.775 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `mu_eff` | 1 | 0.507 | 1 | 3 | 1.26 | 0.845 | `F_n`, `SCOF_mean`, `h_r`, `h_p` | **RETAIN** |
| `w` | 0.517 | 0.347 | 2 | 2.5 | 0.744 | 0.389 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `beta` | 0.506 | 0.328 | 3 | 3.75 | 0.698 | 0.342 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `h` | 0.46 | 0.246 | 3 | 5.25 | 1.2 | 0.189 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `K` | 0.218 | 0.103 | 6 | 6.5 | 1.8 | 0.0992 | `h_p` | **RETAIN** |
| `eps_c` | 0.0976 | 0.071 | 6 | 6.5 | 1.45 | - | - | **freeze** |
| `phi` | 0.0731 | 0.0509 | 5 | 7.25 | 1.29 | - | - | **freeze** |

![Consolidated ranking](consolidated.png)

![Normalised mu* map](heatmap.png)

**Reading.** A `sigma/mu*` above 1.0 flags a factor whose effect is dominated by interactions or by strong non-linearity: Morris does not distinguish the two, only Sobol will. A large gap between the best rank and the mean rank flags a factor specific to one QoI.

## 2. Detail per QoI

### `F_n` -- Normal force (half-model) [N]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.5407 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 2.704 | 1.783 | 3.593 | 1.322 | 0.489 | 0.66 | 2.704 | 9 | RETAIN |
| `w` | 1.278 | 0.7266 | 1.879 | 0.867 | 0.678 | 0.269 | 1.278 | 10 | RETAIN |
| `h` | 1.244 | 0.5117 | 2.414 | 1.234 | 0.992 | 0.189 | 1.244 | 7 | RETAIN? |
| `beta` | 1.159 | 0.5907 | 1.763 | 0.809 | 0.698 | 0.218 | 1.159 | 8 | RETAIN |
| `mu_eff` | 0.5589 | 0.2563 | 0.8261 | 0.3958 | 0.708 | 0.0948 | -0.5537 | 8 | RETAIN? |
| `eps_c` | 0.2638 | 0.1377 | 0.4294 | 0.3456 | 1.31 | 0.0509 | 0.008868 | 8 | freeze |
| `K` | 0.2542 | 0.124 | 0.4194 | 0.3085 | 1.21 | 0.0459 | -0.1421 | 9 | freeze |
| `phi` | 0.1214 | 0.05317 | 0.2119 | 0.154 | 1.27 | 0.0197 | -0.0493 | 7 | freeze |

![Morris F_n](morris_F_n.png)

### `SCOF_mean` -- Apparent friction coefficient [-]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.1299 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `mu_eff` | 0.6496 | 0.5492 | 0.7924 | 0.1716 | 0.264 | 0.845 | 0.6496 | 8 | RETAIN |
| `X` | 0.1031 | 0.06093 | 0.1365 | 0.05399 | 0.524 | 0.0938 | 0.1026 | 9 | freeze |
| `w` | 0.07771 | 0.04332 | 0.1192 | 0.05779 | 0.744 | 0.0667 | -0.07771 | 10 | freeze |
| `beta` | 0.07206 | 0.04091 | 0.1073 | 0.04391 | 0.609 | 0.063 | -0.07206 | 8 | freeze |
| `phi` | 0.03611 | 0.01882 | 0.06049 | 0.04658 | 1.29 | 0.029 | -0.004853 | 7 | freeze |
| `K` | 0.02455 | 0.009133 | 0.04225 | 0.0236 | 0.961 | 0.0141 | -0.02455 | 9 | freeze |
| `eps_c` | 0.0174 | 0.005966 | 0.03083 | 0.0252 | 1.45 | 0.00918 | -0.0003451 | 8 | freeze |
| `h` | 0.01467 | 0.004563 | 0.0296 | 0.01755 | 1.2 | 0.00702 | -0.01283 | 7 | freeze |

![Morris SCOF_mean](morris_SCOF_mean.png)

### `h_r` -- Residual depth [mm]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.003002 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01501 | 0.01164 | 0.01845 | 0.004788 | 0.319 | 0.775 | 0.01501 | 9 | RETAIN |
| `w` | 0.00416 | 0.002644 | 0.005846 | 0.002398 | 0.576 | 0.176 | -0.00416 | 10 | RETAIN? |
| `beta` | 0.004008 | 0.002814 | 0.005594 | 0.001944 | 0.485 | 0.187 | -0.004008 | 8 | RETAIN? |
| `mu_eff` | 0.003894 | 0.0009137 | 0.007986 | 0.004922 | 1.26 | 0.0609 | 0.003862 | 8 | RETAIN? |
| `h` | 0.003516 | 0.001203 | 0.006376 | 0.003201 | 0.911 | 0.0801 | -0.003516 | 7 | RETAIN? |
| `eps_c` | 0.0009948 | 0.000495 | 0.001583 | 0.001274 | 1.28 | 0.033 | -2.504e-05 | 8 | freeze |
| `K` | 0.0009217 | 0.0003004 | 0.002081 | 0.001659 | 1.8 | 0.02 | 8.477e-05 | 9 | freeze |
| `phi` | 0.000449 | 0.0001452 | 0.0008381 | 0.0005359 | 1.19 | 0.00967 | 0.0003386 | 7 | freeze |

![Morris h_r](morris_h_r.png)

### `h_p` -- Lateral pile-up [mm]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.002142 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01071 | 0.007272 | 0.01473 | 0.005216 | 0.487 | 0.679 | 0.01071 | 9 | RETAIN |
| `mu_eff` | 0.006032 | 0.002765 | 0.009165 | 0.006743 | 1.12 | 0.258 | 0.003685 | 8 | RETAIN |
| `w` | 0.005541 | 0.004161 | 0.007562 | 0.002529 | 0.456 | 0.389 | -0.005541 | 10 | RETAIN |
| `beta` | 0.005421 | 0.003666 | 0.00713 | 0.002297 | 0.424 | 0.342 | -0.005421 | 8 | RETAIN |
| `h` | 0.002863 | 0.001625 | 0.003947 | 0.002879 | 1.01 | 0.152 | -0.001704 | 7 | RETAIN? |
| `K` | 0.002339 | 0.001062 | 0.003708 | 0.002537 | 1.08 | 0.0992 | 0.001685 | 9 | RETAIN? |
| `eps_c` | 0.001 | 0.0005821 | 0.001468 | 0.001224 | 1.22 | 0.0544 | -9.055e-05 | 8 | freeze |
| `phi` | 0.0007823 | 0.0001679 | 0.001656 | 0.0009363 | 1.2 | 0.0157 | 0.0007823 | 7 | freeze |

![Morris h_p](morris_h_p.png)

## 3. Decision

**Retained (6):** `X`, `mu_eff`, `w`, `beta`, `h`, `K`

**Retained but marginal (0):** none

**Frozen (2):** `eps_c`, `phi`

> **The threshold is RELATIVE.** It compares each factor to the most influential one of the same QoI, it does not test against zero. Three consequences: the top factor is retained by construction; a `freeze` means *small compared to the largest*, **not** *null*; and if every factor had a real effect of the same order, the rule would freeze none of them. It reduces dimensionality, it does not prove any nullity.

### Sobol

```bash
python3 generate_design.py glassy_pc --method sobol --n 1024 \
        --only X,mu_eff,w,beta,h,K
```

Marginal factors are included as a precaution. Unlisted factors are frozen at mid-range. Going from 8 to 6 factors is what makes 1024 points enough.

## 3bis. Identifiability

### Confounding signature between factors

No pair above the threshold 0.55: no confounding signature detected.

| Pair | index | QoI |
|---|---|---|
| `beta` / `mu_eff` | 0.525 | 4 |
| `w` / `mu_eff` | 0.511 | 4 |
| `X` / `w` | 0.386 | 4 |
| `h` / `mu_eff` | 0.383 | 4 |
| `X` / `beta` | 0.367 | 4 |
| `X` / `h` | 0.129 | 4 |
| `h` / `K` | 0.122 | 4 |
| `beta` / `K` | 0.114 | 4 |

## 3ter. Numerical quality indicators

These are **indicators only**. No run is excluded on any of them: the energy diagnostics are reported so that a systematic drift stays visible, and every parsed run feeds the elementary effects regardless of the values below. On a deliberately coarse mesh, exceeding these limits is expected and is not a reason to discard the run.

| Indicator | Limit | Over limit | Median | p90 | Max | Worst ids |
|---|---|---|---|---|---|---|
| `KE_IE_steady_max` | 5 % | 0 / 80 (0%) | 0.0858 | 0.884 | 3.2 | - |
| `AE_IE_final` | 5 % | 2 / 80 (2.5%) | 1.49 | 3.05 | 7.75 | 00078, 00030 |
| `ETOTAL_drift` | 5 % | 0 / 80 (0%) | 0.202 | 1.05 | 3.18 | - |
| `ALLPW` | 5 % | 3 / 80 (3.75%) | 2.65 | 3.79 | 5.24 | 00075, 00077, 00076 |
| `KE_final_over_IE_peak` | 1 % | 0 / 80 (0%) | 0.00116 | 0.00647 | 0.0162 | - |

**How to read this.** An indicator exceeded by nearly EVERY run points at a systematic setting shared by the whole campaign (element control, mass scaling, ALE frequency, mesh size), not at individual runs. An indicator exceeded by a HANDFUL of runs points at a corner of the factor box; cross the worst ids above with the design to find which factor level they share. In both cases the action is to change the setting or to document the bias, never to drop the runs.

## 4. Caveats

- The ranking holds for the **Drucker-Prager class**, not for any single family: `semicrystalline_*` and `glassy_*` are points of the same dimensionless box. No `mu*` specific to PMMA or PC comes out of it.
- The unified box spans different physical regimes (softening present or absent). A high `sigma` may reflect this mixture rather than an interaction.
- `h` enters via `exp(h*eps^2)` evaluated up to eps_max: strong non-linearity on this factor is expected by construction of the model.
- `phi` is capped below 1 in the sampling box precisely because `phi = 1` switches the friction model from a tabulated Briscoe table to constant Coulomb -- a change of model class, not a variation of a factor. If a design predating that cap is analysed here, the `mu*` of `phi` mixes the two.
- Quality indicators (section 3ter) never exclude a run. A campaign run on a coarse mesh for turnaround will exceed them by construction; that is a known bias to report, not a reason to shrink the design.
