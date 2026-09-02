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
| `X` | 1 | 0.789 | 1 | 1.25 | 0.527 | 0.748 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `mu_eff` | 1 | 0.483 | 1 | 2.75 | 1.42 | 0.843 | `SCOF_mean`, `h_r`, `h_p` | **RETAIN** |
| `w` | 0.414 | 0.31 | 2 | 2.5 | 0.842 | 0.24 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `h` | 0.391 | 0.197 | 3 | 5.5 | 1.27 | 0.148 | `F_n`, `h_p` | **RETAIN** |
| `beta` | 0.379 | 0.259 | 4 | 4 | 0.901 | 0.228 | `F_n`, `h_p` | **RETAIN** |
| `K` | 0.306 | 0.118 | 5 | 6.5 | 1.51 | 0.154 | `h_p` | **RETAIN** |
| `eps_c` | 0.125 | 0.0835 | 6 | 6.5 | 1.5 | - | - | **freeze** |
| `phi` | 0.0816 | 0.0538 | 5 | 7 | 1.52 | - | - | **freeze** |

![Consolidated ranking](consolidated.png)

![Normalised mu* map](heatmap.png)

**Reading.** A `sigma/mu*` above 1.0 flags a factor whose effect is dominated by interactions or by strong non-linearity: Morris does not distinguish the two, only Sobol will. A large gap between the best rank and the mean rank flags a factor specific to one QoI.

## 2. Detail per QoI

### `F_n` -- Normal force (half-model) [N]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.6282 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 3.141 | 2.245 | 4.025 | 1.321 | 0.421 | 0.715 | 3.141 | 9 | RETAIN |
| `w` | 1.3 | 0.6991 | 1.914 | 0.9208 | 0.708 | 0.223 | 1.3 | 10 | RETAIN |
| `h` | 1.228 | 0.4647 | 2.502 | 1.33 | 1.08 | 0.148 | 1.228 | 7 | RETAIN? |
| `beta` | 1.192 | 0.5713 | 1.85 | 0.8774 | 0.736 | 0.182 | 1.192 | 8 | RETAIN? |
| `mu_eff` | 0.4756 | 0.246 | 0.705 | 0.3988 | 0.839 | 0.0783 | -0.4113 | 8 | freeze |
| `eps_c` | 0.2685 | 0.1449 | 0.4231 | 0.3439 | 1.28 | 0.0461 | 0.01725 | 8 | freeze |
| `K` | 0.2284 | 0.1108 | 0.3873 | 0.2013 | 0.881 | 0.0353 | -0.2251 | 9 | freeze |
| `phi` | 0.05595 | 0.02507 | 0.08897 | 0.07037 | 1.26 | 0.00798 | -0.01602 | 7 | freeze |

![Morris F_n](morris_F_n.png)

### `SCOF_mean` -- Apparent friction coefficient [-]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.1288 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `mu_eff` | 0.644 | 0.5426 | 0.7921 | 0.1759 | 0.273 | 0.843 | 0.644 | 8 | RETAIN |
| `X` | 0.09928 | 0.06124 | 0.1308 | 0.05233 | 0.527 | 0.0951 | 0.09772 | 9 | freeze |
| `w` | 0.07719 | 0.04351 | 0.1177 | 0.05633 | 0.73 | 0.0676 | -0.07719 | 10 | freeze |
| `beta` | 0.07415 | 0.04332 | 0.1097 | 0.04374 | 0.59 | 0.0673 | -0.07415 | 8 | freeze |
| `phi` | 0.03265 | 0.01684 | 0.05859 | 0.04321 | 1.32 | 0.0261 | -0.009927 | 7 | freeze |
| `K` | 0.02659 | 0.009488 | 0.04653 | 0.02594 | 0.976 | 0.0147 | -0.02659 | 9 | freeze |
| `eps_c` | 0.01786 | 0.007409 | 0.03065 | 0.02476 | 1.39 | 0.0115 | -0.0009614 | 8 | freeze |
| `h` | 0.01114 | 0.002148 | 0.02415 | 0.01416 | 1.27 | 0.00334 | -0.01072 | 7 | freeze |

![Morris SCOF_mean](morris_SCOF_mean.png)

### `h_r` -- Residual depth [mm]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.002579 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.0129 | 0.009056 | 0.01734 | 0.005888 | 0.457 | 0.702 | 0.0129 | 9 | RETAIN |
| `mu_eff` | 0.005216 | 0.002369 | 0.008421 | 0.003982 | 0.763 | 0.184 | 0.005216 | 8 | RETAIN? |
| `w` | 0.004226 | 0.001997 | 0.006642 | 0.003439 | 0.814 | 0.155 | -0.004226 | 10 | RETAIN? |
| `beta` | 0.002425 | 0.001151 | 0.004248 | 0.002186 | 0.901 | 0.0892 | -0.002425 | 8 | freeze |
| `h` | 0.001962 | 0.0008739 | 0.003237 | 0.001468 | 0.748 | 0.0678 | -0.001962 | 7 | freeze |
| `eps_c` | 0.00123 | 0.0004201 | 0.002366 | 0.001848 | 1.5 | 0.0326 | -0.0002064 | 8 | freeze |
| `phi` | 0.0008389 | 0.0002654 | 0.001951 | 0.001271 | 1.52 | 0.0206 | 0.000627 | 7 | freeze |
| `K` | 0.000677 | 0.0001762 | 0.001339 | 0.001024 | 1.51 | 0.0137 | -0.0003913 | 9 | freeze |

![Morris h_r](morris_h_r.png)

### `h_p` -- Lateral pile-up [mm]

66 elementary effects, 10 missing point(s). Retention threshold: mu* >= 0.002088 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01044 | 0.007807 | 0.01326 | 0.003816 | 0.366 | 0.748 | 0.01044 | 9 | RETAIN |
| `w` | 0.003936 | 0.002505 | 0.005476 | 0.003316 | 0.842 | 0.24 | -0.003164 | 10 | RETAIN |
| `mu_eff` | 0.003926 | 0.001525 | 0.007598 | 0.00558 | 1.42 | 0.146 | -0.001979 | 8 | RETAIN? |
| `beta` | 0.003702 | 0.002382 | 0.005298 | 0.001988 | 0.537 | 0.228 | -0.003702 | 8 | RETAIN |
| `K` | 0.003193 | 0.001606 | 0.004985 | 0.002341 | 0.733 | 0.154 | 0.003193 | 9 | RETAIN? |
| `h` | 0.00238 | 0.001207 | 0.003498 | 0.002897 | 1.22 | 0.116 | -0.0004284 | 7 | RETAIN? |
| `eps_c` | 0.00131 | 0.000364 | 0.002429 | 0.001682 | 1.28 | 0.0349 | -0.0009286 | 8 | freeze |
| `phi` | 0.0008519 | 0.0002795 | 0.001513 | 0.0007792 | 0.915 | 0.0268 | 0.0008519 | 7 | freeze |

![Morris h_p](morris_h_p.png)

## 3. Decision

**Retained (6):** `X`, `mu_eff`, `w`, `h`, `beta`, `K`

**Retained but marginal (0):** none

**Frozen (2):** `eps_c`, `phi`

> **The threshold is RELATIVE.** It compares each factor to the most influential one of the same QoI, it does not test against zero. Three consequences: the top factor is retained by construction; a `freeze` means *small compared to the largest*, **not** *null*; and if every factor had a real effect of the same order, the rule would freeze none of them. It reduces dimensionality, it does not prove any nullity.

### Sobol

```bash
python3 generate_design.py glassy_pc --method sobol --n 1024 \
        --only X,mu_eff,w,h,beta,K
```

Marginal factors are included as a precaution. Unlisted factors are frozen at mid-range. Going from 8 to 6 factors is what makes 1024 points enough.

## 3bis. Identifiability

### Confounding signature between factors

No pair above the threshold 0.55: no confounding signature detected.

| Pair | index | QoI |
|---|---|---|
| `X` / `w` | 0.349 | 4 |
| `X` / `beta` | 0.319 | 4 |
| `w` / `mu_eff` | 0.312 | 4 |
| `beta` / `K` | 0.263 | 4 |
| `beta` / `mu_eff` | 0.231 | 4 |
| `w` / `K` | 0.206 | 4 |
| `h` / `mu_eff` | 0.182 | 4 |
| `eps_c` / `phi` | 0.138 | 4 |

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
