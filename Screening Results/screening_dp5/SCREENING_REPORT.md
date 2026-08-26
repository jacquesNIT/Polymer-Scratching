# Morris screening -- unified Drucker-Prager campaign

| | |
|---|---|
| Campaign | `CDP_drucker_prager_unified` |
| Host family | `glassy_pc` |
| Factors | 8 : `X`, `h`, `w`, `eps_c`, `beta`, `K`, `mu_eff`, `phi` |
| Trajectories r | 10 |
| Step Delta | 0.666667 |
| Planned / usable runs | 90 / 79 |
| Complete trajectories | 5 / 10 |
| Retention rule | mu*_lo / mu*_max >= 0.20 (relative threshold) |
| QoI analysed | 4 |
| Multiplicity correction | alpha/4 over QoI (bootstrap percentile 1.250%) |

> **11 unusable run(s).** Each missing point destroys two elementary effects. Quality is NOT a cause: no run is excluded on an energy criterion.
>
> - **11 never produced (no file)**: 00013, 00016, 00017, 00037, 00065, 00067, 00068, 00069, 00075, 00081, 00087

## 1. Consolidated ranking

Rule applied: a factor is **retained if it exceeds the threshold for at least one QoI**. `mu*` is normalised per QoI (1 = dominant factor for that QoI), which makes the columns comparable across QoI of different units.

This rule is a **union of 4 tests per factor**. Without correction, the risk of wrongly retaining a null factor would reach 19%; the bootstrap threshold is therefore corrected to alpha/4.

| Factor | mu* max | mu* mean | best rank | mean rank | sigma/mu* max | rel_lo max | deciding QoI | Verdict |
|---|---|---|---|---|---|---|---|---|
| `X` | 1 | 0.79 | 1 | 1.25 | 0.529 | 0.781 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `mu_eff` | 1 | 0.502 | 1 | 2.75 | 1.37 | 0.844 | `SCOF_mean`, `h_r`, `h_p` | **RETAIN** |
| `h` | 0.566 | 0.287 | 2 | 5.25 | 1.24 | 0.24 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `beta` | 0.54 | 0.328 | 3 | 3.75 | 0.7 | 0.388 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `w` | 0.513 | 0.342 | 2 | 3 | 0.77 | 0.362 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `K` | 0.315 | 0.128 | 5 | 6.25 | 2.1 | 0.144 | `h_p` | **RETAIN** |
| `eps_c` | 0.123 | 0.0856 | 6 | 6.5 | 1.44 | - | - | **freeze** |
| `phi` | 0.12 | 0.0731 | 5 | 7.25 | 1.9 | - | - | **freeze** |

![Consolidated ranking](consolidated.png)

![Normalised mu* map](heatmap.png)

**Reading.** A `sigma/mu*` above 1.0 flags a factor whose effect is dominated by interactions or by strong non-linearity: Morris does not distinguish the two, only Sobol will. A large gap between the best rank and the mean rank flags a factor specific to one QoI.

## 2. Detail per QoI

### `F_n` -- Normal force (half-model) [N]

63 elementary effects, 11 missing point(s). Retention threshold: mu* >= 0.5609 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 2.804 | 1.94 | 3.65 | 1.255 | 0.448 | 0.692 | 2.804 | 9 | RETAIN |
| `h` | 1.589 | 0.6731 | 3.575 | 1.758 | 1.11 | 0.24 | 1.589 | 5 | RETAIN |
| `w` | 1.324 | 0.6757 | 1.998 | 1.019 | 0.77 | 0.241 | 1.317 | 10 | RETAIN |
| `beta` | 1.139 | 0.5731 | 1.741 | 0.7978 | 0.7 | 0.204 | 1.139 | 8 | RETAIN |
| `mu_eff` | 0.4827 | 0.2538 | 0.7251 | 0.5245 | 1.09 | 0.0905 | -0.2811 | 8 | freeze |
| `eps_c` | 0.3449 | 0.1472 | 0.6331 | 0.4573 | 1.33 | 0.0525 | 0.1283 | 6 | freeze |
| `K` | 0.2954 | 0.1332 | 0.4692 | 0.3231 | 1.09 | 0.0475 | -0.2097 | 9 | freeze |
| `phi` | 0.1301 | 0.0296 | 0.3346 | 0.2469 | 1.9 | 0.0106 | 0.08711 | 8 | freeze |

![Morris F_n](morris_F_n.png)

### `SCOF_mean` -- Apparent friction coefficient [-]

63 elementary effects, 11 missing point(s). Retention threshold: mu* >= 0.1303 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `mu_eff` | 0.6517 | 0.5499 | 0.7933 | 0.1713 | 0.263 | 0.844 | 0.6517 | 8 | RETAIN |
| `X` | 0.103 | 0.06073 | 0.137 | 0.05451 | 0.529 | 0.0932 | 0.1025 | 9 | freeze |
| `w` | 0.07985 | 0.04471 | 0.1231 | 0.05932 | 0.743 | 0.0686 | -0.07985 | 10 | freeze |
| `beta` | 0.0714 | 0.04097 | 0.1059 | 0.04232 | 0.593 | 0.0629 | -0.0714 | 8 | freeze |
| `phi` | 0.06609 | 0.02319 | 0.145 | 0.103 | 1.56 | 0.0356 | -0.03688 | 8 | freeze |
| `K` | 0.02809 | 0.01118 | 0.04713 | 0.02504 | 0.891 | 0.0172 | -0.02809 | 9 | freeze |
| `eps_c` | 0.01991 | 0.003518 | 0.03735 | 0.02875 | 1.44 | 0.0054 | -0.002005 | 6 | freeze |
| `h` | 0.01963 | 0.004801 | 0.04274 | 0.0244 | 1.24 | 0.00737 | -0.01579 | 5 | freeze |

![Morris SCOF_mean](morris_SCOF_mean.png)

### `h_r` -- Residual depth [mm]

63 elementary effects, 11 missing point(s). Retention threshold: mu* >= 0.003026 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01513 | 0.01181 | 0.01865 | 0.004825 | 0.319 | 0.781 | 0.01513 | 9 | RETAIN |
| `w` | 0.003974 | 0.002388 | 0.00581 | 0.002555 | 0.643 | 0.158 | -0.003974 | 10 | RETAIN? |
| `mu_eff` | 0.003965 | 0.001254 | 0.008077 | 0.005429 | 1.37 | 0.0829 | 0.003164 | 8 | RETAIN? |
| `beta` | 0.003859 | 0.002845 | 0.005232 | 0.001648 | 0.427 | 0.188 | -0.003859 | 8 | RETAIN? |
| `h` | 0.003837 | 0.001179 | 0.007533 | 0.003416 | 0.89 | 0.0779 | -0.003837 | 5 | RETAIN? |
| `eps_c` | 0.001025 | 0.0004438 | 0.001748 | 0.001295 | 1.26 | 0.0293 | -0.0003437 | 6 | freeze |
| `K` | 0.0007438 | 6.623e-05 | 0.001864 | 0.001562 | 2.1 | 0.00438 | 0.0001179 | 9 | freeze |
| `phi` | 0.0003743 | 0.0001538 | 0.0006543 | 0.0004736 | 1.27 | 0.0102 | 0.0002008 | 8 | freeze |

![Morris h_r](morris_h_r.png)

### `h_p` -- Lateral pile-up [mm]

63 elementary effects, 11 missing point(s). Retention threshold: mu* >= 0.002222 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01111 | 0.007912 | 0.01494 | 0.004962 | 0.447 | 0.712 | 0.01111 | 9 | RETAIN |
| `mu_eff` | 0.006381 | 0.003371 | 0.009619 | 0.00634 | 0.993 | 0.303 | 0.004589 | 8 | RETAIN |
| `beta` | 0.006005 | 0.004313 | 0.007665 | 0.00225 | 0.375 | 0.388 | -0.006005 | 8 | RETAIN |
| `w` | 0.005699 | 0.004024 | 0.007573 | 0.002651 | 0.465 | 0.362 | -0.005699 | 10 | RETAIN |
| `K` | 0.003503 | 0.001596 | 0.005996 | 0.003759 | 1.07 | 0.144 | 0.002849 | 9 | RETAIN? |
| `h` | 0.003312 | 0.001318 | 0.004737 | 0.003621 | 1.09 | 0.119 | -0.001807 | 5 | RETAIN? |
| `eps_c` | 0.001348 | 0.0006371 | 0.002194 | 0.001617 | 1.2 | 0.0573 | -0.0005721 | 6 | freeze |
| `phi` | 0.001333 | 0.0004873 | 0.002304 | 0.001748 | 1.31 | 0.0439 | 0.0005944 | 8 | freeze |

![Morris h_p](morris_h_p.png)

## 3. Decision

**Retained (6):** `X`, `mu_eff`, `h`, `beta`, `w`, `K`

**Retained but marginal (0):** none

**Frozen (2):** `eps_c`, `phi`

> **The threshold is RELATIVE.** It compares each factor to the most influential one of the same QoI, it does not test against zero. Three consequences: the top factor is retained by construction; a `freeze` means *small compared to the largest*, **not** *null*; and if every factor had a real effect of the same order, the rule would freeze none of them. It reduces dimensionality, it does not prove any nullity.

### Sobol

```bash
python3 generate_design.py glassy_pc --method sobol --n 1024 \
        --only X,mu_eff,h,beta,w,K
```

Marginal factors are included as a precaution. Unlisted factors are frozen at mid-range. Going from 8 to 6 factors is what makes 1024 points enough.

## 3bis. Identifiability

### Confounding signature between factors

No pair above the threshold 0.55: no confounding signature detected.

| Pair | index | QoI |
|---|---|---|
| `beta` / `mu_eff` | 0.452 | 4 |
| `w` / `mu_eff` | 0.443 | 4 |
| `X` / `w` | 0.387 | 4 |
| `X` / `beta` | 0.371 | 4 |
| `h` / `mu_eff` | 0.294 | 4 |
| `beta` / `K` | 0.172 | 4 |
| `w` / `K` | 0.172 | 4 |
| `h` / `K` | 0.146 | 4 |

## 3ter. Numerical quality indicators

These are **indicators only**. No run is excluded on any of them: the energy diagnostics are reported so that a systematic drift stays visible, and every parsed run feeds the elementary effects regardless of the values below. On a deliberately coarse mesh, exceeding these limits is expected and is not a reason to discard the run.

| Indicator | Limit | Over limit | Median | p90 | Max | Worst ids |
|---|---|---|---|---|---|---|
| `KE_IE_steady_max` | 5 % | 0 / 79 (0%) | 0.0852 | 0.947 | 2.97 | - |
| `AE_IE_final` | 5 % | 5 / 79 (6.33%) | 1.51 | 3.23 | 7.45 | 00078, 00028, 00029, 00030, 00066 |
| `ETOTAL_drift` | 5 % | 0 / 79 (0%) | 0.166 | 0.53 | 2.39 | - |
| `ALLPW` | 5 % | 0 / 79 (0%) | 2.58 | 3.39 | 4.22 | - |
| `KE_final_over_IE_peak` | 1 % | 0 / 79 (0%) | 0.0014 | 0.00682 | 0.0175 | - |

**How to read this.** An indicator exceeded by nearly EVERY run points at a systematic setting shared by the whole campaign (element control, mass scaling, ALE frequency, mesh size), not at individual runs. An indicator exceeded by a HANDFUL of runs points at a corner of the factor box; cross the worst ids above with the design to find which factor level they share. In both cases the action is to change the setting or to document the bias, never to drop the runs.

## 4. Caveats

- The ranking holds for the **Drucker-Prager class**, not for any single family: `semicrystalline_*` and `glassy_*` are points of the same dimensionless box. No `mu*` specific to PMMA or PC comes out of it.
- The unified box spans different physical regimes (softening present or absent). A high `sigma` may reflect this mixture rather than an interaction.
- `h` enters via `exp(h*eps^2)` evaluated up to eps_max: strong non-linearity on this factor is expected by construction of the model.
- `phi` is capped below 1 in the sampling box precisely because `phi = 1` switches the friction model from a tabulated Briscoe table to constant Coulomb -- a change of model class, not a variation of a factor. If a design predating that cap is analysed here, the `mu*` of `phi` mixes the two.
- Quality indicators (section 3ter) never exclude a run. A campaign run on a coarse mesh for turnaround will exceed them by construction; that is a known bias to report, not a reason to shrink the design.
