# Morris screening -- unified Drucker-Prager campaign

| | |
|---|---|
| Campaign | `CDP_drucker_prager_unified` |
| Host family | `glassy_pc` |
| Factors | 8 : `X`, `h`, `w`, `eps_c`, `beta`, `K`, `mu_eff`, `phi` |
| Trajectories r | 10 |
| Step Delta | 0.666667 |
| Planned / usable runs | 90 / 81 |
| Complete trajectories | 6 / 10 |
| Retention rule | mu*_lo / mu*_max >= 0.20 (relative threshold) |
| QoI analysed | 4 |
| Multiplicity correction | alpha/4 over QoI (bootstrap percentile 1.250%) |

> **9 unusable run(s).** Each missing point destroys two elementary effects. Quality is NOT a cause: no run is excluded on an energy criterion.
>
> - **9 never produced (no file)**: 00013, 00016, 00017, 00029, 00030, 00037, 00065, 00067, 00069

## 1. Consolidated ranking

Rule applied: a factor is **retained if it exceeds the threshold for at least one QoI**. `mu*` is normalised per QoI (1 = dominant factor for that QoI), which makes the columns comparable across QoI of different units.

This rule is a **union of 4 tests per factor**. Without correction, the risk of wrongly retaining a null factor would reach 19%; the bootstrap threshold is therefore corrected to alpha/4.

| Factor | mu* max | mu* mean | best rank | mean rank | sigma/mu* max | rel_lo max | deciding QoI | Verdict |
|---|---|---|---|---|---|---|---|---|
| `X` | 1 | 0.789 | 1 | 1.25 | 0.497 | 0.741 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `mu_eff` | 1 | 0.484 | 1 | 2.75 | 1.27 | 0.842 | `SCOF_mean`, `h_r`, `h_p` | **RETAIN** |
| `w` | 0.43 | 0.316 | 2 | 2.75 | 0.894 | 0.253 | `F_n`, `h_r`, `h_p` | **RETAIN** |
| `beta` | 0.393 | 0.252 | 3 | 3.75 | 0.829 | 0.2 | `F_n`, `h_p` | **RETAIN** |
| `K` | 0.311 | 0.125 | 5 | 6 | 1.35 | 0.132 | `h_p` | **RETAIN** |
| `h` | 0.24 | 0.137 | 4 | 5.5 | 1.08 | 0.124 | `F_n` | **RETAIN** |
| `eps_c` | 0.107 | 0.0699 | 6 | 7 | 1.52 | - | - | **freeze** |
| `phi` | 0.0692 | 0.0496 | 5 | 7 | 1.34 | - | - | **freeze** |

![Consolidated ranking](consolidated.png)

![Normalised mu* map](heatmap.png)

**Reading.** A `sigma/mu*` above 1.0 flags a factor whose effect is dominated by interactions or by strong non-linearity: Morris does not distinguish the two, only Sobol will. A large gap between the best rank and the mean rank flags a factor specific to one QoI.

## 2. Detail per QoI

### `F_n` -- Normal force (half-model) [N]

65 elementary effects, 9 missing point(s). Retention threshold: mu* >= 0.6161 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 3.08 | 2.178 | 3.973 | 1.325 | 0.43 | 0.707 | 3.08 | 9 | RETAIN |
| `w` | 1.325 | 0.7191 | 1.968 | 0.9618 | 0.726 | 0.233 | 1.325 | 10 | RETAIN |
| `beta` | 1.13 | 0.5231 | 1.769 | 0.8724 | 0.772 | 0.17 | 1.119 | 8 | RETAIN? |
| `h` | 0.7402 | 0.3815 | 1.044 | 0.3834 | 0.518 | 0.124 | 0.7402 | 6 | RETAIN? |
| `mu_eff` | 0.4868 | 0.2863 | 0.7012 | 0.4269 | 0.877 | 0.0929 | -0.3795 | 8 | freeze |
| `eps_c` | 0.2537 | 0.118 | 0.435 | 0.3458 | 1.36 | 0.0383 | 0.02032 | 8 | freeze |
| `K` | 0.2181 | 0.08387 | 0.3827 | 0.2266 | 1.04 | 0.0272 | -0.2034 | 9 | freeze |
| `phi` | 0.07304 | 0.02552 | 0.1337 | 0.09159 | 1.25 | 0.00829 | -0.04545 | 7 | freeze |

![Morris F_n](morris_F_n.png)

### `SCOF_mean` -- Apparent friction coefficient [-]

65 elementary effects, 9 missing point(s). Retention threshold: mu* >= 0.1284 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `mu_eff` | 0.6419 | 0.5407 | 0.7902 | 0.1756 | 0.274 | 0.842 | 0.6419 | 8 | RETAIN |
| `X` | 0.1005 | 0.06309 | 0.1302 | 0.04999 | 0.497 | 0.0983 | 0.09929 | 9 | freeze |
| `w` | 0.07665 | 0.0431 | 0.1175 | 0.05649 | 0.737 | 0.0672 | -0.07665 | 10 | freeze |
| `beta` | 0.07455 | 0.04332 | 0.1097 | 0.04381 | 0.588 | 0.0675 | -0.07455 | 8 | freeze |
| `phi` | 0.03295 | 0.01665 | 0.05978 | 0.04399 | 1.34 | 0.0259 | -0.009694 | 7 | freeze |
| `K` | 0.02685 | 0.01011 | 0.04708 | 0.02587 | 0.964 | 0.0157 | -0.02685 | 9 | freeze |
| `eps_c` | 0.01502 | 0.004313 | 0.02834 | 0.02283 | 1.52 | 0.00672 | 0.0008344 | 8 | freeze |
| `h` | 0.006268 | 0.001661 | 0.01211 | 0.006796 | 1.08 | 0.00259 | -0.005359 | 6 | freeze |

![Morris SCOF_mean](morris_SCOF_mean.png)

### `h_r` -- Residual depth [mm]

65 elementary effects, 9 missing point(s). Retention threshold: mu* >= 0.002644 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01322 | 0.009798 | 0.01728 | 0.005372 | 0.406 | 0.741 | 0.01322 | 9 | RETAIN |
| `w` | 0.004386 | 0.00217 | 0.006854 | 0.003441 | 0.784 | 0.164 | -0.004386 | 10 | RETAIN? |
| `mu_eff` | 0.004144 | 0.001732 | 0.006627 | 0.00323 | 0.78 | 0.131 | 0.004144 | 8 | RETAIN? |
| `h` | 0.001855 | 0.0009965 | 0.002695 | 0.0009659 | 0.521 | 0.0754 | -0.001855 | 6 | freeze |
| `beta` | 0.00177 | 0.0009753 | 0.002917 | 0.001467 | 0.829 | 0.0738 | -0.00168 | 8 | freeze |
| `K` | 0.001024 | 0.0003295 | 0.001992 | 0.001385 | 1.35 | 0.0249 | 0.0007818 | 9 | freeze |
| `phi` | 0.0009143 | 0.0002106 | 0.001784 | 0.0009938 | 1.09 | 0.0159 | 0.0009143 | 7 | freeze |
| `eps_c` | 0.0008866 | 0.0004941 | 0.001267 | 0.001066 | 1.2 | 0.0374 | -0.0001365 | 8 | freeze |

![Morris h_r](morris_h_r.png)

### `h_p` -- Lateral pile-up [mm]

65 elementary effects, 9 missing point(s). Retention threshold: mu* >= 0.002011 (20% of the maximal mu* for this QoI).

| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `X` | 0.01005 | 0.006853 | 0.01353 | 0.004628 | 0.46 | 0.682 | 0.01005 | 9 | RETAIN |
| `mu_eff` | 0.00466 | 0.00225 | 0.008033 | 0.005924 | 1.27 | 0.224 | -0.002007 | 8 | RETAIN |
| `beta` | 0.003953 | 0.00201 | 0.005812 | 0.002487 | 0.629 | 0.2 | -0.003953 | 8 | RETAIN? |
| `w` | 0.003854 | 0.002543 | 0.005562 | 0.003446 | 0.894 | 0.253 | -0.002944 | 10 | RETAIN |
| `K` | 0.003124 | 0.00133 | 0.005119 | 0.002837 | 0.908 | 0.132 | 0.002986 | 9 | RETAIN? |
| `h` | 0.001591 | 0.0007361 | 0.002475 | 0.001698 | 1.07 | 0.0732 | -0.0009726 | 6 | freeze |
| `eps_c` | 0.001073 | 0.0003776 | 0.00186 | 0.001379 | 1.28 | 0.0376 | -0.000577 | 8 | freeze |
| `phi` | 0.0005438 | 0.0001005 | 0.00114 | 0.0007009 | 1.29 | 0.00999 | 0.0005096 | 7 | freeze |

![Morris h_p](morris_h_p.png)

## 3. Decision

**Retained (6):** `X`, `mu_eff`, `w`, `beta`, `K`, `h`

**Retained but marginal (0):** none

**Frozen (2):** `eps_c`, `phi`

> **The threshold is RELATIVE.** It compares each factor to the most influential one of the same QoI, it does not test against zero. Three consequences: the top factor is retained by construction; a `freeze` means *small compared to the largest*, **not** *null*; and if every factor had a real effect of the same order, the rule would freeze none of them. It reduces dimensionality, it does not prove any nullity.

### Sobol

```bash
python3 generate_design.py glassy_pc --method sobol --n 1024 \
        --only X,mu_eff,w,beta,K,h
```

Marginal factors are included as a precaution. Unlisted factors are frozen at mid-range. Going from 8 to 6 factors is what makes 1024 points enough.

## 3bis. Identifiability

### Confounding signature between factors

No pair above the threshold 0.55: no confounding signature detected.

| Pair | index | QoI |
|---|---|---|
| `X` / `w` | 0.344 | 4 |
| `beta` / `K` | 0.338 | 4 |
| `w` / `mu_eff` | 0.338 | 4 |
| `X` / `beta` | 0.313 | 4 |
| `h` / `K` | 0.248 | 4 |
| `h` / `mu_eff` | 0.242 | 4 |
| `w` / `K` | 0.231 | 4 |
| `beta` / `mu_eff` | 0.214 | 4 |

## 3ter. Numerical quality indicators

These are **indicators only**. No run is excluded on any of them: the energy diagnostics are reported so that a systematic drift stays visible, and every parsed run feeds the elementary effects regardless of the values below. On a deliberately coarse mesh, exceeding these limits is expected and is not a reason to discard the run.

| Indicator | Limit | Over limit | Median | p90 | Max | Worst ids |
|---|---|---|---|---|---|---|
| `KE_IE_steady_max` | 5 % | 0 / 81 (0%) | 0.0937 | 1.41 | 4.03 | - |
| `AE_IE_final` | 5 % | 4 / 81 (4.94%) | 1.46 | 3.1 | 7.49 | 00078, 00028, 00068, 00066 |
| `ETOTAL_drift` | 5 % | 10 / 81 (12.3%) | 2.11 | 6.1 | 9.67 | 00057, 00056, 00011, 00058, 00075 |
| `ALLPW` | 5 % | 2 / 81 (2.47%) | 2.59 | 3.87 | 5.41 | 00075, 00077 |
| `KE_final_over_IE_peak` | 1 % | 0 / 81 (0%) | 0.00376 | 0.00892 | 0.0199 | - |

**How to read this.** An indicator exceeded by nearly EVERY run points at a systematic setting shared by the whole campaign (element control, mass scaling, ALE frequency, mesh size), not at individual runs. An indicator exceeded by a HANDFUL of runs points at a corner of the factor box; cross the worst ids above with the design to find which factor level they share. In both cases the action is to change the setting or to document the bias, never to drop the runs.

## 4. Caveats

- The ranking holds for the **Drucker-Prager class**, not for any single family: `semicrystalline_*` and `glassy_*` are points of the same dimensionless box. No `mu*` specific to PMMA or PC comes out of it.
- The unified box spans different physical regimes (softening present or absent). A high `sigma` may reflect this mixture rather than an interaction.
- `h` enters via `exp(h*eps^2)` evaluated up to eps_max: strong non-linearity on this factor is expected by construction of the model.
- `phi` is capped below 1 in the sampling box precisely because `phi = 1` switches the friction model from a tabulated Briscoe table to constant Coulomb -- a change of model class, not a variation of a factor. If a design predating that cap is analysed here, the `mu*` of `phi` mixes the two.
- Quality indicators (section 3ter) never exclude a run. A campaign run on a coarse mesh for turnaround will exceed them by construction; that is a known bias to report, not a reason to shrink the design.
