---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    kernelspec:
      display_name: Python 3
      language: python
      name: python3
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
---

```python
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,redefined-outer-name,trailing-whitespace,too-many-locals,too-many-statements,too-many-arguments,too-many-branches,too-many-positional-arguments,wrong-import-order,wrong-import-position
```

# Previous Methods vs KoMbine

This notebook compares two published recipes for discrete covariate misclassification — Yi's probability weights and Küchenhoff MC-SIMEX — to KoMbine.

KoMbine's reported uncertainty depends on whether group assignments can move:

- **Fixed card:** assignments are determined by the data. Kaplan–Meier bands are the binomial profile, and the hazard-ratio interval is a Wilks $\chi^2_1$ cut (`method="chi2"`).
- **Discrete and Poisson cards:** assignments are free. Hazard-ratio intervals are toy-calibrated Neyman inversions of the profile LRT ($B=19$, 68% and 95%). Kaplan–Meier bands are a decreasing binomial fit to an $S$-grid of toys (11 grid points, no early stop). The $\chi^2$ profile is only the bracket that starts that search, not the reported interval.

**Two families** (n=20 always; n=50 unless CI skips it):

- **n=20** (`*_hr_example*`, discrete $e = 0.20$, $0.25$, $0.40$). Yi, MC-SIMEX, and permutation p-values are minutes-scale unless that scenario is already in the cache. Toy HR intervals and KM bands are a long local run.
- **n=50** (`methods_comparison_*`, regenerated into a gitignored `rebinned/` dir, **4** quantile-bin death-time medians, discrete $e = 0.05$, $0.10$, $0.25$). The same toy calls run after n=20. That pass takes many hours. Default (unset env) runs this family **after** n=20.

**CI.** `KOMBINE_QUICK_COMPARISON=1` skips the n=50 family. `KOMBINE_SKIP_TOY_CALIBRATION=1` skips toy solves and does not draw a $\chi^2$ band in their place; it does not skip p-values. A cache file, if present, is still used: toy KM and HR when `N_MAX` matches, and permutation p-values when `N_PERMUTATIONS` matches.

Fixed and Poisson cards are split at `0.5001` (a density/value cut). Discrete-class cards are split at `1` (the boundary between class indices 0 and 1):
- Fixed Hazard Ratio (deterministic, no measurement error)
- Discrete classes with class probabilities (small/medium/large uncertainty)
- Poisson density with large counts (small relative error)
- Poisson density with moderate counts
- Poisson density with small counts (high relative error)

All three methods use the same measurement model (`observable.probability_in_range`). They differ in what they do with those probabilities.


## Method Overview and Comparison

| Aspect | Yi's Method | MC-SIMEX | KoMbine |
|--------|---|---|---|
| **Core idea** | Weighted KM/logrank using probabilistic group membership | Extra flips of hard labels, then extrapolate the naive estimator to zero error | Full likelihood with explicit group assignment variables |
| **Optimization** | 1-D scalar min of the weighted Breslow 2NLL | Monte Carlo average at each $\lambda$, quadratic fit | Mixed Integer Nonlinear Programming (Gurobi) |
| **Computational cost** | Low | Low–medium | Medium-high (~1 h for the n=50 family) |
| **Accuracy (within model)** | Approximate to the full likelihood | Approximate (simulation + extrapolation) | Exact maximizer within solver tolerance |
| **Uncertainty** | Likelihood-ratio interval of the weighted Breslow 2NLL (no KM bands here) | Sampling CI of the extrapolated number (Wald for HR) | Toy-calibrated interval when assignments are free; Wilks $\chi^2$ when they are fixed |
| **Core assumptions** | Known measurement error distribution; independent errors; fractional group membership is an adequate proxy for uncertain assignment | Known measurement error distribution; independent errors; quadratic extrapolation of the naive hard-label estimator is adequate | Known measurement error distribution; independent errors; patients belong to one group; event times treated as observed and discrete; likelihood model is correctly specified |

### How the three recipes use the same $e_i$
- **Yi averages with weights.** Each patient contributes to both groups in proportion to $P(G\mid\text{data})$.
- **MC-SIMEX extrapolates a naive hard-label estimator.** The observed label $G^*_i$ is flipped with Küchenhoff probability $\bigl(1-(1-2e_i)^{\lambda}\bigr)/2$, the usual KM/logrank/Cox estimator is averaged over simulations, and that curve is fit vs $\lambda$ and evaluated at $\lambda=-1$.
- **KoMbine profiles assignments.** A binary assignment is chosen for each patient, scored by the measurement model, and confidence sets come from the profile likelihood.

### How Yi's Method Works
- Convert each patient's observed biomarker value into a probability of being below or above the threshold using the measurement error model.
- Instead of a global misclassification matrix as Yi describes, we extend her method to compute these probabilities on a per-patient basis (allowing uncertainty to vary by individual measurement).
- Use those per-patient probabilities as weights in the Kaplan-Meier estimator and logrank test.
- Every patient contributes to both groups, in proportion to their probability of belonging there.
- This yields fast, smooth estimates that tend to shrink group differences as measurement uncertainty grows.
- It is an approximation because it does not enforce a single, discrete group assignment for each patient.

### How MC-SIMEX Works
- Keep the same $e_i=1-P(G=G^*_i\mid\text{data})$ that Yi uses, but then discard the probabilities and work with hard labels only.
- At $\lambda=0$ the estimator is the usual naive KM / logrank / Cox fit on $G^*$.
- At $\lambda>0$ additional flips are simulated; averaging the naive estimator traces how it changes as misclassification grows.
- A quadratic in $\lambda$ is extrapolated to $\lambda=-1$ (no misclassification).
- The HR interval is a Wald interval of that extrapolated $\log H$, not a likelihood-ratio interval. The $\Delta$2NLL curve plotted below is the Wald quadratic $(\log H-\widehat{\log H})^2/\hat\sigma^2$.

### How KoMbine Works
- Introduce a binary assignment variable for each patient (low vs high group).
- Combine the survival likelihood with a measurement error penalty that scores how plausible each assignment is.
- Solve a constrained optimization problem that finds the most likely set of assignments and survival parameters together.
- Report a toy-calibrated confidence set when assignments are free. A Wilks $\chi^2$ cut is used only when assignments are fixed, and as the bracket that starts the toy search.
- This is exact for the specified likelihood model but requires heavier computation than Yi or MC-SIMEX.

### Key Comparisons
1. **Kaplan-Meier Curves**: Visual comparison of survival estimates
2. **P-Values**: Logrank test results and statistical significance
3. **Hazard Ratios**: Point estimates and confidence intervals


## Discrete Classes with Class Probabilities

Yi's Section 3.7.1 treats misclassification of a discrete covariate using a
misclassification matrix. We mimic that structure using two classes and a
homogeneous binary matrix,
$$
\Pi = \begin{pmatrix} 1 - e & e \\ e & 1 - e \end{pmatrix}
$$
where $e$ is the misclassification rate shared by all patients.

Each family uses three error levels:

- n=20 (`*_hr_example*`, always): $e = 0.20$, $0.25$, $0.40$
- n=50 (`methods_comparison_*`, default only): $e = 0.05$, $0.10$, $0.25$

`KOMBINE_QUICK_COMPARISON=1` skips the n=50 family.

Each patient keeps the same survival time and censoring as that family's fixed baseline.
Only the class probabilities change: patients in the low group get probabilities
$(1-e, e)$, and patients in the high group get $(e, 1-e)$.

MC-SIMEX uses the same $e$ as the per-patient flip rate.

```python
import pathlib
import sys

# Repo root on path so docs.kombine imports work under nbconvert (cwd is docs/kombine).
_repo_root = pathlib.Path(".").resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from docs.kombine.previous_methods_comparison import (
    CACHE_PATH,
    LOAD_TOY_CACHE,
    N_MAX,
    N_S_GRID,
    QUICK_COMPARISON,
    SCENARIOS_FULL,
    SCENARIOS_QUICK,
    SKIP_TOY,
    TITLE_FULL,
    TITLE_QUICK,
    TOY_CACHE,
    _refit_cached_km,
    load_datacards,
    load_toy_cache,
    plot_hr_mosaic,
    plot_km_mosaic,
    plot_monotone_vs_naive,
    plot_pvalue_bars,
    print_summary_table,
    progress,
    run_hr_analysis,
    run_km_analysis,
    run_pvalue_analysis,
    save_toy_cache,
    show_chi2_vs_toys,
    skip_n50,
)
```

```python
here = pathlib.Path(".").resolve()
test_dir = here.parent.parent / "test" / "kombine"
datacards_dir_quick = test_dir / "datacards" / "simple_examples"

if QUICK_COMPARISON:
    mode_name = "n=20 only (KOMBINE_QUICK_COMPARISON=1; n=50 cells will skip)"
else:
    mode_name = "n=20 then n=50 (default)"

if LOAD_TOY_CACHE and load_toy_cache():
    n_km = _refit_cached_km(TOY_CACHE["n20"]["km"]) + _refit_cached_km(TOY_CACHE["n50"]["km"])
    if n_km:
        save_toy_cache()
    progress(
        f"Loaded toy cache {CACHE_PATH.name}: "
        f"n20 HR {len(TOY_CACHE['n20']['hr'])} / KM {len(TOY_CACHE['n20']['km'])} / p {len(TOY_CACHE['n20']['pv'])}, "
        f"n50 HR {len(TOY_CACHE['n50']['hr'])} / KM {len(TOY_CACHE['n50']['km'])} / p {len(TOY_CACHE['n50']['pv'])}"
        + (f"; re-fit {n_km} KM arms" if n_km else "")
    )
elif LOAD_TOY_CACHE:
    progress(f"No toy cache at {CACHE_PATH}")

progress(f"Notebook 07 mode: {mode_name}")
progress(
    "Toy calibration: "
    + ("skipped (KOMBINE_SKIP_TOY_CALIBRATION=1)" if SKIP_TOY else f"B={N_MAX}, S-grid={N_S_GRID}")
)
progress(f"Loading n=20 hr_example cards ({TITLE_QUICK})")
datacards_quick = load_datacards(datacards_dir_quick, SCENARIOS_QUICK)
```

## Analysis 1: Kaplan-Meier Curves

Compare the Kaplan-Meier survival curves between Yi's method (dashed), MC-SIMEX (dotted), and KoMbine (solid lines with shaded 95% intervals) across all scenarios. Yi and MC-SIMEX are point estimates only. KoMbine shades a toy band when assignments are free, and a binomial $\chi^2$ band on the fixed card.

### n=20 (`*_hr_example*`)

```python
progress(f"Analysis 1 — {TITLE_QUICK}")
km_results_quick = run_km_analysis(SCENARIOS_QUICK, datacards_quick, "n20")
plot_km_mosaic(
    SCENARIOS_QUICK,
    km_results_quick,
    f"Kaplan-Meier Curves: Yi, MC-SIMEX, and KoMbine ({TITLE_QUICK})",
)
plot_monotone_vs_naive(
    SCENARIOS_QUICK,
    km_results_quick,
    f"KoMbine KM: monotone toy fit vs naive per-time intervals ({TITLE_QUICK})",
)
```

### Understanding the Small Counts Anomaly: Why the High Group Can Look Better

In the small-counts scenario, KoMbine’s best-fit *KM curves* can show the “high” group outperforming the “low” group. This is not necessarily a bug: it reflects how a **discrete-assignment likelihood** behaves when measurement error is large and many patients are effectively borderline relative to the threshold.

**What is happening conceptually**
- With high uncertainty, several patients have substantial probability mass on both sides of the threshold.
- KoMbine must choose a single group per patient (within each optimization problem) and will pick the assignment(s) that maximize the likelihood.
- If early events are borderline, the likelihood can increase by assigning them to the group that best explains the observed survival times, which can make the *apparent* group ordering flip.

**A subtle but important detail**
- In this notebook, the two KoMbine KM curves are fit **separately** (one KoMbine optimization for the low range and one for the high range). With large uncertainty, the best-fit assignments for those two separate optimizations need not form a perfectly complementary partition of patients.
- The KoMbine **hazard ratio / p-value** calculations, in contrast, are *joint* two-group fits. Those joint fits are the self-consistent way to summarize “which group is worse” under the KoMbine model.

**Why Yi and MC-SIMEX look different**
- Yi’s method does not pick a single group; it spreads each borderline patient across both groups using probabilistic weights, which tends to shrink the group contrast as $e_i$ grows.
- MC-SIMEX keeps hard labels, adds extra flips, and extrapolates the naive curve; that can move the point estimate *away* from the naive (attenuated) curve rather than averaging the two groups.
- Neither method is a profile over discrete assignments, so both can disagree with KoMbine when membership is weakly identified.

**Takeaway**
When measurement error is large, probabilistic weights (Yi), extrapolated hard-label estimators (MC-SIMEX), and discrete assignment (KoMbine) can tell qualitatively different stories. The right question is not “which is correct?”, but “how sensitive are the conclusions to how uncertain group membership is modeled?”


## Analysis 2: P-Values (Logrank / Likelihood-Ratio Test)

We compare p-values from:

- **Yi**: a *weighted* logrank-style calculation using per-patient probabilistic group membership (fractional membership).
- **MC-SIMEX**: the usual logrank statistic on extra-flipped hard labels, averaged vs $\lambda$ and extrapolated to $\lambda=-1$, then converted to a $\chi^2_1$ p-value.
- **KoMbine**: a *permutation* LRT of HR $=1$ using the full model (**`cox_only=False`**). Assignments are profiled on the observed data and on each shuffle of `(time, censored)`, so the null has the same reassignment freedom as the alternative.

On the **fixed** card, Yi and MC-SIMEX therefore report the same hard-label logrank $\chi^2$ $p$, while KoMbine’s $p$ is a permutation LRT (here $B=99$) of the Cox alternative — they need not match even with zero measurement error.

The three p-values for each scenario are written to the same cache as the toy results. A later run reprints them and does not permute again while `N_PERMUTATIONS` is unchanged.

**What to expect**

- As measurement error grows, **Yi's p-values typically increase** because fractional membership blurs the difference between groups.
- **MC-SIMEX p-values** are those of an extrapolated hard-label statistic; they need not match Yi even though both start from the same $e_i$.
- **KoMbine's permutation p-values** do not get smaller just because assignments become cheaper: the null can re-label too.
- Large disagreements among the three p-values indicate that inference is being driven by how group-membership uncertainty is modeled, not just by sampling noise.

### n=20 (`*_hr_example*`)

```python
progress(f"Analysis 2 — {TITLE_QUICK}")
pvalue_results_quick = run_pvalue_analysis(SCENARIOS_QUICK, datacards_quick, "n20")
plot_pvalue_bars(
    SCENARIOS_QUICK,
    pvalue_results_quick,
    f"P-values: Yi / MC-SIMEX logrank χ² vs KoMbine permutation LRT ({TITLE_QUICK})",
)
```

## Analysis 3: Hazard Ratios

We compare hazard ratios estimated using:

- **Yi**: the weighted Breslow partial likelihood. The table reports a **continuous** MLE in $\log H$ and a likelihood-ratio interval. The curve is that 2NLL on an `N_HR_SCAN`-point log grid, recentered at the continuous MLE.
- **MC-SIMEX**: extrapolated $\widehat{\log H}$ with a Wald CI. The curve below is the Wald quadratic, **not** a profile likelihood. On the **fixed** panel membership is exact and the point HR matches Yi/KoMbine, but the purple scan is still Wald, so it will not overlay the Breslow profiles.
- **KoMbine**: the full model (`cox_only=False`). The curve below is the profile $\Delta$2NLL. On the fixed card the reported interval is a Wilks cut. On the other cards it is the toy-calibrated 95% interval (with the 68% interval shaded inside it). The horizontal $\chi^2=3.84$ line is not KoMbine's cutoff.

### Why the confidence intervals behave differently

As noted in the paper text, Yi’s approach can reduce bias in the *point estimate* of the hazard ratio compared to ignoring measurement error, but the **confidence interval does not necessarily reflect loss of identifiability** when measurement error becomes very large. In the extreme-uncertainty limit we would expect the data to place almost no constraint on the hazard ratio, but Yi-style weighting does not automatically produce that behavior.

MC-SIMEX is in the same family: the Wald interval is the sampling interval of the extrapolated number. It stays finite even when labels are nearly uninformative.

KoMbine's toy interval inverts the same profile LRT used for the permutation test, so the reported set accounts for assignment search. The profile curve is the objective being calibrated, not itself the interval.

### n=20 (`*_hr_example*`)

```python
progress(f"Analysis 3 — {TITLE_QUICK}")
hr_results_quick = run_hr_analysis(SCENARIOS_QUICK, datacards_quick, "n20")
plot_hr_mosaic(
    SCENARIOS_QUICK,
    hr_results_quick,
    f"Hazard Ratio: Yi and KoMbine profiles vs MC-SIMEX Wald ({TITLE_QUICK})",
)
```


## n=50 family (`methods_comparison_*`)

Default (unset `KOMBINE_QUICK_COMPARISON`) repeats Analyses 1–3 on the n=50 cards, including toy HR intervals and monotone KM bands. That is a many-hour run. CI sets the env var, so these cells print a skip message instead of solving.

```python
datacards_full = None
km_results_full = None
pvalue_results_full = None
hr_results_full = None

if QUICK_COMPARISON:
    skip_n50("datacard load")
else:
    from docs.kombine.rebin_methods_comparison_times import ensure_rebinned
    datacards_dir_full = ensure_rebinned(n_bins=4)
    progress(f"Loading n=50 rebinned methods_comparison cards ({TITLE_FULL})")
    datacards_full = load_datacards(datacards_dir_full, SCENARIOS_FULL)
```

### Analysis 1 (n=50 KM)

```python
if QUICK_COMPARISON:
    skip_n50("KM analysis")
else:
    progress(f"Analysis 1 — {TITLE_FULL}")
    km_results_full = run_km_analysis(SCENARIOS_FULL, datacards_full, "n50")
    plot_km_mosaic(
        SCENARIOS_FULL,
        km_results_full,
        f"Kaplan-Meier Curves: Yi, MC-SIMEX, and KoMbine ({TITLE_FULL})",
    )
    plot_monotone_vs_naive(
        SCENARIOS_FULL,
        km_results_full,
        f"KoMbine KM: monotone toy fit vs naive per-time intervals ({TITLE_FULL})",
    )
```

### Analysis 2 (n=50 p-values)

```python
if QUICK_COMPARISON:
    skip_n50("p-value analysis")
else:
    progress(f"Analysis 2 — {TITLE_FULL}")
    pvalue_results_full = run_pvalue_analysis(SCENARIOS_FULL, datacards_full, "n50")
    plot_pvalue_bars(
        SCENARIOS_FULL,
        pvalue_results_full,
        f"P-values: Yi / MC-SIMEX logrank χ² vs KoMbine permutation LRT ({TITLE_FULL})",
    )
```

### Analysis 3 (n=50 hazard ratios)

```python
if QUICK_COMPARISON:
    skip_n50("HR analysis")
else:
    progress(f"Analysis 3 — {TITLE_FULL}")
    hr_results_full = run_hr_analysis(SCENARIOS_FULL, datacards_full, "n50")
    plot_hr_mosaic(
        SCENARIOS_FULL,
        hr_results_full,
        f"Hazard Ratio: Yi and KoMbine Profiles vs MC-SIMEX Wald ({TITLE_FULL})",
    )
```


## χ²₁ cut vs toy intervals

Both intervals use the cached full profile. Patient-wise assignments are free (`cox_only=False`): the scan is `kombine_2nlls` from `compute_2nll_at_hazard_ratio`. The horizontal lines are $\chi^2_1(0.68)$ and $\chi^2_1(0.95)$. The $\chi^2$ interval in the table is the cut main reports with `method="chi2"`: the roots of $\Delta$2NLL = those lines on either side of the scan minimum, or the search bound when that side never crosses. It is read off the 25-point grid. It is not a locked-assignment interval and not a Thomas–Grunkemeier interval.

The shaded bands are the cached toy 68% and 95% intervals of that same profile likelihood-ratio test ($B=19$). The fixed card has no toys, because assignments cannot move there. If the grid has another stretch below a $\chi^2$ line that does not contain the minimum, the table says so and the figure marks it. That extra stretch is not drawn as part of one bar. The reported $\chi^2$ interval is still the one around the minimum.

Read down each family in order of measurement error. Where that error is small, the toy interval and this $\chi^2_1$ cut stay close, to the resolution of $B=19$ and the grid. Where the error is large, they separate. This cell only reads the cache.

```python
show_chi2_vs_toys(
    SCENARIOS_QUICK,
    "n20",
    f"Toy interval vs chi2_1 cut, assignments free ({TITLE_QUICK})",
)
show_chi2_vs_toys(
    SCENARIOS_FULL,
    "n50",
    f"Toy interval vs chi2_1 cut, assignments free ({TITLE_FULL})",
)
```


## Summary of Findings

The key modeling difference is **fractional weights vs extrapolated hard labels vs discrete assignment** under measurement uncertainty:
Yi’s method assigns each patient to both groups with weights, MC-SIMEX extrapolates a naive hard-label estimator, and KoMbine enforces one group per
patient and scores assignments using an explicit measurement-error model.

The two card families use different measurement-error ladders. KoMbine's intervals on the discrete and Poisson cards are toy-calibrated; the fixed card uses a $\chi^2$ cut because assignments do not move. The section above reads each cached full profile, assignments free, and compares a $\chi^2_1$ cut of that scan to the cached toy interval.

### Kaplan–Meier Curves (Qualitative)
- **Fixed observable** (both families): Yi, MC-SIMEX, and KoMbine produce the same curves because group membership is exact. KoMbine's band on this card is the binomial profile.
- **Discrete classes and Poisson counts:** Yi weights patients into both groups, MC-SIMEX extrapolates a hard-label curve, and KoMbine reports a monotone toy band. Those three can disagree when membership is weakly identified, including apparent reversals in the separate KM fits at large error.
- The monotone-vs-naive panels reuse the stored $S$-grid. They do not launch new toys. The fixed panel has no toy grid.

### P-Values and Hazard Ratios
- Yi’s p-values generally increase as uncertainty grows; its best-fit HR drifts toward 1.
  The printed Yi HR and CI are a continuous Breslow profile, not a grid argmin.
- MC-SIMEX p-values and HRs are those of an extrapolated hard-label statistic; the Wald HR interval stays finite.
  The plotted MC-SIMEX curve is that Wald quadratic even when $e_i=0$.
- KoMbine’s plotted $p$ is a permutation LRT with $B=99$.
  On discrete and Poisson cards the shaded HR interval is the toy 95% set (68% inside it). The curve is the profile $\Delta$2NLL, not a $\chi^2$ cutoff.
- Large disagreements among the three indicate that inference is driven by how group-membership
  uncertainty is modeled, not just by sampling noise.

### Runtime
| Family | What runs | When it is skipped |
|---|---|---|
| n=20 | Yi, MC-SIMEX, permutation p-values | Cache hit for this `N_PERMUTATIONS`; otherwise run, including CI |
| n=20 | Toy HR intervals and monotone KM bands | `KOMBINE_SKIP_TOY_CALIBRATION=1` (unless a cache is loaded) |
| n=50 | Same three analyses, including toys | `KOMBINE_QUICK_COMPARISON=1` |

Toy KM and HR on n=50 are a many-hour local run. Toy KM, toy HR, and permutation p-values are written to gitignored `_toy_calibration_cache/methods_comparison_toys.json` after each scenario.

### Practical Takeaways
1. When measurement error is small, the toy hazard-ratio interval and a $\chi^2_1$ cut of
   the same profile agree. Both use the full likelihood, with patient-wise assignments
   free. The close cases are the small-error cards: n=50 discrete $e=0.05$ and $0.10$,
   and the large-count Poisson cards. Larger error separates the two intervals. That
   $\chi^2$ cut is the one main reports (`method="chi2"`). KoMbine’s $p$ is still a
   permutation LRT. The HR scan in Analysis 3 also shows Yi’s profile and MC-SIMEX’s
   Wald quadratic.
2. When measurement error is moderate/large, treat the conclusion as model-dependent and
   report sensitivity to the modeling choice.
3. Yi’s method is fast and often conservative (it blurs separation as uncertainty grows).
4. MC-SIMEX is also fast; its Wald interval is a sampling interval of the extrapolated number.
5. KoMbine reports a toy-calibrated interval when assignments are free, using the same
   profile LRT as the permutation test.

```python
progress(f"Summary table — {TITLE_QUICK}")
print_summary_table(SCENARIOS_QUICK, pvalue_results_quick, hr_results_quick)

if QUICK_COMPARISON:
    skip_n50("summary table")
else:
    progress(f"Summary table — {TITLE_FULL}")
    print_summary_table(SCENARIOS_FULL, pvalue_results_full, hr_results_full)
```
