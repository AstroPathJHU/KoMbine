---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: rocpicker
    language: python
    name: python3
---

```python
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,redefined-outer-name,trailing-whitespace,too-many-locals,wrong-import-order,wrong-import-position
```

# Previous Methods vs KoMbine: Yi, MC-SIMEX, and Profile Likelihood

This notebook compares two published recipes for discrete covariate misclassification — Yi's probability weights and Küchenhoff MC-SIMEX — to KoMbine's profile likelihood over group assignments.

**Two modes** (chosen at runtime):

- **Default / CI** (unset env): `*_hr_example*` cards, $n=20$, ~7 distinct death times — finishes in minutes with real KM bands, p-values, and HR profiles.
- **Full local comparison** (`KOMBINE_FULL_COMPARISON=1`): regenerates `methods_comparison_*` cards on the fly into a gitignored `rebinned/` dir ($n=50$, **4** quantile-bin death-time medians; hours-scale; stronger permutation power).

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
| **Computational cost** | Low | Low–medium | Medium-high |
| **Accuracy (within model)** | Approximate to the full likelihood | Approximate (simulation + extrapolation) | Exact maximizer within solver tolerance |
| **Uncertainty** | Likelihood-ratio interval of the weighted Breslow 2NLL (no KM bands here) | Sampling CI of the extrapolated number (Wald for HR) | Profile likelihood |
| **Core assumptions** | Known measurement error distribution; independent errors; fractional group membership is an adequate proxy for uncertain assignment | Known measurement error distribution; independent errors; quadratic extrapolation of the naive hard-label estimator is adequate | Known measurement error distribution; independent errors; patients belong to one group; event times treated as observed and discrete; likelihood model is correctly specified |

### How the three recipes use the same $e_i$
- **Yi averages with weights.** Each patient contributes to both groups in proportion to $P(G\mid\text{data})$.
- **MC-SIMEX extrapolates a naive hard-label estimator.** The observed label $G^*_i$ is flipped with Küchenhoff probability $\bigl(1-(1-2e_i)^{\lambda}\bigr)/2$, the usual KM/logrank/Cox estimator is averaged over simulations, and that curve is fit vs $\lambda$ and evaluated at $\lambda=-1$.
- **KoMbine profiles assignments.** A binary assignment is chosen for each patient, scored by the measurement model, and confidence sets come from the profile likelihood.

### How Yi's Method Works (Intuition)
- Convert each patient's observed biomarker value into a probability of being below or above the threshold using the measurement error model.
- Instead of a global misclassification matrix as Yi describes, we extend her method to compute these probabilities on a per-patient basis (allowing uncertainty to vary by individual measurement).
- Use those per-patient probabilities as weights in the Kaplan-Meier estimator and logrank test.
- Every patient contributes to both groups, in proportion to their probability of belonging there.
- This yields fast, smooth estimates that tend to shrink group differences as measurement uncertainty grows.
- It is an approximation because it does not enforce a single, discrete group assignment for each patient.

### How MC-SIMEX Works (Intuition)
- Keep the same $e_i=1-P(G=G^*_i\mid\text{data})$ that Yi uses, but then discard the probabilities and work with hard labels only.
- At $\lambda=0$ the estimator is the usual naive KM / logrank / Cox fit on $G^*$.
- At $\lambda>0$ additional flips are simulated; averaging the naive estimator traces how it changes as misclassification grows.
- A quadratic in $\lambda$ is extrapolated to $\lambda=-1$ (no misclassification).
- The HR interval is a Wald interval of that extrapolated $\log H$, not a likelihood-ratio interval. The $\Delta$2NLL curve plotted below is the Wald quadratic $(\log H-\widehat{\log H})^2/\hat\sigma^2$.

### How KoMbine Works (Intuition)
- Introduce a binary assignment variable for each patient (low vs high group).
- Combine the survival likelihood with a measurement error penalty that scores how plausible each assignment is.
- Solve a constrained optimization problem that finds the most likely set of assignments and survival parameters together.
- Compute confidence intervals via profile likelihood, which naturally widens as uncertainty increases.
- This is exact for the specified likelihood model but requires heavier computation.

### What the Assumptions Mean (Plain Language)
- **Known measurement error distribution**: You have a reasonable model for how observed biomarker values deviate from the true value (e.g., Poisson noise).
- **Independent errors**: One patient's measurement error does not influence another patient's measurement error.
- **Fractional membership (Yi)**: It is acceptable to treat a patient as partly in each group, rather than forcing a single group.
- **Hard labels plus extrapolation (MC-SIMEX)**: It is acceptable to keep a single group per simulated data set and correct bias by extrapolating the naive estimator.
- **Discrete membership (KoMbine)**: Each patient ultimately belongs to one group, but the model allows uncertainty in that assignment.
- **Correct likelihood model**: The survival model used (KM/logrank or Cox) is an appropriate description of the data-generating process.

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

We use three error levels chosen so the mid panel keeps KoMbine near the observed-label HR while the large-$e$ panel shows assignment search:

- Default / CI (`*_hr_example*`): $e = 0.20$, $0.25$, $0.40$
- Full local (`KOMBINE_FULL_COMPARISON=1`): $e = 0.05$, $0.10$, $0.25$

Each patient keeps the same survival time and censoring as the fixed baseline.
Only the class probabilities change: patients in the low group get probabilities
$(1-e, e)$, and patients in the high group get $(e, 1-e)$.

MC-SIMEX uses the same $e$ as the per-patient flip rate.

KoMbine stores class $k$ as a piecewise-constant NLL on $[k, k+1)$. The two-group cut must therefore be the integer class boundary $1$, not $0.5001$. That density-style cut would leave part of the class-0 bin in both groups, so KoMbine could reassign class-0 patients at no extra NLL. Yi and MC-SIMEX compare the integer class index to the cut, so $0.5001$ and $1$ are equivalent for them.

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Repo root on path so docs.kombine imports work under nbconvert (cwd is docs/kombine).
_repo_root = pathlib.Path(".").resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from kombine.datacard import Datacard
from kombine.comparisons import YiCorrectionForCoxPH
from docs.kombine.rebin_methods_comparison_times import ensure_rebinned

# Single notebook budget. Edit these if you want denser scans.
N_PERMUTATIONS = 19
N_HR_SCAN = 25
SIMEX_B = 20

# Default / CI: n=20 hr_example cards (minutes).
# Local full: set KOMBINE_FULL_COMPARISON=1 to regenerate n=50 methods_comparison_*
# cards with 4 quantile-binned death times under datacards/.../rebinned/.
# Optional escape hatch: KOMBINE_SKIP_SLOW_ANALYSES skips Analyses 1–2 entirely.
FULL_COMPARISON = bool(os.environ.get("KOMBINE_FULL_COMPARISON"))
SKIP_SLOW_ANALYSES = bool(os.environ.get("KOMBINE_SKIP_SLOW_ANALYSES"))


def placeholder_figure(title: str, figsize=(14, 5)):
    """Stand in for an analysis skipped via KOMBINE_SKIP_SLOW_ANALYSES."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.5, 0.62, title,
        ha="center", va="center", fontsize=20, fontweight="bold",
    )
    ax.text(
        0.5, 0.36,
        "Not computed: KOMBINE_SKIP_SLOW_ANALYSES is set.\n"
        "Run this notebook without that variable to produce this figure.",
        ha="center", va="center", fontsize=16,
    )
    return fig, ax


def format_pvalue(p: float) -> str:
    """Ordinary decimals for p >= 1e-3; scientific notation below that."""
    if not np.isfinite(p):
        return str(p)
    if p >= 1e-3:
        return f'{p:.4g}'
    return f'{p:.3e}'
```

```python
# Setup - Load the comparison datacards
here = pathlib.Path(".").resolve()
test_dir = here.parent.parent / "test" / "kombine"
datacards_dir = test_dir / "datacards" / "simple_examples"

if FULL_COMPARISON:
    # Regenerate K=4 rebinned cards into gitignored rebinned/ (instant).
    datacards_dir = ensure_rebinned(n_bins=4)
    # n=50, 4 quantile-binned death times (hours-scale local run)
    scenarios = {
        'fixed': {
            'file': 'methods_comparison_fixed.txt',
            'label': 'Fixed Observable',
            'description': 'no measurement error',
            'threshold': 0.5001,
        },
        'misclass_small': {
            'file': 'methods_comparison_discrete_e05.txt',
            'label': 'Disc. Classes (e=0.05)',
            'description': 'e = 0.05',
            'threshold': 1.0,
        },
        'misclass_moderate': {
            'file': 'methods_comparison_discrete_e10.txt',
            'label': 'Disc. Classes (e=0.10)',
            'description': 'e = 0.10',
            'threshold': 1.0,
        },
        'misclass_large': {
            'file': 'methods_comparison_discrete_e25.txt',
            'label': 'Disc. Classes (e=0.25)',
            'description': 'e = 0.25',
            'threshold': 1.0,
        },
        'large': {
            'file': 'methods_comparison_poisson_large.txt',
            'label': 'Poisson (large counts)',
            'description': '~2-5% relative error',
            'threshold': 0.5001,
        },
        'moderate': {
            'file': 'methods_comparison_poisson_moderate.txt',
            'label': 'Poisson (moderate counts)',
            'description': '~10-20% relative error',
            'threshold': 0.5001,
        },
        'small': {
            'file': 'methods_comparison_poisson_small.txt',
            'label': 'Poisson (small counts)',
            'description': '~25-70% relative error',
            'threshold': 0.5001,
        },
    }
    mode_name = "FULL_COMPARISON (n=50, binned death times)"
else:
    # n=20 hr_example cards (CI / interactive default)
    scenarios = {
        'fixed': {
            'file': 'fixed_hr_example.txt',
            'label': 'Fixed Observable',
            'description': 'no measurement error',
            'threshold': 0.5001,
        },
        'misclass_small': {
            'file': 'discrete_classes_hr_example_moderate.txt',
            'label': 'Disc. Classes (e=0.20)',
            'description': 'e = 0.20',
            'threshold': 1.0,
        },
        'misclass_moderate': {
            'file': 'discrete_classes_hr_example_large.txt',
            'label': 'Disc. Classes (e=0.25)',
            'description': 'e = 0.25',
            'threshold': 1.0,
        },
        'misclass_large': {
            'file': 'discrete_classes_hr_example_very_large.txt',
            'label': 'Disc. Classes (e=0.40)',
            'description': 'e = 0.40',
            'threshold': 1.0,
        },
        'large': {
            'file': 'poisson_density_hr_example_large.txt',
            'label': 'Poisson (large counts)',
            'description': '~2-3% relative error',
            'threshold': 0.5001,
        },
        'moderate': {
            'file': 'poisson_density_hr_example_moderate.txt',
            'label': 'Poisson (moderate counts)',
            'description': '~5-7% relative error',
            'threshold': 0.5001,
        },
        'small': {
            'file': 'poisson_density_hr_example_small.txt',
            'label': 'Poisson (small counts)',
            'description': '~25-70% relative error',
            'threshold': 0.5001,
        },
    }
    mode_name = "default (n=20 hr_example cards)"

print(f"Notebook 07 mode: {mode_name}", flush=True)

# Load all datacards
datacards = {}
for key, info in scenarios.items():
    filepath = datacards_dir / info['file']
    datacard = Datacard.parse_datacard(filepath)
    datacards[key] = datacard
    n_patients = len(datacard.patients)
    n_deaths = sum(1 for p in datacard.patients if not p.censored)
    uniq_deaths = len({round(p.time, 10) for p in datacard.patients if not p.censored})
    print(
        f"{info['label']}: {n_patients} patients, {n_deaths} deaths, "
        f"{uniq_deaths} distinct death times",
        flush=True,
    )

simex_rng = 0
```

## Analysis 1: Kaplan-Meier Curves

Compare the Kaplan-Meier survival curves between Yi's method (dashed), MC-SIMEX (dotted), and KoMbine (solid lines with shaded 95% confidence intervals) across all scenarios. Yi and MC-SIMEX are point estimates only; they have no fill bands.

```python
# Calculate Yi, MC-SIMEX, and KoMbine KM for each scenario
km_results = {}
label_width = 9

if SKIP_SLOW_ANALYSES:
    print("Skipping the Kaplan-Meier band analysis (KOMBINE_SKIP_SLOW_ANALYSES).")
    km_scenarios = {}
else:
    km_scenarios = scenarios

for scenario_key, scenario_info in km_scenarios.items():
    print(f"\n[{scenario_info['label']}] starting…", flush=True)
    dc = datacards[scenario_key]
    threshold = scenario_info['threshold']
    
    # KoMbine method with confidence bands
    km_low = dc.km_likelihood(
        parameter_min=-np.inf,
        parameter_max=threshold,
    )
    
    km_high = dc.km_likelihood(
        parameter_min=threshold,
        parameter_max=np.inf,
    )
    
    # Use the same time grids for Yi, SIMEX, and KoMbine so curves align
    times_low = sorted(km_low.patient_death_times)
    times_high = sorted(km_high.patient_death_times)
    times_low_plot = [0.0] + times_low
    times_high_plot = [0.0] + times_high
    print(
        f"  death times: low={len(times_low)} high={len(times_high)}",
        flush=True,
    )

    print("  Yi…", flush=True)
    result_low_yi = dc.km_survival_yi(
        parameter_min=-np.inf,
        parameter_max=threshold,
        times_for_plot=times_low_plot,
    )
    
    result_high_yi = dc.km_survival_yi(
        parameter_min=threshold,
        parameter_max=np.inf,
        times_for_plot=times_high_plot,
    )

    print(f"  MC-SIMEX (B={SIMEX_B})…", flush=True)
    result_low_simex = dc.km_survival_mc_simex(
        parameter_min=-np.inf,
        parameter_max=threshold,
        times_for_plot=times_low_plot,
        rng=simex_rng,
        B=SIMEX_B,
    )
    result_high_simex = dc.km_survival_mc_simex(
        parameter_min=threshold,
        parameter_max=np.inf,
        times_for_plot=times_high_plot,
        rng=simex_rng,
        B=SIMEX_B,
    )
    
    # Calculate best-fit and 95% CI for KoMbine
    # Use full likelihood (not binomial_only) to include measurement uncertainty!
    print("  KoMbine low arm (full NLL, CLs=[0.95])…", flush=True)
    best_low, ci_low = km_low.survival_probabilities_likelihood(
        CLs=[0.95],
        times_for_plot=times_low,
        binomial_only=(scenario_key == 'fixed'),  # Only use binomial for fixed observable
    )
    print("  KoMbine high arm (full NLL, CLs=[0.95])…", flush=True)
    best_high, ci_high = km_high.survival_probabilities_likelihood(
        CLs=[0.95],
        times_for_plot=times_high,
        binomial_only=(scenario_key == 'fixed'),  # Only use binomial for fixed observable
    )
    
    km_results[scenario_key] = {
        'yi': {
            'low': result_low_yi,
            'high': result_high_yi,
        },
        'simex': {
            'low': result_low_simex,
            'high': result_high_simex,
        },
        'kombine': {
            'low': {
                'times': times_low,
                'best': best_low,
                'ci': ci_low,
            },
            'high': {
                'times': times_high,
                'best': best_high,
                'ci': ci_high,
            }
        }
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{label_width}} - Low group final survival:  {result_low_yi['survival_probabilities'][-1]:.4f}")
    print(f"  {'Yi':<{label_width}} - High group final survival: {result_high_yi['survival_probabilities'][-1]:.4f}")
    print(f"  {'MC-SIMEX':<{label_width}} - Low group final survival:  {result_low_simex['survival_probabilities'][-1]:.4f}")
    print(f"  {'MC-SIMEX':<{label_width}} - High group final survival: {result_high_simex['survival_probabilities'][-1]:.4f}")
    
    if len(ci_low) > 0:
        ci_low_lower = ci_low[-1, 0, 0]
        ci_low_upper = ci_low[-1, 0, 1]
        print(f"  {'KoMbine':<{label_width}} - Low group final survival:  {best_low[-1]:.4f} [{ci_low_lower:.4f}, {ci_low_upper:.4f}]")
    else:
        print(f"  {'KoMbine':<{label_width}} - Low group final survival:  {best_low[-1]:.4f}")
    
    if len(ci_high) > 0:
        ci_high_lower = ci_high[-1, 0, 0]
        ci_high_upper = ci_high[-1, 0, 1]
        print(f"  {'KoMbine':<{label_width}} - High group final survival: {best_high[-1]:.4f} [{ci_high_lower:.4f}, {ci_high_upper:.4f}]")
    else:
        print(f"  {'KoMbine':<{label_width}} - High group final survival: {best_high[-1]:.4f}")
```


```python
# Define consistent color palette for all plots
colors_palette = {
    ('fixed', 'low'): '#0d47a1',           # Deep blue
    ('fixed', 'high'): '#6d1c1e',          # Deep red
    ('misclass_small', 'low'): '#1b5e20',  # Deep green
    ('misclass_small', 'high'): '#8e0000', # Dark brick
    ('misclass_moderate', 'low'): '#2e7d32',   # Green
    ('misclass_moderate', 'high'): '#b71c1c',  # Dark red
    ('misclass_large', 'low'): '#66bb6a',      # Light green
    ('misclass_large', 'high'): '#e57373',     # Light red
    ('large', 'low'): '#1976d2',           # Strong blue
    ('large', 'high'): '#e53935',          # Strong red
    ('moderate', 'low'): '#26a69a',        # Teal
    ('moderate', 'high'): '#fb8c00',       # Orange
    ('small', 'low'): '#80cbc4',           # Light teal
    ('small', 'high'): '#ffd54f',          # Light amber
}


def _plot_km_in_ax(ax, scenario_key, scenario_info, result):
    """Plot KM curves (Yi dashed, SIMEX dotted, KoMbine solid + CI shading)."""
    color_low = colors_palette[(scenario_key, 'low')]
    color_high = colors_palette[(scenario_key, 'high')]

    times_low_yi = result['yi']['low']['times_for_plot']
    surv_low_yi = result['yi']['low']['survival_probabilities']
    ax.step(times_low_yi, surv_low_yi, where='post', linewidth=2.5,
            color=color_low, alpha=0.7, linestyle='--', label='Yi: Low group')

    times_high_yi = result['yi']['high']['times_for_plot']
    surv_high_yi = result['yi']['high']['survival_probabilities']
    ax.step(times_high_yi, surv_high_yi, where='post', linewidth=2.5,
            color=color_high, alpha=0.7, linestyle='--', label='Yi: High group')

    times_low_simex = result['simex']['low']['times_for_plot']
    surv_low_simex = result['simex']['low']['survival_probabilities']
    ax.step(times_low_simex, surv_low_simex, where='post', linewidth=2.0,
            color=color_low, alpha=0.85, linestyle=':', label='MC-SIMEX: Low group')

    times_high_simex = result['simex']['high']['times_for_plot']
    surv_high_simex = result['simex']['high']['survival_probabilities']
    ax.step(times_high_simex, surv_high_simex, where='post', linewidth=2.0,
            color=color_high, alpha=0.85, linestyle=':', label='MC-SIMEX: High group')

    times_low_kombine = result['kombine']['low']['times']
    best_low_kombine = result['kombine']['low']['best']
    ci_low_kombine = result['kombine']['low']['ci']
    times_plot_low = [times_low_kombine[0]]
    best_plot_low = [1.0]
    for i, t in enumerate(times_low_kombine):
        times_plot_low.append(t)
        best_plot_low.append(best_low_kombine[i])
    ax.step(times_plot_low, best_plot_low, where='post', linewidth=2.5,
            color=color_low, alpha=0.9, label='KoMbine: Low group', zorder=3)
    if getattr(ci_low_kombine, 'size', 0):
        ci_lower_plot_low = [1.0]
        ci_upper_plot_low = [1.0]
        for i, _t in enumerate(times_low_kombine):
            ci_lower_plot_low.append(ci_low_kombine[i, 0, 0])
            ci_upper_plot_low.append(ci_low_kombine[i, 0, 1])
        ax.fill_between(times_plot_low, ci_lower_plot_low, ci_upper_plot_low,
                        step='post', alpha=0.15, color=color_low,
                        label='KoMbine: Low 95% CI', zorder=2)

    times_high_kombine = result['kombine']['high']['times']
    best_high_kombine = result['kombine']['high']['best']
    ci_high_kombine = result['kombine']['high']['ci']
    times_plot_high = [times_high_kombine[0]]
    best_plot_high = [1.0]
    for i, t in enumerate(times_high_kombine):
        times_plot_high.append(t)
        best_plot_high.append(best_high_kombine[i])
    ax.step(times_plot_high, best_plot_high, where='post', linewidth=2.5,
            color=color_high, alpha=0.9, label='KoMbine: High group', zorder=3)
    if getattr(ci_high_kombine, 'size', 0):
        ci_lower_plot_high = [1.0]
        ci_upper_plot_high = [1.0]
        for i, _t in enumerate(times_high_kombine):
            ci_lower_plot_high.append(ci_high_kombine[i, 0, 0])
            ci_upper_plot_high.append(ci_high_kombine[i, 0, 1])
        ax.fill_between(times_plot_high, ci_lower_plot_high, ci_upper_plot_high,
                        step='post', alpha=0.15, color=color_high,
                        label='KoMbine: High 95% CI', zorder=2)

    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title(scenario_info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])


# Layout: fixed centered in row 0; cols aligned by uncertainty level
# Row 0: Fixed (baseline, centered)
# Row 1: Discrete Classes — small / moderate / large error
# Row 2: Poisson Counts  — large / moderate / small counts
mosaic_layout = [
    ['.', 'fixed', '.'],
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['pois_large', 'pois_moderate', 'pois_small'],
]
mosaic_to_scenario = {
    'fixed':       'fixed',
    'dc_small':    'misclass_small',
    'dc_moderate': 'misclass_moderate',
    'dc_large':    'misclass_large',
    'pois_large':  'large',
    'pois_moderate': 'moderate',
    'pois_small':  'small',
}

if SKIP_SLOW_ANALYSES:
    placeholder_figure('Kaplan-Meier Curves: Yi, MC-SIMEX, and KoMbine',
                       figsize=(14, 13))
else:
    fig, axes_dict = plt.subplot_mosaic(mosaic_layout, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
        gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
    )

    for panel_key, scenario_key in mosaic_to_scenario.items():
        _plot_km_in_ax(axes_dict[panel_key], scenario_key,
                       scenarios[scenario_key], km_results[scenario_key])

    # Column headers: uncertainty level above row 1
    col_headers_list = ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty']
    for panel_key, header in zip(['dc_small', 'dc_moderate', 'dc_large'], col_headers_list):
        axes_dict[panel_key].annotate(
            header, xy=(0.5, 1.0), xytext=(0, 30),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#333333', annotation_clip=False,
        )

    # Row labels: observable type on the left of the first column
    for panel_key, row_label in zip(['dc_small', 'pois_large'],
                                     ['Discrete\nClasses', 'Poisson\nCounts']):
        axes_dict[panel_key].annotate(
            row_label, xy=(0, 0.5), xytext=(-52, 0),
            xycoords='axes fraction', textcoords='offset points',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#333333', rotation=90, annotation_clip=False,
        )

    plt.suptitle('Kaplan-Meier Curves: Yi, MC-SIMEX, and KoMbine',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
plt.show()
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

On the **fixed** card, Yi and MC-SIMEX therefore report the same hard-label logrank $\chi^2$ $p$, while KoMbine’s $p$ is a coarse permutation LRT (here $B=19$) of the Cox alternative — they need not match even with zero measurement error.

**What to expect**

- As measurement error grows, **Yi's p-values typically increase** because fractional membership blurs the difference between groups.
- **MC-SIMEX p-values** are those of an extrapolated hard-label statistic; they need not match Yi even though both start from the same $e_i$.
- **KoMbine's permutation p-values** do not get smaller just because assignments become cheaper: the null can re-label too.
- Large disagreements among the three p-values indicate that inference is being driven by how group-membership uncertainty is modeled, not just by sampling noise.

```python
# Calculate p-values (Yi vs MC-SIMEX vs KoMbine) for all scenarios
pvalue_results = {}
label_width = 9

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    threshold = scenario_info['threshold']
    
    # Yi's method
    yi_result = dc.km_p_value_logrank_yi(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )

    simex_result = dc.km_p_value_logrank_mc_simex(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        rng=simex_rng,
        B=SIMEX_B,
    )
    
    # KoMbine (full likelihood; includes patient-wise uncertainty)
    if SKIP_SLOW_ANALYSES:
        pval_kombine = None
    else:
        print(f"  [{scenario_info['label']}] KoMbine permutation LRT (B={N_PERMUTATIONS})…", flush=True)
        kombine_calc = dc.km_p_value(
            parameter_threshold=threshold,
            parameter_min=-np.inf,
            parameter_max=np.inf,
        )
        pval_kombine, _, _ = kombine_calc.solve_and_pvalue(
            cox_only=False,
            n_permutations=N_PERMUTATIONS,
            rng=simex_rng,
        )
    
    pvalue_results[scenario_key] = {
        'yi': yi_result['p_value'],
        'simex': simex_result['p_value'],
        'kombine': pval_kombine
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{label_width}} p-value: {format_pvalue(yi_result['p_value'])}")
    print(f"  {'MC-SIMEX':<{label_width}} p-value: {format_pvalue(simex_result['p_value'])}")
    if pval_kombine is not None:
        print(f"  {'KoMbine':<{label_width}} p-value: {format_pvalue(pval_kombine)}")
```

```python
# Plot p-value comparison
if SKIP_SLOW_ANALYSES:
    placeholder_figure(
        'P-values: Yi / MC-SIMEX logrank χ² vs KoMbine permutation LRT')
else:
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Prepare data
    scenario_keys = list(scenarios.keys())
    scenario_labels = [scenarios[k]['label'] for k in scenario_keys]
    yi_pvals = [pvalue_results[k]['yi'] for k in scenario_keys]
    simex_pvals = [pvalue_results[k]['simex'] for k in scenario_keys]
    kombine_pvals = [pvalue_results[k]['kombine'] for k in scenario_keys]

    # Bar plot
    x = np.arange(len(scenario_labels))
    width = 0.25

    bars1 = ax1.bar(x - width, yi_pvals, width, label="Yi's Method", color='steelblue')
    bars2 = ax1.bar(x, simex_pvals, width, label='MC-SIMEX', color='mediumpurple')
    bars3 = ax1.bar(x + width, kombine_pvals, width, label='KoMbine', color='coral')

    ax1.set_ylabel('P-value', fontsize=12)
    ax1.set_title(
        'P-values: Yi / MC-SIMEX logrank χ² vs KoMbine permutation LRT',
        fontsize=13,
        fontweight='bold',
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    format_pvalue(height), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
plt.show()
```

## Analysis 3: Hazard Ratios

We compare hazard ratios estimated using:

- **Yi**: the weighted Breslow partial likelihood. The table reports a **continuous** MLE in $\log H$ and a likelihood-ratio interval (same `minimize_scalar` / `brentq` recipe as KoMbine). The curve is that 2NLL on an `N_HR_SCAN`-point log grid, recentered at the continuous MLE.
- **MC-SIMEX**: extrapolated $\widehat{\log H}$ with a Wald CI. The curve below is the Wald quadratic, **not** a profile likelihood. On the **fixed** panel membership is exact and the point HR matches Yi/KoMbine, but the purple scan is still Wald, so it will not overlay the Breslow profiles.
- **KoMbine**: the full profile likelihood (**`cox_only=False`**), which jointly optimizes discrete assignments and survival parameters.

### Why the confidence intervals behave differently

As noted in the paper text, Yi’s approach can reduce bias in the *point estimate* of the hazard ratio compared to ignoring measurement error, but the **confidence interval does not necessarily reflect loss of identifiability** when measurement error becomes very large. In the extreme-uncertainty limit we would expect the data to place almost no constraint on the hazard ratio, but Yi-style weighting does not automatically produce that behavior.

MC-SIMEX is in the same family: the Wald interval is the sampling interval of the extrapolated number. It stays finite even when labels are nearly uninformative.

KoMbine’s likelihood framework, by contrast, can naturally widen the profile-likelihood confidence interval as patient-wise uncertainty increases, because the model explicitly accounts for the possibility that the discrete group assignment itself is uncertain.

```python
# Calculate hazard ratios (Yi vs MC-SIMEX vs KoMbine) for all scenarios
hr_results = {}
label_width = 9
hazard_ratios_scan = np.logspace(-2, 2, N_HR_SCAN)  # 0.01 to 100

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    hr_threshold = scenario_info['threshold']

    yi_calc = YiCorrectionForCoxPH(
        patients=dc.patients,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        parameter_threshold=hr_threshold,
    )
    best_hr_yi, yi_lower_ci, yi_upper_ci, yi_best_fit = (
        yi_calc.hazard_ratio_confidence_interval(
            confidence_level=0.95,
            hazard_ratio_min=0.01,
            hazard_ratio_max=100.0,
        )
    )
    yi_2nlls = [
        yi_calc.compute_2nll_at_hazard_ratio(hr).x
        for hr in hazard_ratios_scan
    ]

    simex_calc = dc.km_hazard_ratio_mc_simex(
        parameter_threshold=hr_threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        rng=simex_rng,
        B=SIMEX_B,
    )
    simex_estimate = simex_calc.estimate_hazard_ratio()
    simex_2nlls = [
        simex_calc.compute_2nll_at_hazard_ratio(hr).x
        for hr in hazard_ratios_scan
    ]
    
    # KoMbine (full likelihood; allows assignments to change with HR)
    hr_calc = dc.km_hazard_ratio(
        parameter_threshold=hr_threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    
    best_hr_kombine, lower_ci, upper_ci, _ = hr_calc.hazard_ratio_confidence_interval(
        cox_only=False,
        confidence_level=0.95,
        hazard_ratio_min=0.01,
        hazard_ratio_max=100.0,
    )
    
    # KoMbine profile likelihood scan
    kombine_2nlls = []
    for hr in hazard_ratios_scan:
        result = hr_calc.compute_2nll_at_hazard_ratio(hr, cox_only=False, verbose=False)
        kombine_2nlls.append(result.x)
    
    hr_results[scenario_key] = {
        'yi_best': best_hr_yi,
        'yi_2nlls': yi_2nlls,
        'yi_min_2nll': yi_best_fit.x,
        'yi_lower': yi_lower_ci,
        'yi_upper': yi_upper_ci,
        'simex_best': simex_estimate['hazard_ratio'],
        'simex_2nlls': simex_2nlls,
        'simex_lower': simex_estimate['ci_lower'],
        'simex_upper': simex_estimate['ci_upper'],
        'kombine_best': best_hr_kombine,
        'kombine_2nlls': kombine_2nlls,
        'kombine_lower': lower_ci,
        'kombine_upper': upper_ci,
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{label_width}} best-fit HR: {best_hr_yi:.3f} [{yi_lower_ci:.3f}, {yi_upper_ci:.3f}]")
    print(f"  {'MC-SIMEX':<{label_width}} best-fit HR: {simex_estimate['hazard_ratio']:.3f} [{simex_estimate['ci_lower']:.3f}, {simex_estimate['ci_upper']:.3f}]")
    print(f"  {'KoMbine':<{label_width}} best-fit HR: {best_hr_kombine:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]")
```


```python
# Plot hazard ratio profiles for all scenarios in a grid
mosaic_layout = [
    ['.', 'fixed', '.'],
    ['dc_small', 'dc_moderate', 'dc_large'],
    ['pois_large', 'pois_moderate', 'pois_small'],
]
mosaic_to_scenario = {
    'fixed':         'fixed',
    'dc_small':      'misclass_small',
    'dc_moderate':   'misclass_moderate',
    'dc_large':      'misclass_large',
    'pois_large':    'large',
    'pois_moderate': 'moderate',
    'pois_small':    'small',
}

fig, axes_dict = plt.subplot_mosaic(mosaic_layout, figsize=(14, 13),  # pyright: ignore[reportCallIssue, reportArgumentType]
    gridspec_kw={'hspace': 0.52, 'wspace': 0.35},
)

for panel_key, scenario_key in mosaic_to_scenario.items():
    ax = axes_dict[panel_key]
    result = hr_results[scenario_key]
    info = scenarios[scenario_key]

    yi_2nlls = result['yi_2nlls']
    delta_yi = np.array(yi_2nlls) - result['yi_min_2nll']

    # Already a Wald quadratic (not a profile); do not subtract a scan minimum.
    delta_simex = np.array(result['simex_2nlls'])

    kombine_2nlls = result['kombine_2nlls']
    delta_kombine = np.array(kombine_2nlls) - min(kombine_2nlls)

    ax.plot(hazard_ratios_scan, delta_yi, color='#1976d2', linewidth=2.5, marker='o', markersize=3,
            label="Yi's Method", zorder=3)
    ax.plot(hazard_ratios_scan, delta_simex, color='#7b1fa2', linewidth=2.5, marker='^', markersize=3,
            linestyle=':', label='MC-SIMEX (Wald)', zorder=3)
    ax.plot(hazard_ratios_scan, delta_kombine, color='#d32f2f', linewidth=2.5, marker='s', markersize=3,
            label='KoMbine', zorder=3)
    ax.axvline(result['yi_best'], color='#1976d2', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['simex_best'], color='#7b1fa2', linestyle=':', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['kombine_best'], color='#d32f2f', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axhline(3.84, color='gray', linestyle=':', alpha=0.6, linewidth=2.0,
               label='95% CL (χ²=3.84)', zorder=1)

    ax.set_xlabel('Hazard Ratio', fontsize=10)
    ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=10)
    ax.set_title(info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 100.0])
    ax.set_ylim([0, 10])

# Column headers: uncertainty level above row 1
col_headers_list = ['Small Uncertainty', 'Medium Uncertainty', 'Large Uncertainty']
for panel_key, header in zip(['dc_small', 'dc_moderate', 'dc_large'], col_headers_list):
    axes_dict[panel_key].annotate(
        header, xy=(0.5, 1.0), xytext=(0, 30),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='#333333', annotation_clip=False,
    )

# Row labels: observable type on the left of the first column
for panel_key, row_label in zip(['dc_small', 'pois_large'],
                                 ['Discrete\nClasses', 'Poisson\nCounts']):
    axes_dict[panel_key].annotate(
        row_label, xy=(0, 0.5), xytext=(-52, 0),
        xycoords='axes fraction', textcoords='offset points',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='#333333', rotation=90, annotation_clip=False,
    )

plt.suptitle('Hazard Ratio: Yi and KoMbine Profiles vs MC-SIMEX Wald',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


## Summary of Findings

The key modeling difference is **fractional weights vs extrapolated hard labels vs discrete assignment** under measurement uncertainty:
Yi’s method assigns each patient to both groups with weights, MC-SIMEX extrapolates a naive hard-label estimator, and KoMbine enforces one group per
patient and scores assignments using an explicit measurement-error model.

### Kaplan–Meier Curves (Qualitative)
- **Fixed observable**: Yi, MC-SIMEX, and KoMbine produce identical curves because group membership is exact.
- **Discrete classes**: At $e=0.05$–$0.10$ the KoMbine KM curves stay near the baseline.
  At $e=0.25$, Yi’s curves drift toward each other, MC-SIMEX extrapolates the hard-label KM,
  and KoMbine’s separate KM fits can collapse because assignments are weakly identified.
- **Poisson (large/moderate counts)**: Yi shrinks the group gap; MC-SIMEX is an extrapolated hard-label
  curve; KoMbine confidence bands widen as assignment uncertainty increases.
- **Poisson (small counts)**: Differences can become qualitative (including apparent reversals in the
  individual KM curve fits) because group membership is weakly identified under the error model.

### P-Values and Hazard Ratios
- Yi’s p-values generally increase as uncertainty grows; its best-fit HR drifts toward 1.
  The printed Yi HR and CI are a continuous Breslow profile, not a grid argmin.
- MC-SIMEX p-values and HRs are those of an extrapolated hard-label statistic; the Wald HR interval stays finite.
  The plotted MC-SIMEX curve is that Wald quadratic even when $e_i=0$.
- KoMbine’s plotted $p$ is a permutation LRT. The point HR can stay near the baseline
  while the profile interval widens, and at still larger $e$ the most likely assignment
  can increase apparent separation, but that search is also available under the null.
- Large disagreements among the three indicate that inference is driven by how group-membership
  uncertainty is modeled, not just by sampling noise.

### Practical Takeaways
1. When measurement error is tiny, KM curves and the Cox **point** HR agree. The HR *scan*
   is still Wald (MC-SIMEX) vs two Breslow profiles (Yi, KoMbine), and KoMbine’s $p$ is a
   permutation LRT rather than logrank $\chi^2$.
2. When measurement error is moderate/large, treat the conclusion as model-dependent and
   report sensitivity to the modeling choice.
3. Yi’s method is fast and often conservative (it blurs separation as uncertainty grows).
4. MC-SIMEX is also fast; its Wald interval is a sampling interval of the extrapolated number.
5. KoMbine is likelihood-principled for the specified error model and can reveal when
   parameters become weakly identified via widening profile-likelihood intervals.

```python
# Summary tables — computed live from pvalue_results and hr_results
header = (f"{'Scenario':<36} {'Yi p':>9} {'SIMEX p':>9} {'KoMbine p':>11}"
          f"  {'Yi HR [95% CI]':>20}  {'SIMEX HR [Wald]':>20}  {'KoMbine HR [95% CI]':>22}")
print(header)
print('-' * len(header))
for key, info in scenarios.items():
    pv = pvalue_results[key]
    hr = hr_results[key]
    yi_ci = (f"[{hr['yi_lower']:.3f}, {hr['yi_upper']:.3f}]"
             if not np.isnan(hr['yi_lower']) else '[n/a]')
    simex_ci = f"[{hr['simex_lower']:.3f}, {hr['simex_upper']:.3f}]"
    ko_ci = f"[{hr['kombine_lower']:.3f}, {hr['kombine_upper']:.3f}]"
    yi_hr_str = f"{hr['yi_best']:.3f} {yi_ci}"
    simex_hr_str = f"{hr['simex_best']:.3f} {simex_ci}"
    ko_hr_str = f"{hr['kombine_best']:.3f} {ko_ci}"
    ko_p_str = 'n/a' if pv['kombine'] is None else format_pvalue(pv['kombine'])
    print(f"{info['label']:<36} {format_pvalue(pv['yi']):>9} {format_pvalue(pv['simex']):>9} {ko_p_str:>11}"
          f"  {yi_hr_str:>20}  {simex_hr_str:>20}  {ko_hr_str:>22}")
```
