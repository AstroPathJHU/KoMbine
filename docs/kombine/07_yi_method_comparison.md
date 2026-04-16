---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: rocpicker
    language: python
    name: python3
---

```python
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,redefined-outer-name,trailing-whitespace,too-many-locals,wrong-import-order
```

# Yi's Method vs KoMbine: Comprehensive Comparison

This notebook provides a comprehensive side-by-side comparison between Yi's method for Kaplan-Meier likelihood estimation and KoMbine's approach (MINLP) across multiple measurement scenarios:
- Fixed Hazard Ratio (deterministic, no measurement error)
- Discrete classes with class probabilities (small/medium/large uncertainty)
- Poisson density with large effect size (small relative error ~2-3%)
- Poisson density with moderate effect size (larger relative error ~5-7%)
- Poisson density with small counts (high relative error ~25-70%)
- Discrete class probabilities (controllable uncertainty via class distributions)

Each analysis directly compares both methods to understand how they handle measurement uncertainty differently.


## Method Overview and Comparison

| Aspect | Yi's Method | KoMbine |
|--------|---|---|
| **Core idea** | Weighted KM/logrank using probabilistic group membership | Full likelihood with explicit group assignment variables |
| **Optimization** | Direct probability calculation (no optimization) | Mixed Integer Nonlinear Programming (Gurobi) |
| **Computational cost** | Low | Medium-high |
| **Accuracy (within model)** | Approximate to the full likelihood | Exact maximizer within solver tolerance |
| **Core assumptions** | Known measurement error distribution; independent errors; fractional group membership is an adequate proxy for uncertain assignment | Known measurement error distribution; independent errors; patients belong to one group; event times treated as observed and discrete; likelihood model is correctly specified |

### How Yi's Method Works (Intuition)
- Convert each patient's observed biomarker value into a probability of being below or above the threshold using the measurement error model.
- Instead of a global misclassification matrix as Yi describes, we extend her method to compute these probabilities on a per-patient basis (allowing uncertainty to vary by individual measurement).
- Use those per-patient probabilities as weights in the Kaplan-Meier estimator and logrank test.
- Every patient contributes to both groups, in proportion to their probability of belonging there.
- This yields fast, smooth estimates that tend to shrink group differences as measurement uncertainty grows.
- It is an approximation because it does not enforce a single, discrete group assignment for each patient.

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

To match the binary examples and exercises, we use three error levels:
- Small error: $e = 0.05$
- Medium error: $e = 0.15$
- Large error: $e = 0.30$

Each patient keeps the same survival time and censoring as the fixed baseline.
Only the class probabilities change: patients in the low group get probabilities
$(1-e, e)$, and patients in the high group get $(e, 1-e)$.

```python
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from kombine.datacard import Datacard
```

```python
# Setup - Load the comparison datacards
here = pathlib.Path(".").resolve()
test_dir = here.parent.parent / "test" / "kombine"
datacards_dir = test_dir / "datacards" / "simple_examples"

# Define the scenarios
scenarios = {
    'fixed': {
        'file': 'fixed_hr_example.txt',
        'label': 'Fixed Observable',
        'description': 'no measurement error',
    },
    'misclass_small': {
        'file': 'discrete_classes_hr_example_small.txt',
        'label': 'Disc. Classes (e=0.05)',
        'description': 'e = 0.05',
    },
    'misclass_moderate': {
        'file': 'discrete_classes_hr_example_moderate.txt',
        'label': 'Disc. Classes (e=0.15)',
        'description': 'e = 0.15',
    },
    'misclass_large': {
        'file': 'discrete_classes_hr_example_large.txt',
        'label': 'Disc. Classes (e=0.30)',
        'description': 'e = 0.30',
    },
    'large': {
        'file': 'poisson_density_hr_example_large.txt',
        'label': 'Poisson (large counts)',
        'description': '~2-3% relative error',
    },
    'moderate': {
        'file': 'poisson_density_hr_example_moderate.txt',
        'label': 'Poisson (moderate counts)',
        'description': '~5-7% relative error',
    },
    'small': {
        'file': 'poisson_density_hr_example_small.txt',
        'label': 'Poisson (small counts)',
        'description': '~25-70% relative error',
    },
}

# Load all datacards
datacards = {}
for key, info in scenarios.items():
    filepath = datacards_dir / info['file']
    datacard = Datacard.parse_datacard(filepath)
    datacards[key] = datacard
    n_patients = len(datacard.patients)
    n_deaths = sum(1 for p in datacard.patients if not p.censored)
    print(f"{info['label']}: {n_patients} patients, {n_deaths} deaths")

threshold = 0.5001
```

## Analysis 1: Kaplan-Meier Curves

Compare the Kaplan-Meier survival curves between Yi's method (dashed lines) and KoMbine's approach (solid lines with shaded 95% confidence intervals) across all scenarios. This visualization directly shows how measurement error affects the survival curve estimates and their uncertainties.

```python
# Calculate both Yi's weighted KM and KoMbine KM for each scenario
km_results = {}
label_width = 7

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # KoMbine method with confidence bands
    km_low = dc.km_likelihood(
        parameter_min=-np.inf,
        parameter_max=threshold,
    )
    
    km_high = dc.km_likelihood(
        parameter_min=threshold,
        parameter_max=np.inf,
    )
    
    # Use the same time grids for Yi and KoMbine so curves align
    times_low = sorted(km_low.patient_death_times)
    times_high = sorted(km_high.patient_death_times)
    times_low_plot = [0.0] + times_low
    times_high_plot = [0.0] + times_high

    # Yi's method (each curve is a parameter range, weighted by membership probability)
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
    
    # Calculate best-fit and 95% CI for KoMbine
    # Use full likelihood (not binomial_only) to include measurement uncertainty!
    best_low, ci_low = km_low.survival_probabilities_likelihood(
        CLs=[0.95],
        times_for_plot=times_low,
        binomial_only=(scenario_key == 'fixed'),  # Only use binomial for fixed observable
    )
    
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
    """Plot KM curves (Yi dashed, KoMbine solid + CI shading) in a single axes."""
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

    times_low_kombine = result['kombine']['low']['times']
    best_low_kombine = result['kombine']['low']['best']
    ci_low_kombine = result['kombine']['low']['ci']
    times_plot_low = [times_low_kombine[0]]
    best_plot_low = [1.0]
    ci_lower_plot_low = [1.0]
    ci_upper_plot_low = [1.0]
    for i, t in enumerate(times_low_kombine):
        times_plot_low.append(t)
        best_plot_low.append(best_low_kombine[i])
        ci_lower_plot_low.append(ci_low_kombine[i, 0, 0])
        ci_upper_plot_low.append(ci_low_kombine[i, 0, 1])
    ax.step(times_plot_low, best_plot_low, where='post', linewidth=2.5,
            color=color_low, alpha=0.9, label='KoMbine: Low group', zorder=3)
    ax.fill_between(times_plot_low, ci_lower_plot_low, ci_upper_plot_low,
                    step='post', alpha=0.15, color=color_low, label='KoMbine: Low 95% CI', zorder=2)

    times_high_kombine = result['kombine']['high']['times']
    best_high_kombine = result['kombine']['high']['best']
    ci_high_kombine = result['kombine']['high']['ci']
    times_plot_high = [times_high_kombine[0]]
    best_plot_high = [1.0]
    ci_lower_plot_high = [1.0]
    ci_upper_plot_high = [1.0]
    for i, t in enumerate(times_high_kombine):
        times_plot_high.append(t)
        best_plot_high.append(best_high_kombine[i])
        ci_lower_plot_high.append(ci_high_kombine[i, 0, 0])
        ci_upper_plot_high.append(ci_high_kombine[i, 0, 1])
    ax.step(times_plot_high, best_plot_high, where='post', linewidth=2.5,
            color=color_high, alpha=0.9, label='KoMbine: High group', zorder=3)
    ax.fill_between(times_plot_high, ci_lower_plot_high, ci_upper_plot_high,
                    step='post', alpha=0.15, color=color_high, label='KoMbine: High 95% CI', zorder=2)

    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title(scenario_info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
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

plt.suptitle('Kaplan-Meier Curves: Yi vs KoMbine Across Measurement Scenarios',
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

**Why Yi looks different**
- Yi’s method does not pick a single group; it spreads each borderline patient across both groups using probabilistic weights.
- That softens the group contrast and tends to avoid hard reversals; as uncertainty grows, Yi’s estimates drift toward each other.

**Takeaway**
When measurement error is large, probabilistic assignment (Yi) and discrete assignment (KoMbine) can tell qualitatively different stories. The right question is not “which is correct?”, but “how sensitive are the conclusions to how uncertain group membership is modeled?”


## Analysis 2: P-Values (Logrank / Likelihood-Ratio Test)

We compare p-values from:

- **Yi**: a *weighted* logrank-style calculation using per-patient probabilistic group membership (fractional membership).
- **KoMbine**: a *likelihood-based* p-value using the full model (**`cox_only=False`**), which allows discrete group assignments to change in the fit when measurements are uncertain.

**What to expect**

- As measurement error grows, **Yi's p-values typically increase** because fractional membership blurs the difference between groups.
- **KoMbine's p-values need not increase**: because it enforces discrete membership, it can keep (or even increase) apparent separation by choosing the most likely global assignment consistent with the assumed measurement-error model.
- Large disagreements between the two p-values indicate that inference is being driven by how group-membership uncertainty is modeled, not just by sampling noise.

```python
# Calculate p-values (Yi vs KoMbine) for all scenarios
pvalue_results = {}
label_width = 7

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # Yi's method
    yi_result = dc.km_p_value_logrank_yi(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    
    # KoMbine (full likelihood; includes patient-wise uncertainty)
    kombine_calc = dc.km_p_value(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    pval_kombine, _, _ = kombine_calc.solve_and_pvalue(cox_only=False)
    
    pvalue_results[scenario_key] = {
        'yi': yi_result['p_value'],
        'kombine': pval_kombine
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{label_width}} p-value: {yi_result['p_value']:.4e}")
    print(f"  {'KoMbine':<{label_width}} p-value: {pval_kombine:.4e}")
    rel_diff = abs(yi_result['p_value'] - pval_kombine) / min(yi_result['p_value'], pval_kombine) * 100
    print(f"  Relative diff: {rel_diff:.1f}%")
```

```python
# Plot p-value comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
scenario_keys = list(scenarios.keys())
scenario_labels = [scenarios[k]['label'] for k in scenario_keys]
yi_pvals = [pvalue_results[k]['yi'] for k in scenario_keys]
kombine_pvals = [pvalue_results[k]['kombine'] for k in scenario_keys]

# Bar plot
x = np.arange(len(scenario_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, yi_pvals, width, label="Yi's Method", color='steelblue')
bars2 = ax1.bar(x + width/2, kombine_pvals, width, label='KoMbine', color='coral')

ax1.set_ylabel('P-value', fontsize=12)
ax1.set_title('Logrank Test P-Values Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenario_labels, rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3e}', ha='center', va='bottom', fontsize=9)

# Relative difference plot
differences = [abs(yi_pvals[i] - kombine_pvals[i]) for i in range(len(scenario_labels))]
rel_diffs = [differences[i] / min(yi_pvals[i], kombine_pvals[i]) * 100 for i in range(len(scenario_labels))]

ax2.bar(scenario_labels, rel_diffs, color='green', alpha=0.7)
ax2.set_ylabel('Relative Difference (%)', fontsize=12)
ax2.set_title('Relative Difference: |Yi - KoMbine| / min(Yi, KoMbine)', fontsize=13, fontweight='bold')
ax2.set_xticklabels(scenario_labels, rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (label, val) in enumerate(zip(scenario_labels, rel_diffs)):
    ax2.text(i, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
```

## Analysis 3: Hazard Ratios

We compare hazard ratios estimated using:

- **Yi**: a weighted likelihood / weighted logrank-style construction based on fractional group membership.
- **KoMbine**: the full profile likelihood (**`cox_only=False`**), which jointly optimizes discrete assignments and survival parameters.

### Why the confidence intervals behave differently

As noted in the paper text, Yi’s approach can reduce bias in the *point estimate* of the hazard ratio compared to ignoring measurement error, but the **confidence interval does not necessarily reflect loss of identifiability** when measurement error becomes very large. In the extreme-uncertainty limit we would expect the data to place almost no constraint on the hazard ratio, but Yi-style weighting does not automatically produce that behavior.

KoMbine’s likelihood framework, by contrast, can naturally widen the profile-likelihood confidence interval as patient-wise uncertainty increases, because the model explicitly accounts for the possibility that the discrete group assignment itself is uncertain.

```python
# Calculate hazard ratios (Yi vs KoMbine) for all scenarios
hr_results = {}
label_width = 7
hazard_ratios_scan = np.logspace(-2, 2, 80)  # Match 04: 0.01 to 100 with 80 points
chi2_95 = 3.84  # chi2.ppf(0.95, df=1) for a 95% two-sided CI

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    hr_threshold = threshold
    
    # Yi's method - profile likelihood scan
    yi_2nlls = []
    for hr in hazard_ratios_scan:
        result = dc.km_hazard_ratio_yi(
            parameter_threshold=hr_threshold,
            hazard_ratio=hr,
            parameter_min=-np.inf,
            parameter_max=np.inf,
        )
        yi_2nlls.append(result.x)
    
    best_idx_yi = np.argmin(yi_2nlls)
    best_hr_yi = hazard_ratios_scan[best_idx_yi]
    
    # Calculate Yi's 95% CI from profile likelihood
    min_yi_2nll = min(yi_2nlls)
    delta_yi_2nll = np.array(yi_2nlls) - min_yi_2nll
    # Find HR values where delta crosses the 95% threshold
    below_threshold = delta_yi_2nll < chi2_95
    if np.any(below_threshold):
        yi_ci_indices = np.where(below_threshold)[0]
        yi_lower_ci = hazard_ratios_scan[yi_ci_indices[0]]
        yi_upper_ci = hazard_ratios_scan[yi_ci_indices[-1]]
    else:
        yi_lower_ci = yi_upper_ci = np.nan
    
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
        'yi_lower': yi_lower_ci,
        'yi_upper': yi_upper_ci,
        'kombine_best': best_hr_kombine,
        'kombine_2nlls': kombine_2nlls,
        'kombine_lower': lower_ci,
        'kombine_upper': upper_ci,
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  {'Yi':<{label_width}} best-fit HR: {best_hr_yi:.3f} [{yi_lower_ci:.3f}, {yi_upper_ci:.3f}]")
    print(f"  {'KoMbine':<{label_width}} best-fit HR: {best_hr_kombine:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]")
    rel_hr_diff = abs(best_hr_yi - best_hr_kombine) / best_hr_kombine * 100
    print(f"  Relative HR diff:     {rel_hr_diff:.1f}%")
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
    delta_yi = np.array(yi_2nlls) - min(yi_2nlls)

    kombine_2nlls = result['kombine_2nlls']
    delta_kombine = np.array(kombine_2nlls) - min(kombine_2nlls)

    ax.plot(hazard_ratios_scan, delta_yi, color='#1976d2', linewidth=2.5, marker='o', markersize=3,
            label="Yi's Method", zorder=3)
    ax.plot(hazard_ratios_scan, delta_kombine, color='#d32f2f', linewidth=2.5, marker='s', markersize=3,
            label='KoMbine', zorder=3)
    ax.axvline(result['yi_best'], color='#1976d2', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['kombine_best'], color='#d32f2f', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axhline(3.84, color='gray', linestyle=':', alpha=0.6, linewidth=2.0,
               label='95% CL (χ²=3.84)', zorder=1)

    ax.set_xlabel('Hazard Ratio', fontsize=10)
    ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=10)
    ax.set_title(info['label'], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
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

plt.suptitle('Profile Likelihood for Hazard Ratio: Yi vs KoMbine',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```


## Summary of Findings

The key modeling difference is **fractional vs discrete assignment** under measurement uncertainty:
Yi’s method assigns each patient to both groups with weights, while KoMbine enforces one group per
patient and scores assignments using an explicit measurement-error model.

### Kaplan–Meier Curves (Qualitative)
- **Fixed observable**: Yi and KoMbine produce identical curves because group membership is exact.
- **Discrete classes (small e)**: Curves remain close to the fixed baseline. As e grows, Yi’s curves
  drift toward each other while KoMbine can maintain separation using the most likely discrete assignment.
- **Poisson (large/moderate counts)**: Similar behavior — Yi shrinks the group gap; KoMbine confidence
  bands widen as assignment uncertainty increases.
- **Poisson (small counts)**: Differences can become qualitative (including apparent reversals in the
  individual KM curve fits) because group membership is weakly identified under the error model.

### P-Values and Hazard Ratios
- Yi’s p-values generally increase as uncertainty grows; its best-fit HR drifts toward 1.
- KoMbine’s p-values are more stable; its best-fit HR stays near the baseline, but the
  **confidence interval widens** as uncertainty grows.
- Large disagreements between the two indicate that inference is driven by how group-membership
  uncertainty is modeled, not just by sampling noise.

### Practical Takeaways
1. When measurement error is tiny, both methods agree and the choice is less critical.
2. When measurement error is moderate/large, treat the conclusion as model-dependent and
   report sensitivity to the modeling choice.
3. Yi’s method is fast and often conservative (it blurs separation as uncertainty grows).
4. KoMbine is likelihood-principled for the specified error model and can reveal when
   parameters become weakly identified via widening profile-likelihood intervals.

```python
# Summary tables — computed live from pvalue_results and hr_results
header = (f"{'Scenario':<36} {'Yi p-val':>10} {'KoMbine p-val':>14}"
          f"  {'Yi HR [95% CI]':>22}  {'KoMbine HR [95% CI]':>25}")
print(header)
print('-' * len(header))
for key, info in scenarios.items():
    pv = pvalue_results[key]
    hr = hr_results[key]
    yi_ci = (f"[{hr['yi_lower']:.3f}, {hr['yi_upper']:.3f}]"
             if not np.isnan(hr['yi_lower']) else '[n/a]')
    ko_ci = f"[{hr['kombine_lower']:.3f}, {hr['kombine_upper']:.3f}]"
    yi_hr_str = f"{hr['yi_best']:.3f} {yi_ci}"
    ko_hr_str = f"{hr['kombine_best']:.3f} {ko_ci}"
    print(f"{info['label']:<36} {pv['yi']:>10.3e} {pv['kombine']:>14.3e}"
          f"  {yi_hr_str:>22}  {ko_hr_str:>25}")
```
