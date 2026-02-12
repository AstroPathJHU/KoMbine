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

# Yi's Method vs KoMbine MINLP: Comprehensive Comparison

This notebook provides a comprehensive side-by-side comparison between Yi's method for Kaplan-Meier likelihood estimation and KoMbine's MINLP approach across four measurement scenarios:
- Fixed Hazard Ratio (deterministic, no measurement error)
- Poisson density with large effect size (small relative error ~2-3%)
- Poisson density with moderate effect size (larger relative error ~5-7%)
- Poisson density with small counts (high relative error ~25-70%)

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

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from kombine.datacard import Datacard
```

```python
# Setup - Load the four comparison datacards
here = pathlib.Path(".").resolve()
test_dir = here.parent.parent / "test" / "kombine"
datacards_dir = test_dir / "datacards" / "simple_examples"

# Define the four scenarios
scenarios = {
    'fixed': {
        'file': 'fixed_hr_example.txt',
        'label': 'Fixed Observable (No Error)',
        'description': 'Deterministic group assignments (baseline)'
    },
    'large': {
        'file': 'poisson_density_hr_example_large.txt',
        'label': 'Large Count Poisson',
        'description': 'Small relative error (~2-3%)'
    },
    'moderate': {
        'file': 'poisson_density_hr_example_moderate.txt',
        'label': 'Moderate Count Poisson',
        'description': 'Larger relative error (~5-7%)'
    },
    'small': {
        'file': 'poisson_density_hr_example_small.txt',
        'label': 'Small Count Poisson',
        'description': 'High relative error (~25-70%)'
    }
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

Compare the Kaplan-Meier survival curves between Yi's method (dashed lines) and KoMbine's MINLP approach (solid lines with shaded 95% confidence intervals) across all four scenarios. This visualization directly shows how measurement error affects the survival curve estimates and their uncertainties.

```python
# Calculate both Yi's weighted KM and KoMbine's MINLP KM for each scenario
km_results = {}

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # KoMbine's MINLP method with confidence bands
    km_low = dc.km_likelihood(
        parameter_min=-np.inf,
        parameter_max=threshold,
    )
    
    km_high = dc.km_likelihood(
        parameter_min=threshold,
        parameter_max=np.inf,
    )
    
    # Use the same time grids for Yi and MINLP so curves align
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
    
    # Calculate best-fit and 95% CI for MINLP
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
        'minlp': {
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
    print(f"  Yi    - Low group final survival:  {result_low_yi['survival_probabilities'][-1]:.4f}")
    print(f"  Yi    - High group final survival: {result_high_yi['survival_probabilities'][-1]:.4f}")
    
    if len(ci_low) > 0:
        ci_low_lower = ci_low[-1, 0, 0]
        ci_low_upper = ci_low[-1, 0, 1]
        print(f"  MINLP - Low group final survival:  {best_low[-1]:.4f} [{ci_low_lower:.4f}, {ci_low_upper:.4f}]")
    else:
        print(f"  MINLP - Low group final survival:  {best_low[-1]:.4f}")
    
    if len(ci_high) > 0:
        ci_high_lower = ci_high[-1, 0, 0]
        ci_high_upper = ci_high[-1, 0, 1]
        print(f"  MINLP - High group final survival: {best_high[-1]:.4f} [{ci_high_lower:.4f}, {ci_high_upper:.4f}]")
    else:
        print(f"  MINLP - High group final survival: {best_high[-1]:.4f}")
```


```python
# Define consistent color palette for all plots
# High colors: variations of red-orange (darker to lighter)
# Low colors: variations of blue-green (darker to lighter)
# Fixed (baseline) - darkest; Large and Moderate - mid tones; Small - lightest
colors_palette = {
    ('fixed', 'low'): '#0d47a1',      # Deep blue
    ('fixed', 'high'): '#6d1c1e',     # Deep red
    ('large', 'low'): '#1976d2',      # Strong blue
    ('large', 'high'): '#e53935',     # Strong red
    ('moderate', 'low'): '#26a69a',   # Teal
    ('moderate', 'high'): '#fb8c00',  # Orange
    ('small', 'low'): '#80cbc4',      # Light teal
    ('small', 'high'): '#ffd54f',     # Light amber
}

# Plot KM curves for all four scenarios in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (scenario_key, scenario_info) in enumerate(scenarios.items()):
    ax = axes[idx]
    result = km_results[scenario_key]
    color_low = colors_palette[(scenario_key, 'low')]
    color_high = colors_palette[(scenario_key, 'high')]
    
    # Yi's method - Low group
    times_low_yi = result['yi']['low']['times_for_plot']
    surv_low_yi = result['yi']['low']['survival_probabilities']
    ax.step(times_low_yi, surv_low_yi, where='post', linewidth=2.5, 
            color=color_low, alpha=0.7, linestyle='--', label="Yi: Low group")
    
    # Yi's method - High group
    times_high_yi = result['yi']['high']['times_for_plot']
    surv_high_yi = result['yi']['high']['survival_probabilities']
    ax.step(times_high_yi, surv_high_yi, where='post', linewidth=2.5, 
            color=color_high, alpha=0.7, linestyle='--', label="Yi: High group")
    
    # KoMbine MINLP - Low group with error bands
    times_low_minlp = result['minlp']['low']['times']
    best_low_minlp = result['minlp']['low']['best']
    ci_low_minlp = result['minlp']['low']['ci']
    
    # Create step function coordinates for plotting
    times_plot_low = [times_low_minlp[0]]
    best_plot_low = [1.0]
    ci_lower_plot_low = [1.0]
    ci_upper_plot_low = [1.0]
    
    for i, t in enumerate(times_low_minlp):
        times_plot_low.append(t)
        best_plot_low.append(best_low_minlp[i])
        ci_lower_plot_low.append(ci_low_minlp[i, 0, 0])
        ci_upper_plot_low.append(ci_low_minlp[i, 0, 1])
    
    ax.step(times_plot_low, best_plot_low, where='post', linewidth=2.5, 
            color=color_low, alpha=0.9, label='MINLP: Low group', zorder=3)
    ax.fill_between(times_plot_low, ci_lower_plot_low, ci_upper_plot_low, 
                     step='post', alpha=0.15, color=color_low, label='MINLP: Low 95% CI', zorder=2)
    
    # KoMbine MINLP - High group with error bands
    times_high_minlp = result['minlp']['high']['times']
    best_high_minlp = result['minlp']['high']['best']
    ci_high_minlp = result['minlp']['high']['ci']
    
    times_plot_high = [times_high_minlp[0]]
    best_plot_high = [1.0]
    ci_lower_plot_high = [1.0]
    ci_upper_plot_high = [1.0]
    
    for i, t in enumerate(times_high_minlp):
        times_plot_high.append(t)
        best_plot_high.append(best_high_minlp[i])
        ci_lower_plot_high.append(ci_high_minlp[i, 0, 0])
        ci_upper_plot_high.append(ci_high_minlp[i, 0, 1])
    
    ax.step(times_plot_high, best_plot_high, where='post', linewidth=2.5, 
            color=color_high, alpha=0.9, label='MINLP: High group', zorder=3)
    ax.fill_between(times_plot_high, ci_lower_plot_high, ci_upper_plot_high, 
                     step='post', alpha=0.15, color=color_high, label='MINLP: High 95% CI', zorder=2)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title(f"{scenario_info['label']}\n({scenario_info['description']})", 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.suptitle("Kaplan-Meier Curves: Yi vs KoMbine MINLP Across Measurement Scenarios", 
             fontsize=14, fontweight='bold', y=1.02)
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
- In this notebook, the two KoMbine KM curves are fit **separately** (one MINLP for the low range and one MINLP for the high range). With large uncertainty, the best-fit assignments for those two separate optimizations need not form a perfectly complementary partition of patients.
- The KoMbine **hazard ratio / p-value** calculations, in contrast, are *joint* two-group fits. Those joint fits are the self-consistent way to summarize “which group is worse” under the KoMbine model.

**Why Yi looks different**
- Yi’s method does not pick a single group; it spreads each borderline patient across both groups using probabilistic weights.
- That softens the group contrast and tends to avoid hard reversals; as uncertainty grows, Yi’s estimates drift toward each other.

**Takeaway**
When measurement error is large, probabilistic assignment (Yi) and discrete assignment (KoMbine) can tell qualitatively different stories. The right question is not “which is correct?”, but “how sensitive are the conclusions to how uncertain group membership is modeled?”


## Analysis 2: P-Values (Logrank / Likelihood-Ratio Test)

We compare p-values from:

- **Yi**: a *weighted* logrank-style calculation using per-patient probabilistic group membership (fractional membership).
- **KoMbine**: a *likelihood-based* p-value using the full MINLP model (**`cox_only=False`**), which allows discrete group assignments to change in the fit when measurements are uncertain.

**What to expect**

- As measurement error grows, **Yi's p-values typically increase** because fractional membership blurs the difference between groups.
- **KoMbine's p-values need not increase**: because it enforces discrete membership, it can keep (or even increase) apparent separation by choosing the most likely global assignment consistent with the assumed measurement-error model.
- Large disagreements between the two p-values indicate that inference is being driven by how group-membership uncertainty is modeled, not just by sampling noise.

```python
# Calculate p-values (Yi vs KoMbine) for all four scenarios
pvalue_results = {}

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # Yi's method
    yi_result = dc.km_p_value_logrank_yi(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    
    # KoMbine's MINLP (full likelihood; includes patient-wise uncertainty)
    minlp_calc = dc.km_p_value(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    pval_minlp, _, _ = minlp_calc.solve_and_pvalue(cox_only=False)
    
    pvalue_results[scenario_key] = {
        'yi': yi_result['p_value'],
        'minlp': pval_minlp
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  Yi   p-value: {yi_result['p_value']:.4e}")
    print(f"  MINLP p-value: {pval_minlp:.4e}")
    rel_diff = abs(yi_result['p_value'] - pval_minlp) / min(yi_result['p_value'], pval_minlp) * 100
    print(f"  Relative diff: {rel_diff:.1f}%")
```

```python
# Plot p-value comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
scenario_labels = [scenarios[k]['label'] for k in ['fixed', 'large', 'moderate', 'small']]
yi_pvals = [pvalue_results[k]['yi'] for k in ['fixed', 'large', 'moderate', 'small']]
minlp_pvals = [pvalue_results[k]['minlp'] for k in ['fixed', 'large', 'moderate', 'small']]

# Bar plot
x = np.arange(len(scenario_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, yi_pvals, width, label="Yi's Method", color='steelblue')
bars2 = ax1.bar(x + width/2, minlp_pvals, width, label='KoMbine MINLP', color='coral')

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
differences = [abs(yi_pvals[i] - minlp_pvals[i]) for i in range(len(scenario_labels))]
rel_diffs = [differences[i] / min(yi_pvals[i], minlp_pvals[i]) * 100 for i in range(len(scenario_labels))]

ax2.bar(scenario_labels, rel_diffs, color='green', alpha=0.7)
ax2.set_ylabel('Relative Difference (%)', fontsize=12)
ax2.set_title('Relative Difference: |Yi - MINLP| / min(Yi, MINLP)', fontsize=13, fontweight='bold')
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
- **KoMbine**: the full MINLP profile likelihood (**`cox_only=False`**), which jointly optimizes discrete assignments and survival parameters.

### Why the confidence intervals behave differently

As noted in the paper text, Yi’s approach can reduce bias in the *point estimate* of the hazard ratio compared to ignoring measurement error, but the **confidence interval does not necessarily reflect loss of identifiability** when measurement error becomes very large. In the extreme-uncertainty limit we would expect the data to place almost no constraint on the hazard ratio, but Yi-style weighting does not automatically produce that behavior.

KoMbine’s likelihood framework, by contrast, can naturally widen the profile-likelihood confidence interval as patient-wise uncertainty increases, because the model explicitly accounts for the possibility that the discrete group assignment itself is uncertain.

```python
# Calculate hazard ratios (Yi vs KoMbine) for all four scenarios
hr_results = {}
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
    
    # KoMbine's MINLP (full likelihood; allows assignments to change with HR)
    hr_calc = dc.km_hazard_ratio(
        parameter_threshold=hr_threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    
    best_hr_minlp, lower_ci, upper_ci, _ = hr_calc.hazard_ratio_confidence_interval(
        cox_only=False,
        confidence_level=0.95,
        hazard_ratio_min=0.01,
        hazard_ratio_max=100.0,
    )
    
    # MINLP profile likelihood scan
    minlp_2nlls = []
    for hr in hazard_ratios_scan:
        result = hr_calc.compute_2nll_at_hazard_ratio(hr, cox_only=False, verbose=False)
        minlp_2nlls.append(result.x)
    
    hr_results[scenario_key] = {
        'yi_best': best_hr_yi,
        'yi_2nlls': yi_2nlls,
        'yi_lower': yi_lower_ci,
        'yi_upper': yi_upper_ci,
        'minlp_best': best_hr_minlp,
        'minlp_2nlls': minlp_2nlls,
        'minlp_lower': lower_ci,
        'minlp_upper': upper_ci,
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  Yi best-fit HR:       {best_hr_yi:.3f} [{yi_lower_ci:.3f}, {yi_upper_ci:.3f}]")
    print(f"  MINLP best-fit HR:    {best_hr_minlp:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]")
    rel_hr_diff = abs(best_hr_yi - best_hr_minlp) / best_hr_minlp * 100
    print(f"  Relative HR diff:     {rel_hr_diff:.1f}%")
```


```python
# Plot hazard ratio profiles for all four scenarios in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (scenario_key, scenario_info) in enumerate(scenarios.items()):
    ax = axes[idx]
    result = hr_results[scenario_key]
    
    # Yi profile likelihood
    yi_2nlls = result['yi_2nlls']
    min_yi = min(yi_2nlls)
    delta_yi = np.array(yi_2nlls) - min_yi
    
    # MINLP profile likelihood
    minlp_2nlls = result['minlp_2nlls']
    min_minlp = min(minlp_2nlls)
    delta_minlp = np.array(minlp_2nlls) - min_minlp
    
    # Plot both profile likelihoods
    ax.plot(hazard_ratios_scan, delta_yi, color='#1976d2', linewidth=2.5, marker='o', markersize=3,
            label="Yi's Method", zorder=3)
    ax.plot(hazard_ratios_scan, delta_minlp, color='#d32f2f', linewidth=2.5, marker='s', markersize=3,
            label="KoMbine MINLP", zorder=3)
    
    # Best-fit lines
    ax.axvline(result['yi_best'], color='#1976d2', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    ax.axvline(result['minlp_best'], color='#d32f2f', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)
    
    # Confidence threshold line (95% CI, chi2=3.84)
    ax.axhline(3.84, color='gray', linestyle=':', alpha=0.6, linewidth=2.0, label='95% CL (χ²=3.84)', zorder=1)
    
    ax.set_xlabel('Hazard Ratio', fontsize=11)
    ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=11)
    ax.set_title(f"{scenario_info['label']}",
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_xlim([0.01, 100.0])
    ax.set_ylim([0, 10])

plt.suptitle('Profile Likelihood for Hazard Ratio: Yi vs MINLP', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```


## Summary of Findings (with `cox_only=False` for KoMbine HR/p-values)

The key modeling difference is **fractional vs discrete assignment** under measurement uncertainty: Yi’s method assigns each patient to both groups with weights, while KoMbine enforces one group per patient and scores assignments using an explicit measurement-error model.

### Kaplan–Meier Curves (Qualitative)
- **Fixed / Large-count**: Yi and KoMbine curves are very similar because group membership is effectively known (measurement error is negligible).
- **Moderate-count**: Yi begins to shrink the gap between groups. KoMbine’s best-fit curves can still preserve separation, but the confidence bands widen because assignments are less certain.
- **Small-count**: Differences can become qualitative (including apparent reversals) because group membership is weakly identified under the error model.

### P-Values (Observed in this notebook run)
- Yi’s p-values generally increase as uncertainty grows, reaching near 1.0 in the small-count case.
- KoMbine’s full-likelihood p-values stay relatively stable for fixed/large/moderate and diverge from Yi when assignments become ambiguous.

| Scenario | Yi p-value | KoMbine p-value (MINLP) |
|---|---:|---:|
| Fixed | 4.226e-01 | 2.509e-01 |
| Large Poisson | 3.330e-01 | 2.508e-01 |
| Moderate Poisson | 5.460e-01 | 2.508e-01 |
| Small Poisson | 9.702e-01 | 1.718e-01 |

### Hazard Ratios (Observed in this notebook run)
- Yi’s best-fit HR drifts toward 1 as uncertainty increases, and its profile-based CI in this simple scan does not automatically explode in the large-uncertainty limit.
- KoMbine’s best-fit HR stays near the baseline optimum in these examples, but the **confidence interval widens** as uncertainty grows; in the small-count case the lower bound hits the scan boundary, consistent with the idea that the HR becomes weakly constrained.

| Scenario | Yi best HR [95% CI] | KoMbine best HR [95% CI] |
|---|---|---|
| Fixed | 1.800 [0.586, 8.600] | 2.280 [0.496, 11.156] |
| Large Poisson | 2.043 [0.586, 9.571] | 2.280 [0.496, 11.156] |
| Moderate Poisson | 1.557 [0.586, 6.900] | 2.280 [0.442, 11.156] |
| Small Poisson | 1.071 [0.343, 3.986] | 2.280 [0.100, 11.156] |

### Practical Takeaways
1. When measurement error is tiny, both methods agree and the choice is less critical.
2. When measurement error is moderate/large, treat the conclusion as model-dependent and report sensitivity to the modeling choice.
3. Yi’s method is fast and often conservative (it blurs separation as uncertainty grows).
4. KoMbine is likelihood-principled for the specified error model and can reveal when parameters become weakly identified via widening profile-likelihood intervals.
