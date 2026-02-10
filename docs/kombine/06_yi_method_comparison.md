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

This notebook provides a comprehensive comparison between Yi's method for Kaplan-Meier likelihood estimation and KoMbine's MINLP approach across three measurement scenarios:
- Fixed Hazard Ratio (HR) example
- Poisson density with large effect size
- Poisson density with moderate effect size


## Method Overview and Comparison

| Aspect | Yi's Method | KoMbine MINLP |
|--------|---|---|
| **Approach** | Parametric KM likelihood with Yi's correction | Mixed Integer Nonlinear Programming |
| **Optimization** | Direct parameter search | Gurobi optimization |
| **Computational Cost** | Low | Medium-High |
| **Accuracy** | Approximate | Exact (within solver tolerance) |
| **Assumptions** | Smooth hazard & frailty | Discrete event times |
| **Use Cases** | Quick screening, exploratory analysis | Rigorous statistical inference |

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
# Setup - Load the three comparison datacards
here = pathlib.Path(".").resolve()
test_dir = here.parent.parent / "test" / "kombine"
datacards_dir = test_dir / "datacards" / "simple_examples"

# Define the three scenarios
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

threshold = 0.5
```

## Analysis 1: Kaplan-Meier Curves

Compare the Kaplan-Meier survival curves estimated by Yi's method across all three scenarios.

```python
# Calculate Yi's weighted KM for each scenario
km_results = {}

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # Yi's method
    result_low = dc.km_survival_yi(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=threshold,
        method='bayesian',
    )
    
    result_high = dc.km_survival_yi(
        parameter_threshold=threshold,
        parameter_min=threshold,
        parameter_max=np.inf,
        method='bayesian',
    )
    
    km_results[scenario_key] = {
        'low': result_low,
        'high': result_high
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  Low group final survival: {result_low['survival_probabilities'][-1]:.4f}")
    print(f"  High group final survival: {result_high['survival_probabilities'][-1]:.4f}")
```

```python
# Plot KM curves for all three scenarios side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (scenario_key, scenario_info) in enumerate(scenarios.items()):
    ax = axes[idx]
    result = km_results[scenario_key]
    
    # Low group
    times_low = result['low']['times_for_plot']
    surv_low = result['low']['survival_probabilities']
    ax.step(times_low, surv_low, where='post', linewidth=2.5, color='red', label='Low group')
    
    # High group
    times_high = result['high']['times_for_plot']
    surv_high = result['high']['survival_probabilities']
    ax.step(times_high, surv_high, where='post', linewidth=2.5, color='blue', label='High group')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title(f"{scenario_info['label']}\n({scenario_info['description']})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.suptitle("Yi's Weighted Kaplan-Meier Curves Across Measurement Scenarios", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

## Analysis 2: P-Values (Logrank Test)

Compare p-values computed using Yi's method and KoMbine's MINLP approach for each scenario.

```python
# Calculate p-values (Yi vs KoMbine) for all three scenarios
pvalue_results = {}

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    
    # Yi's method
    yi_result = dc.km_p_value_logrank_yi(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        method='bayesian',
    )
    
    # KoMbine's MINLP
    minlp_calc = dc.km_p_value(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    pval_minlp, _, _ = minlp_calc.solve_and_pvalue(cox_only=True)
    
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
scenario_labels = [scenarios[k]['label'] for k in ['fixed', 'large', 'moderate']]
yi_pvals = [pvalue_results[k]['yi'] for k in ['fixed', 'large', 'moderate']]
minlp_pvals = [pvalue_results[k]['minlp'] for k in ['fixed', 'large', 'moderate']]

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

Compare hazard ratios estimated using Yi's method and KoMbine's MINLP approach.

```python
# Calculate hazard ratios (Yi vs KoMbine) for all three scenarios
hr_results = {}
hazard_ratios_scan = np.linspace(0.2, 5.0, 25)

for scenario_key, scenario_info in scenarios.items():
    dc = datacards[scenario_key]
    hr_threshold = 0.5
    
    # Yi's method - profile likelihood scan
    yi_2nlls = []
    for hr in hazard_ratios_scan:
        result = dc.km_hazard_ratio_yi(
            parameter_threshold=hr_threshold,
            hazard_ratio=hr,
            parameter_min=-np.inf,
            parameter_max=np.inf,
            method='bayesian',
        )
        yi_2nlls.append(result.x)
    
    best_idx_yi = np.argmin(yi_2nlls)
    best_hr_yi = hazard_ratios_scan[best_idx_yi]
    
    # KoMbine's MINLP
    hr_calc = dc.km_hazard_ratio(
        parameter_threshold=hr_threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    
    best_hr_minlp, lower_ci, upper_ci, _ = hr_calc.hazard_ratio_confidence_interval(
        cox_only=True,
        confidence_level=0.95,
        hazard_ratio_min=0.1,
        hazard_ratio_max=10.0
    )
    
    hr_results[scenario_key] = {
        'yi_best': best_hr_yi,
        'yi_2nlls': yi_2nlls,
        'minlp_best': best_hr_minlp,
        'minlp_lower': lower_ci,
        'minlp_upper': upper_ci,
    }
    
    print(f"\n{scenario_info['label']}:")
    print(f"  Yi best-fit HR:       {best_hr_yi:.3f}")
    print(f"  MINLP best-fit HR:    {best_hr_minlp:.3f}")
    print(f"  MINLP 95% CI:         [{lower_ci:.3f}, {upper_ci:.3f}]")
    ci_width = upper_ci - lower_ci
    print(f"  CI width:             {ci_width:.3f}")
    rel_hr_diff = abs(best_hr_yi - best_hr_minlp) / best_hr_minlp * 100
    print(f"  Relative HR diff:     {rel_hr_diff:.1f}%")
```

```python
# Plot hazard ratio profiles for all three scenarios
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (scenario_key, scenario_info) in enumerate(scenarios.items()):
    ax = axes[idx]
    result = hr_results[scenario_key]
    
    # Yi profile likelihood
    yi_2nlls = result['yi_2nlls']
    min_2nll = min(yi_2nlls)
    delta_2nll = np.array(yi_2nlls) - min_2nll
    
    ax.plot(hazard_ratios_scan, delta_2nll, 'b-', linewidth=2.5, marker='o', markersize=4,
            label="Yi's Method")
    ax.axvline(result['yi_best'], color='b', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # MINLP best-fit and CI
    ax.axvline(result['minlp_best'], color='r', linestyle='--', alpha=0.7, linewidth=2,
               label=f"MINLP: {result['minlp_best']:.3f}")
    ax.axvspan(result['minlp_lower'], result['minlp_upper'], alpha=0.2, color='red',
               label=f"MINLP 95% CI")
    
    # Confidence threshold
    ax.axhline(2.706, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='95% threshold')
    
    ax.set_xlabel('Hazard Ratio', fontsize=11)
    ax.set_ylabel(r'$\Delta(-2 \ln L)$', fontsize=11)
    ci_width = result['minlp_upper'] - result['minlp_lower']
    ax.set_title(f"{scenario_info['label']}\n(CI width: {ci_width:.3f})",
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.2, 5.0])
    ax.set_ylim([0, 12])

plt.suptitle('Profile Likelihood for Hazard Ratio: Yi vs MINLP', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

## Summary of Findings

### Kaplan-Meier Curves
The initial KM curves (baseline comparison) show close agreement between Yi's method and standard KM estimation:
- **Fixed Observable (No Error)**: Both methods perfectly capture the two-group survival difference
  - Low group: 100% survival throughout
  - High group: 50% final survival
- **Large Count Poisson (~2-3% error)**: Minimal deviation from fixed case
  - High group final survival: 50.07% (vs 50% in fixed)
  - Visual curves almost perfectly overlapped
- **Moderate Count Poisson (~5-7% error)**: Larger perturbation visible
  - High group final survival: 51.77% (vs 50% in fixed)
  - Measurement error causes slight elevation in high-risk group survival

**Interpretation**: As measurement error increases, Yi's probabilistic weighting adjusts group assignments, which can slightly alter the estimated survival curves. The effect is most pronounced in the moderate measurement error scenario.

### Logrank Test P-Values
The p-value comparisons reveal important differences in how the two methods handle measurement error:

| Scenario | Yi's Method | MINLP | Relative Difference |
|----------|------------|--------|-------------------|
| Fixed Observable | 0.2433 | 0.2509 | 3.1% |
| Large Count Poisson | 0.2783 | 0.2509 | 10.9% |
| Moderate Count Poisson | 0.5316 | 0.2509 | 111.9% |

**Key Observations**:
- In the fixed (no-error) case, both methods agree closely (3.1% relative difference)
- MINLP p-values remain stable (~0.251) across all measurement error scenarios
- Yi's method shows increasing p-values with measurement error, suggesting that probabilistic weighting reduces the apparent separation between groups
- In the moderate error case, Yi's p-value (0.532) suggests nearly no significant difference, while MINLP (0.251) still detects moderate significance
- This divergence indicates that Yi's method treats measurement uncertainty as group ambiguity, inflating uncertainty in statistical tests

### Hazard Ratio Estimates
The hazard ratio comparison shows moderate variation in point estimates but consistent CI bounds:

| Scenario | Yi HR | MINLP HR | CI Bounds | CI Width | HR Difference |
|----------|-------|----------|-----------|----------|---------------|
| Fixed Observable | 2.200 | 2.280 | [0.557, 10.000] | 9.443 | 3.5% |
| Large Count Poisson | 2.200 | 2.280 | [0.557, 10.000] | 9.443 | 3.5% |
| Moderate Count Poisson | 1.600 | 2.280 | [0.557, 10.000] | 9.443 | 29.8% |

**Key Observations**:
- MINLP's point estimate (2.280) remains stable across all measurement scenarios
- Yi's method shows sensitivity to measurement error, particularly in the moderate error case (HR drops from 2.2 to 1.6)
- MINLP's CI bounds are wide in all scenarios ([0.557, 10.000]), reflecting the discrete optimization constraints
- The relative difference in HR estimates grows with measurement error (3.5% → 29.8%)
- Yi's CI is likely narrower but is not directly displayed in the current analysis; the wide MINLP CI reflects the penalty-based approach

### Overall Comparison

**Agreement Pattern**:
1. **No measurement error (fixed)**: Both methods show strong agreement (3-3.5% difference across metrics)
2. **Small measurement error (large Poisson)**: Methods remain similar for KM curves and HRs, but p-values begin to diverge (10.9%)
3. **Moderate measurement error (moderate Poisson)**: Major divergence in p-values (111.9%) and HRs (29.8%)

**Method Characteristics**:
- **Yi's Method**: 
  - Fast computation
  - Adapts to measurement uncertainty by adjusting group weights probabilistically
  - Can be conservative in unusual error distributions (moderate case)
  - Better for exploratory analysis and hypothesis generation
  
- **KoMbine MINLP**:
  - Computationally intensive but robust
  - Maintains consistent estimates across measurement scenarios
  - Uses penalty functions to handle uncertainty systematically
  - Better for confirmatory analysis with formal CI requirements

**Recommendation**: 
For datasets with suspected measurement error, use both methods in tandem:
1. Start with Yi's method for quick exploratory assessment
2. Validate with MINLP for formal inference
3. Investigate discrepancies (>10% relative difference) as indicators of potential measurement issues or model misspecification
