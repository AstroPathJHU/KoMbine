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

# Hazard Ratio Calculation with KoMbine

This notebook demonstrates how to calculate hazard ratios and perform likelihood scans using the KoMbine package.

## Background

When comparing two Kaplan-Meier curves stratified by a biomarker, we often want to quantify the difference using a hazard ratio (HR). The hazard ratio represents the relative rate of events (e.g., death, disease recurrence) between two groups:

- HR = 1: No difference between groups
- HR > 1: High group has higher hazard (worse outcomes)
- HR < 1: Low group has higher hazard (worse outcomes)

KoMbine extends standard Cox proportional hazards methodology by:
1. Using exact binomial likelihoods instead of asymptotic approximations
2. Allowing patient group assignments to be uncertain based on biomarker measurement error
3. Providing profile likelihood confidence intervals via Mixed Integer Nonlinear Programming (MINLP)

```python
import numpy as np
import matplotlib.pyplot as plt
from kombine.datacard import Datacard

# For reproducibility
np.random.seed(42)
```

## Example 1: Fixed Observable Type

We'll start with a simple example using **fixed observable type**, where each patient's biomarker value is known exactly with no measurement uncertainty. This serves as a baseline to understand the Cox partial likelihood contribution to the confidence intervals.

With fixed observables, patient group assignments are deterministic—only the **Cox error** (statistical uncertainty from finite event counts) contributes to confidence intervals.

```python
# Path to example datacards
import pathlib
notebook_dir = pathlib.Path().resolve()
test_dir = notebook_dir.parent.parent / "test" / "kombine"
datacards_dir = test_dir / "datacards" / "simple_examples"

# Load fixed observable datacard
dcfile_fixed = datacards_dir / "fixed_hr_example.txt"
datacard_fixed = Datacard.parse_datacard(dcfile_fixed)

print("=" * 60)
print("FIXED OBSERVABLE TYPE")
print("=" * 60)
print(f"Loaded {len(datacard_fixed.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard_fixed.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard_fixed.patients if p.censored)}")
print("\nWith fixed observables, patient assignments are deterministic")
print("-> Only Cox error (finite events) contributes to CIs")
```

### Calculate Hazard Ratio with Confidence Interval

Let's calculate the best-fit hazard ratio and its confidence interval for the fixed observable case.

**Note on hazard ratio bounds**: The optimizer uses bounds on log(HR) to keep the problem well-conditioned. By default, these are set to [-10, 10], corresponding to HR ∈ [0.000045, 22026]. If your analysis requires exploring more extreme hazard ratios, you can adjust these bounds using the `log_hazard_ratio_bounds` parameter.

```python
# Create hazard ratio calculator for fixed observable datacard
# Threshold of 0.5 splits into low-risk (< 0.5) and high-risk (>= 0.5) groups
hr_calc_fixed = datacard_fixed.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.0,
    parameter_max=1.0,
)

# Calculate 68% and 95% confidence intervals
best_fit_hr_fixed, lower_ci_68_fixed, upper_ci_68_fixed, result_68_fixed = hr_calc_fixed.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.5,
    hazard_ratio_max=10.0,
)

_, lower_ci_95_fixed, upper_ci_95_fixed, _ = hr_calc_fixed.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr_fixed:.3f}")
print(f"68% CI: [{lower_ci_68_fixed:.3f}, {upper_ci_68_fixed:.3f}]")
print(f"95% CI: [{lower_ci_95_fixed:.3f}, {upper_ci_95_fixed:.3f}]")
print(f"\n2NLL at best fit: {result_68_fixed.x:.2f}")
print("\nPatient distribution:")
print(f"  Low group: {result_68_fixed.n_total_low} patients ({result_68_fixed.n_alive_low} alive at end)")
print(f"  High group: {result_68_fixed.n_total_high} patients ({result_68_fixed.n_alive_high} alive at end)")
```

### Likelihood Scan (Fixed Observable)

Let's visualize the likelihood as a function of the hazard ratio for the fixed observable case.

```python
# Perform likelihood scan for fixed observable
hazard_ratios_fixed, twonll_values_fixed, best_fit_result_fixed = hr_calc_fixed.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.5,
    hazard_ratio_max=6.0,
    cox_only=False
)

print(f"Likelihood scan completed over {len(hazard_ratios_fixed)} points")
print(f"Minimum 2NLL: {np.min(twonll_values_fixed):.2f} at HR = {hazard_ratios_fixed[np.argmin(twonll_values_fixed)]:.3f}")

# Store for later comparison
fixed_scan_data = {
    'hrs': hazard_ratios_fixed,
    'twonll': twonll_values_fixed,
    'best_fit': best_fit_hr_fixed,
    'ci_68': (lower_ci_68_fixed, upper_ci_68_fixed),
    'ci_95': (lower_ci_95_fixed, upper_ci_95_fixed),
}
```

## Example 2: Poisson Density with Large Counts

Now let's consider a **poisson_density observable type** where biomarker measurements have Poisson uncertainty. With **large counts** (e.g., hundreds of cells counted in a region), the Poisson error is small (√N/N ~ 2-3%), so:

- Patients are well-localized around their true parameter value
- The **Cox error dominates** over measurement error
- Confidence intervals should be **similar to the fixed case**

This demonstrates that when measurement uncertainty is small, KoMbine's results converge to the Cox proportional hazards case.

```python
# Load Poisson density datacard with large counts
dcfile_poisson_large = datacards_dir / "poisson_density_hr_example_large.txt"
datacard_poisson_large = Datacard.parse_datacard(dcfile_poisson_large)

print("=" * 60)
print("POISSON DENSITY - LARGE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_large.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard_poisson_large.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard_poisson_large.patients if p.censored)}")

# Check the counts to show they're large
nums = [p.observable.numerator for p in datacard_poisson_large.patients]
areas = [p.observable.denominator for p in datacard_poisson_large.patients]
densities = [n/a for n, a in zip(nums, areas)]
print("\nCount statistics:")
print(f"  Mean count: {np.mean(nums):.1f}")
print(f"  Range: [{min(nums)}, {max(nums)}]")
print(f"  Relative uncertainty (√N/N): {np.mean([np.sqrt(n)/n for n in nums]):.1%}")
print("\nWith large counts, Poisson error is small -> Cox error dominates")
```

```python
# Create hazard ratio calculator for Poisson density (large counts)
# For poisson_density, we need parameter_min/max for the density parameter
hr_calc_poisson_large = datacard_poisson_large.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

# Calculate confidence intervals
best_fit_hr_large, lower_ci_68_large, upper_ci_68_large, result_68_large = hr_calc_poisson_large.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.5,
    hazard_ratio_max=10.0,
)

_, lower_ci_95_large, upper_ci_95_large, _ = hr_calc_poisson_large.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr_large:.3f}")
print(f"68% CI: [{lower_ci_68_large:.3f}, {upper_ci_68_large:.3f}]")
print(f"95% CI: [{lower_ci_95_large:.3f}, {upper_ci_95_large:.3f}]")
print(f"\n2NLL at best fit: {result_68_large.x:.2f}")

# Perform likelihood scan
hazard_ratios_large, twonll_values_large, best_fit_result_large = hr_calc_poisson_large.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.5,
    hazard_ratio_max=6.0,
    cox_only=False
)

# Store for comparison
poisson_large_scan_data = {
    'hrs': hazard_ratios_large,
    'twonll': twonll_values_large,
    'best_fit': best_fit_hr_large,
    'ci_68': (lower_ci_68_large, upper_ci_68_large),
    'ci_95': (lower_ci_95_large, upper_ci_95_large),
}
```

### Comparison: Fixed vs. Poisson (Large Counts)

Let's compare the results. With large counts, the Poisson density results should be very similar to the fixed case.

```python
print("Comparison: Fixed vs. Poisson (Large Counts)")
print("=" * 60)
print("                          Fixed      Poisson-Large")
print(f"Best-fit HR:             {best_fit_hr_fixed:6.3f}      {best_fit_hr_large:6.3f}")
print(f"68% CI:          [{lower_ci_68_fixed:5.3f}, {upper_ci_68_fixed:5.3f}]  [{lower_ci_68_large:5.3f}, {upper_ci_68_large:5.3f}]")
print(f"95% CI:          [{lower_ci_95_fixed:5.3f}, {upper_ci_95_fixed:5.3f}]  [{lower_ci_95_large:5.3f}, {upper_ci_95_large:5.3f}]")
print(f"\nDifference in log(HR): {abs(np.log(best_fit_hr_fixed) - np.log(best_fit_hr_large)):.4f}")
print("\nAs expected, with large Poisson counts, results are very similar!")
print("The Cox error dominates, and measurement uncertainty is negligible.")
```

## Example 3: Poisson Density with Moderate Counts

Finally, let's examine a **poisson_density observable** with **moderate counts** (e.g., tens of cells). Now the Poisson error becomes significant (√N/N ~ 15-20%):

- Patients near the threshold have ~10% probability of crossing to the other group
- **Both Cox error and Poisson error** contribute to confidence intervals
- The likelihood scan should show **noticeable differences** from the fixed case
- Confidence intervals will be **wider** due to measurement uncertainty

This demonstrates KoMbine's key advantage: properly accounting for biomarker measurement uncertainty in survival analysis.

```python
# Load Poisson density datacard with moderate counts
dcfile_poisson_moderate = datacards_dir / "poisson_density_hr_example_moderate.txt"
datacard_poisson_moderate = Datacard.parse_datacard(dcfile_poisson_moderate)

print("=" * 60)
print("POISSON DENSITY - MODERATE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_moderate.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard_poisson_moderate.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard_poisson_moderate.patients if p.censored)}")

# Check the counts to show they're moderate
nums = [p.observable.numerator for p in datacard_poisson_moderate.patients]
areas = [p.observable.denominator for p in datacard_poisson_moderate.patients]
densities = [n/a for n, a in zip(nums, areas)]
print("\nCount statistics:")
print(f"  Mean count: {np.mean(nums):.1f}")
print(f"  Range: [{min(nums)}, {max(nums)}]")
print(f"  Relative uncertainty (√N/N): {np.mean([np.sqrt(n)/n for n in nums]):.1%}")

# Estimate probability of crossing threshold for patients near it
near_threshold = [p for p in datacard_poisson_moderate.patients if 0.4 <= p.observable.numerator/p.observable.denominator <= 0.6]
print(f"\nPatients near threshold (density 0.4-0.6): {len(near_threshold)}")
if near_threshold:
    # Rough estimate: probability that observed density ± √N/N crosses threshold
    example = near_threshold[0]
    density = example.observable.numerator / example.observable.denominator
    rel_unc = np.sqrt(example.observable.numerator) / example.observable.numerator
    print(f"  Example: density={density:.3f}, rel. unc.={rel_unc:.1%}")
    print(f"  -> ~{rel_unc*100:.0f}% chance of crossing threshold")
print("\nWith moderate counts, measurement uncertainty is significant!")
```

```python
# Create hazard ratio calculator for Poisson density (moderate counts)
hr_calc_poisson_moderate = datacard_poisson_moderate.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

# Calculate confidence intervals
best_fit_hr_moderate, lower_ci_68_moderate, upper_ci_68_moderate, result_68_moderate = hr_calc_poisson_moderate.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.5,
    hazard_ratio_max=10.0,
)

_, lower_ci_95_moderate, upper_ci_95_moderate, _ = hr_calc_poisson_moderate.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr_moderate:.3f}")
print(f"68% CI: [{lower_ci_68_moderate:.3f}, {upper_ci_68_moderate:.3f}]")
print(f"95% CI: [{lower_ci_95_moderate:.3f}, {upper_ci_95_moderate:.3f}]")
print(f"\n2NLL at best fit: {result_68_moderate.x:.2f}")

# Perform likelihood scan
hazard_ratios_moderate, twonll_values_moderate, best_fit_result_moderate = hr_calc_poisson_moderate.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.5,
    hazard_ratio_max=6.0,
    cox_only=False
)

# Store for comparison
poisson_moderate_scan_data = {
    'hrs': hazard_ratios_moderate,
    'twonll': twonll_values_moderate,
    'best_fit': best_fit_hr_moderate,
    'ci_68': (lower_ci_68_moderate, upper_ci_68_moderate),
    'ci_95': (lower_ci_95_moderate, upper_ci_95_moderate),
}
```

### Comparison: All Three Cases

Now let's compare all three examples to see how measurement uncertainty affects the results.

```python
print("Summary Comparison")
print("=" * 80)
print("                          Fixed      Poisson-Large  Poisson-Moderate")
print(f"Best-fit HR:             {best_fit_hr_fixed:6.3f}      {best_fit_hr_large:6.3f}          {best_fit_hr_moderate:6.3f}")
print(f"68% CI width:            {upper_ci_68_fixed - lower_ci_68_fixed:6.3f}      {upper_ci_68_large - lower_ci_68_large:6.3f}          {upper_ci_68_moderate - lower_ci_68_moderate:6.3f}")
print(f"95% CI width:            {upper_ci_95_fixed - lower_ci_95_fixed:6.3f}      {upper_ci_95_large - lower_ci_95_large:6.3f}          {upper_ci_95_moderate - lower_ci_95_moderate:6.3f}")
print("=" * 80)

# Calculate CI width ratios
print("\nConfidence interval width ratios (relative to fixed):")
print(f"  Poisson-Large   (68%): {(upper_ci_68_large - lower_ci_68_large) / (upper_ci_68_fixed - lower_ci_68_fixed):.2f}x")
print(f"  Poisson-Large   (95%): {(upper_ci_95_large - lower_ci_95_large) / (upper_ci_95_fixed - lower_ci_95_fixed):.2f}x")
print(f"  Poisson-Moderate(68%): {(upper_ci_68_moderate - lower_ci_68_moderate) / (upper_ci_68_fixed - lower_ci_68_fixed):.2f}x")
print(f"  Poisson-Moderate(95%): {(upper_ci_95_moderate - lower_ci_95_moderate) / (upper_ci_95_fixed - lower_ci_95_fixed):.2f}x")

print("\nKey observations:")
print("1. Large counts → CIs similar to fixed (measurement error negligible)")
print("2. Moderate counts → CIs noticeably wider (measurement error significant)")
print("3. Best-fit HRs remain similar (same underlying survival distributions)")
```

## Visualizing the Three Likelihood Scans

Let's create a comprehensive visualization comparing all three cases.

```python
# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chi-squared thresholds
chi2_68 = 1.0
chi2_95 = 3.84

# Plot 1: All three scans overlaid (linear scale)
ax = axes[0, 0]
twonll_min_fixed = np.min(twonll_values_fixed)
ax.plot(hazard_ratios_fixed, twonll_values_fixed - twonll_min_fixed, 'b-', linewidth=2, label='Fixed', alpha=0.8)
ax.plot(hazard_ratios_large, twonll_values_large - np.min(twonll_values_large), 'g-', linewidth=2, label='Poisson-Large', alpha=0.8)
ax.plot(hazard_ratios_moderate, twonll_values_moderate - np.min(twonll_values_moderate), 'r-', linewidth=2, label='Poisson-Moderate', alpha=0.8)

ax.axhline(chi2_68, color='orange', linestyle=':', alpha=0.7, label='68% CI threshold')
ax.axhline(chi2_95, color='purple', linestyle=':', alpha=0.7, label='95% CI threshold')

ax.set_xlabel('Hazard Ratio', fontsize=12)
ax.set_ylabel('Δ(2NLL) = 2NLL - 2NLL_min', fontsize=12)
ax.set_title('Likelihood Comparison (Linear Scale)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Plot 2: All three scans overlaid (log scale)
ax = axes[0, 1]
ax.plot(hazard_ratios_fixed, twonll_values_fixed - twonll_min_fixed, 'b-', linewidth=2, label='Fixed', alpha=0.8)
ax.plot(hazard_ratios_large, twonll_values_large - np.min(twonll_values_large), 'g-', linewidth=2, label='Poisson-Large', alpha=0.8)
ax.plot(hazard_ratios_moderate, twonll_values_moderate - np.min(twonll_values_moderate), 'r-', linewidth=2, label='Poisson-Moderate', alpha=0.8)

ax.axhline(chi2_68, color='orange', linestyle=':', alpha=0.7, label='68% CI threshold')
ax.axhline(chi2_95, color='purple', linestyle=':', alpha=0.7, label='95% CI threshold')

ax.set_xlabel('Hazard Ratio', fontsize=12)
ax.set_ylabel('Δ(2NLL) = 2NLL - 2NLL_min', fontsize=12)
ax.set_title('Likelihood Comparison (Log Scale)', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 10])

# Plot 3: Confidence interval comparison (68%)
ax = axes[1, 0]
cases = ['Fixed', 'Poisson\nLarge', 'Poisson\nModerate']
best_fits = [best_fit_hr_fixed, best_fit_hr_large, best_fit_hr_moderate]
ci_68_lower = [lower_ci_68_fixed, lower_ci_68_large, lower_ci_68_moderate]
ci_68_upper = [upper_ci_68_fixed, upper_ci_68_large, upper_ci_68_moderate]

x_pos = np.arange(len(cases))
ax.errorbar(x_pos, best_fits, 
            yerr=[np.array(best_fits) - np.array(ci_68_lower), 
                  np.array(ci_68_upper) - np.array(best_fits)],
            fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2,
            color='darkblue', ecolor='orange', label='68% CI')

ax.set_xticks(x_pos)
ax.set_xticklabels(cases)
ax.set_ylabel('Hazard Ratio', fontsize=12)
ax.set_title('68% Confidence Intervals', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Plot 4: Confidence interval comparison (95%)
ax = axes[1, 1]
ci_95_lower = [lower_ci_95_fixed, lower_ci_95_large, lower_ci_95_moderate]
ci_95_upper = [upper_ci_95_fixed, upper_ci_95_large, upper_ci_95_moderate]

ax.errorbar(x_pos, best_fits, 
            yerr=[np.array(best_fits) - np.array(ci_95_lower), 
                  np.array(ci_95_upper) - np.array(best_fits)],
            fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2,
            color='darkblue', ecolor='purple', label='95% CI')

ax.set_xticks(x_pos)
ax.set_xticklabels(cases)
ax.set_ylabel('Hazard Ratio', fontsize=12)
ax.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
plt.savefig('hazard_ratio_likelihood_scan_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()

print("\nFigure saved as 'hazard_ratio_likelihood_scan_comparison.pdf'")
```

## Key Insights

This notebook demonstrated three important scenarios for hazard ratio analysis:

### 1. Fixed Observable (Baseline)
- Patient group assignments are deterministic
- Only **Cox error** (finite events) contributes to uncertainty
- Serves as a reference for comparison

### 2. Poisson Density with Large Counts
- Measurement uncertainty is small (√N/N ~ 2-3%)
- Results converge to the fixed case
- Demonstrates that KoMbine correctly reduces to Cox proportional hazards when measurement error is negligible

### 3. Poisson Density with Moderate Counts
- Measurement uncertainty is significant (√N/N ~ 15-20%)
- ~10% of patients near threshold have ambiguous group assignment
- Confidence intervals are **noticeably wider** than the fixed case
- Demonstrates KoMbine's key advantage: properly accounting for biomarker measurement uncertainty

## Interpreting the Results

The hazard ratio quantifies the relative risk between two groups:

- **Best-fit HR**: Maximum likelihood estimate of the hazard ratio
- **Confidence intervals**: Derived from profile likelihood, accounting for:
  - Statistical uncertainty (finite number of patients - Cox error)
  - Measurement error in the biomarker (Poisson uncertainty)
- **Width of CIs**: Reflects total uncertainty from both sources

## When Does Measurement Uncertainty Matter?

From our examples:
- **Large counts** (√N/N < 5%): Measurement error negligible, standard Cox methods sufficient
- **Moderate counts** (√N/N ~ 10-20%): Measurement error significant, KoMbine's full model needed
- **Patients near threshold**: Most susceptible to group assignment uncertainty

## Summary

This notebook demonstrated:

1. ✅ Calculating hazard ratios with confidence intervals for different observable types
2. ✅ Comparing fixed vs. Poisson density observables
3. ✅ Showing how measurement uncertainty widens confidence intervals
4. ✅ Visualizing likelihood scans across multiple scenarios
5. ✅ Understanding when KoMbine's full model provides value over standard Cox analysis

The hazard ratio provides a clinically interpretable effect size that complements the p-value from hypothesis testing, and KoMbine ensures that biomarker measurement uncertainty is properly incorporated into the analysis.
