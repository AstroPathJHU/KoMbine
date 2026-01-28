---
jupyter:
  jupytext:
    formats: ipynb,md,py
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

## Load Example Data

We'll use a simple example dataset with Poisson-distributed biomarker measurements.

```python
# Path to example datacard with clear separation between groups
# Using pathlib to construct a path relative to this notebook
import pathlib
notebook_dir = pathlib.Path().resolve()
test_dir = notebook_dir.parent.parent / "test" / "kombine"
dcfile = test_dir / "datacards" / "simple_examples" / "fixed_hr_example.txt"

# Load the datacard
datacard = Datacard.parse_datacard(dcfile)

print(f"Loaded {len(datacard.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard.patients if p.censored)}")
```

## Calculate Hazard Ratio with Confidence Interval

First, let's calculate the best-fit hazard ratio and its confidence interval.

**Note on hazard ratio bounds**: The optimizer uses bounds on log(HR) to keep the problem well-conditioned. By default, these are set to [-10, 10], corresponding to HR ∈ [0.000045, 22026]. If your analysis requires exploring more extreme hazard ratios, you can adjust these bounds using the `log_hazard_ratio_bounds` parameter.

```python
# Create hazard ratio calculator using the datacard factory method
# For "fixed" observable type, threshold divides patients into two groups
# Threshold of 0.5 splits into low-risk (< 0.5) and high-risk (>= 0.5) groups

hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.0,
    parameter_max=1.0,
    # log_hazard_ratio_bounds=(-10.0, 10.0),  # Default, can be adjusted if needed
)

# Calculate 68% confidence interval (1 sigma)
best_fit_hr, lower_ci_68, upper_ci_68, result_68 = hr_calc.hazard_ratio_confidence_interval(
    cox_only=False,  # Allow patient assignments to float
    confidence_level=0.68,
    hazard_ratio_min=0.5,
    hazard_ratio_max=10.0,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr:.3f}")
print(f"68% CI: [{lower_ci_68:.3f}, {upper_ci_68:.3f}]")
print(f"2NLL at best fit: {result_68.x:.2f}")
print(f"\nPatient distribution:")
print(f"  Low group: {result_68.n_total_low} patients ({result_68.n_alive_low} alive at end)")
print(f"  High group: {result_68.n_total_high} patients ({result_68.n_alive_high} alive at end)")
```

```python
# Also calculate 95% confidence interval
_, lower_ci_95, upper_ci_95, _ = hr_calc.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
)

print(f"95% CI: [{lower_ci_95:.3f}, {upper_ci_95:.3f}]")
```

## Likelihood Scan

Now let's visualize the likelihood as a function of the hazard ratio.

```python
# Perform likelihood scan
# Use a range that includes typical hazard ratios
hazard_ratios, twonll_values, best_fit_result = hr_calc.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.5,
    hazard_ratio_max=6.0,
    cox_only=False
)

print(f"Likelihood scan completed over {len(hazard_ratios)} points")
print(f"Minimum 2NLL: {np.min(twonll_values):.2f} at HR = {hazard_ratios[np.argmin(twonll_values)]:.3f}")

# Verify the best fit from scan matches the minimum
best_fit_from_scan = best_fit_result.hazard_ratio
print(f"Best fit HR from scan result: {best_fit_from_scan:.3f}")
print(f"Best fit HR from CI calculation: {best_fit_hr:.3f}")
print(f"Difference in log(HR): {abs(np.log(best_fit_from_scan) - np.log(best_fit_hr)):.4f}")

# Verify the reported minimum is actually the minimum
assert best_fit_result.hazard_ratio == hazard_ratios[np.argmin(twonll_values)], \
    "Best fit result should correspond to minimum of scan"
```

```python
# Plot the likelihood scan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: 2NLL vs Hazard Ratio (linear scale)
ax1.plot(hazard_ratios, twonll_values, 'b-', linewidth=2, label='2NLL')
# Use the best fit from the actual scan
ax1.axvline(best_fit_from_scan, color='r', linestyle='--', label=f'Best fit: {best_fit_from_scan:.2f}')

# Mark confidence intervals
chi2_68 = 1.0  # chi^2(1 df, 68% CL) ≈ 1.0
chi2_95 = 3.84  # chi^2(1 df, 95% CL) ≈ 3.84
twonll_min = np.min(twonll_values)

ax1.axhline(twonll_min + chi2_68, color='orange', linestyle=':', alpha=0.7, label='68% CI threshold')
ax1.axhline(twonll_min + chi2_95, color='green', linestyle=':', alpha=0.7, label='95% CI threshold')

# Mark CI boundaries
ax1.axvline(lower_ci_68, color='orange', linestyle=':', alpha=0.5)
ax1.axvline(upper_ci_68, color='orange', linestyle=':', alpha=0.5)
ax1.axvline(lower_ci_95, color='green', linestyle=':', alpha=0.5)
ax1.axvline(upper_ci_95, color='green', linestyle=':', alpha=0.5)

ax1.set_xlabel('Hazard Ratio', fontsize=12)
ax1.set_ylabel('Twice Negative Log Likelihood (2NLL)', fontsize=12)
ax1.set_title('Profile Likelihood for Hazard Ratio', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Same but with log scale for hazard ratio
ax2.plot(hazard_ratios, twonll_values, 'b-', linewidth=2, label='2NLL')
ax2.axvline(best_fit_from_scan, color='r', linestyle='--', label=f'Best fit: {best_fit_from_scan:.2f}')
ax2.axhline(twonll_min + chi2_68, color='orange', linestyle=':', alpha=0.7, label='68% CI threshold')
ax2.axhline(twonll_min + chi2_95, color='green', linestyle=':', alpha=0.7, label='95% CI threshold')

ax2.axvline(lower_ci_68, color='orange', linestyle=':', alpha=0.5)
ax2.axvline(upper_ci_68, color='orange', linestyle=':', alpha=0.5)
ax2.axvline(lower_ci_95, color='green', linestyle=':', alpha=0.5)
ax2.axvline(upper_ci_95, color='green', linestyle=':', alpha=0.5)

ax2.set_xlabel('Hazard Ratio', fontsize=12)
ax2.set_ylabel('Twice Negative Log Likelihood (2NLL)', fontsize=12)
ax2.set_title('Profile Likelihood (Log Scale)', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hazard_ratio_likelihood_scan.pdf', bbox_inches='tight', dpi=300)
plt.show()

print(f"\nFigure saved as 'hazard_ratio_likelihood_scan.pdf'")
```

## Comparison with p-value Calculation

Let's verify consistency with the p-value calculation from the KoMbine package.

```python
# Create p-value calculator using the datacard factory method
pval_calc = datacard.km_p_value(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

# Calculate p-value
p_value, result_null, result_alt = pval_calc.solve_and_pvalue(cox_only=False)

print(f"p-value: {p_value:.4f}")
print(f"\nNull hypothesis (HR = 1):")
print(f"  2NLL: {result_null.x:.2f}")
print(f"  Hazard ratio: {result_null.hazard_ratio:.3f}")

print(f"\nAlternative hypothesis (HR free to float):")
print(f"  2NLL: {result_alt.x:.2f}")
print(f"  Hazard ratio: {result_alt.hazard_ratio:.3f}")

print(f"\nLikelihood ratio test statistic: {result_null.x - result_alt.x:.2f}")

# Compare with our hazard ratio calculation (in log scale for proper comparison)
print(f"\nConsistency check:")
print(f"  HR from p-value calc: {result_alt.hazard_ratio:.3f}")
print(f"  HR from profile likelihood: {best_fit_hr:.3f}")
print(f"  Difference in log(HR): {abs(np.log(result_alt.hazard_ratio) - np.log(best_fit_hr)):.4f}")
```

## Cox-Only Mode

KoMbine can also fix patient assignments to their nominal groups (based on observed biomarker values) and only use the Cox partial likelihood. This is similar to standard Cox regression but with exact binomial likelihoods.

```python
# Calculate with cox_only=True
best_fit_hr_cox, lower_ci_cox, upper_ci_cox, result_cox = hr_calc.hazard_ratio_confidence_interval(
    cox_only=True,  # Fix patient assignments
    confidence_level=0.68,
)

print(f"\nCox-only mode (fixed patient assignments):")
print(f"Best-fit hazard ratio: {best_fit_hr_cox:.3f}")
print(f"68% CI: [{lower_ci_cox:.3f}, {upper_ci_cox:.3f}]")

print(f"\nComparison:")
print(f"  Full model (floating assignments): {best_fit_hr:.3f} [{lower_ci_68:.3f}, {upper_ci_68:.3f}]")
print(f"  Cox-only (fixed assignments):      {best_fit_hr_cox:.3f} [{lower_ci_cox:.3f}, {upper_ci_cox:.3f}]")
```

## Interpreting the Results

The hazard ratio quantifies the relative risk between the two groups:

- **Best-fit HR**: The maximum likelihood estimate of the hazard ratio
- **Confidence intervals**: Derived from the profile likelihood, accounting for both statistical uncertainty (finite number of patients) and measurement error in the biomarker
- **Cox-only vs. Full model**: The full model allows patients to move between groups based on their biomarker uncertainty, which can lead to wider confidence intervals

## Summary

This notebook demonstrated:

1. ✅ Calculating the best-fit hazard ratio with confidence intervals
2. ✅ Performing a profile likelihood scan over hazard ratio values
3. ✅ Visualizing the likelihood surface
4. ✅ Comparing with p-value calculations for consistency
5. ✅ Using Cox-only mode for comparison with standard approaches

The hazard ratio provides a clinically interpretable effect size that complements the p-value from hypothesis testing.
