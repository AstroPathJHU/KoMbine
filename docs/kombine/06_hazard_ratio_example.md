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

# Plot the likelihood scan
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Chi-squared thresholds for confidence intervals
chi2_68 = 1.0
chi2_95 = 3.84

# Compute delta 2NLL
twonll_min_fixed = np.min(twonll_values_fixed)
delta_twonll = twonll_values_fixed - twonll_min_fixed

# Plot the profile likelihood
ax.plot(hazard_ratios_fixed, delta_twonll, 'b-', linewidth=2.5, label='Fixed Observable')
ax.axhline(chi2_68, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI threshold')
ax.axhline(chi2_95, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='95% CI threshold')
ax.axvline(best_fit_hr_fixed, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Best fit: {best_fit_hr_fixed:.2f}')

ax.set_xlabel('Hazard Ratio', fontsize=12)
ax.set_ylabel(r'$-2 \Delta \ln L$', fontsize=12)
ax.set_title('Profile Likelihood: Fixed Observable', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 6.0)
ax.set_ylim(0, 15)

plt.tight_layout()
plt.show()
```

## Example 2: Poisson Density with Large Counts

Now let's consider a **poisson_density observable type** where biomarker measurements have Poisson uncertainty. With **large counts** (e.g., hundreds of cells counted in a region), the Poisson error is small (√N/N ~ 2-3%), so:

- Patients are well-localized around their true parameter value
- The **Cox error dominates** over measurement error
- Confidence intervals should be **similar to the fixed case**

This demonstrates that when measurement uncertainty is small, KoMbine's results converge to the Cox proportional hazards case.

```python
# Load Poisson density datacard with large counts
dcfile_poisson_large_counts = datacards_dir / "poisson_density_hr_example_large.txt"
datacard_poisson_large_counts = Datacard.parse_datacard(dcfile_poisson_large_counts)

print("=" * 60)
print("POISSON DENSITY - LARGE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_large_counts.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard_poisson_large_counts.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard_poisson_large_counts.patients if p.censored)}")

# Check the counts to show they're large
nums = [p.observable.numerator for p in datacard_poisson_large_counts.patients]  # type: ignore[union-attr]
areas = [p.observable.denominator for p in datacard_poisson_large_counts.patients]  # type: ignore[union-attr]
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
hr_calc_poisson_large_counts = datacard_poisson_large_counts.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

# Calculate confidence intervals
best_fit_hr_large, lower_ci_68_large, upper_ci_68_large, result_68_large = hr_calc_poisson_large_counts.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.5,
    hazard_ratio_max=10.0,
)

_, lower_ci_95_large, upper_ci_95_large, _ = hr_calc_poisson_large_counts.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr_large:.3f}")
print(f"68% CI: [{lower_ci_68_large:.3f}, {upper_ci_68_large:.3f}]")
print(f"95% CI: [{lower_ci_95_large:.3f}, {upper_ci_95_large:.3f}]")
print(f"\n2NLL at best fit: {result_68_large.x:.2f}")

# Perform likelihood scan
hazard_ratios_large, twonll_values_large, best_fit_result_large = hr_calc_poisson_large_counts.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.5,
    hazard_ratio_max=6.0,
    cox_only=False
)

# Store for comparison
poisson_large_counts_scan_data = {
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
print("=" * 70)
print(f"{'':30} {'Fixed':>15} {'Poisson-Large':>15}")
print(f"{'Best-fit HR':30} {best_fit_hr_fixed:>15.3f} {best_fit_hr_large:>15.3f}")
print(f"{'68% CI':30} [{lower_ci_68_fixed:5.3f}, {upper_ci_68_fixed:5.3f}] [{lower_ci_68_large:5.3f}, {upper_ci_68_large:5.3f}]")
print(f"{'95% CI':30} [{lower_ci_95_fixed:5.3f}, {upper_ci_95_fixed:5.3f}] [{lower_ci_95_large:5.3f}, {upper_ci_95_large:5.3f}]")
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
dcfile_poisson_moderate_counts = datacards_dir / "poisson_density_hr_example_moderate.txt"
datacard_poisson_moderate_counts = Datacard.parse_datacard(dcfile_poisson_moderate_counts)

print("=" * 60)
print("POISSON DENSITY - MODERATE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_moderate_counts.patients)} patients")
print(f"Number of deaths: {sum(1 for p in datacard_poisson_moderate_counts.patients if not p.censored)}")
print(f"Number of censored: {sum(1 for p in datacard_poisson_moderate_counts.patients if p.censored)}")

# Check the counts to show they're moderate
nums = [p.observable.numerator for p in datacard_poisson_moderate_counts.patients]  # type: ignore[union-attr]
areas = [p.observable.denominator for p in datacard_poisson_moderate_counts.patients]  # type: ignore[union-attr]
densities = [n/a for n, a in zip(nums, areas)]
print("\nCount statistics:")
print(f"  Mean count: {np.mean(nums):.1f}")
print(f"  Range: [{min(nums)}, {max(nums)}]")
print(f"  Relative uncertainty (√N/N): {np.mean([np.sqrt(n)/n for n in nums]):.1%}")

# Estimate probability of crossing threshold for patients near it
near_threshold = [p for p in datacard_poisson_moderate_counts.patients if 0.4 <= p.observable.numerator/p.observable.denominator <= 0.6]  # type: ignore[union-attr]
print(f"\nPatients near threshold (density 0.4-0.6): {len(near_threshold)}")
if near_threshold:
    # Rough estimate: probability that observed density ± √N/N crosses threshold
    example = near_threshold[0]
    density = example.observable.numerator / example.observable.denominator  # type: ignore[union-attr]
    rel_unc = np.sqrt(example.observable.numerator) / example.observable.numerator  # type: ignore[union-attr]
    print(f"  Example: density={density:.3f}, rel. unc.={rel_unc:.1%}")
    print(f"  -> ~{rel_unc*100:.0f}% chance of crossing threshold")
print("\nWith moderate counts, measurement uncertainty is significant!")
```

```python
# Create hazard ratio calculator for Poisson density (moderate counts)
hr_calc_poisson_moderate_counts = datacard_poisson_moderate_counts.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

# Calculate confidence intervals with extended bounds to find thresholds
best_fit_hr_moderate, lower_ci_68_moderate, upper_ci_68_moderate, result_68_moderate = hr_calc_poisson_moderate_counts.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.1,  # Extended from 0.5
    hazard_ratio_max=15.0,  # Extended from 10.0
)

_, lower_ci_95_moderate, upper_ci_95_moderate, _ = hr_calc_poisson_moderate_counts.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=0.1,  # Extended
    hazard_ratio_max=15.0,  # Extended
)

# Format CI values for display: show "<value" or ">value" if boundary not found
def format_ci_bound(value, bound_type, extended_bound):
    """Format CI bound, showing < or > if at extended boundary"""
    if bound_type == "lower":
        if abs(value - extended_bound) < 0.01:  # At lower boundary
            return f"<{extended_bound}"
        return f"{value:.3f}"
    else:  # upper
        if abs(value - extended_bound) < 0.01:  # At upper boundary
            return f">{extended_bound}"
        return f"{value:.3f}"

print(f"\nBest-fit hazard ratio: {best_fit_hr_moderate:.3f}")
print(f"68% CI: [{format_ci_bound(lower_ci_68_moderate, 'lower', 0.1)}, {format_ci_bound(upper_ci_68_moderate, 'upper', 15.0)}]")
print(f"95% CI: [{format_ci_bound(lower_ci_95_moderate, 'lower', 0.1)}, {format_ci_bound(upper_ci_95_moderate, 'upper', 15.0)}]")
print(f"\n2NLL at best fit: {result_68_moderate.x:.2f}")

# Perform likelihood scan with extended bounds
hazard_ratios_moderate, twonll_values_moderate, best_fit_result_moderate = hr_calc_poisson_moderate_counts.likelihood_scan_hazard_ratio(
    n_points=50,
    hazard_ratio_min=0.1,  # Extended from 0.5
    hazard_ratio_max=15.0,  # Extended from 6.0
    cox_only=False
)

# Store for comparison
poisson_moderate_counts_scan_data = {
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
# Helper function to draw error bars with arrows for unbounded CIs
def plot_confidence_intervals(ax, x_pos, best_fits, ci_lower, ci_upper, color, label, extended_lower=None, extended_upper=None):
    """
    Plot error bars with arrows at ends where CI bounds hit extended search limits
    """
    # Determine which bounds are at extended limits
    bounded_lower = [True] * len(x_pos)
    bounded_upper = [True] * len(x_pos)
    
    if extended_lower is not None:
        for i in range(len(x_pos)):
            if isinstance(ci_lower[i], str) or abs(ci_lower[i] - extended_lower) < 0.01:
                bounded_lower[i] = False
                ci_lower[i] = extended_lower if isinstance(ci_lower[i], str) else ci_lower[i]
    
    if extended_upper is not None:
        for i in range(len(x_pos)):
            if isinstance(ci_upper[i], str) or abs(ci_upper[i] - extended_upper) < 0.01:
                bounded_upper[i] = False
                ci_upper[i] = extended_upper if isinstance(ci_upper[i], str) else ci_upper[i]
    
    # Convert to numpy arrays for arithmetic
    ci_lower_vals = np.array([float(c) if isinstance(c, str) else c for c in ci_lower])
    ci_upper_vals = np.array([float(c) if isinstance(c, str) else c for c in ci_upper])
    
    # Plot main error bars
    for i in range(len(x_pos)):
        lower_err = best_fits[i] - ci_lower_vals[i]
        upper_err = ci_upper_vals[i] - best_fits[i]
        
        # Plot bar
        ax.plot([x_pos[i], x_pos[i]], [ci_lower_vals[i], ci_upper_vals[i]], 
                color=color, linewidth=2, zorder=1)
        
        # Plot point
        ax.plot(x_pos[i], best_fits[i], 'o', markersize=10, color='darkblue', zorder=3)
        
        # Add arrows at ends where bounds are unbounded
        arrow_size = 0.12  # relative to y-axis on log scale
        
        if not bounded_lower[i]:
            # Draw downward arrow at lower end
            ax.annotate('', xy=(x_pos[i], ci_lower_vals[i]), 
                       xytext=(x_pos[i], ci_lower_vals[i] * 1.1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        else:
            # Draw cap at lower end
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [ci_lower_vals[i], ci_lower_vals[i]], 
                   color=color, linewidth=2, zorder=2)
        
        if not bounded_upper[i]:
            # Draw upward arrow at upper end
            ax.annotate('', xy=(x_pos[i], ci_upper_vals[i]), 
                       xytext=(x_pos[i], ci_upper_vals[i] / 1.1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        else:
            # Draw cap at upper end
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [ci_upper_vals[i], ci_upper_vals[i]], 
                   color=color, linewidth=2, zorder=2)
    
    # Add dummy point for legend
    ax.plot([], [], 'o-', color=color, linewidth=2, markersize=10, label=label)

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
plot_confidence_intervals(ax, x_pos, best_fits, ci_68_lower, ci_68_upper, 
                         'orange', '68% CI', extended_lower=0.1, extended_upper=15.0)

ax.set_xticks(x_pos)
ax.set_xticklabels(cases)
ax.set_yscale("log")
ax.set_ylabel('Hazard Ratio', fontsize=12)
ax.set_title('68% Confidence Intervals', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Plot 4: Confidence interval comparison (95%)
ax = axes[1, 1]
ci_95_lower = [lower_ci_95_fixed, lower_ci_95_large, lower_ci_95_moderate]
ci_95_upper = [upper_ci_95_fixed, upper_ci_95_large, upper_ci_95_moderate]

plot_confidence_intervals(ax, x_pos, best_fits, ci_95_lower, ci_95_upper, 
                         'purple', '95% CI', extended_lower=0.1, extended_upper=15.0)

ax.set_xticks(x_pos)
ax.set_xticklabels(cases)
ax.set_ylabel('Hazard Ratio', fontsize=12)
ax.set_yscale("log")
ax.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
plt.savefig('hazard_ratio_likelihood_scan_comparison.pdf', bbox_inches='tight', dpi=300)
plt.show()

print("\nFigure saved as 'hazard_ratio_likelihood_scan_comparison.pdf'")
```

## Comparing MINLP vs. Yi's Discrete Covariate Misclassification Method

Now that we've visualized the three likelihood scans, let's quantitatively compare KoMbine's MINLP approach with the discrete covariate misclassification method presented in Yi's *Statistical Analysis with Measurement Error or Misclassification* (Springer 2017, §3.7.1). This method estimates a single misclassification matrix Π that applies uniformly to all patients, while MINLP optimizes each patient's group assignment individually.

```python
# Compare all three datasets: fixed, large counts, and moderate counts
# Gurobi output is suppressed for readability; progress dots will be shown.
threshold = 0.5
hazard_ratios_to_test = [1.0, 1.5, 2.0, 2.5, 3.0]

for datacard_name, datacard, hr_calc in [
    ("FIXED (No Measurement Uncertainty)", datacard_fixed, hr_calc_fixed),
    ("LARGE COUNTS (Small Poisson Errors)", datacard_poisson_large_counts, hr_calc_poisson_large_counts),
    ("MODERATE COUNTS (Larger Poisson Errors)", datacard_poisson_moderate_counts, hr_calc_poisson_moderate_counts)
]:
    print(f"\n{datacard_name} - Comparing MINLP vs Yi's Method")
    print("=" * 70)
    print(f"{'HR':<8} {'MINLP 2NLL':<15} {'Yi 2NLL':<15} {'Difference':<12}")
    print("-" * 70)
    for hr_test in hazard_ratios_to_test:
        # MINLP approach (suppress Gurobi output, show progress)
        result_minlp = hr_calc.compute_2nll_at_hazard_ratio(
            hazard_ratio=hr_test,
            cox_only=False,
            print_progress=False,
            verbose=False
        )
        # Yi's method
        result_yi = datacard.km_hazard_ratio_yi(
            parameter_threshold=threshold,
            hazard_ratio=hr_test,
            parameter_min=0.01,
            parameter_max=0.99,
            method='bayesian',
        )
        diff = result_minlp.x - result_yi['x']
        print(f"{hr_test:<8.1f} {result_minlp.x:<15.4f} {result_yi['x']:<15.4f} {diff:<12.4f}")
```

## Analyzing Measurement Error Effects

The comparison table above shows how closely MINLP and the discrete covariate misclassification method (Yi §3.7.1) agree across different measurement scenarios. Now let's examine the underlying mechanisms: the misclassification matrices and the full likelihood profile.


### Understanding Yi's Per-Patient Probability Weighting

Yi's method (§3.7.1) uses inverse probability weighting to correct for measurement error. Our implementation computes individual probabilities for each patient rather than using an aggregate misclassification matrix. This per-patient approach is more accurate as it accounts for individual measurement uncertainty.


### Likelihood Scan Comparison: KoMbine MINLP vs. Yi's Method

The following plots show the profile likelihood (-2ΔlnL) and absolute NLL for both methods across all three scenarios. In the fixed case, the agreement is perfect (to numerical precision) in both -2ΔlnL and absolute NLL. For large counts, the agreement is good. For moderate counts, differences emerge when measurement error is large, as we will discuss below.

```python
# Compare likelihood scans for all three datasets
# Gurobi output is suppressed for readability; progress dots will be shown.
hazard_ratios_scan = np.linspace(0.5, 4.0, 50)

# Create 3x2 plot grid (3 rows for fixed/large/moderate, 2 columns for delta and absolute)
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

for row_idx, (datacard_name, datacard, hr_calc) in enumerate([
    ("Fixed", datacard_fixed, hr_calc_fixed),
    ("Large Counts", datacard_poisson_large_counts, hr_calc_poisson_large_counts),
    ("Moderate Counts", datacard_poisson_moderate_counts, hr_calc_poisson_moderate_counts)
]):
    nll_minlp = []
    nll_yi = []
    # Compute likelihood scans
    for hr in hazard_ratios_scan:
        # MINLP approach (suppress Gurobi output, show progress)
        result_minlp = hr_calc.compute_2nll_at_hazard_ratio(
            hazard_ratio=hr,
            cox_only=False,
            print_progress=False,
            verbose=False
        )
        nll_minlp.append(result_minlp.x)
        # Yi's method
        result_yi = datacard.km_hazard_ratio_yi(
            parameter_threshold=threshold,
            hazard_ratio=hr,
            parameter_min=0.01,
            parameter_max=0.99,
            method='bayesian',
        )
        nll_yi.append(result_yi['x'])
    # Convert to arrays and compute minima
    nll_minlp = np.array(nll_minlp)
    nll_yi = np.array(nll_yi)
    minlp_min = np.min(nll_minlp)
    yi_min = np.min(nll_yi)
    # Compute -2Δ(NLL) relative to each method's minimum
    delta_nll_minlp = nll_minlp - minlp_min
    delta_nll_yi = nll_yi - yi_min
    # Left column: -2Δ ln L (CMS convention)
    ax_delta = axes[row_idx, 0]
    ax_delta.plot(hazard_ratios_scan, delta_nll_minlp, 'b-', label='MINLP', linewidth=2)
    ax_delta.plot(hazard_ratios_scan, delta_nll_yi, 'r--', label="Yi's Method", linewidth=2)
    ax_delta.set_xlabel('Hazard Ratio', fontsize=11)
    ax_delta.set_ylabel(r'$-2 \Delta \ln L$', fontsize=11)
    ax_delta.set_title(f'{datacard_name}: Profile Likelihood', fontsize=12, fontweight='bold')
    ax_delta.legend(fontsize=10)
    ax_delta.grid(True, alpha=0.3)
    ax_delta.set_ylim([0, max(np.max(delta_nll_minlp), np.max(delta_nll_yi)) * 1.1])
    # Right column: Absolute -2 ln L (shows offset)
    ax_abs = axes[row_idx, 1]
    ax_abs.plot(hazard_ratios_scan, nll_minlp, 'b-', label='MINLP', linewidth=2)
    ax_abs.plot(hazard_ratios_scan, nll_yi, 'r--', label="Yi's Method", linewidth=2)
    ax_abs.set_xlabel('Hazard Ratio', fontsize=11)
    ax_abs.set_ylabel(r'$-2 \ln L$ (absolute)', fontsize=11)
    ax_abs.set_title(f'{datacard_name}: Absolute Likelihood', fontsize=12, fontweight='bold')
    ax_abs.legend(fontsize=10)
    ax_abs.grid(True, alpha=0.3)
    # Print statistics
    diff_absolute = nll_minlp - nll_yi
    diff_delta = delta_nll_minlp - delta_nll_yi
    print(f"\n{datacard_name} Statistics:")
    print(f"  Absolute -2 ln L offset: {np.mean(diff_absolute):.4f} ± {np.std(diff_absolute):.4f}")
    print(f"  Max absolute diff in -2Δ ln L: {np.max(np.abs(diff_delta)):.6f}")
    print(f"  Mean absolute diff in -2Δ ln L: {np.mean(np.abs(diff_delta)):.6f}")

plt.tight_layout()
plt.show()
```

### Comparing MINLP vs. Yi's Method

**Yi's Per-Patient Probability Weighting (Improved Implementation):**
- Computes each patient's individual probability P(true group = high | observed data)
- Weights each patient's contributions to Cox risk sets by their specific probabilities
- Accounts for individual variation in measurement uncertainty

**KoMbine's MINLP Method (Individual Optimization):**
- Simultaneously optimizes both the parameters AND each patient's true group assignment
- Each patient's optimal group is determined based on their specific observable data and error model
- Patients with different measurements can be assigned different groups, even if they're in the same observed group
- This accounts for individual variation in measurement uncertainty

**Key Insight:** With our improved per-patient probability implementation of Yi's method, the methods should agree well across all scenarios since both now account for individual patient measurement uncertainty rather than using aggregate assumptions.


## Key Takeaways

### Methodological Insights

**Improved Yi's Method Implementation:**
- **Original approach**: Used aggregate misclassification matrix Π applied uniformly to all patients
- **Improved approach**: Computes individual probability P(true group = high | observed data) for each patient
- **Advantage**: Accounts for individual variation in measurement uncertainty, eliminating the need for homogeneity assumptions

**KoMbine's MINLP approach:**
- Optimizes each patient's group assignment individually, accounting for their specific measurement error
- Uses integer optimization with patient-wise NLL penalties

**Method Comparison:**
- **Yi's improved implementation**: Probabilistic weighting using per-patient probabilities (no optimization required)
- **MINLP**: Integer optimization over discrete assignments with NLL penalties
- **Agreement**: Both methods now account for individual patient uncertainty, leading to better agreement across all scenarios

### Practical Results

- **Fixed observable**: No measurement error. MINLP and Yi's method agree perfectly as expected.
- **Large counts**: Small measurement error (~2-3% relative uncertainty). Agreement between methods is excellent; confidence intervals and best-fit HRs are nearly identical to the fixed case.
- **Moderate counts**: Measurement error is significant (~15-20% relative uncertainty). With per-patient probability weighting, Yi's method now agrees much better with MINLP compared to aggregate matrix approaches. Confidence intervals widen substantially compared to fixed case, demonstrating the impact of measurement uncertainty.

### Implementation Advantages

The improved per-patient probability implementation of Yi's method offers several benefits:
- **More accurate**: Accounts for individual patient measurement uncertainty
- **Simpler**: Eliminates matrix inversion complexity
- **More robust**: No assumptions about population homogeneity required
- **Computationally efficient**: Direct probability calculations without optimization

### Practical Guidance

- **Large counts** (relative error < 5%): Standard Cox methods suffice; measurement error is negligible
- **Moderate counts** (relative error 10-20%): KoMbine's full MINLP model or improved Yi's method provide accurate inference
- **Heterogeneous populations**: Both improved Yi's method and MINLP handle patient heterogeneity well by accounting for individual measurement uncertainty
- **Method selection**: 
  - Yi's method: Fast, no optimization needed, good for most applications
  - MINLP: Provides additional flexibility with discrete optimization, useful for complex scenarios

**Conclusion:**
KoMbine provides a robust, likelihood-based approach to survival analysis with measurement error. The improved implementation of Yi's method using per-patient probability weighting offers a practical and theoretically sound alternative that agrees well with MINLP's optimization-based approach while being computationally more efficient. Both methods properly account for individual patient measurement uncertainty, making them suitable for real-world applications with heterogeneous patient populations.
