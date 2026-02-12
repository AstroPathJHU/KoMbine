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

# Comprehensive P-value and Hazard Ratio Analysis with KoMbine

This notebook provides a comprehensive guide to calculating p-values, hazard ratios, and performing likelihood scans using the KoMbine package.

## Contents
1. Basic p-value calculations using logrank tests
2. Hazard ratio estimation with confidence intervals
3. Profile likelihood scans
4. Detailed comparisons across fixed, large count, and moderate count measurement scenarios

```python
import warnings
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from kombine.datacard import Datacard, FixedObservable, PoissonDensityObservable

# For reproducibility
np.random.seed(42)
```

## Part 1: Basic Analysis - P-values and Hazard Ratios

### Loading Data

First, we load patient data from a datacard file. Datacard files specify patient survival times, censoring status, and biomarker measurements.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent / "test" / "kombine" / "datacards" / "simple_examples" / "fixed_km_censoring.txt"

# Display the datacard format
with open(datacardfile) as f:
    content = f.read()
    print("Datacard file contents:")
    print(content)
```

```python
# Parse the datacard
datacard = Datacard.parse_datacard(datacardfile)
print(f"Loaded {len(datacard.patients)} patients")
print(f"Deaths: {sum(1 for p in datacard.patients if not p.censored)}")
print(f"Censored: {sum(1 for p in datacard.patients if p.censored)}")
```

### KoMbine P-value Calculation

KoMbine provides a likelihood-based p-value that accounts for measurement uncertainty when comparing survival curves between two groups defined by a biomarker threshold. For comparison, we also calculate the conventional logrank test p-value.

```python
# Define biomarker threshold to split patients into two groups
threshold = 0.5

# KoMbine likelihood p-value
km_p_value_minlp = datacard.km_p_value(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
)
kombine_p_value, _, _ = km_p_value_minlp.solve_and_pvalue(cox_only=False)

# Conventional logrank p-value
logrank_p_value = km_p_value_minlp.survival_curves_pvalue_logrank()

print("P-value comparison:")
print(f"  KoMbine likelihood p-value: {kombine_p_value:.4e}")
print(f"  Logrank p-value:           {logrank_p_value:.4e}")
```

### Basic Hazard Ratio Estimation

The hazard ratio quantifies the relative instantaneous risk of an event between two groups:
- HR = 1: No difference
- HR > 1: High group has higher hazard (worse outcomes)
- HR < 1: Low group has higher hazard (better outcomes)

```python
# Create hazard ratio calculator
hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
)

# Calculate best-fit HR and 95% confidence interval
best_fit_hr, lower_ci_95, upper_ci_95, result_95 = hr_calc.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
)

print("Hazard Ratio Analysis:")
print(f"  Best-fit HR: {best_fit_hr:.4f}")
print(f"  95% CI: [{lower_ci_95:.4f}, {upper_ci_95:.4f}]")
```

## Part 2: Detailed Hazard Ratio Analysis

### Scenario 1: Fixed Observable Type

We'll start with **fixed observable type**, where each patient's biomarker value is known exactly with no measurement uncertainty. This serves as a baseline to understand the Cox partial likelihood contribution to the confidence intervals.

```python
# Path to example datacards
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

```python
# Create hazard ratio calculator for fixed observable datacard
hr_calc_fixed = datacard_fixed.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.0,
    parameter_max=1.0,
)

hr_min = 0.01
hr_max = 100.0

# Calculate 68% and 95% confidence intervals
best_fit_hr_fixed, lower_ci_68_fixed, upper_ci_68_fixed, result_68_fixed = hr_calc_fixed.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

_, lower_ci_95_fixed, upper_ci_95_fixed, _ = hr_calc_fixed.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

print(f"\nBest-fit hazard ratio: {best_fit_hr_fixed:.3f}")
print(f"68% CI: [{lower_ci_68_fixed:.3f}, {upper_ci_68_fixed:.3f}]")
print(f"95% CI: [{lower_ci_95_fixed:.3f}, {upper_ci_95_fixed:.3f}]")
```

```python
# Perform likelihood scan for fixed observable
(
    hazard_ratios_fixed,
    twonll_values_fixed,
    best_fit_result_fixed,
    assignments_low_fixed,
    assignments_high_fixed,
) = hr_calc_fixed.likelihood_scan_hazard_ratio(
    n_points=80,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
    cox_only=False,
)

low_counts_fixed = assignments_low_fixed.sum(axis=1)
high_counts_fixed = assignments_high_fixed.sum(axis=1)

# Plot the likelihood scan with group sizes
fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05},
)
ax_main, ax_low, ax_high = axes

chi2_68 = 1.0
chi2_95 = 3.84
twonll_min_fixed = np.min(twonll_values_fixed)
delta_twonll = twonll_values_fixed - twonll_min_fixed

ax_main.plot(hazard_ratios_fixed, delta_twonll, 'b-', linewidth=2.5, label='Fixed Observable')
ax_main.axhline(chi2_68, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI')
ax_main.axhline(chi2_95, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='95% CI')
ax_main.axvline(best_fit_hr_fixed, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Best fit: {best_fit_hr_fixed:.2f}')

ax_main.set_xscale('log')
ax_main.set_ylabel("$-2 \\Delta \\ln L$", fontsize=12)
ax_main.set_title('Profile Likelihood: Fixed Observable', fontsize=14, fontweight='bold')
ax_main.legend(fontsize=10)
ax_main.grid(True, alpha=0.3, which='both')
ax_main.set_ylim(0, 15)

ax_low.step(hazard_ratios_fixed, low_counts_fixed, where='mid', color='tab:blue')
ax_low.set_ylabel('Low N', fontsize=10)
ax_low.grid(True, alpha=0.3, which='both')
ax_low.set_xscale('log')

ax_high.step(hazard_ratios_fixed, high_counts_fixed, where='mid', color='tab:green')
ax_high.set_ylabel('High N', fontsize=10)
ax_high.set_xlabel('Hazard Ratio', fontsize=12)
ax_high.grid(True, alpha=0.3, which='both')
ax_high.set_xscale('log')
ax_high.set_xlim(0.01, 100.0)

plt.tight_layout()
plt.show()
```

### Scenario 2: Poisson Density with Large Counts

Now let's consider measurements with large counts (hundreds of cells), where Poisson error is small (√N/N ~ 2-3%). The **Cox error dominates** over measurement error.


The likelihood scans below include a main profile-likelihood panel and two narrow subplots showing how many patients are assigned to the low and high groups at each hazard ratio. These assignment counts can change with hazard ratio when measurement uncertainty allows reassignment.

```python
# Load Poisson density datacard with large counts
dcfile_poisson_large_counts = datacards_dir / "poisson_density_hr_example_large.txt"
datacard_poisson_large_counts = Datacard.parse_datacard(dcfile_poisson_large_counts)

print("=" * 60)
print("POISSON DENSITY - LARGE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_large_counts.patients)} patients")

# Check the counts
nums = []
for p in datacard_poisson_large_counts.patients:
    o = p.observable
    assert isinstance(o, PoissonDensityObservable)
    nums.append(o.numerator)
print(f"  Mean count: {np.mean(nums):.1f}")
print(f"  Relative uncertainty (√N/N): {np.mean([np.sqrt(n)/n for n in nums]):.1%}")
print("\nWith large counts, Poisson error is small -> Cox error dominates")
```

```python
# Create hazard ratio calculator for Poisson (large counts)
hr_calc_poisson_large = datacard_poisson_large_counts.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)

hr_min = 0.01
hr_max = 100.0

best_fit_hr_large, lower_ci_68_large, upper_ci_68_large, _ = hr_calc_poisson_large.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

_, lower_ci_95_large, upper_ci_95_large, _ = hr_calc_poisson_large.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

print(f"Best-fit hazard ratio: {best_fit_hr_large:.3f}")
print(f"68% CI: [{lower_ci_68_large:.3f}, {upper_ci_68_large:.3f}]")
print(f"95% CI: [{lower_ci_95_large:.3f}, {upper_ci_95_large:.3f}]")
```

```python
# Likelihood scan for Poisson (large counts)
(
    hazard_ratios_large,
    twonll_values_large,
    _,
    assignments_low_large,
    assignments_high_large,
) = hr_calc_poisson_large.likelihood_scan_hazard_ratio(
    n_points=80,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
    cox_only=False,
)

twonll_min_large = np.min(twonll_values_large)
delta_twonll_large = twonll_values_large - twonll_min_large
low_counts_large = assignments_low_large.sum(axis=1)
high_counts_large = assignments_high_large.sum(axis=1)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05},
)
ax_main, ax_low, ax_high = axes

ax_main.plot(hazard_ratios_large, delta_twonll_large, 'b-', linewidth=2.5, label='Poisson Large Counts')
ax_main.axhline(chi2_68, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI')
ax_main.axhline(chi2_95, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='95% CI')
ax_main.axvline(best_fit_hr_large, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Best fit: {best_fit_hr_large:.2f}')

ax_main.set_xscale('log')
ax_main.set_ylabel("$-2 \\Delta \\ln L$", fontsize=12)
ax_main.set_title('Profile Likelihood: Poisson Large Counts', fontsize=14, fontweight='bold')
ax_main.legend(fontsize=10)
ax_main.grid(True, alpha=0.3, which='both')
ax_main.set_ylim(0, 15)

ax_low.step(hazard_ratios_large, low_counts_large, where='mid', color='tab:blue')
ax_low.set_ylabel('Low N', fontsize=10)
ax_low.grid(True, alpha=0.3, which='both')
ax_low.set_xscale('log')

ax_high.step(hazard_ratios_large, high_counts_large, where='mid', color='tab:green')
ax_high.set_ylabel('High N', fontsize=10)
ax_high.set_xlabel('Hazard Ratio', fontsize=12)
ax_high.grid(True, alpha=0.3, which='both')
ax_high.set_xscale('log')
ax_high.set_xlim(0.01, 100.0)

plt.tight_layout()
plt.show()
```

At hazard ratios around 4, the likelihood surface is fairly flat with respect to a few borderline patients, so the optimizer can switch their group assignments without a meaningful change in the objective. This shows up as small, brief changes in the high-group count even though the overall profile likelihood remains smooth.


### Scenario 3: Poisson Density with Moderate Counts

Finally, examine **moderate counts** (tens of cells) with significant Poisson error (√N/N ~ 15-20%). Now **both Cox error and Poisson error** contribute to confidence intervals, which will be noticeably **wider**.

```python
# Load Poisson density datacard with moderate counts
dcfile_poisson_moderate_counts = datacards_dir / "poisson_density_hr_example_moderate.txt"
datacard_poisson_moderate_counts = Datacard.parse_datacard(dcfile_poisson_moderate_counts)

print("=" * 60)
print("POISSON DENSITY - MODERATE COUNTS")
print("=" * 60)
print(f"Loaded {len(datacard_poisson_moderate_counts.patients)} patients")

# Check the counts
nums = []
for p in datacard_poisson_moderate_counts.patients:
    o = p.observable
    assert isinstance(o, PoissonDensityObservable)
    nums.append(o.numerator)
print(f"  Mean count: {np.mean(nums):.1f}")
print(f"  Relative uncertainty (√N/N): {np.mean([np.sqrt(n)/n for n in nums]):.1%}")
print("\nWith moderate counts, measurement uncertainty is significant!")

# KoMbine p-value for a nontrivial datacard (Poisson errors)
km_p_value_poisson = datacard_poisson_moderate_counts.km_p_value(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
)
kombine_p_value_poisson, _, _ = km_p_value_poisson.solve_and_pvalue(cox_only=False)
print(f"KoMbine p-value (Poisson errors): {kombine_p_value_poisson:.4e}")
```

```python
# Create hazard ratio calculator for Poisson (moderate counts)
hr_calc_poisson_moderate = datacard_poisson_moderate_counts.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-35.0, 35.0),
)

hr_min = 1e-12
hr_max = 1e12

best_fit_hr_moderate, lower_ci_68_moderate, upper_ci_68_moderate, _ = hr_calc_poisson_moderate.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

_, lower_ci_95_moderate, upper_ci_95_moderate, _ = hr_calc_poisson_moderate.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.95,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
)

def format_ci_value(value: float, *, bound: float, is_lower: bool) -> str:
    if np.isclose(value, bound):
        return f"< {bound:g}" if is_lower else f"> {bound:g}"
    return f"{value:.3f}"

lower_68_str = format_ci_value(lower_ci_68_moderate, bound=hr_min, is_lower=True)
upper_68_str = format_ci_value(upper_ci_68_moderate, bound=hr_max, is_lower=False)
lower_95_str = format_ci_value(lower_ci_95_moderate, bound=hr_min, is_lower=True)
upper_95_str = format_ci_value(upper_ci_95_moderate, bound=hr_max, is_lower=False)

print(f"Best-fit hazard ratio: {best_fit_hr_moderate:.3f}")
print(f"68% CI: [{lower_68_str}, {upper_68_str}]")
print(f"95% CI: [{lower_95_str}, {upper_95_str}]")

if np.isclose(lower_ci_95_moderate, hr_min) or np.isclose(upper_ci_95_moderate, hr_max):
    print("Note: 95% CI is open-ended at the scan bounds.")
```

```python
# Likelihood scan for Poisson (moderate counts)
(
    hazard_ratios_moderate,
    twonll_values_moderate,
    _,
    assignments_low_moderate,
    assignments_high_moderate,
) = hr_calc_poisson_moderate.likelihood_scan_hazard_ratio(
    n_points=80,
    hazard_ratio_min=hr_min,
    hazard_ratio_max=hr_max,
    cox_only=False,
)

twonll_min_moderate = np.min(twonll_values_moderate)
delta_twonll_moderate = twonll_values_moderate - twonll_min_moderate
low_counts_moderate = assignments_low_moderate.sum(axis=1)
high_counts_moderate = assignments_high_moderate.sum(axis=1)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05},
)
ax_main, ax_low, ax_high = axes

ax_main.plot(hazard_ratios_moderate, delta_twonll_moderate, 'b-', linewidth=2.5, label='Poisson Moderate Counts')
ax_main.axhline(chi2_68, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='68% CI')
ax_main.axhline(chi2_95, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='95% CI')
ax_main.axvline(best_fit_hr_moderate, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Best fit: {best_fit_hr_moderate:.2f}')

ax_main.set_xscale('log')
ax_main.set_ylabel("$-2 \\Delta \\ln L$", fontsize=12)
ax_main.set_title('Profile Likelihood: Poisson Moderate Counts', fontsize=14, fontweight='bold')
ax_main.legend(fontsize=10)
ax_main.grid(True, alpha=0.3, which='both')
ax_main.set_ylim(0, 15)

ax_low.step(hazard_ratios_moderate, low_counts_moderate, where='mid', color='tab:blue')
ax_low.set_ylabel('Low N', fontsize=10)
ax_low.grid(True, alpha=0.3, which='both')
ax_low.set_xscale('log')

ax_high.step(hazard_ratios_moderate, high_counts_moderate, where='mid', color='tab:green')
ax_high.set_ylabel('High N', fontsize=10)
ax_high.set_xlabel('Hazard Ratio', fontsize=12)
ax_high.grid(True, alpha=0.3, which='both')
ax_high.set_xscale('log')
ax_high.set_xlim(hr_min, hr_max)

plt.tight_layout()
plt.show()
```

### Note on the Moderate-Count Likelihood Scan Shape

The discontinuities come from **assignment changes** as the hazard ratio is forced to extreme values:
- The low group remains fixed (patients with the smallest observed parameters).
- The high group shrinks from two patients to one, and then becomes empty at very large hazard ratios.
- Many patients are excluded from either curve because the MINLP is allowed to drop assignments when that reduces the total penalty.

That discontinuous re-assignment creates visible kinks and plateaus in the profile likelihood at extreme hazard ratios.

#### Why can a group become empty?

In this scan, the optimizer is allowed to place each patient in **low**, **high**, or **neither** (dropped). Dropping a patient costs a *patient-wise penalty*, but it can still be beneficial if keeping that patient would make the Cox/Breslow (survival-curve) part of the objective much worse at a forced hazard ratio.

There is also **no constraint that forces both groups to stay non-empty**, so at extreme forced hazard ratios the optimizer can decide that the best option is to shrink (or even empty) one group.

#### What does an empty group mean for the hazard ratio?

A hazard ratio only has content when you are comparing *two* groups. If the optimizer empties one group, then the survival-data part of the objective no longer has leverage to prefer one hazard ratio over another.

Concretely, the Cox/Breslow likelihood used here depends on the risk sets through terms like:

$$\log\big(r_{\mathrm{low}}(t) + \mathrm{HR}\,r_{\mathrm{high}}(t) + \epsilon\big).$$

- If the **high** group is empty, then $r_{\mathrm{high}}(t)=0$ (and there are no high-group deaths), so HR drops out of these terms.
- If the **low** group is empty, the dependence on HR cancels in the same way in the ideal Cox expression; in the implementation a small $\epsilon$ is included to keep the log well-defined, so any remaining HR dependence is purely a numerical safeguard and does not provide a meaningful constraint.

#### Why does that create a plateau (“ceiling”) in the scan?

As HR is forced to more extreme values, the best *two-group* assignment can become very expensive in NLL. But the optimizer has an escape hatch: it can drop borderline patients (and in the limit, empty one group), pay the patient-wise penalties, and move to a configuration where the objective no longer depends on HR.

So the patient-wise penalty effectively sets the *height of the plateau*: beyond some HR, the optimizer prefers a roughly fixed penalty + an HR-insensitive Cox/Breslow term, rather than letting $-2\Delta\ln L$ continue to grow.

**Interpretation:** in the plateau region, the hazard ratio is **not constrained by the data at a higher confidence level than the plateau height**. Intuitively, the data do not force the model to keep any patients in the depleted group, so they also cannot force a particular hazard ratio between the two groups.


### Summary Comparison

Let's compare all three scenarios to see how measurement uncertainty affects the results.

```python
def format_ci_width(lower: float, upper: float, *, min_bound: float | None = None, max_bound: float | None = None) -> str:
    if min_bound is not None and np.isclose(lower, min_bound):
        return "open"
    if max_bound is not None and np.isclose(upper, max_bound):
        return "open"
    return f"{upper - lower:6.3f}"

print("Summary Comparison")
print("=" * 80)
print("                          Fixed      Poisson-Large  Poisson-Moderate")

width_68_fixed = format_ci_width(lower_ci_68_fixed, upper_ci_68_fixed)
width_68_large = format_ci_width(lower_ci_68_large, upper_ci_68_large)
width_68_moderate = format_ci_width(lower_ci_68_moderate, upper_ci_68_moderate, min_bound=hr_min, max_bound=hr_max)

width_95_fixed = format_ci_width(lower_ci_95_fixed, upper_ci_95_fixed)
width_95_large = format_ci_width(lower_ci_95_large, upper_ci_95_large)
width_95_moderate = format_ci_width(lower_ci_95_moderate, upper_ci_95_moderate, min_bound=hr_min, max_bound=hr_max)

print(f"Best-fit HR:             {best_fit_hr_fixed:6.3f}      {best_fit_hr_large:6.3f}          {best_fit_hr_moderate:6.3f}")
print(f"68% CI width:            {width_68_fixed:>6}      {width_68_large:>6}          {width_68_moderate:>6}")
print(f"95% CI width:            {width_95_fixed:>6}      {width_95_large:>6}          {width_95_moderate:>6}")
print("=" * 80)

print("\nConfidence interval width ratios (relative to fixed):")
print(f"  Poisson-Large   (68%): {(upper_ci_68_large - lower_ci_68_large) / (upper_ci_68_fixed - lower_ci_68_fixed):.2f}x")
print(f"  Poisson-Large   (95%): {(upper_ci_95_large - lower_ci_95_large) / (upper_ci_95_fixed - lower_ci_95_fixed):.2f}x")
print(f"  Poisson-Moderate(68%): {(upper_ci_68_moderate - lower_ci_68_moderate) / (upper_ci_68_fixed - lower_ci_68_fixed):.2f}x")
print(f"  Poisson-Moderate(95%): {(upper_ci_95_moderate - lower_ci_95_moderate) / (upper_ci_95_fixed - lower_ci_95_fixed):.2f}x")

hr_bounds = (hr_min, hr_max)
moderate_hits_bounds = (
    np.isclose(lower_ci_95_moderate, hr_bounds[0]) or
    np.isclose(upper_ci_95_moderate, hr_bounds[1])
)

print("\nKey observations:")
print("1. Large counts → CIs similar to fixed (measurement error negligible)")
print("2. Moderate counts → CIs widen substantially (measurement error significant)")
if moderate_hits_bounds:
    print("3. Moderate-count HR estimate is weakly constrained (95% CI reaches scan bounds)")
else:
    print("3. Moderate-count HR estimate remains less constrained than fixed/large-count cases")
```

## Summary

This notebook demonstrated:
1. Loading patient data from datacard files
2. Computing logrank test p-values
3. Estimating hazard ratios with confidence intervals
4. Visualizing profile likelihood curves
5. Understanding how measurement uncertainty affects statistical inference

**Key takeaways:**
- Fixed observables provide a baseline for Cox-error-only analysis
- Large count measurements behave similarly to fixed, showing measurement error is negligible
- Moderate count measurements show substantial widening of confidence intervals due to measurement uncertainty
- KoMbine properly accounts for this measurement uncertainty in survival analysis

