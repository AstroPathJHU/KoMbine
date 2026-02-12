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

This notebook provides a comprehensive guide to calculating p-values and hazard ratios using the KoMbine package.

## Contents
1. Basic p-value calculations using logrank tests
2. Hazard ratio estimation with confidence intervals
3. Scenario likelihood scans (fixed, large, moderate, small)
4. Restricted range analysis (0.01 to 0.99)

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
threshold = 0.5001

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

## Scenario Likelihood Scans (Fixed, Large, Moderate, Small)
These scans use the full likelihood (`cox_only=False`) with wide parameter bounds and a shared hazard-ratio grid.

```python
from pathlib import Path

datacard_root = Path("../../test/kombine/datacards/simple_examples").resolve()

scenario_specs = [
    ("Fixed counts", "fixed_hr_example.txt"),
    ("Large counts", "poisson_density_hr_example_large.txt"),
    ("Moderate counts", "poisson_density_hr_example_moderate.txt"),
    ("Small counts", "poisson_density_hr_example_small.txt"),
]

hazard_ratio_scan = np.logspace(np.log10(0.01), np.log10(100), 80)

scenario_results = []
for name, filename in scenario_specs:
    dc = Datacard.parse_datacard(datacard_root / filename)
    km_pvalue = dc.km_p_value(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
    )
    km_hr = dc.km_hazard_ratio(
        parameter_threshold=threshold,
        parameter_min=-np.inf,
        parameter_max=np.inf,
        log_hazard_ratio_bounds=(-30.0, 30.0),
    )

    hazard_ratios, twonll_values, _, assignments_low, assignments_high = (
        km_hr.likelihood_scan_hazard_ratio(
            hazard_ratio_scan,
            cox_only=False,
        )
    )
    delta_2nll = twonll_values - np.min(twonll_values)
    low_scan = assignments_low.sum(axis=1)
    high_scan = assignments_high.sum(axis=1)

    scenario_results.append(
        {
            "name": name,
            "km_pvalue": km_pvalue,
            "km_hr": km_hr,
            "hazard_ratio_scan": hazard_ratios,
            "delta_2nll": delta_2nll,
            "low_scan": low_scan,
            "high_scan": high_scan,
        }
    )
```

```python
def _tight_patient_ylim(ax, values):
    ymin = np.min(values)
    ymax = np.max(values)
    if ymin == ymax:
        pad = 0.25
    else:
        pad = 0.1 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

def _add_confidence_lines(ax):
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.axhline(3.84, color="0.5", linestyle=":", linewidth=1.0)

fig, axes = plt.subplots(
    6,
    2,
    figsize=(12, 14),
    sharex="col",
    gridspec_kw={"height_ratios": [2.2, 0.7, 0.7, 2.2, 0.7, 0.7]},
)

scenario_pairs = [
    (scenario_results[0], scenario_results[1]),
    (scenario_results[2], scenario_results[3]),
]

for pair_index, (left, right) in enumerate(scenario_pairs):
    base_row = pair_index * 3
    for col, scenario in enumerate([left, right]):
        hr = scenario["hazard_ratio_scan"]

        axes[base_row, col].plot(hr, scenario["delta_2nll"], color="black")
        _add_confidence_lines(axes[base_row, col])
        axes[base_row, col].set_xscale("log")
        axes[base_row, col].set_title(f"{scenario['name']}: -2Δ ln L")
        axes[base_row, col].set_ylabel(r"$-2\Delta\ln L$")
        axes[base_row, col].set_ylim(bottom=0)

        axes[base_row + 1, col].plot(hr, scenario["low_scan"], color="#1f77b4")
        axes[base_row + 1, col].set_xscale("log")
        axes[base_row + 1, col].set_title(f"{scenario['name']}: low-risk patients")
        axes[base_row + 1, col].set_ylabel("Patients")
        _tight_patient_ylim(axes[base_row + 1, col], scenario["low_scan"])

        axes[base_row + 2, col].plot(hr, scenario["high_scan"], color="#d62728")
        axes[base_row + 2, col].set_xscale("log")
        axes[base_row + 2, col].set_title(f"{scenario['name']}: high-risk patients")
        axes[base_row + 2, col].set_ylabel("Patients")
        axes[base_row + 2, col].set_xlabel("Hazard ratio")
        _tight_patient_ylim(axes[base_row + 2, col], scenario["high_scan"])

plt.tight_layout()
```

The NLL profiles align in shape across scenarios, while the patient-count panels show how assignments shift as the hazard ratio changes.
As counts decrease, the NLL curves become flatter and the patient-count curves show more step-like changes, reflecting the discrete nature of the assignments.
Fixed-count data stay comparatively stable across hazard ratios, whereas smaller counts show earlier shifts in who is assigned to the high-risk group.


## Restricted Range Analysis (0.01 to 0.99)
This section shows how restricting the fitted parameter range affects the moderate-count scan.

```python
moderate_path = datacard_root / "poisson_density_hr_example_moderate.txt"
moderate_dc = Datacard.parse_datacard(moderate_path)
moderate_hr_restricted = moderate_dc.km_hazard_ratio(
    parameter_threshold=threshold,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-30.0, 30.0),
 )

hazard_ratio_scan_restricted = np.logspace(-12, 12, 120)

twonll_restricted = np.full_like(hazard_ratio_scan_restricted, np.nan, dtype=float)
low_restricted = np.full_like(hazard_ratio_scan_restricted, np.nan, dtype=float)
high_restricted = np.full_like(hazard_ratio_scan_restricted, np.nan, dtype=float)

for idx, hr in enumerate(hazard_ratio_scan_restricted):
    try:
        result = moderate_hr_restricted.compute_2nll_at_hazard_ratio(hr, cox_only=False)
    except ValueError:
        continue
    twonll_restricted[idx] = result.x
    low_restricted[idx] = len(result.patients_low)
    high_restricted[idx] = len(result.patients_high)

valid_mask = ~np.isnan(twonll_restricted)
delta_2nll_restricted = twonll_restricted - np.nanmin(twonll_restricted)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 0.7, 0.7]},
)
axes[0].plot(
    hazard_ratio_scan_restricted[valid_mask],
    delta_2nll_restricted[valid_mask],
    color="black",
)
_add_confidence_lines(axes[0])
axes[0].set_xscale("log")
axes[0].set_ylabel(r"$-2\Delta\ln L$")
axes[0].set_title("Moderate counts (restricted fit range): -2Δ ln L")
axes[0].set_ylim(bottom=0)

axes[1].plot(
    hazard_ratio_scan_restricted[valid_mask],
    low_restricted[valid_mask],
    color="#1f77b4",
)
axes[1].set_xscale("log")
axes[1].set_ylabel("Patients")
axes[1].set_title("Moderate counts (restricted fit range): low-risk patients")
_tight_patient_ylim(axes[1], low_restricted[valid_mask])

axes[2].plot(
    hazard_ratio_scan_restricted[valid_mask],
    high_restricted[valid_mask],
    color="#d62728",
)
axes[2].set_xscale("log")
axes[2].set_ylabel("Patients")
axes[2].set_xlabel("Hazard ratio")
axes[2].set_title("Moderate counts (restricted fit range): high-risk patients")
_tight_patient_ylim(axes[2], high_restricted[valid_mask])

plt.tight_layout()
```

When the fitted parameter is restricted to 0.01 to 0.99, the optimizer can exclude some assignments that would otherwise be available, which introduces broader plateaus in the NLL curve.
The patient-count plots show where the fit is forced to hold a group size fixed because the best-fit parameter would lie outside the allowed range.
This effect becomes more pronounced when many groups are fit together, since a single restricted parameter can force the shared hazard ratio to favor different subsets of patients across groups.

### Why can a group become empty?

In this scan, the optimizer is allowed to place each patient in **low**, **high**, or **neither** (dropped). Dropping a patient costs a patient-wise penalty, but it can still be beneficial if keeping that patient would make the Cox/Breslow part of the objective much worse at a forced hazard ratio.

There is also no constraint that forces both groups to stay non-empty, so at extreme forced hazard ratios the optimizer can decide that the best option is to shrink (or even empty) one group.

### What does an empty group mean for the hazard ratio?

A hazard ratio only has content when you are comparing two groups. If the optimizer empties one group, then the survival-data part of the objective no longer has leverage to prefer one hazard ratio over another.

Concretely, the Cox/Breslow likelihood used here depends on the risk sets through terms like:

$$
\log\big(r_{\mathrm{low}}(t) + \mathrm{HR}\,r_{\mathrm{high}}(t) + \epsilon\big).
$$

- If the high group is empty, then $r_{\mathrm{high}}(t)=0$ (and there are no high-group deaths), so HR drops out of these terms.
- If the low group is empty, the dependence on HR cancels in the same way in the ideal Cox expression; in the implementation a small $\epsilon$ is included to keep the log well-defined, so any remaining HR dependence is purely a numerical safeguard and does not provide a meaningful constraint.

### Why does that create a plateau in the scan?

As HR is forced to more extreme values, the best two-group assignment can become very expensive in NLL. But the optimizer has an escape hatch: it can drop borderline patients (and in the limit, empty one group), pay the patient-wise penalties, and move to a configuration where the objective no longer depends on HR.

So the patient-wise penalty effectively sets the height of the plateau: beyond some HR, the optimizer prefers a roughly fixed penalty plus an HR-insensitive Cox/Breslow term, rather than letting $-2\Delta\ln L$ continue to grow.

**Interpretation:** in the plateau region, the hazard ratio is not constrained by the data at a higher confidence level than the plateau height.



