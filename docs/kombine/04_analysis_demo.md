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
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,unspecified-encoding,wrong-import-order,wrong-import-position
```

# Comprehensive P-value and Hazard Ratio Analysis with KoMbine

This notebook demonstrates likelihood-based p-values and hazard ratio estimation with KoMbine, then explores how the likelihood behaves across different data-count scenarios and parameter restrictions.

## Contents
- [Part 1: Basic analysis - p-values and hazard ratios](#part-1-basic-analysis---p-values-and-hazard-ratios)
  - [Loading data](#loading-data)
  - [KoMbine p-value calculation](#kombine-p-value-calculation)
  - [Basic hazard ratio estimation](#basic-hazard-ratio-estimation)
- [Scenario likelihood scans (fixed, large, moderate, small)](#scenario-likelihood-scans-fixed-large-moderate-small)
- [Restricted range analysis (0.01 to 0.99)](#restricted-range-analysis-001-to-099)
  - [Why can a group become empty?](#why-can-a-group-become-empty)
  - [What does an empty group mean for the hazard ratio?](#what-does-an-empty-group-mean-for-the-hazard-ratio)
  - [Why does that create a plateau in the scan?](#why-does-that-create-a-plateau-in-the-scan)

```python
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from kombine.datacard import Datacard

# For reproducibility
np.random.seed(42)
```

## Part 1: Basic analysis - p-values and hazard ratios

### Loading data

We load a small example datacard with fixed observables. The file encodes survival times, censoring, and biomarker measurements for a 12-patient cohort.

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

### KoMbine p-value calculation

KoMbine provides a likelihood-based p-value that accounts for measurement uncertainty when comparing survival curves across a biomarker threshold. For context, we also compute the standard logrank p-value.

For this dataset and threshold, the likelihood p-value is $2.67\times 10^{-2}$ and the logrank p-value is $1.61\times 10^{-2}$.

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

### Basic hazard ratio estimation

The hazard ratio (HR) summarizes the relative instantaneous event risk between the high- and low-risk groups:
- HR = 1: no difference
- HR > 1: high group has higher hazard (worse outcomes)
- HR < 1: high group has lower hazard (better outcomes)

For this example, the best-fit HR is 0.1959 with a 95% CI of [0.0284, 0.8386], indicating lower hazard in the high group at this threshold.

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

## Scenario likelihood scans (fixed, large, moderate, small)
These scans use the full likelihood (`cox_only=False`) with wide parameter bounds and a shared hazard-ratio grid from 0.01 to 100. In some scenarios the best-fit HR lands at the lower scan bound, so extending the grid can be useful for a tighter minimum.

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

The fixed and large-count cases show smooth, convex $-2\Delta\ln L$ profiles with a clear minimum, while the moderate case remains smooth but with more pronounced curvature changes.
The small-count scenario is noticeably rougher, with shallow local structure before rising at large HR.

Patient assignments behave accordingly: fixed counts remain constant across the scan, large counts shift once, and moderate/small counts show multiple step-like changes as the optimizer reassigns borderline patients.


## Restricted range analysis (0.01 to 0.99)
This section repeats the moderate-count scan while restricting the fitted parameter to [0.01, 0.99], showing how parameter bounds can change the likelihood surface and the optimal patient assignments.

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

With the 0.01 to 0.99 restriction, the scan shows a long plateau at low HR, a sharp dip to the minimum at intermediate HR, and a rapid rise at extreme HR. The low-risk count stays nearly fixed, while the high-risk group steadily shrinks and can hit zero near the extreme end.

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

So the patient-wise penalty effectively sets the height of the plateau: beyond some HR, the optimizer prefers a fixed penalty plus an HR-insensitive Cox/Breslow term, rather than letting $-2\Delta\ln L$ continue to grow.

**Interpretation:** in the plateau region, the hazard ratio is not constrained by the data at a higher confidence level than the plateau height.
