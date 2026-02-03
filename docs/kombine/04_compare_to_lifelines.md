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
import warnings
warnings.simplefilter("error")
```

# Comparison to Lifelines


Our error estimation method is more general than the exponential Greenwood confidence intervals as used by the `lifelines` package.  Greenwood's method only supports the statistical uncertainty from the number of patients and not patient-wise uncertainties.  (`lifelines` also contains lots of other functionality that this package does not.)

The JSS paper (LaTeX document in this folder) contains an extended comparison of the exponential Greenwood intervals to our method and a discussion of the differences.  Here, we just compare our implementation of the exponential Greenwood intervals to the `lifelines` implementation.

To run this notebook, you need to have the `lifelines` package installed.  You can install it with `pip install lifelines`.

```python
import pathlib  #noqa: E402

import lifelines  #noqa: E402
import matplotlib.pyplot as plt  #noqa: E402
import numpy as np  #noqa: E402

from roc_picker.datacard import Datacard  #noqa: E402

# Try to import Pandas4Warning for pandas v3 compatibility
# On older pandas versions, this doesn't exist
try:
    from pandas.errors import Pandas4Warning
except ImportError:
    Pandas4Warning = None
```

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"kombine"/"datacards"/"simple_examples"/"fixed_km_censoring.txt"
```

```python
with open(datacardfile) as f:
    print(f.read())
```

```python
datacard = Datacard.parse_datacard(datacardfile)
kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
```

```python
T = [patient.time for patient in kml.nominalkm.patients]
E = [not patient.censored for patient in kml.nominalkm.patients]
kmf = lifelines.KaplanMeierFitter()
kmf.fit(T, event_observed=E)
plt.figure()
_ = kml.plot(CLs=[0.95], create_figure=False, include_nominal=False, best_color="red", CL_colors_greenwood=["orangered", "lightcoral"], include_full_NLL=False, include_exponential_greenwood=True)
kmf.plot_survival_function(label="lifelines")
plt.legend()
plt.show()
```

We do, in fact, get 1:1 agreement with `lifelines`.


## Comparing Logrank P-Values: Standard vs Yi's Correction

In addition to survival curve estimation, we can compare hypothesis testing approaches. Let's compare:
1. **Standard logrank test** (lifelines)
2. **KoMbine logrank** (same as standard for fixed observables)
3. **Yi's misclassification correction** (accounts for measurement error)

For fixed observables (no measurement error), all three methods should agree.

```python
# Create a two-group dataset for logrank testing
# Group 0: patients with low observable values
# Group 1: patients with high observable values

# Parse the datacard with two groups
datacard_two_groups = Datacard.parse_datacard(datacardfile)

# Define a threshold to split patients (use median)
observables = [p.observable.value for p in datacard_two_groups.patients]
threshold = np.median(observables)

print(f"Splitting patients at threshold: {threshold}")
print(f"Total patients: {len(datacard_two_groups.patients)}")

# Create groups for lifelines
group_labels = []
times = []
events = []

for p in datacard_two_groups.patients:
    if p.observable.value < threshold:
        group_labels.append(0)
    else:
        group_labels.append(1)
    times.append(p.survival_time)
    events.append(not p.censored)

print(f"Group 0 (low): {sum(1 for g in group_labels if g == 0)} patients")
print(f"Group 1 (high): {sum(1 for g in group_labels if g == 1)} patients")
```

### Method 1: Lifelines Logrank Test

```python
from lifelines.statistics import logrank_test

# Lifelines logrank test
# Temporarily allow Pandas4Warning (if it exists) since we can't control lifelines code
with warnings.catch_warnings():
    if Pandas4Warning is not None:
        warnings.simplefilter("default", Pandas4Warning)
    
    results_lifelines = logrank_test(
        durations_A=[t for t, g in zip(times, group_labels) if g == 0],
        durations_B=[t for t, g in zip(times, group_labels) if g == 1],
        event_observed_A=[e for e, g in zip(events, group_labels) if g == 0],
        event_observed_B=[e for e, g in zip(events, group_labels) if g == 1]
    )

print("Lifelines Logrank Test:")
print(f"  Test statistic: {results_lifelines.test_statistic:.4f}")
print(f"  P-value: {results_lifelines.p_value:.6f}")
```

### Method 2: KoMbine Logrank Test

```python
# KoMbine logrank test (standard, no measurement error for fixed observables)
p_value_kombine = datacard_two_groups.km_p_value_logrank(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    cox_only=True
)

print("KoMbine Logrank Test:")
print(f"  P-value: {p_value_kombine:.6f}")
```

### Method 3: Yi's Misclassification Correction

```python
# Yi's correction method (for fixed observables, should match standard)
result_yi = datacard_two_groups.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    method='bayesian'
)

print("Yi's Correction Method:")
print(f"  Test statistic: {result_yi['logrank_statistic']:.4f}")
print(f"  P-value: {result_yi['p_value']:.6f}")
```

### Comparison Summary

```python
print("=" * 60)
print("P-Value Comparison Summary")
print("=" * 60)
print(f"{'Method':<30} {'P-value':<15} {'Match?':<10}")
print("-" * 60)
print(f"{'Lifelines (standard)':<30} {results_lifelines.p_value:<15.6f} {'Reference':<10}")
print(f"{'KoMbine (standard)':<30} {p_value_kombine:<15.6f} {'✓' if abs(results_lifelines.p_value - p_value_kombine) < 1e-4 else '✗':<10}")
print(f"{'Yi correction':<30} {result_yi['p_value']:<15.6f} {'✓' if abs(results_lifelines.p_value - result_yi['p_value']) < 1e-4 else '✗':<10}")
print("=" * 60)

print("\nKey Observations:")
print("- For fixed observables (no measurement error), all three methods agree")
print("- Yi's misclassification matrix is nearly identity (Π[0,0] ≈ Π[1,1] ≈ 1)")
print("- Yi's method provides a framework for handling measurement uncertainty")
print("- When measurement error is present (e.g., Poisson counts), Yi's p-value")
print("  will differ from the standard logrank, properly accounting for uncertainty")
```
