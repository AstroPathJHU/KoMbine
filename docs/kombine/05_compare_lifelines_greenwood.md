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

from kombine.datacard import Datacard  #noqa: E402
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


## P-value and Hazard Ratio Comparisons

For completeness, KoMbine also provides logrank test p-values and hazard ratio calculations that match `lifelines` for fixed observables. These comparisons demonstrate that our implementation of standard survival analysis methods is correct.

For measurement error corrections using Yi's method, see **notebook 07_yi_method_comparison.ipynb**.


### Fixed Observable: P-value and Hazard Ratio vs lifelines
We split patients at a fixed threshold and compare the conventional logrank p-value and Cox PH hazard ratio (and CI when available) between KoMbine and `lifelines`.

```python
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

threshold = 0.5
patients = datacard.patients
times = np.array([p.time for p in patients], dtype=float)
events = np.array([not p.censored for p in patients], dtype=bool)
observed = np.array([p.observed_parameter for p in patients], dtype=float)
group_high = observed >= threshold
group_low = ~group_high

if group_high.all() or group_low.all():
    raise ValueError("Threshold does not split patients into two groups")

# Lifelines logrank p-value
lifelines_logrank = logrank_test(
    times[group_low],
    times[group_high],
    event_observed_A=events[group_low],
    event_observed_B=events[group_high],
)
lifelines_p_value = float(lifelines_logrank.p_value)

# KoMbine logrank p-value
kombine_p_value = datacard.km_p_value_logrank(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    cox_only=True,
)

# Lifelines Cox PH hazard ratio
df = pd.DataFrame({
    "T": times,
    "E": events.astype(int),
    "group": group_high.astype(int),
})
cph = CoxPHFitter()
cph.fit(df, duration_col="T", event_col="E", show_progress=False)
lifelines_hr = float(np.exp(cph.params_["group"]))
summary = cph.summary
lifelines_ci = (
    float(summary.loc["group", "exp(coef) lower 95%"]),
    float(summary.loc["group", "exp(coef) upper 95%"]),
)

# KoMbine hazard ratio and CI (Cox only for fixed observables)
hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
)
kombine_hr, kombine_ci_low, kombine_ci_high, _ = hr_calc.hazard_ratio_confidence_interval(
    cox_only=True,
    confidence_level=0.95,
)

print("Logrank p-value comparison (fixed observable)")
print(f"  KoMbine:  {kombine_p_value:.6g}")
print(f"  lifelines:{lifelines_p_value:.6g}")
print("")
print("Hazard ratio comparison (fixed observable)")
print(f"  KoMbine HR:   {kombine_hr:.6g} [{kombine_ci_low:.6g}, {kombine_ci_high:.6g}]")
print(f"  lifelines HR: {lifelines_hr:.6g} [{lifelines_ci[0]:.6g}, {lifelines_ci[1]:.6g}]")
```

**Why the hazard ratio differs slightly**

The logrank p-values match because the group split is identical and the logrank statistic is computed the same way. The Cox PH hazard ratio still differs because `lifelines` uses its own partial-likelihood tie handling (Efron by default in this version), while KoMbine's Cox-only hazard ratio is derived from a Breslow-style penalty inside the likelihood model. With tied event times, Efron vs Breslow produces small shifts in the point estimate and CI, so a modest mismatch is expected even when the groups are the same.
