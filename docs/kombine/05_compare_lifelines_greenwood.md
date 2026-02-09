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

from kombine.datacard import Datacard, FixedObservable  #noqa: E402


# Try to import Pandas4Warning for pandas v3 compatibility
# On older pandas versions, this doesn't exist
try:
    from pandas.errors import Pandas4Warning  # pyright: ignore[reportAttributeAccessIssue]
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


## P-value and Hazard Ratio Comparisons

For completeness, KoMbine also provides logrank test p-values and hazard ratio calculations that match `lifelines` for fixed observables. These comparisons demonstrate that our implementation of standard survival analysis methods is correct.

For measurement error corrections using Yi's method, see **notebook 06_yi_method_comparison.ipynb**.
