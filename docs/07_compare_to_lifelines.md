---
jupyter:
  jupytext:
    formats: ipynb,md,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
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


Our error estimation method is more general than the exponential Greenwood confidence intervals as used by the `lifelines` package.  Greenwood's method only supports the statistical uncertainty from the number of patients and not patient-wise uncertainties.  (`lifelines` also contains lots of other functionality that this package does not.)  Here, we apply our method with a fixed observable and compare the results to the exponential Greenwood intervals.

```python
import pathlib  #noqa: E402

try:
  import lifelines  #noqa: E402
except ImportError:
  lifelines = None
import matplotlib.pyplot as plt  #noqa: E402
import numpy as np  #noqa: E402

from roc_picker.datacard import Datacard  #noqa: E402
```

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"fixed_km_censoring.txt"
```

```python
with open(datacardfile) as f:
    print(f.read())
```

```python
datacard = Datacard.parse_datacard(datacardfile)
kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
```

First, we want to compare our implementation of the exponential Greenwood confidence intervals to `lifelines`.  We should get a 1:1 match.

(Note that this cell requires the `lifelines` package, which you can install with `pip install lifelines`.)

```python
if lifelines is not None:
  T = [patient.time for patient in kml.nominalkm.patients]
  E = [not patient.censored for patient in kml.nominalkm.patients]
  kmf = lifelines.KaplanMeierFitter()
  kmf.fit(T, event_observed=E)
  plt.figure()
  _ = kml.plot(CLs=[0.95], create_figure=False, include_nominal=False, best_color="red", CL_colors_greenwood=["orangered", "lightcoral"], include_full_NLL=False, include_greenwood=True)
  kmf.plot_survival_function(label="lifelines")
  plt.legend()
  plt.show()
```

Since we get 1:1 agreement, we proceed to use our implementation for the rest of the notebook.

```python
_ = kml.plot(create_figure=False, include_nominal=False, best_color="red", CL_colors=["orangered", "lightcoral"], include_greenwood=True, CL_colors_greenwood=["dodgerblue", "skyblue"])
plt.legend()
plt.show()
```

Our code gives nonzero errors for the first and last time points, while `lifelines` does not.  In the middle, the error bars from our code generally track with the error bars from `lifelines`, but do not exactly match.  To understand this better, let's look at a case with many patients.

(Please note, this plot takes about two minutes to run.)

```python
datacardfile_manypatients = here.parent/"test"/"datacards"/"simple_examples"/"fixed_km_censoring_many_patients.txt"
with open(datacardfile_manypatients) as f:
    print(f.read())
```

```python
datacard_manypatients = Datacard.parse_datacard(datacardfile_manypatients)
kml_manypatients = datacard_manypatients.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
```

```python
plt.figure()
_ = kml_manypatients.plot(create_figure=False, include_nominal=False, best_color="red", CL_colors=["orangered", "lightcoral"], include_greenwood=True, CL_colors_greenwood=["dodgerblue", "skyblue"], print_progress=True)
plt.legend()
plt.show()
```

Here, the middle points match lifelines much better.  What is happening is as follows:

Lifelines uses exponential Greenwood confidence intervals ([mathematical notes](https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf), linked from the [`lifelines` documentation](https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html)).  This is a good approximation and, crucially, can be computed instantaneously for any number of patients who were censored or died.  The approximation gets better as the number of patients increases and, as pointed out in the mathematical notes, behaves well for sample sizes as small as 25.  Note that our first example had 12 patients and our second had 100 patients.

Our method uses the full binomial log likelihood, which is valid for any sample size and can compute accurate confidence intervals even for the first and last time points.  When there is more than one binomial factor (which happens when patients are censored), our method requires numerical optimization and becomes much slower.



