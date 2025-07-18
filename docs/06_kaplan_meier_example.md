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

# KoMbine


KoMbine, the part of this package that deals with Kaplan-Meier curves, uses datacards similar to the ROC Picker datacards.

```python
import pathlib  #noqa: E402
import matplotlib.pyplot as plt  #noqa: E402
import numpy as np  #noqa: E402
from roc_picker.datacard import Datacard  #noqa: E402
```

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"poisson_ratio_km_censoring.txt"
```

```python
with open(datacardfile) as f:
    print(f.read())
```

The `observable_type` and `observable` lines work the same was as in the ROC Picker datacards.  The `survival_time` and `censored` lines are specific to Kaplan-Meier curves.  The `survival_time` line contains the time when the patient was censored or died, and the `censored` line indicates whether the patient was censored (1) or not (0).

```python
datacard = Datacard.parse_datacard(datacardfile)
```

We typically divide the patients into groups based on their `observable` value and plot the Kaplan-Meier curve for each group.  The goal is to see whether one group of patients fares better than the other.

```python
kml_low = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=0.45)
kml_high = datacard.km_likelihood(parameter_min=0.45, parameter_max=np.inf)
```

```python
_ = kml_low.plot()
_ = kml_high.plot()
```

Or, to display them both on the same plot:

```python
plt.figure()
_ = kml_low.plot(
    best_color="blue",
    CL_colors=["dodgerblue", "skyblue"],
    best_label=f"observable < 0.45",
    create_figure=False,
    include_nominal=False,
)
_ = kml_high.plot(
    best_color="red",
    CL_colors=["orangered", "lightcoral"],
    best_label=f"observable >= 0.45",
    create_figure=False,
    include_nominal=False,
)
plt.legend()
plt.show()
```

Note that, counterintuitively, the best fit does not agree with the nominal survival probability, and at some points the nominal probability is not even within the 1-sigma band.  This is not a bug.  For more information, see the math in the LaTeX documentation.


We can also display the individual contributions of the binomial and patient-wise errors to the total error band.

```python
_ = kml_low.plot(include_patient_wise_only=True)
_ = kml_low.plot(include_binomial_only=True)
```

```python

```
