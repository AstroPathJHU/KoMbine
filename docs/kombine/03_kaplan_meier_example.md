---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: rocpicker
    language: python
    name: python3
---

```python
# pylint: disable=bad-indentation,line-too-long,missing-module-docstring,unspecified-encoding,wrong-import-position
```

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
datacardfile = here.parent.parent/"test"/"kombine"/"datacards"/"simple_examples"/"poisson_ratio_km_censoring.txt"
```

```python
with open(datacardfile) as f:
    print(f.read())
```

The first line gives the observable type.
Options are:
* `fixed`: The observable for each patient is a fixed number.  It may be modified by systematics in the systematics section, but has no internal uncertainty.
* `poisson`: The observable for each patient is a count, which has an associated Poisson uncertainty.  It may have additional uncertainties defined in the systematics section.
* `poisson_density`: The observable for each patient is a count, which has an associated Poisson uncertainty, divided by a fixed area, which is assumed to have no error.
* `poisson_ratio`: The observable for each patient is a ratio of two counts.  Again, it may have additional uncertainties defined in the systematics section.
* `discrete_classes`: The observable for each patient is a probability distribution over discrete class indices.

Next is the list of patients.
- `survival_time`: the time when the patient was censored or died
- `censored`: indicates whether the patient was censored (1) or not (0).
- The observables for each patient, which depends on the observable_type given above.
  - For `fixed`, the line should be labeled `observable`
  - For `poisson`, it should be labeled `count`
  - For `poisson_ratio`, there should be two lines labeled `num` and `denom`, as in the example here
  - For `poisson_density`, there should be two lines labeled `num` and `area`
  - For `discrete_classes`, there should be one line per class labeled `prob0`, `prob1`, ...

Below, you can put a list of systematic uncertainties.  These are documented in the example notebook in the ROC Picker documentation.  For KoMbine, uncertainties cannot yet be correlated between patients, so each uncertainty can only apply to a single patient.

```python
datacard = Datacard.parse_datacard(datacardfile)
```

We typically divide the patients into groups based on their `observable` value and plot the Kaplan-Meier curve for each group.  The goal is to see whether one group of patients fares better than the other.

```python
kml_low = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=0.45)
kml_high = datacard.km_likelihood(parameter_min=0.45, parameter_max=np.inf)
```

First, let's display both curves with nominal survival probabilities (no error bands):

```python
# Plot both curves with nominal only (no error bands)
# The time_unit is automatically inherited from the datacard
plt.figure()
_ = kml_low.plot(
    nominal_color="blue",
    nominal_label="observable < 0.45",
    create_figure=False,
    include_nominal=True,
    include_full_NLL=False,
    include_best_fit=False,
)
_ = kml_high.plot(
    nominal_color="red",
    nominal_label="observable >= 0.45",
    create_figure=False,
    include_nominal=True,
    include_full_NLL=False,
    include_best_fit=False,
)
plt.legend()
plt.show()
```

Now we will show error bands to visualize the uncertainty in the survival probability estimates.

```python
# time_unit is automatically inherited from the datacard
_ = kml_low.plot()
_ = kml_high.plot()
```

Note that, counterintuitively, the best fit does not agree with the nominal survival probability, and at some points the nominal probability is not even within the 1-sigma band.  This is not a bug.  For more information, see the math in the LaTeX documentation.


Or, to display them both on the same plot:

```python
# time_unit is automatically inherited from the datacard
plt.figure()
_ = kml_low.plot(
    best_color="blue",
    CL_colors=["dodgerblue", "skyblue"],
    best_label="observable < 0.45",
    create_figure=False,
    include_nominal=False,
)
_ = kml_high.plot(
    best_color="red",
    CL_colors=["orangered", "lightcoral"],
    best_label="observable >= 0.45",
    create_figure=False,
    include_nominal=False,
)
plt.legend()
plt.show()
```

We can also display the individual contributions of the binomial and patient-wise errors to the total error band.

```python
# time_unit is automatically inherited from the datacard
_ = kml_low.plot(include_patient_wise_only=True)
_ = kml_low.plot(include_binomial_only=True)
```

```python

```
