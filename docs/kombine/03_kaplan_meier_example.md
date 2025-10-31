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

Next is the list of patients.
- `survival_time`: the time when the patient was censored or died
- `censored`: indicates whether the patient was censored (1) or not (0).
- The observables for each patient, which depends on the observable_type given above.
  - For `fixed`, the line should be labeled `observable`
  - For `poisson`, it should be labeled `count`
  - For `poisson_ratio`, there should be two lines labeled `num` and `denom`, as in the example here
  - For `poisson_density`, there should be two lines labeled `num` and `area`

Below, you can put a list of systematic uncertainties.  These are documented in the example notebook in the ROC Picker documentation.  For KoMbine, uncertainties cannot yet be correlated between patients, so each uncertainty can only apply to a single patient.

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

## Additional Features

### Limiting the x-axis range

You can limit the x-axis range of plots using the `xmax` parameter:

```python
_ = kml_low.plot(xmax=50.0)
```

This limits the plot to the time range [0, xmax], which is useful for focusing on the early portion of the survival curve or when you want to compare plots with different time scales.

### Customizing colors

Colors can be customized for plots. For single plots, use the `best_color` and `CL_colors` parameters in the plot configuration. From the command line, use the `--color` option:

```bash
kombine datacard.txt output.pdf --color green
```

For two-group plots, you can specify different colors for each group:

```bash
kombine_twogroups datacard.txt output.pdf --parameter-threshold 0.45 --high-color purple --low-color orange
```

Available color options include: blue, red, green, purple, orange, teal, brown, and pink.

```python

```
