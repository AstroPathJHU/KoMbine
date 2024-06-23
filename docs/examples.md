---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pathlib
from roc_picker.datacard import Datacard
```

ROC Picker works using datacards.  The datacard format is inspired by the CMS Collaboration's Combine tool.  See [arXiv:2404.06614 [data-an]](https://arxiv.org/abs/2404.06614) and the [Github repo](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit).

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"datacard_example_2.txt"
```

```python
with open(datacardfile) as f:
    print(f.read())
```

The first line gives the observable type.
Options are:
* `fixed`: The observable for each patient is a fixed number.  It may be modified by systematics in the systematics section, but has no internal uncertainty.
* `poisson`: The observable for each patient is a count, which has an associated Poisson uncertainty.  It may have additional uncertainties defined in the systematics section.
* `poisson_ratio`: The observable for each patient is a ratio of two counts.  Again, it may have additional uncertainties defined in the systematics section.

Next is the list of patients.
- `bin`: This might be used in the future to group patients (into, for example, immunotherapy and chemo).  It currently doesn't do anything.
- `response`: Options are `responder` and `non-responder`.  These define the two categories the observable is meant to separate between and are plotted on the two axes of the ROC curve.
- The observables for each patient, which depends on the observable_type given above.
  - For `fixed`, the line should be labeled `observable`, as it is in the example here
  - For `poisson`, it should be labeled `count`
  - For `poisson_ratio`, there should be two lines labeled `numerator` and `denominator`

Next is the systematics section, which will be described in more detail below.  This simple example datacard doesn't have any systematics.

We'll now parse the datacard.

```python
datacard = Datacard.parse_datacard(datacardfile)
```

The first two methods implemented in ROC Picker only take into account the statistical uncertainty on the number of patients, and do not currently support systematics or any `observable_type` besides `fixed`.  If all the `scipy.optimize` methods involved converge, both should give identical results.  The mathematical details are in the latex document in this folder.

```python
_ = datacard.discrete().plot_roc(show=True)
```

We produced three plots:
1. The ROC curve
2. A likelihood scan for the area under the curve (AUC)
3. The ROC curve again, this time with error bands obtained using the likelihood scan

See the latex document for more on these plots.

We can also produce the same plots using the $\delta$ functions method.

```python
_ = datacard.discrete().plot_roc(show=True)
```

# Systematics


The third method currently implemented addresses sample-wise uncertainty.  This includes counting uncertainty on the observable as well as systematic uncertainties.

Here's an example datacard.  This is from an actual analysis of the density of CD8+FoxP3+-like neighborhoods in non-small-cell lung cancer patients, which we found to correlate with response to anti-PD1 immunotherapy.  Please see `doi:10.1136/jitc-2023-SITC2023.0121`.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"datacard_neighborhoods.txt"

with open(datacardfile) as f:
    print(f.read())
```

```python

```
