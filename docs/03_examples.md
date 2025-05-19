---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import warnings
warnings.simplefilter("error")
```

# ROC Picker


This notebook is meant to be a technical introduction to ROC Picker, with examples.  The math and discussion is in the latex document, also in this folder.

```python
import pathlib
from roc_picker.datacard import Datacard
```

## A simple datacard


ROC Picker works using datacards.  The datacard format is inspired by the CMS Collaboration's Combine tool.  See [arXiv:2404.06614 [physics.data-an]](https://arxiv.org/abs/2404.06614), the [Github repo](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit), and the [manual](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/).

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"simple_examples"/"datacard_example_2.txt"
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
- `bin`: This might be used in the future to group patients (for example, into immunotherapy- and chemo-treated patients).  It currently doesn't do anything, and doesn't have to be included.
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
_ = datacard.discrete_roc().make_plots(show=True)
```

We produced three plots:
1. The ROC curve
2. A likelihood scan for the area under the curve (AUC)
3. The ROC curve again, this time with error bands obtained using the likelihood scan

See the latex document for more on these plots.

We can also produce the same plots using the $\delta$ functions method.

```python
_ = datacard.delta_functions_roc().make_plots(show=True)
```

## Systematics


The third method currently implemented addresses sample-wise uncertainty.  This includes counting uncertainty on the observable as well as systematic uncertainties.

Here's an example datacard.  This is from an actual analysis of the density of CD8+FoxP3+-like neighborhoods in biopsies of non-small-cell lung cancer patients, which we found to correlate with response to anti-PD1 immunotherapy.  Please see [`doi:10.1136/jitc-2023-SITC2023.0121`](https://doi.org/10.1136/jitc-2023-SITC2023.0121) for more details.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent/"test"/"datacards"/"lung"/"datacard_neighborhoods.txt"

with open(datacardfile) as f:
    print(f.read())

datacard = Datacard.parse_datacard(datacardfile)
```

This time, the `observable_type` is a Poisson ratio, meaning that we are counting two features and characterizing patients based on their ratio.  Both of those counts have a Poisson uncertainty that we want to take into account in our analysis.

Additionally, we have systematic uncertainties on each sample.  Each line contains two introductory fields, the name of the systematic `sys_batchN` and the model type.  Currently `lnN` is the only option for the model type.  It indicates a log-normal systematic, which is often a good choice for multiplicative corrections.

These systematics are estimated using the overlap area between adjacent high-power fields, and include features such as residual flatfielding and warping effects that remain after calibration, which are correlated among samples in a batch, and edge effects from the segmentation algorithm, which are correlated among all the samples.  For this example, we assume that the effects are fully correlated within a batch but uncorrelated between batches.  For example, the first systematic line indicates that, when the systematic parameter varies up by $1\sigma$, two of the responders will have their counts multiplied by $0.948$ and $1.011$, and two of the non-responders will have *their* counts multiplied by $0.909$ and $0.878$.

Each of the systematic parameters can vary independently.


The method currently implemented for systematics uses a Monte Carlo method to vary the parameters and produce a family of ROC curves.  The 68% and 95% CL regions are estimated using quantiles of this family of ROC curves.  This does not take into account the statistical uncertainty on the number of samples, which was described in the previous section, but it does include the sample-wise statistical uncertainty on the numerator and denominator counts.

```python
_ = datacard.systematics_mc_roc(flip_sign=True).generate(size=10000, random_state=123456).plot(show=True)
```
