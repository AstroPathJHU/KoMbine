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

```python
import pathlib   #noqa: E402
from roc_picker.datacard import Datacard  #noqa: E402
```

# Introduction


This is from an actual analysis of the density of CD8+FoxP3+ cells and CD8+FoxP3+-like neighborhoods in biopsies of non-small-cell lung cancer patients, which we found to correlate with response to anti-PD1 immunotherapy [`doi:10.1136/jitc-2023-SITC2023.0121`](https://doi.org/10.1136/jitc-2023-SITC2023.0121).

For a technical introduction to using ROC Picker, see the `examples` notebook in this folder.  For a discussion of the math, see the latex document in this folder.


# CD8+FoxP3+ cells

CD8+FoxP3+ cells are a rare phenotype that can appear in the TME.  Naively, one could expect that they would act as Tregs, by analogy with CD4+FoxP3+.  On the other hand, some previous studies, including ours in melannoma [`doi:10.1126/science.aba2609`](https://doi.org/10.1126/science.aba2609), have shown positive correlation between CD8+FoxP3+ cells and immune respose.  These cells are difficult to study because of their rarity, as the uncertainty analysis in this notebook will show.

We find that they show positive correlation with response in lung cancer as well, and we would like to understand the uncertainties on this correlation.


We can divide the uncertainties in the analysis into three categories:
1. The statistical uncertainty arising from the finite number of patients
2. The statistical uncertainty arising from the finite number of cells in each patient's biopsy
3. Systematic uncertainties


## Statistical uncertainty from the number of patients

We estimate this uncertainty using the discrete method.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"roc_picker"/"datacards"/"lung"/"datacard_cells_binomial.txt"
datacard = Datacard.parse_datacard(datacardfile)
_ = datacard.discrete_roc(flip_sign=True).make_plots(show=[False, False, True])
```

## Statistical uncertainty from the number of cells

We propagate this uncertainty using the Monte Carlo method.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"roc_picker"/"datacards"/"lung"/"datacard_cells_poisson.txt"
datacard = Datacard.parse_datacard(datacardfile)
_ = datacard.systematics_mc_roc(flip_sign=True).generate(size=10000, random_state=123456).plot(show=True)
```

## Discussion

The statistical uncertainty from the number of patients included in the study ($n=25$) dominates, but the statistical uncertainty from the number of cells in each patient's biopsy is of comparable size.

It's clear how to reduce the first uncertainty: do a bigger study with more patients.  The second uncertainty, however, will remain the same.  It could, in principle, be reduced by taking a bigger biopsy.  However, this is not always feasible, particularly in the lungs.


# CD8+FoxP3+-like neighborhoods

Instead, we identify "CD8+FoxP3+-like neighborhoods", which are characteristic of the region around a CD8+FoxP3+ cell.  The definition of these neighborhoods is discussed further in [`doi:10.1136/jitc-2023-SITC2023.0121`](https://doi.org/10.1136/jitc-2023-SITC2023.0121) and in a forthcoming longer paper, but for the purposes of illustrating ROC Picker, we just take their existence and density in these patient samples as given.  They are about $200\times$ as common as CD8+FoxP3+ cells.


## Statistical uncertainty from the number of patients

We again use the discrete method.  The ROC curve error bands and AUC error ranges are similar to the ones obtained when we were counting cells.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"roc_picker"/"datacards"/"lung"/"datacard_neighborhoods_binomial.txt"
datacard = Datacard.parse_datacard(datacardfile)
_ = datacard.discrete_roc(flip_sign=True).make_plots(show=[False, False, True])
```

## Statistical uncertainty from the number of cells

We use the Monte Carlo method.  Because the neighborhoods are so much more common than the CD8+FoxP3+ cells, the associated uncertainty decreases significantly.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"roc_picker"/"datacards"/"lung"/"datacard_neighborhoods_poisson.txt"
datacard = Datacard.parse_datacard(datacardfile)
_ = datacard.systematics_mc_roc(flip_sign=True).generate(size=10000, random_state=123456).plot(show=True)
```

## Systematic uncertainty

We illustrate just one example systematic uncertainty.  The samples are scanned on the microscope as a collection of high-power fields, or HPFs, which are stitched together to make up the whole image.  We find that, when we average over many HPFs, the distribution of CD8+FoxP3+-like neighborhoods is not uniform, but varies as a function of position within an HPF.  The neighborhoods slightly more common at the edges of an HPF than in the middle.

This can come from a variety of sources, including residual flatfielding and warping effects that remain after calibration, which are correlated among samples in a batch, and edge effects from the segmentation algorithm, which are correlated among all the samples.  For this example, we assume that the effects are fully correlated within a batch but uncorrelated between batches.

In future work, we will estimate the systematic uncertainties from each of these sources (and others!) separately.  For our present purposes, the rough approximation is sufficient.

We use the Monte Carlo method again and find that this uncertainty is larger than the statistical uncertainty from the number of neighborhoods.  The statistical uncertainty from the finite number of patients is still the largest.

```python
here = pathlib.Path(".").resolve()
datacardfile = here.parent.parent/"test"/"roc_picker"/"datacards"/"lung"/"datacard_neighborhoods_systematics.txt"
datacard = Datacard.parse_datacard(datacardfile)
_ = datacard.systematics_mc_roc(flip_sign=True).generate(size=10000, random_state=123456).plot(show=True)
```
