---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md,py
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
---

# ROC Picker Documentation - Table of Contents

 - `01_table_of_contents.html`
   - this file
 - `02_rocpicker.pdf`
   - This LaTeX document contains a detailed explanation of the math that
     goes into all the ROC analysis methods.
 - `02_examples.html`
   - examples of how to run the various ROC analysis methods included in ROC Picker
     using the datacard interface.
 - `03_small_perturbation.html`
   - an illustration of small perturbations to one of the
     observables and how that affects the ROC results.
 - `04_lung_example.html`
   - an example ROC analysis of statistical and systematic uncertainties
     using AstroPath lung cancer data

# Compilation instructions

The documentation is compiled with Github Actions.

If you want to compile it yourself:

 - First, install ROC Picker with `pip install`.
 - LaTeX:
   - Run `compile_roc_plots.sh`
   - Then compile the LaTeX using `xelatex` and `biber`.
 - Jupyter notebooks
   - Run `jupytext --sync *.md` to convert the markdown files to `.ipynb`
