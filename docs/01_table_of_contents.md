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

# Table of Contents

 - `01_table_of_contents.html`
   - this file
 - `02_rocpicker.pdf`
   - This LaTeX document contains a detailed explanation of the math that
     goes into all the methods used in ROC Picker.
 - `03_examples.html`
   - examples of how to run the various methods included in ROC Picker
     using the datacard interface.
 - `04_small_perturbation.html`
   - an illustration of small perturbations to one of the
     observables and how that affects the results.
 - `05_lung_example.html`
   - an example analysis of statistical and systematic uncertainties
     using AstroPath lung cancer data
 - `06_kaplan_meier_example.html`
   - an example of how to use the likelihood method for uncertainties on Kaplan-Meier curves
 - `07_compare_to_lifelines.html`
   - a comparison of our Kaplan-Meier likelihood method to the `lifelines` package

# Compilation instructions

The documentation is all compiled with Github Actions and provided as a single
zip file in the artifacts.  The latest version from the `main` branch is
[here](https://nightly.link/AstroPathJHU/ROCPicker/workflows/test_and_docs/main/docs.zip).

If you want to compile it yourself anyway, here are the instructions.

 - First, install ROC Picker with `pip install`.
 - LaTeX:
   - Run `compile_plots.sh`
   - Then compile the LaTeX.
     - Use `xelatex` and `biber`.  It will take a few iterations, as usual with
       LaTeX.
     - I included the relevant magic comments and find that it compiles out of
       the box in TeXstudio.
 - Jupyter notebooks
   - Run `jupytext --sync *.md` to convert the markdown files to `.ipynb`
   - This one (the table of contents) is trivial, but the rest contain functioning
     code that you can play with.
