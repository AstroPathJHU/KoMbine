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
      jupytext_version: 1.17.2
---

# KoMbine Documentation - Table of Contents

 - `01_table_of_contents.html`
   - this file
 - `02_kaplan_meier_paper.pdf`
   - This LaTeX document contains a detailed explanation of the math that
     goes into the Kaplan-Meier likelihood methods.
 - `02_kaplan_meier_example.html`
   - an example of how to use the likelihood method for uncertainties on Kaplan-Meier curves
 - `03_compare_to_lifelines.html`
   - a comparison of our Kaplan-Meier likelihood method to the `lifelines` package

# Compilation instructions

The documentation is compiled with Github Actions.

If you want to compile it yourself:

 - First, install KoMbine with `pip install`.
 - LaTeX:
   - Run `compile_km_plots.sh`
   - Then compile the LaTeX using `xelatex` and `biber`.
 - Jupyter notebooks
   - Run `jupytext --sync *.md` to convert the markdown files to `.ipynb`