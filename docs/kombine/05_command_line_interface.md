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
      jupytext_version: 1.18.1
---

# KoMbine Command Line Interface

This document describes the command line interface for KoMbine's Kaplan-Meier likelihood methods.

## Commands

KoMbine provides two command line tools:

- `kombine` - Generate Kaplan-Meier likelihood plots for a single group
- `kombine_twogroups` - Generate Kaplan-Meier likelihood plots comparing two groups (high vs. low parameter values)

## Common Arguments

### Required Arguments

Both commands require these positional arguments:

- `datacard` - Path to the datacard file containing patient data
- `output_file` - Path to save the output plot (PDF format)

### Error Band Options

Control which error bands are displayed in the plot:

- `--include-binomial-only` - Include error bands for the binomial error alone (mutually exclusive with `--include-patient-wise-only`)
- `--include-patient-wise-only` - Include error bands for the patient-wise error alone (mutually exclusive with `--include-binomial-only`)
- `--include-exponential-greenwood` - Include the binomial-only exponential Greenwood error band in the plot
- `--exclude-full-nll` - Exclude the full NLL (negative log-likelihood) from the plot
- `--exclude-nominal` - Exclude the nominal line from the plot

### Plot Appearance Options

Customize the visual appearance of the plot:

- `--color {blue,red,green,purple,orange,teal,brown,pink}` - Color scheme for the plot (default: blue) [kombine only]
- `--high-color {blue,red,green,purple,orange,teal,brown,pink}` - Color scheme for the high group (default: blue) [kombine_twogroups only]
- `--low-color {blue,red,green,purple,orange,teal,brown,pink}` - Color scheme for the low group (default: red) [kombine_twogroups only]
- `--figsize WIDTH HEIGHT` - Figure size in inches (default: 10 7)
- `--no-tight-layout` - Do not use tight layout for the plot
- `--xmax XMAX` - Maximum time for x-axis range. Limits the plot to [0, xmax]

### Legend and Labels

Customize text elements in the plot:

- `--title TITLE` - Title for the plot (default: "Kaplan-Meier Curves")
- `--xlabel XLABEL` - Label for the x-axis (default: "Time")
- `--ylabel YLABEL` - Label for the y-axis (default: "Survival Probability")
- `--legend-loc LEGEND_LOC` - Location of the legend in the plot (mutually exclusive with `--legend-saveas`)
- `--legend-saveas LEGEND_SAVEAS` - Path to save the legend separately. If provided, the legend will be left off the main plot (mutually exclusive with `--legend-loc`)
- `--include-median-survival` - Include the median survival time in the legend
- `--no-n-in-legend` - Do not include number of patients in each group in the legend labels [kombine_twogroups only]

### Legend Suffixes

Customize suffixes added to legend labels:

- `--patient-wise-only-suffix PATIENT_WISE_ONLY_SUFFIX` - Suffix to add to the patient-wise-only label in the legend (default: "Patient-wise only")
- `--binomial-only-suffix BINOMIAL_ONLY_SUFFIX` - Suffix to add to the binomial-only label in the legend (default: "Binomial only")
- `--full-nll-suffix FULL_NLL_SUFFIX` - Suffix to add to the full NLL label in the legend (default: "")
- `--exponential-greenwood-suffix EXPONENTIAL_GREENWOOD_SUFFIX` - Suffix to add to the exponential Greenwood label in the legend (default: "Binomial only, exp. Greenwood")

### Font Sizes

Adjust font sizes for various text elements:

- `--legend-fontsize LEGEND_FONTSIZE` - Font size for legend text (default: 10)
- `--label-fontsize LABEL_FONTSIZE` - Font size for axis labels (default: 12)
- `--title-fontsize TITLE_FONTSIZE` - Font size for the plot title (default: 14)
- `--tick-fontsize TICK_FONTSIZE` - Font size for the tick labels (default: 10)
- `--pvalue-fontsize PVALUE_FONTSIZE` - Font size for p value text (default: 12) [kombine_twogroups only]

### Computational Options

Control the likelihood calculation:

- `--parameter-min PARAMETER_MIN` - Minimum parameter value for filtering patients
- `--parameter-max PARAMETER_MAX` - Maximum parameter value for filtering patients
- `--endpoint-epsilon ENDPOINT_EPSILON` - Endpoint epsilon for the likelihood calculation (default: 1e-6)
- `--log-zero-epsilon LOG_ZERO_EPSILON` - Log zero epsilon for the likelihood calculation (default: 1e-10)
- `--dont-collapse-consecutive-deaths` - Disable collapsing of consecutive death times with no intervening censoring (slower but may be more accurate)
- `--print-progress` - Print progress messages during the computation

## kombine_twogroups Specific Arguments

These arguments are specific to the `kombine_twogroups` command:

### Group Separation

- `--parameter-threshold PARAMETER_THRESHOLD` - **Required.** The parameter threshold for separating high and low groups

### P-value Options

Control p-value calculation and display:

- `--exclude-p-value` - Exclude p value calculation and display from the plot
- `--include-logrank-pvalue` - Include conventional logrank p value for comparison with likelihood method
- `--p-value-tie-handling {breslow}` - Method for handling ties in p value calculation (default: breslow)
- `--pvalue-format PVALUE_FORMAT` - Format string for p value display (default: '.3g')

## Examples

### Basic single group plot

```
kombine datacard.txt output.pdf
```

### Single group with custom color and x-axis limit

```
kombine datacard.txt output.pdf --color green --xmax 50.0
```

### Single group with patient-wise error only

```
kombine datacard.txt output.pdf --include-patient-wise-only
```

### Two group comparison

```
kombine_twogroups datacard.txt output.pdf --parameter-threshold 0.45
```

### Two groups with custom colors

```
kombine_twogroups datacard.txt output.pdf --parameter-threshold 0.45 --high-color purple --low-color orange
```

### Two groups with custom parameter range and x-axis limit

```
kombine_twogroups datacard.txt output.pdf \
  --parameter-threshold 0.45 \
  --parameter-min 0.0 \
  --parameter-max 1.0 \
  --xmax 100.0
```

### Advanced: Custom plot appearance

```
kombine datacard.txt output.pdf \
  --color teal \
  --xmax 75.0 \
  --title "Survival Analysis" \
  --xlabel "Time (months)" \
  --ylabel "Survival Rate" \
  --figsize 12 8 \
  --legend-fontsize 12 \
  --label-fontsize 14
```
