#!/usr/bin/env python3
"""
Compare Kaplan-Meier error bands with collapse_consecutive_deaths=True and
collapse_consecutive_deaths=False.

This script compares the Kaplan-Meier likelihood method with different settings for
the collapse_consecutive_deaths parameter, plotting both curves on the same figure.
Uses the likelihood method for confidence intervals instead of Greenwood.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from roc_picker.datacard import Datacard
from roc_picker.kaplan_meier_likelihood import KaplanMeierPlotConfig


def main():
  """
  Run the comparison and generate the plot.
  """
  parser = argparse.ArgumentParser(description=main.__doc__)
  parser.add_argument("datacard", help="Path to datacard file")
  parser.add_argument("output", help="Output PDF file")
  parser.add_argument("--parameter-min", type=float, default=-np.inf,
                     help="Minimum parameter value (default: -inf)")
  parser.add_argument("--parameter-max", type=float, default=np.inf,
                     help="Maximum parameter value (default: inf)")
  parser.add_argument("--title", type=str, default="Collapse Consecutive Deaths Comparison",
                     help="Plot title")

  # Styling arguments matching compile_plots.sh COMMON_ARGS
  parser.add_argument("--figsize", nargs=2, type=float, default=[7, 7],
                     help="Figure size in inches")
  parser.add_argument("--legend-fontsize", type=float, default=16,
                     help="Font size for legend text")
  parser.add_argument("--title-fontsize", type=float, default=16,
                     help="Font size for the plot title")
  parser.add_argument("--label-fontsize", type=float, default=16,
                     help="Font size for axis labels")
  parser.add_argument("--tick-fontsize", type=float, default=16,
                     help="Font size for the tick labels")
  parser.add_argument("--xlabel", type=str, default="Time",
                     help="X-axis label")
  parser.add_argument("--ylabel", type=str, default="Survival Probability",
                     help="Y-axis label")
  parser.add_argument("--legend-loc", type=str, default="lower right",
                     help="Legend location")

  # Options for confidence interval methods
  g = parser.add_mutually_exclusive_group()
  g.add_argument("--full-nll", action="store_true",
                 help="Use the full negative log-likelihood confidence bands (default)")
  g.add_argument("--binomial-only", action="store_true",
                 help="Use the binomial-only confidence bands.")
  g.add_argument("--patient-wise-only", action="store_true",
                 help="Use the patient-wise only confidence bands.")
  g.add_argument("--greenwood", action="store_true",
                 help="Use the Greenwood method.")

  args = parser.parse_args()

  if not (args.full_nll or args.binomial_only or args.patient_wise_only or args.greenwood):
    args.full_nll = True  # Default to full NLL if no option specified

  include_best_fit = False  # Don't show best fit either if no bands

  # Handle confidence band options
  include_full_nll = args.full_nll
  include_binomial_only = args.binomial_only
  include_patient_wise_only = args.patient_wise_only
  include_exponential_greenwood = args.greenwood

  # Load the datacard
  datacard = Datacard.parse_datacard(args.datacard)

  # Create KM likelihood objects with different collapse_consecutive_deaths settings
  kml_true = datacard.km_likelihood(
    parameter_min=args.parameter_min,
    parameter_max=args.parameter_max,
    collapse_consecutive_deaths=True
  )

  kml_false = datacard.km_likelihood(
    parameter_min=args.parameter_min,
    parameter_max=args.parameter_max,
    collapse_consecutive_deaths=False
  )

  # Create plot configurations matching plot_km_likelihood_two_groups approach
  # First plot (with collapse_consecutive_deaths=True) - similar to "high" group
  config_true = KaplanMeierPlotConfig(
    create_figure=True,
    close_figure=False,
    show=False,
    saveas=None,
    best_label=f"With consecutive deaths collapsed (n={len(kml_true.nominalkm.patients)})",
    best_color="blue",
    nominal_label=f"With consecutive deaths collapsed (n={len(kml_true.nominalkm.patients)})",
    nominal_color="blue",
    CL_colors=["dodgerblue", "skyblue"],
    include_full_NLL=include_full_nll,
    include_binomial_only=include_binomial_only,
    include_patient_wise_only=include_patient_wise_only,
    include_exponential_greenwood=include_exponential_greenwood,
    include_nominal=True,
    include_best_fit=include_best_fit,  # Only show one curve per setting to avoid clutter
    figsize=tuple(args.figsize),
    legend_fontsize=args.legend_fontsize,
    title_fontsize=args.title_fontsize,
    label_fontsize=args.label_fontsize,
    tick_fontsize=args.tick_fontsize,
    title=args.title,
    xlabel=args.xlabel,
    ylabel=args.ylabel,
    legend_loc=args.legend_loc,
  )

  # Plot first curve with create_figure=True
  kml_true.plot(config=config_true)

  # Second plot (with collapse_consecutive_deaths=False) - similar to "low" group
  config_false = KaplanMeierPlotConfig(
    create_figure=False,  # Add to existing figure
    close_figure=False,
    show=False,
    saveas=None,
    best_label=f"Without consecutive deaths collapsed (n={len(kml_false.nominalkm.patients)})",
    best_color="red",
    nominal_label=f"Without consecutive deaths collapsed (n={len(kml_false.nominalkm.patients)})",
    nominal_color="red",
    CL_colors=["orangered", "lightcoral"],
    include_full_NLL=include_full_nll,
    include_binomial_only=include_binomial_only,
    include_patient_wise_only=include_patient_wise_only,
    include_exponential_greenwood=include_exponential_greenwood,
    include_nominal=True,
    include_best_fit=include_best_fit,  # Only show one curve per setting to avoid clutter
    figsize=tuple(args.figsize),
    legend_fontsize=args.legend_fontsize,
    title_fontsize=args.title_fontsize,
    label_fontsize=args.label_fontsize,
    tick_fontsize=args.tick_fontsize,
    title=args.title,
    xlabel=args.xlabel,
    ylabel=args.ylabel,
    legend_loc=args.legend_loc,
  )

  # Plot second curve on the same figure
  kml_false.plot(config=config_false)

  # Save the plot
  plt.savefig(args.output)
  print(f"Plot saved to {args.output}")


if __name__ == "__main__":
  main()
