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
  parser.add_argument("--include-full-nll", action="store_true",
                     help="Include full negative log-likelihood confidence bands "
                     "(requires full Gurobi license)")
  parser.add_argument("--include-binomial-only", action="store_true", default=True,
                     help="Include binomial-only confidence bands "
                     "(default, works with restricted Gurobi license)")
  parser.add_argument("--no-confidence-bands", action="store_true",
                     help="Use only Greenwood method "
                     "(minimal confidence bands for large datasets)")

  args = parser.parse_args()

  # Handle confidence band options
  if args.no_confidence_bands:
    # Use Greenwood method as fallback when no other methods are wanted
    # This is the least computationally intensive method
    include_full_nll = False
    include_binomial_only = False
    include_exponential_greenwood = True
    include_best_fit = False  # Don't show best fit either if no bands
  else:
    include_full_nll = args.include_full_nll
    include_binomial_only = args.include_binomial_only
    include_exponential_greenwood = False
    include_best_fit = False  # Only show one curve per setting to avoid clutter

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
