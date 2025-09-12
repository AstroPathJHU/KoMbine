#!/usr/bin/env python3
"""
Compare Kaplan-Meier error bands with collapse_consecutive_deaths=True and 
collapse_consecutive_deaths=False.

This script compares the Kaplan-Meier likelihood method with different settings for 
the collapse_consecutive_deaths parameter, plotting both curves on the same figure.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from roc_picker.datacard import Datacard


def main():
  """
  Run the comparison and generate the plot.
  """
  parser = argparse.ArgumentParser(description=main.__doc__)
  parser.add_argument("datacard", help="Path to datacard file")
  parser.add_argument("output", help="Output PDF file")
  parser.add_argument("--parameter-min", type=str, default="-inf",
                     help="Minimum parameter value (default: -inf)")
  parser.add_argument("--parameter-max", type=str, default="inf",
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

  args = parser.parse_args()

  # Convert string infinity values to float
  if args.parameter_min == "-inf":
    parameter_min = -np.inf
  elif args.parameter_min == "inf":
    parameter_min = np.inf
  else:
    parameter_min = float(args.parameter_min)

  if args.parameter_max == "-inf":
    parameter_max = -np.inf
  elif args.parameter_max == "inf":
    parameter_max = np.inf
  else:
    parameter_max = float(args.parameter_max)

  # Load the datacard
  datacard = Datacard.parse_datacard(args.datacard)

  # Create KM likelihood objects with different collapse_consecutive_deaths settings
  kml_true = datacard.km_likelihood(
    parameter_min=parameter_min,
    parameter_max=parameter_max,
    collapse_consecutive_deaths=True
  )

  kml_false = datacard.km_likelihood(
    parameter_min=parameter_min,
    parameter_max=parameter_max,
    collapse_consecutive_deaths=False
  )

  # Set up the plot with styling consistent with compile_plots.sh
  _, ax = plt.subplots(figsize=tuple(args.figsize))

  # Get the maximum time for plotting
  all_times = []
  for patient in kml_true.nominalkm.patients:
    all_times.append(patient.time)
  max_time = max(all_times) if all_times else 10
  times_for_plot = np.linspace(0, max_time * 1.1, 100)

  # Get the nominal KM curve points for both settings
  x_true, y_true = kml_true.nominalkm.points_for_plot(times_for_plot=times_for_plot)
  x_false, y_false = kml_false.nominalkm.points_for_plot(times_for_plot=times_for_plot)

  # Plot the curves with colors matching plot_km_likelihood_two_groups
  # Use blue for collapse_consecutive_deaths=True (like "high" group)
  ax.plot(x_true, y_true, color="blue", linewidth=2,
         label="With consecutive deaths collapsed")

  # Use red for collapse_consecutive_deaths=False (like "low" group)
  ax.plot(x_false, y_false, color="red", linewidth=2,
         label="Without consecutive deaths collapsed")

  # Add censored patient markers
  for patient in kml_true.nominalkm.patients:
    if patient.censored:
      # Find the survival probability at the censoring time
      idx = np.searchsorted(x_true, patient.time)
      if idx < len(y_true):
        survival_prob = y_true[idx]
        ax.plot(patient.time, survival_prob, 'o', color='blue', markersize=4)

  for patient in kml_false.nominalkm.patients:
    if patient.censored:
      # Find the survival probability at the censoring time
      idx = np.searchsorted(x_false, patient.time)
      if idx < len(y_false):
        survival_prob = y_false[idx]
        ax.plot(patient.time, survival_prob, 'o', color='red', markersize=4)

  # Apply styling consistent with compile_plots.sh
  ax.set_xlabel(args.xlabel, fontsize=args.label_fontsize)
  ax.set_ylabel(args.ylabel, fontsize=args.label_fontsize)
  ax.set_title(args.title, fontsize=args.title_fontsize)
  ax.tick_params(labelsize=args.tick_fontsize)
  ax.grid(True)
  ax.legend(loc=args.legend_loc, fontsize=args.legend_fontsize)

  # Set reasonable axis limits
  ax.set_xlim(0, max_time * 1.1)
  ax.set_ylim(0, 1.05)

  # Save the plot
  plt.tight_layout()
  plt.savefig(args.output)
  print(f"Plot saved to {args.output}")


if __name__ == "__main__":
  main()
