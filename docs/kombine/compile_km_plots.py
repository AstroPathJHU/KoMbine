#!/usr/bin/env python3
"""
Generate all plots for 02_kombine.tex using matplotlib subplots.

This script creates combined figures with
subplots instead of individual PDF files. Each combined figure includes
Nature-style subfigure labels (A, B, C, etc.).

Command line options:
  --km-example         Generate only the KM example plot
  --greenwood          Generate only the Greenwood comparison plot
  --p-value            Generate only the p-value comparison plot
  --lung               Generate only the lung cancer dataset plot
  --testing            Use smaller datasets for faster testing

If no plot options are specified, all plots are generated.
Multiple plot options can be combined to generate specific subsets.
"""

import argparse
import os
import pathlib
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kombine.datacard import Datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

# Set matplotlib backend before importing kombine modules
matplotlib.use('Agg')

# Suppress warnings as in original script
warnings.filterwarnings('error')
os.environ['PYTHONUNBUFFERED'] = '1'

# Navigate to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Font sizes for all plots
FONT_SIZES = {
  'legend': 18,
  'title': 20,  # Subplot titles should be larger than legend and label
  'label': 18,
  'tick': 16,
  'suptitle': 24,  # For main titles
}

# Common configurations
FIGSIZE_BIG = (7, 7)
FIGSIZE_SMALL = (5, 5)


def add_subfigure_label(ax, label, fontsize=18, x=-0.15, y=1.05):
  """
  Add a subfigure label (A, B, C, etc.) to an axes in Nature style.

  Parameters
  ----------
  ax : matplotlib.axes.Axes
    The axes to add the label to
  label : str
    The label text (e.g., 'A', 'B', 'C')
  fontsize : float
    Font size for the label
  x, y : float
    Position of the label in axes coordinates
  """
  ax.text(
    x, y, label,
    transform=ax.transAxes,
    fontsize=fontsize,
    fontweight='bold',
    va='top',
    ha='right'
  )


def plot_km_example():
  """Generate the single Kaplan-Meier example plot."""
  print("Generating km_example.pdf...")

  dc_file = pathlib.Path(
    "../../test/kombine/datacards/simple_examples/poisson_ratio_km_censoring.txt"
  )

  datacard = Datacard.parse_datacard(dc_file)

  kml = datacard.km_likelihood(
    parameter_min=0.45,
    parameter_max=np.inf
  )

  config = KaplanMeierPlotConfig(
    create_figure=True,
    close_figure=False,
    show=False,
    saveas="km_example.pdf",
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title="Kaplan–Meier Example",
  )

  kml.plot(config=config)
  plt.savefig("km_example.pdf")
  plt.close()
  print("  Saved km_example.pdf")


def plot_compare_to_greenwood():
  """Generate comparison to Greenwood plots as a single figure with 2 subplots."""
  print("Generating comparison_to_greenwood.pdf...")

  fig, axes = plt.subplots(1, 2, figsize=(16, 7))

  # Add main title
  fig.suptitle('Comparison to exponential Greenwood confidence intervals',
               fontsize=FONT_SIZES['suptitle'], fontweight='bold')

  # Common configuration for exponential Greenwood plots
  common_config = {
    'include_nominal': False,
    'include_exponential_greenwood': True,
    'include_binomial_only': True,
    'include_full_NLL': False,
    'binomial_only_suffix': 'KoMbine',
    'exponential_greenwood_suffix': 'e. G.',
  }

  # Panel A: Small N (12 patients)
  dc_file = pathlib.Path("../../test/kombine/datacards/simple_examples/fixed_km_censoring.txt")
  datacard_small = Datacard.parse_datacard(
    dc_file
  )
  kml_small = datacard_small.km_likelihood(
    parameter_min=-np.inf,
    parameter_max=np.inf
  )

  plt.sca(axes[0])
  config_small = KaplanMeierPlotConfig(
    create_figure=False,
    close_figure=False,
    show=False,
    saveas=None,
    legend_saveas=None,
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title='$N=12$',
    legend_loc=None,  # Remove legend from left plot
    **common_config
  )
  kml_small.plot(config=config_small)
  # Remove legend if it exists
  legend = axes[0].get_legend()
  if legend is not None:
    legend.remove()
  add_subfigure_label(axes[0], 'A')

  # Panel B: Large N (100 patients)
  dc_file = pathlib.Path(
    "../../test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt"
  )
  datacard_large = Datacard.parse_datacard(dc_file)
  kml_large = datacard_large.km_likelihood(
    parameter_min=-np.inf,
    parameter_max=np.inf
  )

  plt.sca(axes[1])
  config_large = KaplanMeierPlotConfig(
    create_figure=False,
    close_figure=False,
    show=False,
    saveas=None,
    legend_saveas=None,
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title='$N=100$',
    **common_config
  )
  kml_large.plot(config=config_large)
  add_subfigure_label(axes[1], 'B')

  plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for suptitle
  plt.savefig("comparison_to_greenwood.pdf")
  plt.close()
  print("  Saved comparison_to_greenwood.pdf")


def plot_compare_p_value(testing=False):
  """
  Generate p-value comparison plots as a single figure with 3x2 subplots.

  Args:
    testing: If True, use 10 patients for all plots for faster testing
  """
  # pylint: disable=too-many-locals
  print("Generating p_value_comparison.pdf...")

  # Import here to ensure script directory is in path
  sys.path.insert(0, SCRIPT_DIR)
  # pylint: disable=import-error, import-outside-toplevel
  from compare_p_value import simulate_pvalues, plot_pvalue_comparison, PlotConfig

  fig, axes = plt.subplots(2, 3, figsize=(16, 11))

  # Add main title
  fig.suptitle('$p$ value comparison', fontsize=FONT_SIZES['suptitle'], fontweight='bold')

  # Configuration for each subplot
  configs = [
    # Top row (with ties)
    {'n_patients': 10, 'allow_ties': True, 'label': 'A'},
    {'n_patients': 100, 'allow_ties': True, 'label': 'B'},
    {'n_patients': 1000, 'allow_ties': True, 'label': 'C'},
    # Bottom row (no ties)
    {'n_patients': 10, 'allow_ties': False, 'label': 'D'},
    {'n_patients': 100, 'allow_ties': False, 'label': 'E'},
    None,  # Empty subplot in bottom right - will be used for legend
  ]

  seed = 123456
  n_trials = 100

  plot_config = PlotConfig(
    figsize=FIGSIZE_SMALL,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
  )

  legend_handles = None
  legend_labels = None

  for idx, config in enumerate(configs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    if config is None:
      # This is the empty subplot - will be used for legend
      ax.axis('off')
      continue

    # Use 10 patients for all plots in testing mode
    n_patients = 10 if testing else config['n_patients']

    print(f"  Simulating {n_patients} patients " +
          f"({'with ties' if config['allow_ties'] else 'no ties'})...")

    pvalues = simulate_pvalues(
      n_patients=n_patients,
      n_trials=n_trials,
      seed=seed + idx,
      time_is_integer=config['allow_ties'],
      verbose=False,
    )

    # Calculate correlation for subtitle
    minlp_vals, logrank_vals = pvalues[:, 0], pvalues[:, 1]
    r = np.corrcoef(minlp_vals, logrank_vals)[0, 1]

    # Display original n_patients in title even in testing mode
    title_suffix = "with ties" if config['allow_ties'] else "no ties"
    title = f"{config['n_patients']} patients ({title_suffix}) ($r={r:.3f}$)"

    # Use the refactored function from compare_p_value.py
    _, handles, labels = plot_pvalue_comparison(
      pvalues,
      title=title,
      ax=ax,
      config=plot_config,
      add_legend=False,
    )

    # Update y-axis label to add newline
    ax.set_ylabel("MINLP $p$ value\n(Cox penalty only)", fontsize=plot_config.label_fontsize)

    # Store legend handles and labels from the first plot
    if legend_handles is None:
      legend_handles, legend_labels = handles, labels

    add_subfigure_label(ax, config['label'])

  # Add legend to the bottom-right empty subplot
  if legend_handles:
    legend_ax = axes[1, 2]
    legend_ax.legend(legend_handles, legend_labels, loc='center',
                     fontsize=plot_config.legend_fontsize * 1.5,
                     frameon=True, fancybox=True, shadow=True)

  plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for suptitle
  # Adjust spacing between rows
  plt.subplots_adjust(hspace=0.4)
  plt.savefig("p_value_comparison.pdf")
  plt.close()
  print("  Saved p_value_comparison.pdf")


def plot_lung_dataset(testing=False):
  """
  Generate lung dataset plot as a single 2x3 figure combining cells and donuts.

  Args:
    testing: If True, use test datacards with fewer patients for faster testing
  """
  # pylint: disable=too-many-locals, too-many-statements
  print("Generating lung dataset plot...")

  survival_type = 'RFS'

  if survival_type == 'RFS':
    ylabel = "RFS Probability"  # Abbreviated to avoid overlap
    cell_threshold = 0.4
    donut_threshold = 1130
  elif survival_type == 'OS':
    ylabel = "Overall Survival Probability"
    cell_threshold = 0.4
    donut_threshold = 350
  else:
    raise ValueError(f"Unknown survival type: {survival_type}")

  # Load datacards - use simple test datacards in testing mode
  if testing:
    # Use simple fixed datacards for testing (they work with restricted Gurobi license)
    dc_file_cells = pathlib.Path(
      "../../test/kombine/datacards/test_compile_km_plots/test_lung_cells.txt"
    )
    dc_file_donuts = pathlib.Path(
      "../../test/kombine/datacards/test_compile_km_plots/test_lung_donuts.txt"
    )
  else:
    dc_file_cells = pathlib.Path(
      f"../../test/kombine/datacards/lung/datacard_cells_{survival_type}.txt"
    )
    dc_file_donuts = pathlib.Path(
      f"../../test/kombine/datacards/lung/datacard_donuts_{survival_type}.txt"
    )

  datacard_cells = Datacard.parse_datacard(dc_file_cells)
  datacard_donuts = Datacard.parse_datacard(dc_file_donuts)

  # Helper function to create KM plot for a specific configuration
  def create_km_subplot(ax, datacard, threshold, title, include_full_nll,
                        include_patient_wise, include_binomial, label):
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """Create a single KM subplot."""
    plt.sca(ax)

    # Create KM likelihood objects for high and low groups
    kml_low = datacard.km_likelihood(
      parameter_min=-np.inf,
      parameter_max=threshold,
    )
    kml_high = datacard.km_likelihood(
      parameter_min=threshold,
      parameter_max=np.inf,
    )

    # Common plot configuration
    common_config = {
      'close_figure': False,
      'show': False,
      'saveas': None,
      'legend_saveas': None,
      'tight_layout': False,  # Disable tight_layout for subplots in GridSpec
      'figsize': FIGSIZE_SMALL,
      'legend_fontsize': FONT_SIZES['legend'],
      'title_fontsize': FONT_SIZES['title'],
      'label_fontsize': FONT_SIZES['label'],
      'tick_fontsize': FONT_SIZES['tick'],
      'xlabel': 'Time (Months)',
      'ylabel': ylabel,
      'pvalue_fontsize': 16,
      'pvalue_format': '.2f',
      'include_nominal': False,
      'include_full_NLL': include_full_nll,
      'include_patient_wise_only': include_patient_wise,
      'include_binomial_only': include_binomial,
    }

    # Plot high group (blue)
    config_high = KaplanMeierPlotConfig(
      **common_config,
      create_figure=False,
      best_label=f"High (n={len(kml_high.nominalkm.patients)})",
      best_color='blue',
      CL_colors=['dodgerblue', 'skyblue'],
      title=title,
    )
    kml_high.plot(config=config_high)

    # Plot low group (red) on same axes
    config_low = KaplanMeierPlotConfig(
      **common_config,
      create_figure=False,
      best_label=f"Low (n={len(kml_low.nominalkm.patients)})",
      best_color='red',
      CL_colors=['orangered', 'lightcoral'],
      title=None,  # Don't repeat title
    )
    kml_low.plot(config=config_low)

    # Calculate and add p-value
    p_value_minlp = datacard.km_p_value(
      parameter_min=-np.inf,
      parameter_threshold=threshold,
      parameter_max=np.inf,
      tie_handling='breslow',
    )

    if include_full_nll:
      p_value, *_ = p_value_minlp.solve_and_pvalue()
    else:
      p_value, *_ = p_value_minlp.solve_and_pvalue(cox_only=True)

    ax.text(
      0.95, 0.95, f"$p$ = {p_value:.2f}",
      ha="right", va="top",
      transform=ax.transAxes,
      fontsize=16,
    )

    add_subfigure_label(ax, label)

  # Create single figure with 2 rows, 3 columns (cells in row 0, donuts in row 1)
  fig = plt.figure(figsize=(21, 12))  # Increased height for more space
  gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)  # Optimized hspace for legend placement

  # Row 0: CD8+FoxP3+ Cells
  print("  Processing cells...")
  ax_cells_a = fig.add_subplot(gs[0, 0])
  create_km_subplot(
    ax_cells_a, datacard_cells, cell_threshold,
    title='CD8+FoxP3+ Cells',
    include_full_nll=True,
    include_patient_wise=False,
    include_binomial=False,
    label='A'
  )

  ax_cells_b = fig.add_subplot(gs[0, 1])
  create_km_subplot(
    ax_cells_b, datacard_cells, cell_threshold,
    title='CD8+FoxP3+ Cells, Patient-Wise Errors',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='B'
  )

  ax_cells_c = fig.add_subplot(gs[0, 2])
  create_km_subplot(
    ax_cells_c, datacard_cells, cell_threshold,
    title='CD8+FoxP3+ Cells, Binomial Errors',
    include_full_nll=False,
    include_patient_wise=False,
    include_binomial=True,
    label='C'
  )

  # Row 1: DONUTS
  print("  Processing donuts...")
  ax_donuts_a = fig.add_subplot(gs[1, 0])
  create_km_subplot(
    ax_donuts_a, datacard_donuts, donut_threshold,
    title='DONUTS',
    include_full_nll=True,
    include_patient_wise=False,
    include_binomial=False,
    label='D'
  )

  ax_donuts_b = fig.add_subplot(gs[1, 1])
  create_km_subplot(
    ax_donuts_b, datacard_donuts, donut_threshold,
    title='DONUTS, Patient-Wise Errors',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='E'
  )

  ax_donuts_c = fig.add_subplot(gs[1, 2])
  create_km_subplot(
    ax_donuts_c, datacard_donuts, donut_threshold,
    title='DONUTS, Binomial Errors',
    include_full_nll=False,
    include_patient_wise=False,
    include_binomial=True,
    label='F'
  )

  # Remove legends from all subplots - we'll add separate legends for cells and donuts
  for ax in [ax_cells_a, ax_cells_b, ax_cells_c, ax_donuts_a, ax_donuts_b, ax_donuts_c]:
    legend = ax.get_legend()
    if legend is not None:
      legend.remove()

  # Get legend handles and labels from cells row
  handles_cells, labels_cells = ax_cells_a.get_legend_handles_labels()
  if handles_cells:
    # Add legend for cells row (centered between rows, no title)
    # Position optimized so distance from top row = distance from bottom row
    fig.legend(handles_cells, labels_cells, loc='center', ncol=len(handles_cells),
               fontsize=FONT_SIZES['legend'], bbox_to_anchor=(0.5, 0.48),
               frameon=True, fancybox=True)

  # Get legend handles and labels from donuts row
  handles_donuts, labels_donuts = ax_donuts_a.get_legend_handles_labels()
  if handles_donuts:
    # Add legend for donuts row (bottom, no title)
    fig.legend(handles_donuts, labels_donuts, loc='lower center', ncol=len(handles_donuts),
               fontsize=FONT_SIZES['legend'], bbox_to_anchor=(0.5, 0.01),
               frameon=True, fancybox=True)

  output_file = f"lung_km_{survival_type}.pdf"
  # Use bbox_inches='tight' with bbox_extra_artists to include the legends
  if fig.legends:
    plt.savefig(output_file, bbox_inches='tight', bbox_extra_artists=fig.legends)
  else:
    plt.savefig(output_file, bbox_inches='tight')
  plt.close()
  print(f"  Saved {output_file}")


def main():
  """Generate plots for the paper based on command line arguments."""
  parser = argparse.ArgumentParser(
    description='Generate plots for 02_kombine.tex',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python compile_km_plots.py                    # Generate all plots
  python compile_km_plots.py --testing          # Generate all plots with test data
  python compile_km_plots.py --lung --testing   # Generate only lung plot with test data
  python compile_km_plots.py --greenwood --p-value  # Generate only Greenwood and p-value plots
    """
  )

  # Plot selection options
  parser.add_argument('--km-example', action='store_true',
                      help='Generate only the KM example plot')
  parser.add_argument('--greenwood', action='store_true',
                      help='Generate only the Greenwood comparison plot')
  parser.add_argument('--p-value', action='store_true',
                      help='Generate only the p-value comparison plot')
  parser.add_argument('--lung', action='store_true',
                      help='Generate only the lung cancer dataset plot')

  # Testing option
  parser.add_argument('--testing', action='store_true',
                      help='Use smaller datasets for faster testing')

  args = parser.parse_args()

  # If no plot options specified, generate all plots
  generate_all = not (args.km_example or args.greenwood or args.p_value or args.lung)

  print("Starting plot generation for 02_kombine.tex")
  print("=" * 60)
  if args.testing:
    print("TESTING MODE: Using smaller datasets")
    print("=" * 60)

  if generate_all or args.km_example:
    plot_km_example()

  if generate_all or args.greenwood:
    plot_compare_to_greenwood()

  if generate_all or args.p_value:
    plot_compare_p_value(testing=args.testing)

  if generate_all or args.lung:
    plot_lung_dataset(testing=args.testing)

  print("=" * 60)
  print("Plot generation complete!")


if __name__ == "__main__":
  main()
