#!/usr/bin/env python3
"""
Generate all plots for 02_kombine.tex using matplotlib subplots.

This script replaces compile_km_plots.sh and creates combined figures with
subplots instead of individual PDF files. Each combined figure includes
Nature-style subfigure labels (A, B, C, etc.).
"""

import os
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
  'legend': 16,
  'title': 16,
  'label': 16,
  'tick': 16,
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

  datacard = Datacard.parse_datacard(
    "../../test/kombine/datacards/simple_examples/poisson_ratio_km_censoring.txt"
  )

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

  _, axes = plt.subplots(1, 2, figsize=(14, 7))

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
  datacard_small = Datacard.parse_datacard(
    "../../test/kombine/datacards/simple_examples/fixed_km_censoring.txt"
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
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title='Comparison to Greenwood, $N=12$',
    **common_config
  )
  kml_small.plot(config=config_small)
  add_subfigure_label(axes[0], 'A')

  # Panel B: Large N (100 patients)
  datacard_large = Datacard.parse_datacard(
    "../../test/kombine/datacards/simple_examples/fixed_km_censoring_many_patients.txt"
  )
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
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title='Comparison to Greenwood, $N=100$',
    **common_config
  )
  kml_large.plot(config=config_large)
  add_subfigure_label(axes[1], 'B')

  plt.tight_layout()
  plt.savefig("comparison_to_greenwood.pdf")
  plt.close()
  print("  Saved comparison_to_greenwood.pdf")


def plot_compare_p_value():
  """Generate p-value comparison plots as a single figure with 3x2 subplots."""
  # pylint: disable=too-many-locals, too-many-statements
  print("Generating p_value_comparison.pdf...")

  # Import here to ensure script directory is in path
  sys.path.insert(0, SCRIPT_DIR)
  # pylint: disable=import-error, import-outside-toplevel
  from compare_p_value import simulate_pvalues, PlotConfig

  _, axes = plt.subplots(2, 3, figsize=(15, 10))

  # Configuration for each subplot
  configs = [
    # Top row (with ties)
    {'n_patients': 10, 'allow_ties': True, 'label': 'A'},
    {'n_patients': 100, 'allow_ties': True, 'label': 'B'},
    {'n_patients': 1000, 'allow_ties': True, 'label': 'C'},
    # Bottom row (no ties)
    {'n_patients': 10, 'allow_ties': False, 'label': 'D'},
    {'n_patients': 100, 'allow_ties': False, 'label': 'E'},
    None,  # Empty subplot in bottom right
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

  for idx, config in enumerate(configs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    if config is None:
      # Hide the empty subplot
      ax.axis('off')
      continue

    print(f"  Simulating {config['n_patients']} patients " +
          f"({'with ties' if config['allow_ties'] else 'no ties'})...")

    pvalues = simulate_pvalues(
      n_patients=config['n_patients'],
      n_trials=n_trials,
      seed=seed + idx,
      time_is_integer=config['allow_ties'],
      verbose=False,
    )

    minlp_vals, logrank_vals = pvalues[:, 0], pvalues[:, 1]
    r = np.corrcoef(minlp_vals, logrank_vals)[0, 1]

    # Plot on the specific axes
    plt.sca(ax)
    ax.scatter(logrank_vals, minlp_vals, alpha=0.6, s=20)
    ax.plot([0, 1], [0, 1], "r--", label="$y=x$")

    ax.set_xlabel("Conventional log-rank $p$ value", fontsize=plot_config.label_fontsize)
    ax.set_ylabel("MINLP (Cox penalty only) $p$ value", fontsize=plot_config.label_fontsize)

    title_suffix = "with ties" if config['allow_ties'] else "no ties"
    ax.set_title(
      f"$p$ value comparison for\n{config['n_patients']} patients ({title_suffix})\n$r={r:.3f}$",
      fontsize=plot_config.title_fontsize
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=plot_config.tick_fontsize)
    ax.legend(fontsize=plot_config.legend_fontsize)
    ax.grid(True)

    # Add zoomed inlay for better visibility of small p-values
    inlay_upper_limit = 0.1
    inlay_left = 0.59
    inlay_bottom = 0.07
    inlay_right = 0.95
    inlay_top = 0.43

    inlay_ax = ax.inset_axes(
      (inlay_left, inlay_bottom, inlay_right - inlay_left, inlay_top - inlay_bottom)
    )

    inlay_ax.scatter(logrank_vals, minlp_vals, alpha=0.6, s=5)
    inlay_ax.plot([0, inlay_upper_limit], [0, inlay_upper_limit], "r--", linewidth=0.8)

    inlay_ax.set_xlim(0, inlay_upper_limit)
    inlay_ax.set_ylim(0, inlay_upper_limit)
    inlay_ax.set_aspect('equal', adjustable='box')
    inlay_ticks = np.linspace(0, inlay_upper_limit, 3)
    inlay_ax.set_xticks(inlay_ticks)
    inlay_ax.set_yticks(inlay_ticks)
    inlay_minor_ticks = np.linspace(0, inlay_upper_limit, 5)
    inlay_ax.set_xticks(inlay_minor_ticks, minor=True)
    inlay_ax.set_yticks(inlay_minor_ticks, minor=True)

    inlay_ax.tick_params(axis='both', which='major', labelsize=plot_config.tick_fontsize * 0.7)
    inlay_ax.grid(True, alpha=0.5, which='both')

    for spine in inlay_ax.spines.values():
      spine.set_edgecolor('black')
      spine.set_linewidth(1.5)

    add_subfigure_label(ax, config['label'])

  plt.tight_layout()
  plt.savefig("p_value_comparison.pdf")
  plt.close()
  print("  Saved p_value_comparison.pdf")


def plot_lung_dataset():
  """Generate lung dataset plots as two figures with 1x3 subplots plus legend."""
  # pylint: disable=too-many-locals, too-many-statements
  print("Generating lung dataset plots...")

  survival_type = 'RFS'

  if survival_type == 'RFS':
    ylabel = "Regression-Free Survival Probability"
    cell_threshold = 0.4
    donut_threshold = 1130
  elif survival_type == 'OS':
    ylabel = "Overall Survival Probability"
    cell_threshold = 0.4
    donut_threshold = 350
  else:
    raise ValueError(f"Unknown survival type: {survival_type}")

  # Load datacards
  datacard_cells = Datacard.parse_datacard(
    f"../../test/kombine/datacards/lung/datacard_cells_{survival_type}.txt"
  )
  datacard_donuts = Datacard.parse_datacard(
    f"../../test/kombine/datacards/lung/datacard_donuts_{survival_type}.txt"
  )

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
    try:
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
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"    Warning: Could not calculate p-value: {exc}")

    add_subfigure_label(ax, label)

  # Create figures for cells and donuts
  for dataset_name, datacard, threshold, name in [
    ('cells', datacard_cells, cell_threshold, 'CD8+FoxP3+ Cells'),
    ('donuts', datacard_donuts, donut_threshold, 'DONUTS'),
  ]:
    print(f"  Processing {dataset_name}...")

    # Create figure with 1 row, 4 columns (3 plots + 1 legend)
    fig = plt.figure(figsize=(21, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.2], wspace=0.35)

    # Panel A: Full NLL
    ax_a = fig.add_subplot(gs[0, 0])
    create_km_subplot(
      ax_a, datacard, threshold,
      title=name,
      include_full_nll=True,
      include_patient_wise=False,
      include_binomial=False,
      label='A'
    )

    # Panel B: Patient-wise only
    ax_b = fig.add_subplot(gs[0, 1])
    create_km_subplot(
      ax_b, datacard, threshold,
      title=f"{name}, Patient-Wise Errors",
      include_full_nll=False,
      include_patient_wise=True,
      include_binomial=False,
      label='B'
    )

    # Panel C: Binomial only
    ax_c = fig.add_subplot(gs[0, 2])
    create_km_subplot(
      ax_c, datacard, threshold,
      title=f"{name}, Binomial Errors",
      include_full_nll=False,
      include_patient_wise=False,
      include_binomial=True,
      label='C'
    )

    # Add legend in the rightmost column
    legend_ax = fig.add_subplot(gs[0, 3])
    legend_ax.axis('off')

    # Get legend from one of the plots and display it separately
    handles, labels = ax_a.get_legend_handles_labels()
    if handles:
      # Remove legend from the subplots
      for ax in [ax_a, ax_b, ax_c]:
        legend = ax.get_legend()
        if legend is not None:
          legend.remove()

      # Add combined legend to the legend axes
      legend_ax.legend(handles, labels, loc='center', fontsize=FONT_SIZES['legend'])

    output_file = f"lung_{dataset_name}_km_{survival_type}.pdf"
    plt.savefig(output_file)
    plt.close()
    print(f"    Saved {output_file}")


def main():
  """Generate all plots for the paper."""
  print("Starting plot generation for 02_kombine.tex")
  print("=" * 60)

  # Generate km_example
  try:
    plot_km_example()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  WARNING: Failed to generate km_example.pdf: {exc}")

  # Generate comparison to Greenwood
  try:
    plot_compare_to_greenwood()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  WARNING: Failed to generate comparison_to_greenwood.pdf: {exc}")

  # Generate p-value comparison
  try:
    plot_compare_p_value()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  WARNING: Failed to generate p_value_comparison.pdf: {exc}")

  # Generate lung dataset plots
  try:
    plot_lung_dataset()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  WARNING: Failed to generate lung dataset plots: {exc}")

  print("=" * 60)
  print("Plot generation complete!")


if __name__ == "__main__":
  main()
