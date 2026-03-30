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
import re
import subprocess
import sys
import warnings

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

from kombine.datacard import Datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

from .compare_p_value import simulate_pvalues, plot_pvalue_comparison, PlotConfig

# Set matplotlib backend before importing kombine modules
matplotlib.use('Agg')

# Suppress warnings as in original script
warnings.filterwarnings('error')
os.environ['PYTHONUNBUFFERED'] = '1'

# Get script directory for relative file operations
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATACARDS_DIR = SCRIPT_DIR / "../../test/kombine/datacards"

# When run with `python -m docs.kombine.compile_km_plots`, the current directory
# is already the repo root. We keep it there for imports to work correctly.
# File operations will use SCRIPT_DIR for relative paths.

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


def plot_km_example(testing=False):
  """Generate the single Kaplan-Meier example plot."""
  print("Generating km_example.pdf...")

  if testing:
    # Use minimal datacard for testing (4 patients)
    dc_file = DATACARDS_DIR / "simple_examples/simple_km_few_deaths.txt"
  else:
    dc_file = DATACARDS_DIR / "simple_examples/poisson_ratio_km_censoring.txt"
  datacard = Datacard.parse_datacard(dc_file)

  kml = datacard.km_likelihood(
    parameter_min=0.45 if not testing else -np.inf,
    parameter_max=np.inf
  )

  output_file = SCRIPT_DIR / "km_example.pdf"
  config = KaplanMeierPlotConfig(
    create_figure=True,
    close_figure=False,
    show=False,
    saveas=str(output_file),
    figsize=FIGSIZE_BIG,
    legend_fontsize=FONT_SIZES['legend'],
    title_fontsize=FONT_SIZES['title'],
    label_fontsize=FONT_SIZES['label'],
    tick_fontsize=FONT_SIZES['tick'],
    title="Kaplan–Meier Example",
  )

  kml.plot(config=config)
  plt.savefig(output_file)
  plt.close()
  print(f"  Saved {output_file}")


def plot_compare_to_greenwood(testing=False):
  """
  Generate comparison to Greenwood plots as a single figure with 2 subplots.

  Args:
    testing: If True, use minimal datacards (4 timepoints for testing)
  """
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

  # Panel A: Small N (12 patients in production, 4 in testing)
  if testing:
    dc_file = DATACARDS_DIR / "simple_examples/simple_km_few_deaths.txt"
    n_small_label = '$N=4$'
  else:
    dc_file = DATACARDS_DIR / "simple_examples/fixed_km_censoring.txt"
    n_small_label = '$N=12$'
  datacard_small = Datacard.parse_datacard(dc_file)
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
    title=n_small_label,
    legend_loc=None,  # Remove legend from left plot
    **common_config
  )
  kml_small.plot(config=config_small)
  # Remove legend if it exists
  legend = axes[0].get_legend()
  if legend is not None:
    legend.remove()
  add_subfigure_label(axes[0], 'A')

  # Panel B: Large N (100 patients in production, 4 in testing)
  if testing:
    # Use same minimal datacard for testing
    dc_file = DATACARDS_DIR / "simple_examples/simple_km_few_deaths.txt"
    n_label = '$N=4$'
  else:
    dc_file = DATACARDS_DIR / "simple_examples/fixed_km_censoring_many_patients.txt"
    n_label = '$N=100$'

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
    title=n_label,
    **common_config
  )
  kml_large.plot(config=config_large)
  add_subfigure_label(axes[1], 'B')

  plt.tight_layout(rect=(0, 0, 1, 0.96))  # Make room for suptitle
  output_file = SCRIPT_DIR / "comparison_to_greenwood.pdf"
  plt.savefig(output_file)
  plt.close()
  print(f"  Saved {output_file}")


def plot_compare_p_value(testing=False):
  """
  Generate p-value comparison plots as a single figure with 3x2 subplots.

  Args:
    testing: If True, use 10 patients for all plots for faster testing
  """
  # pylint: disable=too-many-locals
  print("Generating p_value_comparison.pdf...")

  _, axes = plt.subplots(2, 3, figsize=(16, 11))

  # Add main title
  # fig.suptitle('$p$ value comparison', fontsize=FONT_SIZES['suptitle'], fontweight='bold')

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

    # Use 3 patients for all plots in testing mode (to work with restricted Gurobi license)
    n_patients = 3 if testing else config['n_patients']

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

  plt.tight_layout()  # Make room for suptitle
  # Adjust spacing between rows
  plt.subplots_adjust(hspace=0.4)
  output_file = SCRIPT_DIR / "p_value_comparison.pdf"
  plt.savefig(output_file)
  plt.close()
  print(f"  Saved {output_file}")

def plot_lung_dataset(testing: bool | tuple[bool, ...] =False, survival_type: str ='RFS'):
  """
  Generate lung dataset plot with 10 panels (A-J) in 3 rows:
  - Row 1: A, B (cells), F, G (DONUTS)
  - Row 2: C, D (cells), H, I (DONUTS)
  - Row 3: E (cells), J (DONUTS) - larger plots

  Args:
    testing: If True (or a single bool), use test datacards for all panels.
             If a tuple of 10 bools, use test datacards for each panel individually
             (order: A, B, C, D, E, F, G, H, I, J)
  survival_type: 'RFS' or 'OS' to select which survival type to plot
  """
  # pylint: disable=too-many-locals, too-many-statements
  print(f"Generating lung dataset plot for {survival_type}...")

  # Convert testing to a tuple of 10 bools if it's a single bool
  if isinstance(testing, bool):
    testing_flags = (testing,) * 10
  elif isinstance(testing, (tuple, list)) and len(testing) == 10:
    testing_flags = tuple(testing)
  else:
    raise ValueError("testing must be a bool or a tuple/list of 10 bools")

  # Generate systematic datacards if needed (only if any panel is in production mode)
  if not all(testing_flags):
    print("  Generating systematic datacards...")
    script_path = DATACARDS_DIR / "lung/generate_systematic_datacards.py"
    subprocess.run([sys.executable, str(script_path)], check=True, text=True)

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

  # Define lung datacard root directory
  lung_datacard_root = DATACARDS_DIR / "lung"

  # Helper function to get datacard path based on testing flag for that panel
  def get_datacard_path(panel_idx, cell_type, datacard_type, survival_type):
    """
    Get datacard path for a specific panel.

    Args:
      panel_idx: Index in testing_flags (0-9 for panels A-J)
      cell_type: 'cells' or 'donuts'
      datacard_type: 'poisson', 'flatfield_before', 'flatfield_after', or 'combined'
      survival_type: 'RFS' or 'OS'
    """
    if testing_flags[panel_idx]:
      # Use test datacards
      return lung_datacard_root / f"test_small_dataset/test_lung_{cell_type}.txt"

    # Use production datacards
    if datacard_type == 'poisson':
      return lung_datacard_root / f"poisson/datacard_{cell_type}_{survival_type}.txt"
    if datacard_type == 'flatfield_before':
      return (lung_datacard_root /
              f"uncorrected_flatfielding_systematic/datacard_{cell_type}_{survival_type}.txt")
    if datacard_type == 'flatfield_after':
      return (lung_datacard_root /
              f"flatfielding_systematic/datacard_{cell_type}_{survival_type}.txt")
    if datacard_type == 'combined':
      return (lung_datacard_root /
              f"poisson_and_flatfielding/datacard_{cell_type}_{survival_type}.txt")
    raise ValueError(f"Unknown datacard_type: {datacard_type}")

  # Load datacards for each panel
  # Panel B uses cells poisson; Panel G uses donuts poisson
  dc_file_cells_poisson = get_datacard_path(1, 'cells', 'poisson', survival_type)  # Panel A
  dc_file_donuts_poisson = get_datacard_path(6, 'donuts', 'poisson', survival_type)  # Panel F

  #Panels A, F use binomial uncertainties (from poisson datacard)
  dc_file_cells_binomial = get_datacard_path(0, 'cells', 'poisson', survival_type)  # Panel A
  dc_file_donuts_binomial = get_datacard_path(5, 'donuts', 'poisson', survival_type)  # Panel F

  # Panels C, D use cells flatfield before/after; Panels H, I use donuts flatfield before/after
  dc_file_cells_flatfield_before = get_datacard_path(
    2, 'cells', 'flatfield_before', survival_type
  )  # Panel C
  dc_file_cells_flatfield_after = get_datacard_path(
    3, 'cells', 'flatfield_after', survival_type
  )  # Panel D
  dc_file_donuts_flatfield_before = get_datacard_path(
    7, 'donuts', 'flatfield_before', survival_type
  )  # Panel H
  dc_file_donuts_flatfield_after = get_datacard_path(
    8, 'donuts', 'flatfield_after', survival_type
  )  # Panel I

  # Panels E, J use cells/donuts combined
  dc_file_cells_combined = get_datacard_path(4, 'cells', 'combined', survival_type)  # Panel E
  dc_file_donuts_combined = get_datacard_path(9, 'donuts', 'combined', survival_type)  # Panel J
  # Parse all datacards
  print(f"  Loading datacards (testing flags: {testing_flags})...")
  datacard_cells_poisson = Datacard.parse_datacard(dc_file_cells_poisson)
  datacard_donuts_poisson = Datacard.parse_datacard(dc_file_donuts_poisson)
  datacard_cells_binomial = Datacard.parse_datacard(dc_file_cells_binomial)
  datacard_donuts_binomial = Datacard.parse_datacard(dc_file_donuts_binomial)
  datacard_cells_flatfield_before = Datacard.parse_datacard(dc_file_cells_flatfield_before)
  datacard_donuts_flatfield_before = Datacard.parse_datacard(dc_file_donuts_flatfield_before)
  datacard_cells_flatfield_after = Datacard.parse_datacard(dc_file_cells_flatfield_after)
  datacard_donuts_flatfield_after = Datacard.parse_datacard(dc_file_donuts_flatfield_after)
  datacard_cells_combined = Datacard.parse_datacard(dc_file_cells_combined)
  datacard_donuts_combined = Datacard.parse_datacard(dc_file_donuts_combined)

  # Helper function to create KM plot for a specific configuration
  def create_km_subplot(ax, datacard, threshold, title, include_full_nll,
                        include_patient_wise, include_binomial, label, label_x=-0.15,
                        rerun_until_convergence=False):
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
      'rerun_until_convergence': rerun_until_convergence,
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

    if not include_patient_wise:
      # Calculate and add p-value
      p_value_minlp = datacard.km_p_value(
        parameter_min=-np.inf,
        parameter_threshold=threshold,
        parameter_max=np.inf,
        tie_handling='breslow',
      )

      if include_full_nll:
        p_value, *_ = p_value_minlp.solve_and_pvalue()
        text = f"$p$ = {p_value:.2f}"
      else:
        p_value, *_ = p_value_minlp.solve_and_pvalue(cox_only=True)
        text = f"$p$ (Cox only) = {p_value:.2f}"

      ax.text(
        0.95, 0.95, text,
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=16,
      )

    add_subfigure_label(ax, label, x=label_x)

  # Create figure with 3 rows and 4 columns
  # Row 1: A, B (cells) | F, G (DONUTS) - 4 small plots
  # Row 2: C, D (cells) | H, I (DONUTS) - 4 small plots
  # Row 3: E (cells) | J (DONUTS) - 2 larger plots (taller to keep them square)
  fig = plt.figure(figsize=(21, 21))  # Increased height for taller bottom plots
  # Increased wspace to 0.50 to create more space between columns, preventing right column
  # subfigure labels (F, H, J) and axis labels from overlapping the vertical separator line
  gs = fig.add_gridspec(3, 4, hspace=0.25, wspace=0.25, height_ratios=[1, 1, 2])

  print("  Processing cells...")

  # Row 0, Col 0: Panel A - Binomial uncertainties (cells)
  print(f"    Creating panel A (binomial, cells, testing={testing_flags[0]})...")
  ax_a = fig.add_subplot(gs[0, 0])
  create_km_subplot(
    ax_a, datacard_cells_binomial, cell_threshold,
    title='Binomial Uncertainties',
    include_full_nll=False,
    include_patient_wise=False,
    include_binomial=True,
    label='A'
  )

  # Row 0, Col 1: Panel B - Poisson uncertainties (cells)
  print(f"    Creating panel B (Poisson, cells, testing={testing_flags[1]})...")
  ax_b = fig.add_subplot(gs[0, 1])
  create_km_subplot(
    ax_b, datacard_cells_poisson, cell_threshold,
    title='Poisson Uncertainties',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='B'
  )

  # Row 1, Col 0: Panel C - Flatfielding before correction (cells)
  print(f"    Creating panel C (flatfield pre, cells, testing={testing_flags[2]})...")
  ax_c = fig.add_subplot(gs[1, 0])
  create_km_subplot(
    ax_c, datacard_cells_flatfield_before, cell_threshold,
    title='Flatfielding\n(Pre-Correction)',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='C'
  )

  # Row 1, Col 1: Panel D - Flatfielding after correction (cells)
  print(f"    Creating panel D (flatfield post, cells, testing={testing_flags[3]})...")
  ax_d = fig.add_subplot(gs[1, 1])
  create_km_subplot(
    ax_d, datacard_cells_flatfield_after, cell_threshold,
    title='Flatfielding\n(Post-Correction)',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='D'
  )

  # Row 2, Col 0-1: Panel E - Combined uncertainties (cells) - spans 2 columns
  print(f"    Creating panel E (combined, cells, testing={testing_flags[4]})...")
  ax_e = fig.add_subplot(gs[2, 0:2])
  create_km_subplot(
    ax_e, datacard_cells_combined, cell_threshold,
    title='Combined Uncertainties',
    include_full_nll=True,
    include_patient_wise=False,
    include_binomial=False,
    label='E',
    label_x=-0.08  # Adjusted for 2-column span to align with A and C
  )

  print("  Processing donuts...")

  # Row 0, Col 2: Panel F - Binomial uncertainties (DONUTS)
  print(f"    Creating panel F (binomial, donuts, testing={testing_flags[5]})...")
  ax_f = fig.add_subplot(gs[0, 2])
  create_km_subplot(
    ax_f, datacard_donuts_binomial, donut_threshold,
    title='Binomial Uncertainties',
    include_full_nll=False,
    include_patient_wise=False,
    include_binomial=True,
    label='F'
  )

  # Row 0, Col 3: Panel G - Poisson uncertainties (DONUTS)
  print(f"    Creating panel G (Poisson, donuts, testing={testing_flags[6]})...")
  ax_g = fig.add_subplot(gs[0, 3])
  create_km_subplot(
    ax_g, datacard_donuts_poisson, donut_threshold,
    title='Poisson Uncertainties',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='G'
  )

  # Row 1, Col 2: Panel H - Flatfielding before correction (DONUTS)
  print(f"    Creating panel H (flatfield pre, donuts, testing={testing_flags[7]})...")
  ax_h = fig.add_subplot(gs[1, 2])
  create_km_subplot(
    ax_h, datacard_donuts_flatfield_before, donut_threshold,
    title='Flatfielding\n(Pre-Correction)',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='H',
    rerun_until_convergence=True
  )

  # Row 1, Col 3: Panel I - Flatfielding after correction (DONUTS)
  print(f"    Creating panel I (flatfield post, donuts, testing={testing_flags[8]})...")
  ax_i = fig.add_subplot(gs[1, 3])
  create_km_subplot(
    ax_i, datacard_donuts_flatfield_after, donut_threshold,
    title='Flatfielding\n(Post-Correction)',
    include_full_nll=False,
    include_patient_wise=True,
    include_binomial=False,
    label='I'
  )

  # Row 2, Col 2-3: Panel J - Combined uncertainties (DONUTS) - spans 2 columns
  print(f"    Creating panel J (combined, donuts, testing={testing_flags[9]})...")
  ax_j = fig.add_subplot(gs[2, 2:4])
  create_km_subplot(
    ax_j, datacard_donuts_combined, donut_threshold,
    title='Combined Uncertainties',
    include_full_nll=True,
    include_patient_wise=False,
    include_binomial=False,
    label='J',
    label_x=-0.08  # Adjusted for 2-column span to align with F and H
  )

  # Remove legends from all subplots
  all_axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j]
  for ax in all_axes:
    legend = ax.get_legend()
    if legend is not None:
      legend.remove()

  output_file = SCRIPT_DIR / f"lung_km_{survival_type}.pdf"

  # Adjust layout to make room for column titles, legends, and subplot labels
  # Small margins to accommodate subplot labels (A,C,E on left; F,H,J on right)
  # More space at top for titles, space at bottom for legends
  plt.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.09)

  # Calculate column title positions based on actual subplot positions
  # Get the actual bounding boxes INCLUDING labels to find where the gap really is
  # Use column 1 (ax_b) and column 2 (ax_f) to find the boundary between left and right groups
  # Need to draw the canvas first so matplotlib calculates the text bounding boxes
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

  # get_tightbbox includes axis labels, tick labels, etc.
  orig_bbox_a = ax_a.get_tightbbox(renderer)
  orig_bbox_b = ax_b.get_tightbbox(renderer)
  orig_bbox_f = ax_f.get_tightbbox(renderer)
  orig_bbox_g = ax_g.get_tightbbox(renderer)
  assert orig_bbox_a is not None
  assert orig_bbox_b is not None
  assert orig_bbox_f is not None
  assert orig_bbox_g is not None
  bbox_a = orig_bbox_a.transformed(fig.transFigure.inverted())
  bbox_b = orig_bbox_b.transformed(fig.transFigure.inverted())
  bbox_f = orig_bbox_f.transformed(fig.transFigure.inverted())
  bbox_g = orig_bbox_g.transformed(fig.transFigure.inverted())

  # The center line should be in the middle of the gap between these two columns
  # x1 is right edge of B (with labels), x0 is left edge of F (with labels)
  center_line = (bbox_b.x1 + bbox_f.x0) / 2

  # For column titles, find the centers of the left and right groups
  # Left column spans from left edge of A to right edge of B (including labels)
  left_col_center = (bbox_a.x0 + bbox_b.x1) / 2
  # Right column spans from left edge of F to right edge of G (including labels)
  right_col_center = (bbox_f.x0 + bbox_g.x1) / 2

  # Add column titles at the top, centered above their respective columns
  fig.text(left_col_center, 0.99, 'CD8+FoxP3+ Cells', ha='center', va='top',
           fontsize=FONT_SIZES['suptitle'], fontweight='bold',
           transform=fig.transFigure)
  fig.text(right_col_center, 0.99, 'DONUTS', ha='center', va='top',
           fontsize=FONT_SIZES['suptitle'], fontweight='bold',
           transform=fig.transFigure)

  # Add a vertical line to separate cells (left) from DONUTS (right)
  # The line goes in the middle of the gap between columns 1 and 2
  # Extend from bottom (0.0) to top of column titles (0.99) to cover legend area
  line_x = center_line
  fig.add_artist(matplotlib.lines.Line2D(
    [line_x, line_x], [0.0, 0.99],
    transform=fig.transFigure,
    color='black', linewidth=2, linestyle='-')
  )

  # Get legend handles and labels from the combined plots (no "Binomial only" text)
  handles_cells, labels_cells = ax_e.get_legend_handles_labels()
  handles_donuts, labels_donuts = ax_j.get_legend_handles_labels()

  # Reorder legend items to display in rows: High group in top row, Low group in bottom row
  # Matplotlib with ncol=3 fills column-by-column, so we need to reorder for row-by-row display
  def reorder_legend_items(handles, labels):
    """Reorder legend items so they display in rows when matplotlib fills by columns."""

    if len(handles) != 6:
      raise ValueError(f"Expected 6 legend items, got {len(handles)}")

    # Validate labels are in the expected input order
    # Input order: [High, 68CL, 95CL, Low, 68CL, 95CL]
    if not re.match(r'^High \(n=[0-9]+\)$', labels[0]):
      raise ValueError(f"Expected labels[0] to be 'High (n=...)', got '{labels[0]}'")
    if labels[1] != '68% CL':
      raise ValueError(f"Expected labels[1] to be '68% CL', got '{labels[1]}'")
    if labels[2] != '95% CL':
      raise ValueError(f"Expected labels[2] to be '95% CL', got '{labels[2]}'")
    if not re.match(r'^Low \(n=[0-9]+\)$', labels[3]):
      raise ValueError(f"Expected labels[3] to be 'Low (n=...)', got '{labels[3]}'")
    if labels[4] != '68% CL':
      raise ValueError(f"Expected labels[4] to be '68% CL', got '{labels[4]}'")
    if labels[5] != '95% CL':
      raise ValueError(f"Expected labels[5] to be '95% CL', got '{labels[5]}'")

    # With ncol=3, matplotlib fills by columns:
    # Position: [0, 1, 2, 3, 4, 5] displays as:
    #   0  2  4
    #   1  3  5
    # We want:
    #   High  68%  95%
    #   Low   68%  95%
    # So reorder to: [High, Low, 68%, 68%, 95%, 95%]
    # Mapping: [0, 3, 1, 4, 2, 5]
    new_order = [0, 3, 1, 4, 2, 5]
    return [handles[i] for i in new_order], [labels[i] for i in new_order]

  handles_cells, labels_cells = reorder_legend_items(handles_cells, labels_cells)
  handles_donuts, labels_donuts = reorder_legend_items(handles_donuts, labels_donuts)

  # Add separate legends for each column (2 rows each)
  if handles_cells:
    # Calculate ncol to get 2 rows: ncol = ceil(n_items / 2)
    ncol_cells = (len(handles_cells) + 1) // 2
    fig.legend(handles_cells, labels_cells, loc='lower left',
               ncol=ncol_cells, fontsize=FONT_SIZES['legend'],
               bbox_to_anchor=(left_col_center - 0.18, 0.005), frameon=True, fancybox=True)

  if handles_donuts:
    ncol_donuts = (len(handles_donuts) + 1) // 2
    fig.legend(handles_donuts, labels_donuts, loc='lower right',
               ncol=ncol_donuts, fontsize=FONT_SIZES['legend'],
               bbox_to_anchor=(right_col_center + 0.18, 0.005), frameon=True, fancybox=True)

  # Save the figure
  plt.savefig(output_file)
  plt.close()
  print(f"  Saved {output_file}")


def main():
  """Generate plots for the paper based on command line arguments."""
  parser = argparse.ArgumentParser(
    description='Generate plots for 02_kombine.tex',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples (run from parent directory):
  python -m docs.kombine.compile_km_plots                    # Generate all plots
  python -m docs.kombine.compile_km_plots --testing          # Generate all plots with test data
  python -m docs.kombine.compile_km_plots --lung --testing   # Generate only lung plot with test data
  python -m docs.kombine.compile_km_plots --lung --lung-production-panel A E  # Lung plot: panels A,E in production, others in testing
  python -m docs.kombine.compile_km_plots --greenwood --p-value  # Generate Greenwood and p-value plots
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
  parser.add_argument('--lung-production-panel', type=str, nargs='+', metavar='PANEL',
                      help='For lung plot, use production data for specified panel(s) '
                           '(A-J, space-separated) and testing data for others. '
                           'Example: --lung-production-panel A E')
  parser.add_argument('--lung-survival-type', type=str, choices=['RFS', 'OS'], default='RFS',
                      help='Survival type for lung dataset plot: RFS (default) or OS')

  args = parser.parse_args()

  # If no plot options specified, generate all plots
  generate_all = not (args.km_example or args.greenwood or args.p_value or args.lung)

  print("Starting plot generation for 02_kombine.tex")
  print("=" * 60)
  if args.testing:
    print("TESTING MODE: Using smaller datasets")
    print("=" * 60)

  if generate_all or args.km_example:
    plot_km_example(testing=args.testing)

  if generate_all or args.greenwood:
    plot_compare_to_greenwood(testing=args.testing)

  if generate_all or args.p_value:
    plot_compare_p_value(testing=args.testing)

  if generate_all or args.lung:
    # Handle lung plot testing options
    lung_testing = args.testing
    if args.lung_production_panel:
      # Parse panel specification (e.g., ["A", "E"])
      production_panels = [p.strip().upper() for p in args.lung_production_panel]
      panel_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

      # Validate panel names
      invalid_panels = [p for p in production_panels if p not in panel_names]
      if invalid_panels:
        parser.error(f"Invalid panel name(s): {', '.join(invalid_panels)}. "
                     f"Must be one of: {', '.join(panel_names)}")

      # Create testing flags tuple (True = testing, False = production)
      lung_testing = tuple(p not in production_panels for p in panel_names)
      print(f"LUNG PLOT: Production panels: {', '.join(production_panels)}, "
            f"Testing panels: {', '.join(p for p, t in zip(panel_names, lung_testing) if t)}")
      print("=" * 60)

    plot_lung_dataset(testing=lung_testing, survival_type=args.lung_survival_type)

  print("=" * 60)
  print("Plot generation complete!")


if __name__ == "__main__":
  main()
