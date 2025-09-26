"""
Command line interface functions for KoMbine.
"""

import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from .datacard import Datacard
from .kaplan_meier_likelihood import KaplanMeierPlotConfig
from .utilities import LOG_ZERO_EPSILON_DEFAULT


def _make_common_parser(description: str) -> argparse.ArgumentParser:
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("datacard", type=pathlib.Path, help="Path to the datacard file.")
  parser.add_argument("output_file", type=pathlib.Path, help="Path to the output file for the plot.")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--include-binomial-only", action="store_true", help="Include error bands for the binomial error alone.")
  group.add_argument("--include-patient-wise-only", action="store_true", help="Include error bands for the patient-wise error alone.")
  parser.add_argument("--include-exponential-greenwood", action="store_true", dest="include_exponential_greenwood", help="Include the binomial-only exponential Greenwood error band in the plot.")
  parser.add_argument("--exclude-full-nll", action="store_false", dest="include_full_NLL", default=True, help="Exclude the full NLL from the plot.")
  parser.add_argument("--exclude-nominal", action="store_false", dest="include_nominal", default=True, help="Exclude the nominal line from the plot.")
  parser.add_argument("--include-median-survival", action="store_true", dest="include_median_survival", help="Include the median survival line in the plot.")
  parser.add_argument("--print-progress", action="store_true", dest="print_progress", help="Print progress messages during the computation.")
  parser.add_argument("--endpoint-epsilon", type=float, dest="endpoint_epsilon", default=1e-6, help="Endpoint epsilon for the likelihood calculation.")
  parser.add_argument("--log-zero-epsilon", type=float, dest="log_zero_epsilon", default=LOG_ZERO_EPSILON_DEFAULT, help="Log zero epsilon for the likelihood calculation.")
  parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), help="Figure size in inches.", default=KaplanMeierPlotConfig.figsize)
  parser.add_argument("--no-tight-layout", action="store_false", help="Do not use tight layout for the plot.", default=True, dest="tight_layout")
  parser.add_argument("--legend-fontsize", type=float, help="Font size for legend text.", default=KaplanMeierPlotConfig.legend_fontsize)
  parser.add_argument("--label-fontsize", type=float, help="Font size for axis labels.", default=KaplanMeierPlotConfig.label_fontsize)
  parser.add_argument("--title-fontsize", type=float, help="Font size for the plot title.", default=KaplanMeierPlotConfig.title_fontsize)
  parser.add_argument("--tick-fontsize", type=float, help="Font size for the tick labels.", default=KaplanMeierPlotConfig.tick_fontsize)
  g = parser.add_mutually_exclusive_group()
  g.add_argument("--legend-loc", type=str, help="Location of the legend in the plot.", default=KaplanMeierPlotConfig.legend_loc)
  g.add_argument("--legend-saveas", type=pathlib.Path, help="Path to save the legend separately. If provided, the legend will be left off the main plot.", default=None)
  parser.add_argument("--title", type=str, help="Title for the plot.", default=KaplanMeierPlotConfig.title)
  parser.add_argument("--xlabel", type=str, help="Label for the x-axis.", default=KaplanMeierPlotConfig.xlabel)
  parser.add_argument("--ylabel", type=str, help="Label for the y-axis.", default=KaplanMeierPlotConfig.ylabel)
  parser.add_argument("--patient-wise-only-suffix", type=str, help="Suffix to add to the patient-wise-only label in the legend.", default=KaplanMeierPlotConfig.patient_wise_only_suffix)
  parser.add_argument("--binomial-only-suffix", type=str, help="Suffix to add to the binomial-only label in the legend.", default=KaplanMeierPlotConfig.binomial_only_suffix)
  parser.add_argument("--full-nll-suffix", type=str, help="Suffix to add to the full NLL label in the legend.", default=KaplanMeierPlotConfig.full_NLL_suffix)
  parser.add_argument("--exponential-greenwood-suffix", type=str, help="Suffix to add to the exponential Greenwood label in the legend.", default=KaplanMeierPlotConfig.exponential_greenwood_suffix)
  # pylint: enable=line-too-long
  return parser


def _validate_plot_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
  if (
    not args.include_full_NLL
    and not args.include_binomial_only
    and not args.include_patient_wise_only
  ):
    parser.error(
      "If --exclude-full-nll is set, at least one of "
      "--include-binomial-only or --include-patient-wise-only must be set."
    )


def _extract_common_plot_config_args(args: argparse.Namespace) -> dict:
  return {
    "include_patient_wise_only": args.__dict__.pop("include_patient_wise_only"),
    "include_binomial_only": args.__dict__.pop("include_binomial_only"),
    "include_exponential_greenwood": args.__dict__.pop("include_exponential_greenwood"),
    "include_full_NLL": args.__dict__.pop("include_full_NLL"),
    "include_nominal": args.__dict__.pop("include_nominal"),
    "include_median_survival": args.__dict__.pop("include_median_survival"),
    "print_progress": args.__dict__.pop("print_progress"),
    "figsize": args.__dict__.pop("figsize"),
    "tight_layout": args.__dict__.pop("tight_layout"),
    "legend_fontsize": args.__dict__.pop("legend_fontsize"),
    "label_fontsize": args.__dict__.pop("label_fontsize"),
    "title_fontsize": args.__dict__.pop("title_fontsize"),
    "tick_fontsize": args.__dict__.pop("tick_fontsize"),
    "title": args.__dict__.pop("title"),
    "xlabel": args.__dict__.pop("xlabel"),
    "ylabel": args.__dict__.pop("ylabel"),
    "legend_loc": args.__dict__.pop("legend_loc"),
    "patient_wise_only_suffix": args.__dict__.pop("patient_wise_only_suffix"),
    "binomial_only_suffix": args.__dict__.pop("binomial_only_suffix"),
    "full_NLL_suffix": args.__dict__.pop("full_nll_suffix"),
    "exponential_greenwood_suffix": args.__dict__.pop("exponential_greenwood_suffix"),
    "show_grid": True,
  }


def plot_km_likelihood():
  """
  Run Kaplan-Meier likelihood method from a datacard.
  """
  # pylint: disable=line-too-long
  parser = _make_common_parser("Run Kaplan-Meier likelihood method from a datacard.")
  parser.add_argument("--parameter-min", type=float, dest="parameter_min", default=-np.inf)
  parser.add_argument("--parameter-max", type=float, dest="parameter_max", default=np.inf)
  parser.add_argument("--dont-collapse-consecutive-deaths", action="store_true", dest="dont_collapse_consecutive_deaths", help="Disable collapsing of consecutive death times with no intervening censoring (slower but may be more accurate)")
  # pylint: enable=line-too-long
  args = parser.parse_args()
  _validate_plot_args(args, parser)

  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  kml = datacard.km_likelihood(
    parameter_min=args.__dict__.pop("parameter_min"),
    parameter_max=args.__dict__.pop("parameter_max"),
    endpoint_epsilon=args.__dict__.pop("endpoint_epsilon"),
    log_zero_epsilon=args.__dict__.pop("log_zero_epsilon"),
    collapse_consecutive_deaths=not args.__dict__.pop("dont_collapse_consecutive_deaths"),
  )

  plot_config = KaplanMeierPlotConfig(
    **_extract_common_plot_config_args(args),
    saveas=args.__dict__.pop("output_file"),
    legend_saveas=args.__dict__.pop("legend_saveas"),
  )

  kml.plot(config=plot_config)

  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")


def plot_km_likelihood_two_groups(): # pylint: disable=too-many-locals, too-many-statements
  """
  Run Kaplan-Meier likelihood method from a datacard, and plot Kaplan-Meier
  curves for two groups separated into high and low values of the parameter.
  """
  # pylint: disable=line-too-long
  parser = _make_common_parser("Run Kaplan-Meier likelihood method from a datacard, and plot Kaplan-Meier curves for two groups separated into high and low values of the parameter.")
  parser.add_argument("--parameter-threshold", type=float, dest="parameter_threshold", required=True, help="The parameter threshold for separating high and low groups.")
  parser.add_argument("--parameter-min", type=float, dest="parameter_min", default=-np.inf, help="The minimum parameter value for the low group.")
  parser.add_argument("--parameter-max", type=float, dest="parameter_max", default=np.inf, help="The maximum parameter value for the high group.")
  parser.add_argument("--exclude-p-value", action="store_false", dest="include_p_value", default=True, help="Exclude p value calculation and display from the plot.")
  parser.add_argument("--include-logrank-pvalue", action="store_true", dest="include_logrank_pvalue", help="Include conventional logrank p value for comparison with likelihood method.")
  parser.add_argument("--p-value-tie-handling", type=str, dest="p_value_tie_handling", choices=["breslow"], default="breslow", help="Method for handling ties in p value calculation: currently only option is 'breslow' (Breslow approximation).")
  parser.add_argument("--pvalue-fontsize", type=float, dest="pvalue_fontsize", default=KaplanMeierPlotConfig.pvalue_fontsize, help="Font size for p value text.")
  parser.add_argument("--pvalue-format", type=str, dest="pvalue_format", default=KaplanMeierPlotConfig.pvalue_format, help="Format string for p value display (e.g., '.3g', '.2f').")
  parser.add_argument("--dont-collapse-consecutive-deaths", action="store_true", dest="dont_collapse_consecutive_deaths", help="Disable collapsing of consecutive death times with no intervening censoring (slower but may be more accurate)")
  # pylint: enable=line-too-long
  args = parser.parse_args()
  _validate_plot_args(args, parser)

  datacard = Datacard.parse_datacard(args.__dict__.pop("datacard"))
  parameter_min = args.__dict__.pop("parameter_min")
  threshold = args.__dict__.pop("parameter_threshold")
  parameter_max = args.__dict__.pop("parameter_max")
  log_zero_epsilon = args.__dict__.pop("log_zero_epsilon")
  endpoint_epsilon = args.__dict__.pop("endpoint_epsilon")
  collapse_consecutive_deaths = not args.__dict__.pop("dont_collapse_consecutive_deaths")
  include_p_value = args.__dict__.pop("include_p_value")
  include_logrank_pvalue = args.__dict__.pop("include_logrank_pvalue")
  p_value_tie_handling = args.__dict__.pop("p_value_tie_handling")
  pvalue_fontsize = args.__dict__.pop("pvalue_fontsize")
  pvalue_format = args.__dict__.pop("pvalue_format")

  kml_low = datacard.km_likelihood(
    parameter_min=parameter_min,
    parameter_max=threshold,
    log_zero_epsilon=log_zero_epsilon,
    endpoint_epsilon=endpoint_epsilon,
    collapse_consecutive_deaths=collapse_consecutive_deaths,
  )
  kml_high = datacard.km_likelihood(
    parameter_min=threshold,
    parameter_max=parameter_max,
    log_zero_epsilon=log_zero_epsilon,
    endpoint_epsilon=endpoint_epsilon,
    collapse_consecutive_deaths=collapse_consecutive_deaths,
  )

  common_plot_kwargs = _extract_common_plot_config_args(args)

  config_high = KaplanMeierPlotConfig(
    **common_plot_kwargs,
    create_figure=True,
    close_figure=False,
    show=False,
    saveas=None,
    legend_saveas=os.devnull,
    best_label=f"High (n={len(kml_high.nominalkm.patients)})",
    best_color="blue",
    CL_colors=["dodgerblue", "skyblue"],
    pvalue_fontsize=pvalue_fontsize,
    pvalue_format=pvalue_format,
  )
  kml_high.plot(config=config_high)

  config_low = KaplanMeierPlotConfig(
    **common_plot_kwargs,
    create_figure=False,
    close_figure=False,
    show=False,
    saveas=None,
    legend_saveas=args.__dict__.pop("legend_saveas"),
    best_label=f"Low (n={len(kml_low.nominalkm.patients)})",
    best_color="red",
    CL_colors=["orangered", "lightcoral"],
    pvalue_fontsize=pvalue_fontsize,
    pvalue_format=pvalue_format,
  )
  kml_low.plot(config=config_low)

  if include_p_value:
    # Calculate and display p values based on options
    p_value_texts = []

    if (common_plot_kwargs["include_full_NLL"]
        or common_plot_kwargs["include_binomial_only"]
        or common_plot_kwargs["include_patient_wise_only"]):
      p_value_minlp = datacard.km_p_value(
        parameter_min=parameter_min,
        parameter_threshold=threshold,
        parameter_max=parameter_max,
        log_zero_epsilon=log_zero_epsilon,
        tie_handling=p_value_tie_handling,
      )

      if common_plot_kwargs["include_full_NLL"]:
        p_value, *_ = p_value_minlp.solve_and_pvalue()
        p_value_texts.append(f"$p$ = {p_value:{pvalue_format}}")

      if common_plot_kwargs["include_binomial_only"]:
        p_value_binomial, *_ = p_value_minlp.solve_and_pvalue(cox_only=True)
        p_value_texts.append(f"$p$ (Cox only) = {p_value_binomial:{pvalue_format}}")

      #Patient-wise p value is not implemented
      #if common_plot_kwargs["include_patient_wise_only"]:
      #  p_value_patient_wise, *_ = p_value_minlp.solve_and_pvalue(patient_wise_only=True)
      #  p_value_texts.append(f"$p$ (patient-wise only) = {p_value_patient_wise:{pvalue_format}}")

    # Add logrank p value if requested
    if include_logrank_pvalue:
      p_value_logrank = datacard.km_p_value_logrank(
        parameter_threshold=threshold,
        parameter_min=parameter_min,
        parameter_max=parameter_max,
        cox_only=True,
      )
      p_value_texts.append(f"$p$ (logrank) = {p_value_logrank:{pvalue_format}}")

    # Display p value text(s) on the plot
    if p_value_texts:
      ax = plt.gca()
      # Position multiple p values vertically, starting from top-right
      for i, text in enumerate(p_value_texts):
        y_pos = 0.95 - (i * 0.05)  # Each line is 5% down from the previous
        ax.text(
          0.95, y_pos, text,
          ha="right", va="top",
          transform=ax.transAxes,
          fontsize=pvalue_fontsize,
        )

  plt.savefig(args.__dict__.pop("output_file"))

  if args.__dict__:
    raise ValueError(f"Unused arguments: {args.__dict__}")
