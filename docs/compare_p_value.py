"""
Compare p-values from custom MINLP method (binomial-only) and conventional log-rank test
by generating many Monte Carlo trials.
"""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from roc_picker.kaplan_meier_p_value_MINLP import KaplanMeierPatientNLL, MINLPforKMPValue

@dataclass
class PlotConfig:
  """Configuration for plot styling."""
  figsize: tuple[float, float] = (6, 6)
  legend_fontsize: float = 12
  title_fontsize: float = 14
  label_fontsize: float = 12
  tick_fontsize: float = 10

def simulate_pvalues( #pylint: disable=too-many-locals
  n_patients: int,
  n_trials: int,
  seed: int | None = None,
  time_is_integer: bool = True,
  verbose: bool = False,
) -> np.ndarray:
  """
  Generate synthetic patients and compare p-values from custom MINLP method
  (binomial-only) and conventional log-rank test.

  Returns
  -------
  np.ndarray of shape (n_trials, 2)
    Column 0: p-values from MINLP
    Column 1: p-values from log-rank
  """
  rng = np.random.default_rng(seed)
  results = np.zeros((n_trials, 2))

  for trial in range(n_trials):
    if verbose:
      print("Trial", trial+1, "of", n_trials)
    patients = []
    any_patients = {1: False, 2: False}
    while not all(any_patients.values()):
      any_patients = {1: False, 2: False}
      for _ in range(n_patients):
        time = float(rng.integers(1, 11)) if time_is_integer else rng.uniform(1, 11)
        censored = rng.random() < 0.1
        parameter = 1 if rng.random() < 0.5 else 2
        any_patients[parameter] = True
        patients.append(
          KaplanMeierPatientNLL.from_fixed_observable(
            time=time,
            censored=censored,
            observable=parameter,
          )
        )

    # custom MINLP p-value
    minlp_breslow = MINLPforKMPValue(patients, parameter_threshold=1.5, tie_handling="breslow")
    pval_breslow, _, _ = minlp_breslow.solve_and_pvalue(binomial_only=True)

    # log-rank p-value
    pval_logrank = minlp_breslow.survival_curves_pvalue_logrank()

    results[trial, 0] = pval_breslow
    results[trial, 1] = pval_logrank

  return results

def plot_pvalue_comparison( #pylint: disable=too-many-arguments
  pvalues: np.ndarray,
  title: str = "Comparison of p-value methods",
  *,
  saveas: os.PathLike | str | None = None,
  show: bool | None = None,
  config: PlotConfig | None = None,
  inlay_upper_limit: float = 0.1,
) -> float:
  """
  Make scatter plot and compute correlation coefficient.
  
  Parameters
  ----------
  pvalues : np.ndarray
    Array of shape (n_trials, 2) with p-values from MINLP and log-rank methods
  title : str
    Plot title
  saveas : os.PathLike | str | None
    Filename to save plot
  show : bool | None  
    Whether to show plot
  config : PlotConfig | None
    Plot styling configuration
  inlay_upper_limit : float
    Upper limit for the zoomed inlay (default 0.1)
    
  Returns
  -------
  float
    Correlation coefficient
  """
  if show is None:
    show = saveas is None
  if config is None:
    config = PlotConfig()

  minlp_vals, logrank_vals = pvalues[:, 0], pvalues[:, 1]
  r = np.corrcoef(minlp_vals, logrank_vals)[0, 1]

  _, ax = plt.subplots(figsize=config.figsize)
  ax.scatter(logrank_vals, minlp_vals, alpha=0.6, label="Data points")
  ax.plot([0, 1], [0, 1], "r--", label="y = x")

  ax.set_xlabel("Conventional log-rank p-value", fontsize=config.label_fontsize)
  ax.set_ylabel("MINLP (Cox penalty only) p-value", fontsize=config.label_fontsize)
  ax.set_title(f"{title} (r={r:.3f})", fontsize=config.title_fontsize)

  # Set limits to [0,1] and ensure square aspect ratio
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_aspect('equal', adjustable='box')

  # Configure tick labels
  ax.tick_params(axis='both', which='major', labelsize=config.tick_fontsize)

  # Add legend
  ax.legend(fontsize=config.legend_fontsize)

  ax.grid(True)

  # Add zoomed inlay
  # Create inset axes using Axes.inset_axes method
  # [left, bottom, width, height] in axes coordinates
  inlay_ax = ax.inset_axes((0.57, 0.05, 0.4, 0.4))

  # Plot the same data in the inlay but with zoomed limits
  inlay_ax.scatter(logrank_vals, minlp_vals, alpha=0.6, s=10)  # Smaller points for inlay
  inlay_ax.plot([0, inlay_upper_limit], [0, inlay_upper_limit], "r--", linewidth=0.8)

  # Set zoomed limits
  inlay_ax.set_xlim(0, inlay_upper_limit)
  inlay_ax.set_ylim(0, inlay_upper_limit)
  inlay_ax.set_aspect('equal', adjustable='box')
  inlay_ticks = np.linspace(0, inlay_upper_limit, 6) # Generates 6 ticks from 0 to the upper limit
  inlay_ax.set_xticks(inlay_ticks)
  inlay_ax.set_yticks(inlay_ticks)

  # Style the inlay
  inlay_ax.tick_params(axis='both', which='major', labelsize=config.tick_fontsize * 0.8)
  inlay_ax.grid(True, alpha=0.5)

  # Add border to make inlay stand out
  for spine in inlay_ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

  if saveas is not None:
    plt.savefig(saveas)
  if show:
    plt.show()
  plt.close()

  return r

def main(args=None):
  """
  Run simulations and plot results.
  """
  # pylint: disable=C0301
  p = argparse.ArgumentParser(description=main.__doc__)
  p.add_argument("--n-trials", type=int, default=100, help="Number of trials to simulate")
  p.add_argument("--seed", type=int, default=123456, help="Random seed")
  p.add_argument("--n-patients", type=int, required=True, help="Number of patients per trial")
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("--allow-ties", action="store_true", help="Allow ties in patient event times")
  g.add_argument("--no-ties", action="store_false", dest="allow_ties", help="Do not allow ties in patient event times")
  p.add_argument("--save-as", type=str, default=None, help="Filename to save plot (default: do not save)")

  # Styling arguments (consistent with kombine/kombine_twogroups)
  p.add_argument("--figsize", nargs=2, type=float, default=[6, 6], help="Figure size in inches")
  p.add_argument("--legend-fontsize", type=float, default=12, help="Font size for legend text")
  p.add_argument("--title-fontsize", type=float, default=14, help="Font size for the plot title")
  p.add_argument("--label-fontsize", type=float, default=12, help="Font size for axis labels")
  p.add_argument("--tick-fontsize", type=float, default=10, help="Font size for the tick labels")
  p.add_argument("--inlay-upper-limit", type=float, default=0.1, help="Upper limit for the zoomed inlay (default: 0.1)")
  # pylint: enable=C0301

  args = p.parse_args(args=args)
  n_trials = args.n_trials
  seed = args.seed
  n_patients = args.n_patients
  print(f"Simulating {n_trials} trials with {n_patients} patients each...")
  pvalues = simulate_pvalues(
    n_patients=n_patients,
    n_trials=n_trials,
    seed=seed,
    time_is_integer=args.allow_ties,
  )
  title_suffix = "with ties" if args.allow_ties else "without ties"
  title = f"$p$ value comparison for {n_patients} patients ({title_suffix})"

  # Create plot configuration
  config = PlotConfig(
    figsize=tuple(args.figsize),
    legend_fontsize=args.legend_fontsize,
    title_fontsize=args.title_fontsize,
    label_fontsize=args.label_fontsize,
    tick_fontsize=args.tick_fontsize,
  )

  r = plot_pvalue_comparison(
    pvalues,
    saveas=args.save_as,
    show=False,
    title=title,
    config=config,
    inlay_upper_limit=args.inlay_upper_limit,
  )
  print(f"Correlation coefficient (n_patients={n_patients}): r = {r:.3f}\n")

if __name__ == "__main__":
  main()
