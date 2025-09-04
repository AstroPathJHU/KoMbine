"""
Compare p-values from custom MINLP method (binomial-only) and conventional log-rank test
by generating many Monte Carlo trials.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from roc_picker.kaplan_meier_p_value_MINLP import KaplanMeierPatientNLL, MINLPforKMPValue

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
        time = rng.integers(1, 11) if time_is_integer else rng.uniform(1, 11)
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

def plot_pvalue_comparison(
  pvalues: np.ndarray,
  title: str = "Comparison of p-value methods",
  saveas: os.PathLike | str | None = None,
  show: bool | None = None,
) -> float:
  """
  Make scatter plot and compute correlation coefficient.
  """
  if show is None:
    show = saveas is None

  minlp_vals, logrank_vals = pvalues[:, 0], pvalues[:, 1]
  r = np.corrcoef(minlp_vals, logrank_vals)[0, 1]

  plt.figure(figsize=(6, 6))
  plt.scatter(logrank_vals, minlp_vals, alpha=0.6)
  plt.xlabel("Conventional log-rank p-value")
  plt.ylabel("MINLP (hypergeometric penalty only) p-value")
  plt.title(f"{title} (r={r:.3f})")
  plt.plot([0, 1], [0, 1], "r--")
  plt.grid(True)
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
  r = plot_pvalue_comparison(pvalues, saveas=args.save_as, show=False, title=title)
  print(f"Correlation coefficient (n_patients={n_patients}): r = {r:.3f}\n")

if __name__ == "__main__":
  main()
