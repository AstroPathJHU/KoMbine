"""
Test the continuous distributions module, and generate the figures
for that section of the documentation.
"""

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import roc_picker.continuous_distributions

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

h = .6

NX = NY = 3

def X(t):
  """
  The X coordinates of the ROC curve, which comes from the non-responder CDF.
  """
  return (
    scipy.stats.norm.cdf(t, loc=-1, scale=h)
      + 2*scipy.stats.norm.cdf(t, loc=2, scale=h)
  ) / 3 * NX
def Y(t):
  """
  The Y coordinates of the ROC curve, which comes from the responder CDF.
  """
  return (
    scipy.stats.norm.cdf(t, loc=1, scale=h)
      + 2*scipy.stats.norm.cdf(t, loc=-2, scale=h)
  ) / 3 * NY
def Xdot(t):
  """
  The derivative of the X coordinates of the ROC curve.
  """
  return (
    scipy.stats.norm.pdf(t, loc=-1, scale=h)
      + 2*scipy.stats.norm.pdf(t, loc=2, scale=h)
  ) / 3 * NX
def Ydot(t):
  """
  The derivative of the Y coordinates of the ROC curve.
  """
  return (
    scipy.stats.norm.pdf(t, loc=1, scale=h)
      + 2*scipy.stats.norm.pdf(t, loc=-2, scale=h)
  ) / 3 * NY

t_plot = np.linspace(-10, 10, 1001)
dt_plot = t_plot[1] - t_plot[0]
NX = X(np.inf)
NY = Y(np.inf)
AUC = np.sum(Y(t_plot)/NY * Xdot(t_plot)/NX) * dt_plot

def run(   # pylint: disable=dangerous-default-value, too-many-locals
  target_AUC,
  verbose=True,
  prev_rocs={AUC: (t_plot, X, Y, None, None)},
):
  """
  Run the optimization for a given target AUC.

  Parameters
  ----------
  target_AUC : float
    The target AUC.
  verbose : bool
    Whether to print and plot the results.
  prev_rocs : dict
    The previous results, which are used to guess the initial parameters.
    This gets updated automatically if the optimization is successful.
  """
  AUC_for_guess = min(prev_rocs.keys(), key=lambda x: abs(x-target_AUC))
  t_for_guess, X_for_guess, Y_for_guess, Lambda_guess, prev_result = prev_rocs[AUC_for_guess]
  if AUC_for_guess == target_AUC and prev_result is not None:
    optimize_result = prev_result
    xy_guess = optimize_result.y
  else:
    xy_guess = roc_picker.continuous_distributions.xy_guess(
      X=X_for_guess, Y=Y_for_guess, t_guess=t_for_guess, AUC=target_AUC
    )

    if Lambda_guess is None:
      if target_AUC < AUC:
        Lambda_guess = 2
      else:
        Lambda_guess = -2

    optimize_result = roc_picker.continuous_distributions.optimize(
      X=X, Y=Y, Xdot=Xdot, Ydot=Ydot,
      AUC=target_AUC, Lambda_scaling=1, Lambda_guess=Lambda_guess,
      guess=xy_guess, t_guess=t_for_guess
    )

  t = optimize_result.x
  x, y = xy = optimize_result.y
  #xd, yd = optimize_result.yp
  Lambda, c1, c2 = optimize_result.p

  if verbose:
    plt.scatter(X(t_plot)/NX, Y(t_plot)/NY, label="X, Y")
    plt.scatter(xy_guess[0], xy_guess[1], label="guess")
    plt.scatter(xy[0], xy[1], label="optimized")
    print("=========================")
    print("should be 0:", x[0], y[0])
    print("should be 1:", x[-1], y[-1])
    print("should be equal:", target_AUC, 1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1])))
    print("should be equal:", c1, 2*NX - Lambda*target_AUC)
    print("should be equal:", c2, 2*NY + Lambda*(1-target_AUC))
    print("=========================")
    plt.legend()
    plt.show()

  if (
    optimize_result is not prev_result
    and optimize_result.success
    and np.isclose(
      target_AUC,
      1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1])),
      rtol=0, atol=1e-3,
    )
  ):
    prev_rocs[target_AUC] = t, x, y, Lambda, optimize_result

  return optimize_result

def make_continuous_distribution_plots(*, skip_aucs=(), show=False):
  """
  Generate the plots of the fitted parameters and negative log likelihood
  as a function of the target AUC.
  """
  results = {
    "target_aucs": [],
    "delta_aucs": [],
    "L": [],
    "c1": [],
    "c2": [],
    "NLL": []
  }
  linspaces = [
    [AUC] + [_ for _ in np.linspace(0, 1, 1001) if _ >= AUC],
    [AUC] + [_ for _ in np.linspace(1, 0, 1001) if _ <= AUC],
  ]
  for linspace in linspaces:
    last_failed = False
    for target_auc in linspace:
      if target_auc in skip_aucs:
        continue
      result = run(target_auc, verbose=False)
      x, y = result.y
      auc = 1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1]))
      delta_auc = auc - target_auc
      if not result.success or abs(delta_auc) > 1e-2:
        print("failed", target_auc)
        if last_failed:
          break
        last_failed = True
        continue
      last_failed = False
      results["target_aucs"].append(target_auc)
      results["delta_aucs"].append(delta_auc)
      results["L"].append(result.p[0])
      results["c1"].append(result.p[1])
      results["c2"].append(result.p[2])
      results["NLL"].append(result.NLL)

  plot_params(
    target_aucs=results["target_aucs"],
    L=results["L"],
    c1=results["c1"],
    c2=results["c2"],
    show=show
  )
  plot_scan(
    target_aucs=results["target_aucs"],
    NLL=results["NLL"],
    show=show
  )

  return results

def plot_params(target_aucs, L, c1, c2, show=False):
  """
  Generate the plot of the fitted parameters as a function of the target AUC.
  """
  plt.figure(figsize=(5,5))
  #plt.scatter(target_aucs, delta_aucs, label=r"$\Delta$AUC")
  plt.scatter(target_aucs, L, label=r"$\Lambda$")
  plt.scatter(target_aucs, c1, label="$c_1$")
  plt.scatter(target_aucs, c2, label="$c_2$")
  plt.ylim(-10, 10)
  plt.xlabel("AUC")
  plt.ylabel("Parameters")
  plt.legend()
  plt.savefig(docsfolder/"exampleparameters.pdf", bbox_inches="tight")
  if show:
    plt.show()
  plt.close()

def plot_scan(NLL, target_aucs, show=False):
  """
  Generate the plot of the scan of the negative log likelihood.
  """
  target_aucs = np.asarray(target_aucs)
  deltaNLL = np.asarray(NLL)
  deltaNLL -= np.nanmin(deltaNLL)
  plt.figure(figsize=(5,5))
  plt.scatter(target_aucs, 2*deltaNLL, label=r"$-2\Delta\ln{L}$")
  slc = np.isclose(deltaNLL, np.nanmin(deltaNLL))
  plt.scatter(target_aucs[slc], 2*deltaNLL[slc], label="best fit")
  xlow, xhigh = plt.xlim()
  plt.plot([xlow, xhigh], [1, 1], label="68% CL")
  plt.plot([xlow, xhigh], [3.84, 3.84], label="95% CL")
  plt.legend()
  plt.xlabel("AUC")
  plt.ylabel(r"$-2\Delta\ln{L}$")
  plt.savefig(docsfolder/"examplescan.pdf")
  if show:
    plt.show()
  plt.close()

def plots(show=False):
  """
  Generate the illustrations of the continuous distributions used for
  the example in the documentation.

  Parameters
  ----------
  show : bool
    Whether to show the plots.
  """
  plt.figure(figsize=(5, 5))
  plt.scatter(t_plot, Xdot(t_plot), label=r"$\dot{X}$")
  plt.scatter(t_plot, Ydot(t_plot), label=r"$\dot{Y}$")
  plt.xlabel("$t$")
  plt.ylabel(r"$\dot{X}$, $\dot{Y}$")
  plt.legend()
  plt.savefig(docsfolder/"exampleXdotYdot.pdf")
  if show:
    plt.show()
  plt.close()

  plt.figure(figsize=(5, 5))
  plt.scatter(t_plot, X(t_plot), label=r"$X$")
  plt.scatter(t_plot, Y(t_plot), label=r"$Y$")
  plt.xlabel("$t$")
  plt.ylabel(r"$X$, $Y$")
  plt.legend()
  plt.savefig(docsfolder/"exampleXY.pdf")
  if show:
    plt.show()
  plt.close()

  plt.figure(figsize=(5, 5))
  plt.scatter(X(t_plot)/NX, Y(t_plot)/NY)
  plt.xlabel("$X$")
  plt.ylabel("$Y$")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.savefig(docsfolder/"exampleroc.pdf")
  if show:
    plt.show()
  plt.close()

if __name__ == "__main__":
  make_continuous_distribution_plots()
  plots()
