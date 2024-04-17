import matplotlib.pyplot as plt, numpy as np, pathlib
import roc_picker.discrete

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responders = np.array([-2, -2, 1])
nonresponders = -responders

def plot_params(responders, nonresponders, *, skip_aucs=[], show=False, yupperlim=None):
  target_aucs = []
  aucs = []
  delta_aucs = []
  NLL = []

  optimizer = roc_picker.discrete.Discrete(responders=responders, nonresponders=nonresponders)

  t = np.asarray(sorted(set(responders) | set(nonresponders) | {-np.inf, np.inf}))

  @np.vectorize
  def X(t): return sum(1 for n in nonresponders if n < t)
  @np.vectorize
  def Y(t): return sum(1 for r in responders if r < t)

  xx = X(t) / len(nonresponders)
  yy = Y(t) / len(responders)
  AUC = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))

  linspaces = [
    [AUC] + [_ for _ in np.linspace(0, 1, 1001) if _ >= AUC],
    [AUC] + [_ for _ in np.linspace(1, 0, 1001) if _ <= AUC],
  ]

  plt.figure(figsize=(5, 5))
  for linspace in linspaces:
    last_failed = False
    for target_auc in linspace:
      if target_auc in skip_aucs: continue
      result = optimizer.optimize(AUC=target_auc)
      xx = result.x
      yy = result.y
      auc = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))
      delta_auc = auc - target_auc
      if abs(delta_auc) > 1e-4 or not result.success:
        if last_failed:
          break
        else:
          last_failed = True
          continue
      last_failed = False
      if target_auc == AUC and linspace is linspaces[0]:
        plt.scatter(xx, yy)
      target_aucs.append(target_auc)
      delta_aucs.append(delta_auc)
      NLL.append(result.NLL)
      if yupperlim is not None and 2*(result.NLL - min(NLL)) > yupperlim: break
  plt.xlabel("X (Fraction of non-responders)")
  plt.ylabel("Y (Fraction of responders)")
  plt.savefig(docsfolder/"discrete_exampleroc.pdf")
  if show:
    plt.show()
  plt.close()

  target_aucs = np.asarray(target_aucs)
  print(NLL)
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
  plt.ylim(top=yupperlim)
  plt.savefig(docsfolder/"discrete_scan.pdf")
  if show:
    plt.show()
  plt.close()

  #find the 68% and 95% bands
  for (nsigma, d2NLLcut) in ((1, 1), (2, 3.84)):
    withinsigma = 2 * deltaNLL < d2NLLcut

    from_above_to_below = withinsigma[:-1] & ~withinsigma[1:]
    from_above_to_below_left = np.concatenate((from_above_to_below, [False]))
    from_above_to_below_right = np.concatenate(([False], from_above_to_below))
    np.testing.assert_equal(sum(from_above_to_below_left), 1)
    np.testing.assert_equal(sum(from_above_to_below_right), 1)

    from_below_to_above = ~withinsigma[:-1] & withinsigma[1:]
    from_below_to_above_left = np.concatenate((from_below_to_above, [False]))
    from_below_to_above_right = np.concatenate(([False], from_below_to_above))
    np.testing.assert_equal(sum(from_below_to_above_left), 1)
    np.testing.assert_equal(sum(from_below_to_above_right), 1)

    def tosolve(target_auc):
      result = optimizer.optimize(AUC=target_auc)
      return 2 * (result.NLL - np.nanmin(NLL)) - d2NLLcut

    left_auc = scipy.optimize.root_scalar(tosolve, bracket=[from_above_to_below_left, from_above_to_below_right])
    left_result = optimizer.optimize(AUC=left_auc)
    right_auc = scipy.optimize.root_scalar(tosolve, bracket=[from_below_to_above_left, from_below_to_above_right])
    left_result = optimizer.optimize(AUC=left_auc)

def main():
  plot_params(responders=responders, nonresponders=nonresponders, yupperlim=20)
