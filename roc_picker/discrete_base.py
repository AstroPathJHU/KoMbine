"""
Base class for ROC curve optimization using discrete points.
This includes the discrete and delta functions methods.
The plotting code is implemented here.
"""

import abc, collections, matplotlib.pyplot as plt, numpy as np, scipy.optimize

class DiscreteROCBase(abc.ABC):
  """
  Base class for ROC curve optimization using discrete points.
  This includes the discrete and delta functions methods.
  The plotting code is implemented here.

  Parameters
  ----------
  responders: array-like
    The values of the observable for the responders.
  nonresponders: array-like
    The values of the observable for the non-responders.
  flip_sign: bool, optional
    If True, the sign of the observable is flipped.
  """
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.responders = responders
    self.nonresponders = nonresponders
    self.flip_sign = flip_sign

  @abc.abstractmethod
  def optimize(self, *, AUC=None):
    """
    Optimize the ROC curve, either unconditionally or for a given AUC.
    """

  def plot_roc(
    self, *,
    show=False,
    rocfilename=None,
    scanfilename=None,
    rocerrorsfilename=None,
    yupperlim=None,
    npoints=100
  ):
    """
    Plot the optimized ROC curve, the scan of the NLL,
    and the 68% and 95% CL bands.

    Parameters
    ----------
    show: bool or tuple of bool, optional
      Whether to show the plots.
      If a tuple, the first element is for the ROC curve,
      the second for the scan of the NLL,
      and the third for the 68% and 95% CL bands.
    rocfilename: os.PathLike, optional
      The filename to save the ROC curve plot.
    scanfilename: os.PathLike, optional
      The filename to save the scan of the NLL plot.
    rocerrorsfilename: os.PathLike, optional
      The filename to save the 68% and 95% CL bands plot.
    yupperlim: float, optional
      The upper limit for the y-axis of the scan of the NLL plot.
    npoints: int, optional
      The number of points to scan for the NLL.
    """
    if not isinstance(show, collections.abc.Sequence):
      show = [show, show, show]
    show_roc, show_scan, show_rocerrors = show

    target_aucs = []
    NLL = []

    sign = 1
    t = np.asarray(sorted(set(self.responders) | set(self.nonresponders) | {-np.inf, np.inf}))
    if self.flip_sign:
      sign = -1
      t = t[::-1]

    @np.vectorize
    def X(t):
      return sum(1 for n in self.nonresponders if n*sign < t*sign)
    @np.vectorize
    def Y(t):
      return sum(1 for r in self.responders if r*sign < t*sign)

    xx = X(t) / len(self.nonresponders)
    yy = Y(t) / len(self.responders)
    AUC = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))

    linspaces = [
      [AUC] + [_ for _ in np.linspace(0, 1, npoints+1) if _ >= AUC],
      [AUC] + [_ for _ in np.linspace(1, 0, npoints+1) if _ <= AUC],
    ]

    plt.figure(figsize=(5, 5))
    for linspace in linspaces:
      last_failed = False
      for target_auc in linspace:
        result = self.optimize(AUC=target_auc)
        xx = result.x
        yy = result.y
        if not result.success:
          if last_failed:
            break
          last_failed = True
          continue
        last_failed = False
        if target_auc == AUC and linspace is linspaces[0]:
          plt.scatter(xx, yy)
        target_aucs.append(target_auc)
        NLL.append(result.NLL)
        if yupperlim is not None and 2*(result.NLL - min(NLL)) > yupperlim:
          break
    plt.xlabel("X (Fraction of non-responders)")
    plt.ylabel("Y (Fraction of responders)")
    if rocfilename is not None:
      plt.savefig(rocfilename)
    if show_roc:
      plt.show()
    plt.close()

    target_aucs = np.asarray(target_aucs)
    NLL = np.asarray(NLL)

    sortslice = np.argsort(target_aucs)
    target_aucs = target_aucs[sortslice]
    NLL = NLL[sortslice]

    deltaNLL = NLL.copy()
    deltaNLL -= np.nanmin(deltaNLL)
    plt.figure(figsize=(5, 5))
    plt.scatter(target_aucs, 2*deltaNLL, label=r"$-2\Delta\ln{L}$")
    slc = np.isclose(deltaNLL, np.nanmin(deltaNLL))
    plt.scatter(target_aucs[slc], 2*deltaNLL[slc], label="best fit")
    plt.xlabel("AUC")
    plt.ylabel(r"$-2\Delta\ln{L}$")
    plt.xlim(0, 1)
    plt.ylim(0, yupperlim)
    xlow, xhigh = plt.xlim()
    plt.plot([xlow, xhigh], [1, 1], label="68% CL")
    plt.plot([xlow, xhigh], [3.84, 3.84], label="95% CL")
    plt.legend()
    if scanfilename is not None:
      plt.savefig(scanfilename)
    if show_scan:
      plt.show()
    plt.close()

    #find the 68% and 95% bands
    error_band_results = {}
    for (nsigma, d2NLLcut) in ((1, 1), (2, 3.84)):
      withinsigma = 2 * deltaNLL < d2NLLcut

      from_below_to_above = withinsigma[:-1] & ~withinsigma[1:]
      from_below_to_above_left = np.concatenate((from_below_to_above, [False]))
      from_below_to_above_right = np.concatenate(([False], from_below_to_above))
      np.testing.assert_equal(sum(from_below_to_above_left), 1)
      np.testing.assert_equal(sum(from_below_to_above_right), 1)

      from_above_to_below = ~withinsigma[:-1] & withinsigma[1:]
      from_above_to_below_left = np.concatenate((from_above_to_below, [False]))
      from_above_to_below_right = np.concatenate(([False], from_above_to_below))
      np.testing.assert_equal(sum(from_above_to_below_left), 1)
      np.testing.assert_equal(sum(from_above_to_below_right), 1)

      def tosolve(target_auc, d2NLLcut=d2NLLcut):
        result = self.optimize(AUC=target_auc)
        return 2 * (result.NLL - np.nanmin(NLL)) - d2NLLcut

      left_auc_left_bracket = target_aucs[from_below_to_above_left].item()
      left_auc_right_bracket = target_aucs[from_below_to_above_right].item()
      right_auc_left_bracket = target_aucs[from_above_to_below_left].item()
      right_auc_right_bracket = target_aucs[from_above_to_below_right].item()

      left_auc = scipy.optimize.root_scalar(
        tosolve,
        bracket=[left_auc_left_bracket, left_auc_right_bracket]
      )
      assert left_auc.converged, left_auc
      left_result = self.optimize(AUC=left_auc.root)
      right_auc = scipy.optimize.root_scalar(
        tosolve,
        bracket=[right_auc_left_bracket, right_auc_right_bracket]
      )
      assert right_auc.converged, right_auc
      right_result = self.optimize(AUC=right_auc.root)

      error_band_results[nsigma] = left_result, right_result

    nominal = self.optimize(AUC=AUC)
    m68, p68 = error_band_results[1]
    m95, p95 = error_band_results[2]

    x_n = nominal.x
    x_m68 = m68.x
    x_p68 = p68.x
    x_m95 = m95.x
    x_p95 = p95.x
    y_n = nominal.y
    y_m68 = m68.y
    y_p68 = p68.y
    y_m95 = m95.y
    y_p95 = p95.y

    xx_pm68 = []
    yy_p68 = []
    yy_m68 = []
    for x in sorted(set(x_m68) | set(x_p68)):
      addyy_p68 = list(y_p68[x_p68 == x])
      addyy_m68 = list(y_m68[x_m68 == x])
      xx_pm68 += [x] * max(len(addyy_p68), len(addyy_m68))
      if not addyy_p68:
        addyy_p68 = [np.interp(x, x_p68, y_p68)] * len(addyy_m68)
      elif not addyy_m68:
        addyy_m68 = [np.interp(x, x_m68, y_m68)] * len(addyy_p68)
      np.testing.assert_equal(len(addyy_p68), len(addyy_m68))
      yy_p68 += addyy_p68
      yy_m68 += addyy_m68

    xx_pm95 = []
    yy_p95 = []
    yy_m95 = []
    for x in sorted(set(x_m95) | set(x_p95)):
      addyy_p95 = list(y_p95[x_p95 == x])
      addyy_m95 = list(y_m95[x_m95 == x])
      xx_pm95 += [x] * max(len(addyy_p95), len(addyy_m95))
      if not addyy_p95:
        addyy_p95 = [np.interp(x, x_p95, y_p95)] * len(addyy_m95)
      elif not addyy_m95:
        addyy_m95 = [np.interp(x, x_m95, y_m95)] * len(addyy_p95)
      np.testing.assert_equal(len(addyy_p95), len(addyy_m95))
      yy_p95 += addyy_p95
      yy_m95 += addyy_m95

    #xx_pm68 = np.array(sorted(set(x_m68) | set(x_p68)))
    #yy_p68 = np.interp(xx_pm68, x_p68, y_p68)
    #yy_m68 = np.interp(xx_pm68, x_m68, y_m68)
    #xx_pm95 = np.array(sorted(set(x_m95) | set(x_p95)))
    #yy_p95 = np.interp(xx_pm95, x_p95, y_p95)
    #yy_m95 = np.interp(xx_pm95, x_m95, y_m95)

    plt.figure(figsize=(5, 5))
    colornominal="blue"
    color68="dodgerblue"
    color95="skyblue"
    #plt.plot(x_m95, y_m95, label=r"$-2\sigma$")
    #plt.plot(x_m68, y_m68, label=r"$-1\sigma$")
    plt.plot(
      x_n, y_n,
      label=f"nominal\nAUC={nominal.AUC:.2f}",
      color=colornominal
    )
    #plt.plot(x_p68, y_p68, label=r"$+1\sigma$")
    #plt.plot(x_p95, y_p95, label=r"$+2\sigma$")
    lowAUC_68, highAUC_68 = sorted((m68.AUC, p68.AUC))
    lowAUC_95, highAUC_95 = sorted((m95.AUC, p95.AUC))
    plt.fill_between(
      xx_pm68, yy_m68, yy_p68,
      color=color68, alpha=0.5,
      label=f"68% CL\nAUC$\\in$({lowAUC_68:.2f}, {highAUC_68:.2f})",
    )
    plt.fill_between(
      xx_pm95, yy_m95, yy_p95,
      color=color95, alpha=0.5,
      label=f"95% CL\nAUC$\\in$({lowAUC_95:.2f}, {highAUC_95:.2f})",
    )
    plt.legend()

    plt.xlabel("X (Fraction of non-responders)")
    plt.ylabel("Y (Fraction of responders)")

    if rocerrorsfilename is not None:
      plt.savefig(rocerrorsfilename)
    if show_rocerrors:
      plt.show()
    plt.close()

    return {
      "nominal": nominal,
      "m68": m68,
      "p68": p68,
      "m95": m95,
      "p95": p95,
    }
