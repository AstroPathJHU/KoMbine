import matplotlib.pyplot as plt, numpy as np, pathlib, scipy.optimize
import roc_picker.discrete

here = pathlib.Path(__file__).parent
docsfolder = here.parent/"docs"

responders = np.array([-2, -2, 1])
nonresponders = -responders

def plot_params(responders, nonresponders, *, skip_aucs=[], show=False, yupperlim=None):
  target_aucs = []
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
    [AUC] + [_ for _ in np.linspace(1, 0, 101) if _ <= AUC],
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
      NLL.append(result.NLL)
      if yupperlim is not None and 2*(result.NLL - min(NLL)) > yupperlim: break
  plt.xlabel("X (Fraction of non-responders)")
  plt.ylabel("Y (Fraction of responders)")
  plt.savefig(docsfolder/"discrete_exampleroc.pdf")
  if show:
    plt.show()
  plt.close()

  target_aucs = np.asarray(target_aucs)
  NLL = np.asarray(NLL)

  sortslice = np.argsort(target_aucs)
  target_aucs = target_aucs[sortslice]
  NLL = NLL[sortslice]

  deltaNLL = NLL.copy()
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

    def tosolve(target_auc):
      result = optimizer.optimize(AUC=target_auc)
      return 2 * (result.NLL - np.nanmin(NLL)) - d2NLLcut

    left_auc_left_bracket = target_aucs[from_below_to_above_left].item()
    left_auc_right_bracket = target_aucs[from_below_to_above_right].item()
    right_auc_left_bracket = target_aucs[from_above_to_below_left].item()
    right_auc_right_bracket = target_aucs[from_above_to_below_right].item()

    left_auc = scipy.optimize.root_scalar(tosolve, bracket=[left_auc_left_bracket, left_auc_right_bracket])
    assert left_auc.converged, left_auc
    left_result = optimizer.optimize(AUC=left_auc.root)
    right_auc = scipy.optimize.root_scalar(tosolve, bracket=[right_auc_left_bracket, right_auc_right_bracket])
    assert right_auc.converged, right_auc
    right_result = optimizer.optimize(AUC=right_auc.root)

    error_band_results[nsigma] = left_result, right_result

  nominal = optimizer.optimize(AUC=AUC)
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
    if len(addyy_p68) < len(addyy_m68):
      addyy_p68 = [max(y_p68[x_p68 < x])] * (len(addyy_m68) - len(addyy_p68)) + addyy_p68
    elif len(addyy_m68) < len(addyy_p68):
      addyy_m68 = [max(y_m68[x_m68 < x])] * (len(addyy_p68) - len(addyy_m68)) + addyy_m68
    yy_p68 += addyy_p68
    yy_m68 += addyy_m68

  xx_pm95 = []
  yy_p95 = []
  yy_m95 = []
  for x in sorted(set(x_m95) | set(x_p95)):
    addyy_p95 = list(y_p95[x_p95 == x])
    addyy_m95 = list(y_m95[x_m95 == x])
    xx_pm95 += [x] * max(len(addyy_p95), len(addyy_m95))
    if len(addyy_p95) < len(addyy_m95):
      addyy_p95 = [max(y_p95[x_p95 < x])] * (len(addyy_m95) - len(addyy_p95)) + addyy_p95
    elif len(addyy_m95) < len(addyy_p95):
      addyy_m95 = [max(y_m95[x_m95 < x])] * (len(addyy_p95) - len(addyy_m95)) + addyy_m95
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
  plt.plot(x_n, y_n, label=f"nominal\nAUC={nominal.AUC:.3f}", color=colornominal)
  #plt.plot(x_p68, y_p68, label=r"$+1\sigma$")
  #plt.plot(x_p95, y_p95, label=r"$+2\sigma$")
  plt.fill_between(xx_pm68, yy_m68, yy_p68, alpha=0.5, label=f"68% CL\nAUC$\\in$({m68.AUC:.3f}, {p68.AUC:.3f})", color=color68)
  plt.fill_between(xx_pm95, yy_m95, yy_p95, alpha=0.5, label=f"95% CL\nAUC$\\in$({m95.AUC:.3f}, {p95.AUC:.3f})", color=color95)
  plt.legend()

  plt.savefig(docsfolder/"discrete_exampleroc_errors.pdf")
  if show:
    plt.show()
  plt.close()

def main():
  plot_params(responders=responders, nonresponders=nonresponders, yupperlim=20)
