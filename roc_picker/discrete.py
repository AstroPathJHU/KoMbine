import collections, functools, matplotlib.pyplot as plt, numpy as np, scipy.optimize

class DiscreteROC:
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.responders = responders
    self.nonresponders = nonresponders
    self.flip_sign = flip_sign

  @functools.cached_property
  def ts(self):
    return sorted(set(self.responders) | set(self.nonresponders))
  @functools.cached_property
  def Xscr(self):
    return collections.Counter(self.nonresponders)
  @functools.cached_property
  def Yscr(self):
    return collections.Counter(self.responders)

  def checkvalidity(self, xscr, yscr):
    for t in xscr:
      if xscr[t] != 0 and self.Xscr[t] == 0:
        raise ValueError(f"xscr has nonzero at t={t} but Xscr does not")
    for t in yscr:
      if yscr[t] != 0 and self.Yscr[t] == 0:
        raise ValueError(f"yscr has nonzero at t={t} but Yscr does not")
    np.testing.assert_allclose([sum(xscr.values()), sum(yscr.values())], 1)

  def evalNLL(self, xscr, yscr):
    self.checkvalidity(xscr, yscr)
    NLL = 0
    for t in self.ts:
      if self.Xscr[t] != 0:
        NLL -= self.Xscr[t] * np.log(xscr[t])
      if self.Yscr[t] != 0:
        NLL -= self.Yscr[t] * np.log(yscr[t])
    return NLL

  def buildroc(self, xscr, yscr):
    self.checkvalidity(xscr, yscr)
    x = np.zeros(shape=len(self.ts)+2)
    y = np.zeros(shape=len(self.ts)+2)
    sign = 1
    ts = [-np.inf] + self.ts + [np.inf]
    if self.flip_sign:
      sign = -1
      ts = ts[::-1]
    for i, t in enumerate(ts):
      x[i] = sum(v for k, v in xscr.items() if k*sign < t*sign)
      y[i] = sum(v for k, v in yscr.items() if k*sign < t*sign)
      if x[-1]:
        x /= x[-1]
      if y[-1]:
        y /= y[-1]
    return x, y

  def evalAUC(self, xscr, yscr):
    xx, yy = self.buildroc(xscr, yscr)
    return np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]))

  def optimize(self, AUC=None):
    def unpackxy(xy):
      np.testing.assert_equal(len(xy), sum((self.Xscr[t] != 0) + (self.Yscr[t] != 0) for t in self.ts))
      xy_iterator = iter(xy)
      xscr = collections.Counter()
      yscr = collections.Counter()
      for t in self.ts:
        if self.Xscr[t] != 0:
          xscr[t] = next(xy_iterator)
      for t in self.ts:
        if self.Yscr[t] != 0:
          yscr[t] = next(xy_iterator)
      try:
        next(xy_iterator)
      except StopIteration:
        pass
      else:
        assert False

      xsum = sum(xscr.values())
      ysum = sum(yscr.values())
      for k in xscr: xscr[k] /= xsum
      for k in yscr: yscr[k] /= ysum

      return xscr, yscr

    def NLL(xy):
      xscr, yscr = unpackxy(xy)
      return self.evalNLL(xscr, yscr)

    guess = []
    for t in self.ts:
      if self.Xscr[t] != 0:
        guess.append(self.Xscr[t])
    for t in self.ts:
      if self.Yscr[t] != 0:
        guess.append(self.Yscr[t])

    def sumxscr(xy):
      xscr, yscr = unpackxy(xy)
      return sum(xscr) - 1
    def sumyscr(xy):
      xscr, yscr = unpackxy(xy)
      return sum(yscr) - 1
    constraints = [
      #{
      #  "type": "eq",
      #  "fun": sumxscr,
      #}, {
      #  "type": "eq",
      #  "fun": sumyscr,
      #}
    ]
    if AUC is not None:
      def constraintfunction(xy):
        xscr, yscr = unpackxy(xy)
        return self.evalAUC(xscr, yscr) - AUC
      constraints.append({
        "type": "eq",
        "fun": constraintfunction,
      })

    result = scipy.optimize.minimize(NLL, guess, constraints=constraints, method="SLSQP")
    result["xscryscr"] = xscryscr = result.pop("x")
    xscr, yscr = unpackxy(xscryscr)
    result["x"], result["y"] = self.buildroc(xscr, yscr)
    result["AUC"] = self.evalAUC(xscr, yscr)
    result["NLL"] = result["fun"]
    if AUC is not None:
      if abs(constraintfunction(xscryscr)) > 1e-4:
        result["success"] = False
    return result

  def plot_roc(self, *, show=False, rocfilename=None, scanfilename=None, rocerrorsfilename=None, yupperlim=None, npoints=100):
    if not show and rocfilename is None and scanfilename is None and rocerrorsfilename is None:
      raise RuntimeError("If you're not showing or saving the plots, there's nothing to do")

    target_aucs = []
    NLL = []

    sign = 1
    t = np.asarray(sorted(set(self.responders) | set(self.nonresponders) | {-np.inf, np.inf}))
    if self.flip_sign:
      sign = -1
      t = t[::-1]

    @np.vectorize
    def X(t): return sum(1 for n in self.nonresponders if n*sign < t*sign)
    @np.vectorize
    def Y(t): return sum(1 for r in self.responders if r*sign < t*sign)

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
        auc = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))
        delta_auc = auc - target_auc
        if not result.success:
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
    if rocfilename is not None:
      plt.savefig(rocfilename)
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
    plt.xlim(0, 1)
    plt.ylim(0, yupperlim)
    if scanfilename is not None:
      plt.savefig(scanfilename)
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
        result = self.optimize(AUC=target_auc)
        return 2 * (result.NLL - np.nanmin(NLL)) - d2NLLcut

      left_auc_left_bracket = target_aucs[from_below_to_above_left].item()
      left_auc_right_bracket = target_aucs[from_below_to_above_right].item()
      right_auc_left_bracket = target_aucs[from_above_to_below_left].item()
      right_auc_right_bracket = target_aucs[from_above_to_below_right].item()

      left_auc = scipy.optimize.root_scalar(tosolve, bracket=[left_auc_left_bracket, left_auc_right_bracket])
      assert left_auc.converged, left_auc
      left_result = self.optimize(AUC=left_auc.root)
      right_auc = scipy.optimize.root_scalar(tosolve, bracket=[right_auc_left_bracket, right_auc_right_bracket])
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
      if not len(addyy_p68):
        addyy_p68 = [np.interp(x, x_p68, y_p68)] * len(addyy_m68)
      elif not len(addyy_m68):
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
      if not len(addyy_p95):
        addyy_p95 = [np.interp(x, x_p95, y_p95)] * len(addyy_m95)
      elif not len(addyy_m95):
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
    plt.plot(x_n, y_n, label=f"nominal\nAUC={nominal.AUC:.2f}", color=colornominal)
    #plt.plot(x_p68, y_p68, label=r"$+1\sigma$")
    #plt.plot(x_p95, y_p95, label=r"$+2\sigma$")
    plt.fill_between(xx_pm68, yy_m68, yy_p68, alpha=0.5, label=f"68% CL\nAUC$\\in$({m68.AUC:.2f}, {p68.AUC:.2f})", color=color68)
    plt.fill_between(xx_pm95, yy_m95, yy_p95, alpha=0.5, label=f"95% CL\nAUC$\\in$({m95.AUC:.2f}, {p95.AUC:.2f})", color=color95)
    plt.legend()

    if rocerrorsfilename is not None:
      plt.savefig(rocerrorsfilename)
    if show:
      plt.show()
    plt.close()


