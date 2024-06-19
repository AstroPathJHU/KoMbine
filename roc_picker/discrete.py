import collections, functools, numpy as np, scipy.optimize
from .discrete_base import DiscreteROCBase

class DiscreteROC(DiscreteROCBase):
  def __init__(self, *args, check_validity=False, **kwargs):
    self.__check_validity = check_validity
    super().__init__(*args, **kwargs)

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
    if not self.__check_validity: return
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
