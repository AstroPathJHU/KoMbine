import collections, functools, numpy as np, scipy.optimize
from .discrete_base import DiscreteROCBase

class DiscreteROC(DiscreteROCBase):
  def __init__(self, *args, check_validity=False, **kwargs):
    self.__check_validity = check_validity
    super().__init__(*args, **kwargs)

  @functools.cached_property
  def ts(self):
    return np.array(sorted(set(self.responders) | set(self.nonresponders)))
  @functools.cached_property
  def Xscr(self):
    counter = collections.Counter(self.nonresponders)
    return np.array([counter[t] for t in self.ts])
  @functools.cached_property
  def Yscr(self):
    counter = collections.Counter(self.responders)
    return np.array([counter[t] for t in self.ts])

  @functools.cached_property
  def nonzeroXindices(self):
    return self.Xscr != 0
  @functools.cached_property
  def nonzeroYindices(self):
    return self.Yscr != 0
  @functools.cached_property
  def numnonzeroXindices(self):
    return np.count_nonzero(self.nonzeroXindices)

  @functools.cached_property
  def nonzeroXscr(self):
    return self.Xscr[self.nonzeroXindices]
  @functools.cached_property
  def nonzeroYscr(self):
    return self.Yscr[self.nonzeroYindices]

  def checkvalidity(self, xscr, yscr):
    if not self.__check_validity: return
    np.testing.assert_array_equal(xscr[self.Xscr==0], 0)
    np.testing.assert_array_equal(yscr[self.Yscr==0], 0)
    np.testing.assert_allclose([xscr.sum(), yscr.sum()], 1)

  def evalNLL(self, xscr, yscr):
    self.checkvalidity(xscr, yscr)

    nonzeroxscr = xscr[self.nonzeroXindices]
    nonzeroyscr = yscr[self.nonzeroYindices]
    if np.min(nonzeroxscr) <= 0 or np.min(nonzeroyscr) <= 0:
      return np.inf

    NLL = 0
    NLL -= (self.nonzeroXscr * np.log(nonzeroxscr)).sum()
    NLL -= (self.nonzeroYscr * np.log(nonzeroyscr)).sum()

    return NLL

  def buildroc(self, xscr, yscr):
    self.checkvalidity(xscr, yscr)
    x = np.zeros(shape=len(self.ts)+1)
    y = np.zeros(shape=len(self.ts)+1)

    if self.flip_sign:
      xscr = xscr[::-1]
      yscr = yscr[::-1]

    x[1:] = np.cumsum(xscr)
    y[1:] = np.cumsum(yscr)
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
      length = len(self.ts)
      xscr = np.zeros(length)
      yscr = np.zeros(length)

      xscr[self.nonzeroXindices] = xy[:self.numnonzeroXindices]
      yscr[self.nonzeroYindices] = xy[self.numnonzeroXindices:]

      xscr /= xscr.sum()
      yscr /= yscr.sum()

      return xscr, yscr

    def NLL(xy):
      xscr, yscr = unpackxy(xy)
      return self.evalNLL(xscr, yscr)

    guess = np.concatenate([self.Xscr[self.nonzeroXindices], self.Yscr[self.nonzeroYindices]])

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
