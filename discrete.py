import collections, functools, numpy as np, scipy.optimize

class Discrete:
  def __init__(self, responders, nonresponders):
    self.responders = responders
    self.nonresponders = nonresponders

  @functools.cached_property
  def ts(self):
    return sorted(set(self.responders) | set(self.nonresponders))
  @functools.cached_property
  def Xdot(self):
    return collections.Counter(self.nonresponders)
  @functools.cached_property
  def Ydot(self):
    return collections.Counter(self.responders)

  def checkvalidity(self, xdot, ydot):
    for t in xdot:
      if xdot[t] != 0 and self.Xdot[t] == 0:
        raise ValueError(f"xdot has nonzero at t={t} but Xdot does not")
    for t in ydot:
      if ydot[t] != 0 and self.Ydot[t] == 0:
        raise ValueError(f"ydot has nonzero at t={t} but Ydot does not")
    np.testing.assert_allclose([sum(xdot.values()), sum(ydot.values())], 1)

  def evalNLL(self, xdot, ydot):
    self.checkvalidity(xdot, ydot)
    NLL = 0
    for t in self.ts:
      if self.Xdot[t] != 0:
        NLL -= self.Xdot[t] * np.log(xdot[t])
      if self.Ydot[t] != 0:
        NLL -= self.Ydot[t] * np.log(ydot[t])
    return NLL

  def buildroc(self, xdot, ydot):
    self.checkvalidity(xdot, ydot)
    x = np.zeros(shape=len(self.ts)+2)
    y = np.zeros(shape=len(self.ts)+2)
    for i, t in enumerate([-np.inf] + self.ts + [np.inf]):
      x[i] = sum(v for k, v in xdot.items() if k < t)
      y[i] = sum(v for k, v in ydot.items() if k < t)
      x /= x[-1]
      y /= y[-1]
    return x, y

  def evalAUC(self, xdot, ydot):
    xx, yy = self.buildroc(xdot, ydot)
    return np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]))

  def optimize(self):
    def unpackxy(xy):
      np.testing.assert_equal(len(xy), sum((self.Xdot[t] != 0) + (self.Ydot[t] != 0) for t in self.ts))
      xy_iterator = iter(xy)
      xdot = collections.Counter()
      ydot = collections.Counter()
      for t in self.ts:
        if self.Xdot[t] != 0:
          xdot[t] = next(xy_iterator)
      for t in self.ts:
        if self.Ydot[t] != 0:
          ydot[t] = next(xy_iterator)
      try:
        next(xy_iterator)
      except StopIteration:
        pass
      else:
        assert False

      xsum = sum(xdot.values())
      ysum = sum(ydot.values())
      for k in xdot: xdot[k] /= xsum
      for k in ydot: ydot[k] /= ysum

      return xdot, ydot

    def NLL(xy):
      xdot, ydot = unpackxy(xy)
      return self.evalNLL(xdot, ydot)

    guess = []
    for t in self.ts:
      if self.Xdot[t] != 0:
        guess.append(self.Xdot[t])
    for t in self.ts:
      if self.Ydot[t] != 0:
        guess.append(self.Ydot[t])

    return scipy.optimize.minimize(NLL, guess)

