import functools, numpy as np, scipy.optimize

class DeltaFunctions:
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.responders = responders
    self.nonresponders = nonresponders
    self.flip_sign = flip_sign

  @functools.cached_property
  def sign(self):
    if self.flip_sign: return -1
    return 1
  @functools.cached_property
  def ts(self):
    return sorted(set(self.responders) | set(self.nonresponders) | {np.inf, -np.inf})

  def X(self, t):
    return sum(1 for ni in self.nonresponders if ni*self.sign < t*self.sign)
  def Y(self, t):
    return sum(1 for ri in self.responders if ri*self.sign < t*self.sign)

  def xy(self, c1, c5, Lambda):
    c3 = c4 = 0
    if self.flip_sign:
      c3 = c4 = 1

    @np.vectorize
    def x(t):
      return c4 + c5 * self.sign * sum(
        np.exp(
          -self.sign * sum(
            Lambda / (2*c1 - Lambda * self.X(ri) + Lambda * self.Y(ri))
            for ri in self.responders
            if ri < nj
          )
        )
        for nj in self.nonresponders
        if nj < t
      )

    @np.vectorize
    def y(t):
      return c3 + 2/c5 * self.sign * sum(
        np.exp(
          self.sign * sum(
            Lambda / (2*c1 - Lambda * self.X(ri) + Lambda * self.Y(ri))
            for ri in self.responders
            if ri < rj
          )
        ) / (
          2*c1 - Lambda * self.X(rj) + Lambda * self.Y(rj)
        )
        for rj in self.responders
        if rj < t
      )

    return x, y

  def findparams(self, *, AUC, c1_guess, c5_guess, Lambda_guess):
    def bc(params):
      if not self.flip_sign:
        target_at_inf = 1
      else:
        target_at_inf = 0

      c1, c5, Lambda = params
      x, y = self.xy(c1=c1, c5=c5, Lambda=Lambda)

      xx = x(self.ts)
      yy = y(self.ts)
      auc = np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1])) * self.sign

      return [
        x(np.inf)-target_at_inf,
        y(np.inf)-target_at_inf,
        auc - AUC,
      ]

    guess = [c1_guess, c5_guess, Lambda_guess]
    return scipy.optimize.fsolve(bc, guess)

  def optimize(self, *, AUC, c1_guess=1, c5_guess=1, Lambda_guess=1):
    c1, c5, Lambda = self.findparams(AUC=AUC, c1_guess=c1_guess, c5_guess=c5_guess, Lambda_guess=Lambda_guess)
    x, y = self.xy(c1, c5, Lambda)

    NLL = 0
    #sum(-Xdot ln xdot - Ydot ln ydot)
    for r in self.responders:
      NLL -= np.log(self.sign*(y(r+0.00001) - y(r-0.00001)))

    for n in self.nonresponders:
      NLL -= np.log(self.sign*(x(n+0.00001) - x(n-0.00001)))

    xx = x(self.ts)
    yy = y(self.ts)
    auc = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))

    return scipy.optimize.OptimizeResult(
      xfun=x,
      yfun=y,
      x=x(self.ts),
      y=y(self.ts),
      c1=c1,
      c5=c5,
      Lambda=Lambda,
      NLL=NLL,
      AUC=AUC,
      success=abs(auc-AUC) < 1e-4 and abs(xx[-1]-1) < 1e-4 and abs(yy[-1]-1) < 1e-4,
    )
