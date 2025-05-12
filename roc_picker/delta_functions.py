"""
Optimize the discrete ROC curve using the Lagrangian method
applied to delta functions.  This is a sanity check and should
be equivalent to the discrete method.  See docs/02_rocpicker.tex
for the math details and docs/03_examples.md for usage examples.
"""

import functools
import typing
import warnings

import numpy as np
import scipy.optimize

from .discrete_base import DiscreteROCBase

class DeltaFunctions(DiscreteROCBase):
  """
  Optimize the discrete ROC curve using the Lagrangian method
  applied to delta functions.  This is a sanity check and should
  be equivalent to the discrete method.  See docs/02_rocpicker.tex
  for the math details and docs/03_examples.md for usage examples.

  Parameters
  ----------
  responders: array-like
    The parameter values for the responders.
  nonresponders: array-like
    The parameter values for the nonresponders.
  flip_sign: bool, optional
    If True, the sign of the parameter is flipped.
    Default is False.
  """
  @functools.cached_property
  def sign(self):
    """
    The sign to multiply the parameter values by.
    """
    return -1 if self.flip_sign else 1
  @functools.cached_property
  def ts(self):
    """
    The parameter values for all patients.
    """
    return sorted(set(self.responders) | set(self.nonresponders) | {np.inf, -np.inf})

  def X(self, t):
    """
    The X coordinate of the nominal ROC curve at the parameter value t,
    which represents the number of nonresponders at or below t.
    """
    return sum(1 for ni in self.nonresponders if ni*self.sign < t*self.sign)
  def Y(self, t):
    """
    The Y coordinate of the nominal ROC curve at the parameter value t,
    which represents the number of responders at or below t.
    """
    return sum(1 for ri in self.responders if ri*self.sign < t*self.sign)

  def xy(self, c1, c5, Lambda):
    """
    The x and y functions for the fitted ROC curve,
    given the parameters c1, c5, and Lambda.
    (The optimization is to determine those parameters.)
    c3 and c4 are trivial to calculate from the boundary conditions.
    """
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
    """
    Find the parameters c1, c5, and Lambda that satisfy the boundary conditions.
    """
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
    sol = typing.cast(np.ndarray, scipy.optimize.fsolve(bc, guess, full_output=False))
    return sol

  def optimize(self, *, AUC=None, c1_guess=1, c5_guess=1, Lambda_guess=1): #pylint: disable=too-many-locals
    """
    Optimize the ROC curve to match the given AUC.

    Parameters
    ----------
    AUC: float
      The target AUC.
    c1_guess: float, optional
      The initial guess for the parameter c1.
      Default is 1.
    c5_guess: float, optional
      The initial guess for the parameter c5.
      Default is 1.
    Lambda_guess: float, optional
      The initial guess for the parameter Lambda.
      Default is 1.
    """
    with warnings.catch_warnings():
      warnings.filterwarnings(
        "ignore",
        "The number of calls to function has reached maxfev = "
        "|"
        "The iteration is not making good progress"
      )
      c1, c5, Lambda = self.findparams(
        AUC=AUC,
        c1_guess=c1_guess,
        c5_guess=c5_guess,
        Lambda_guess=Lambda_guess,
      )

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
