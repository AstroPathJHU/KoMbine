import numpy as np, scipy.optimize

def xy(responders, nonresponders, c1, c5, Lambda):
  c3 = c4 = 0

  def X(t):
    return sum(1 for ni in nonresponders if ni < t)
  def Y(t):
    return sum(1 for ri in responders if ri < t)

  @np.vectorize
  def x(t):
    return c4 + c5 * sum(
      np.exp(
        -sum(
          Lambda / (2*c1 - Lambda * X(ri) + Lambda * Y(ri))
          for ri in responders
          if ri < nj
        )
      )
      for nj in nonresponders
      if nj < t
    )

  @np.vectorize
  def y(t):
    return c3 + 2/c5 * sum(
      np.exp(
        sum(
          Lambda / (2*c1 - Lambda * X(ri) + Lambda * Y(ri))
          for ri in responders
          if ri < rj
        )
      ) / (
        2*c1 - Lambda * X(rj) + Lambda * Y(rj)
      )
      for rj in responders
      if rj < t
    )

  return x, y

def findparams(responders, nonresponders, *, AUC, c1_guess, c5_guess, Lambda_guess):
  def bc(params):
    c1, c5, Lambda = params
    x, y = xy(responders, nonresponders, c1=c1, c5=c5, Lambda=Lambda)
    
    t = np.asarray(sorted(set(responders) | set(nonresponders) | {-np.inf, np.inf}))
    xx = x(t)
    yy = y(t)
    auc = np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]))
  
    return [
      x(np.inf)-1,
      y(np.inf)-1,
      auc - AUC,
    ]

  guess = [c1_guess, c5_guess, Lambda_guess]
  return scipy.optimize.fsolve(bc, guess)

def findxy(responders, nonresponders, *, AUC, c1_guess, c5_guess, Lambda_guess):
  c1, c5, Lambda = findparams(responders, nonresponders, AUC=AUC, c1_guess=c1_guess, c5_guess=c5_guess, Lambda_guess=Lambda_guess)
  x, y = xy(responders, nonresponders, c1, c5, Lambda)
  return scipy.optimize.OptimizeResult(
    x=x,
    y=y,
    c1=c1,
    c5=c5,
    Lambda=Lambda,
  )
