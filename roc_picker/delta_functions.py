import numpy as np, scipy.optimize

def xy(responders, nonresponders, c1, c5, Lambda, *, flip_sign=False):
  c3 = c4 = 0
  sign = 1
  if flip_sign:
    sign = -1
    c3 = c4 = 1

  def X(t):
    return sum(1 for ni in nonresponders if ni*sign < t*sign)
  def Y(t):
    return sum(1 for ri in responders if ri*sign < t*sign)

  @np.vectorize
  def x(t):
    return c4 + c5 * sign * sum(
      np.exp(
        -sign * sum(
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
    return c3 + 2/c5 * sign * sum(
      np.exp(
        sign * sum(
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

def findparams(responders, nonresponders, *, AUC, c1_guess, c5_guess, Lambda_guess, flip_sign):
  def bc(params):
    if not flip_sign:
      sign = 1
      target_at_inf = 1
    else:
      sign = -1
      target_at_inf = 0

    c1, c5, Lambda = params
    x, y = xy(responders, nonresponders, c1=c1, c5=c5, Lambda=Lambda, flip_sign=flip_sign)
    
    t = np.asarray(sorted(set(responders) | set(nonresponders) | {-np.inf, np.inf}))
    xx = x(t)
    yy = y(t)
    auc = np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1])) * sign

    return [
      x(np.inf)-target_at_inf,
      y(np.inf)-target_at_inf,
      auc - AUC,
    ]

  guess = [c1_guess, c5_guess, Lambda_guess]
  return scipy.optimize.fsolve(bc, guess)

def findxy(responders, nonresponders, *, AUC, c1_guess, c5_guess, Lambda_guess, flip_sign=False):
  c1, c5, Lambda = findparams(responders, nonresponders, AUC=AUC, c1_guess=c1_guess, c5_guess=c5_guess, Lambda_guess=Lambda_guess, flip_sign=flip_sign)
  x, y = xy(responders, nonresponders, c1, c5, Lambda, flip_sign=flip_sign)

  sign = 1
  if flip_sign:
    sign = -1

  NLL = 0
  #sum(-Xdot ln xdot - Ydot ln ydot)
  for r in responders:
    NLL -= np.log(sign*(y(r+0.00001) - y(r-0.00001)))
    
  for n in nonresponders:
    NLL -= np.log(sign*(x(n+0.00001) - x(n-0.00001)))

  return scipy.optimize.OptimizeResult(
    x=x,
    y=y,
    c1=c1,
    c5=c5,
    Lambda=Lambda,
    NLL=NLL,
  )
