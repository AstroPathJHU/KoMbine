import numpy as np, scipy.integrate

def optimize(*, X, Y, Xdot, Ydot, AUC, Lambda_guess, guess=None, Lambda_scaling=1):
  def fun(t, xy, params):
    x, y = xy
    Lambda, c1, c2 = params
    Lambda *= Lambda_scaling

    xdot = 2 * Xdot(t) / (+Lambda * y + c1)
    ydot = 2 * Ydot(t) / (-Lambda * x + c2)

    return [xdot, ydot]

  def bc(xyminusinfinity, xyplusinfinity, params):
    xminusinfinity, yminusinfinity = xyminusinfinity
    xplusinfinity, yplusinfinity = xyplusinfinity
    Lambda, c1, c2 = params
    Lambda *= Lambda_scaling

    bcs = [xminusinfinity, yminusinfinity, xplusinfinity-1, yplusinfinity-1, Lambda * AUC + c1 - 2, -Lambda * (1-AUC) + c2 - 2]
    return np.asarray(bcs[:-1])

  t_plot = np.linspace(-10, 10, 1001)
  if guess is None:
    guess = xy_guess(X=X, Y=Y, t_plot=t_plot, AUC=AUC)

  Lambda_guess /= Lambda_scaling
  c1_guess = 2 - Lambda_guess * AUC
  c2_guess = 2 + Lambda_guess * (1-AUC)
  params_guess = np.array([Lambda_guess, c1_guess, c2_guess])

  result = scipy.integrate.solve_bvp(fun=fun, bc=bc, x=t_plot, y=guess, p=params_guess, max_nodes=100000)
  return result

def xy_guess(X, Y, t_plot, AUC):
  if not 0 <= AUC <= 1:
    raise ValueError(f"AUC={AUC} is not between 0 and 1")

  if callable(X):
    X = X(t_plot)
  if callable(Y):
    Y = Y(t_plot)
  XplusY = X + Y
  XminusY = X - Y

  xplusy = XplusY

  def xminusy_s(s):
    max_allowed_xminusy = np.min([xplusy, 2 - xplusy], axis=0)
    min_allowed_xminusy = -max_allowed_xminusy

    if s >= 0:
      xminusy = XminusY * (1-s) + min_allowed_xminusy * s
    elif s < 0:
      xminusy = XminusY * (1+s) + max_allowed_xminusy * (-s)

    return xminusy

  def AUCresidual_s(s):
    xminusy = xminusy_s(s)
    x = (xplusy + xminusy) / 2
    y = (xplusy - xminusy) / 2

    AUC_s = 1/2 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
    return AUC_s - AUC

  np.testing.assert_allclose(AUCresidual_s(1), 1-AUC, atol=1e-5, rtol=0)
  np.testing.assert_allclose(AUCresidual_s(-1), -AUC, atol=1e-5, rtol=0)

  if AUC == 1:
    s = 1
  elif AUC == 0:
    s = -1
  else:
    result = scipy.optimize.root_scalar(AUCresidual_s, method="bisect", bracket=[-1, 1])
    if not result.converged:
      raise RuntimeError("Optimize didn't converge")
    s = result.root

  xminusy = xminusy_s(s)
  x = (xplusy + xminusy) / 2
  y = (xplusy - xminusy) / 2
  return np.array([x, y])
