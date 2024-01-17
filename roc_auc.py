import numpy as np, scipy.integrate

def optimize(X, Y, Xdot, Ydot, AUC):
  def fun(t, xy, params):
    x, y = xy
    Lambda, c1, c2 = params

    xdot = 2 * Xdot(t) / (+Lambda * y + c1)
    ydot = 2 * Ydot(t) / (-Lambda * x + c2)

    return [xdot, ydot]

  def bc(xyminusinfinity, xyplusinfinity, params):
    xminusinfinity, yminusinfinity = xyminusinfinity
    xplusinfinity, yplusinfinity = xyplusinfinity
    Lambda, c1, c2 = params
    bcs = [xminusinfinity, yminusinfinity, xplusinfinity-1, yplusinfinity-1, Lambda * AUC + c1 - 2, -Lambda * (1-AUC) + c2 - 2]
    return np.asarray(bcs[1:])

  t_plot = np.linspace(-10, 10, 1001)

  xy_guess = np.asarray([X(t_plot), Y(t_plot)])
  Lambda_guess = 1
  c1_guess = 2 - Lambda_guess * AUC
  c2_guess = 2 + Lambda_guess * (1-AUC)
  params_guess = np.array([Lambda_guess, c1_guess, c2_guess])

  return scipy.integrate.solve_bvp(fun=fun, bc=bc, x=t_plot, y=xy_guess, p=params_guess, max_nodes=100000)

