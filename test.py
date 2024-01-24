import matplotlib.pyplot as plt, numpy as np, scipy.integrate, scipy.stats
import roc_auc

h = .6

def X(t):
  return (scipy.stats.norm.cdf(t, loc=-1, scale=h) + 2*scipy.stats.norm.cdf(t, loc=2, scale=h)) / 3
def Y(t):
  return (scipy.stats.norm.cdf(t, loc=1, scale=h) + 2*scipy.stats.norm.cdf(t, loc=-2, scale=h)) / 3
def Xdot(t):
  return (scipy.stats.norm.pdf(t, loc=-1, scale=h) + 2*scipy.stats.norm.pdf(t, loc=2, scale=h)) / 3
def Ydot(t):
  return (scipy.stats.norm.pdf(t, loc=1, scale=h) + 2*scipy.stats.norm.pdf(t, loc=-2, scale=h)) / 3

t_plot = np.linspace(-10, 10, 1001)
dt_plot = t_plot[1] - t_plot[0]
AUC = np.sum(Y(t_plot) * Xdot(t_plot)) * dt_plot

def run(target_AUC, verbose=True, prev_rocs={AUC: (t_plot, X, Y, None, None)}):
  AUC_for_guess = min(prev_rocs.keys(), key=lambda x: abs(x-target_AUC))
  t_for_guess, X_for_guess, Y_for_guess, Lambda_guess, prev_result = prev_rocs[AUC_for_guess]
  if AUC_for_guess == target_AUC and prev_result is not None:
    optimize_result = prev_result
    xy_guess = optimize_result.y
  else:
    xy_guess = roc_auc.xy_guess(X=X_for_guess, Y=Y_for_guess, t_guess=t_for_guess, AUC=target_AUC)

    if Lambda_guess is None:
      if target_AUC < AUC:
        Lambda_guess = 2
      else:
        Lambda_guess = -2

    optimize_result = roc_auc.optimize(X=X, Y=Y, Xdot=Xdot, Ydot=Ydot, AUC=target_AUC, Lambda_scaling=1, Lambda_guess=Lambda_guess, guess=xy_guess, t_guess=t_for_guess)

  t = optimize_result.x
  x, y = xy = optimize_result.y
  xd, yd = xyd = optimize_result.yp
  Lambda, c1, c2 = params = optimize_result.p

  if verbose:  
    plt.scatter(X(t_plot), Y(t_plot), label="X, Y")
    plt.scatter(xy_guess[0], xy_guess[1], label="guess")
    plt.scatter(xy[0], xy[1], label="optimized")
    print("=========================")
    print("should be 0:", x[0], y[0])
    print("should be 1:", x[-1], y[-1])
    print("should be equal:", target_AUC, 1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1])))
    print("should be equal:", c1, 2 - Lambda*target_AUC)
    print("should be equal:", c2, 2 + Lambda*(1-target_AUC))
    print("=========================")
    plt.legend()
    plt.show()

  if optimize_result is not prev_result and optimize_result.success and np.isclose(target_AUC, 1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1])), rtol=0, atol=1e-5):
    slc = slice(None)#(xd>0) & (yd>0)
    prev_rocs[target_AUC] = t[slc], x[slc], y[slc], Lambda, optimize_result

  return optimize_result

def plot_params(*, skip_aucs=[]):
  target_aucs = []
  aucs = []
  delta_aucs = []
  L = []
  c1 = []
  c2 = []
  NLL = []
  for linspace in np.linspace(AUC, 0.99999, 501), np.linspace(AUC, 0.00001, 501):
    for target_auc in linspace:
      if target_auc in skip_aucs: continue
      result = run(target_auc, verbose=False)
      print(target_auc, result.success)
      if not result.success: break
      target_aucs.append(target_auc)
      x, y = result.y
      auc = 1/2 * np.sum((y[1:]+y[:-1]) * (x[1:] - x[:-1]))
      delta_aucs.append(auc - target_auc)
      L.append(result.p[0])
      c1.append(result.p[1])
      c2.append(result.p[2])
      NLL.append(result.NLL)
  #plt.scatter(target_aucs, delta_aucs, label="$\Delta$AUC")
  plt.scatter(target_aucs, L, label="$\Lambda$")
  plt.scatter(target_aucs, c1, label="$c_1$")
  plt.scatter(target_aucs, c2, label="$c_2$")
  plt.ylim(-3, 3)
  plt.legend()
  plt.show()
  deltaNLL = np.asarray(NLL)
  deltaNLL -= np.nanmin(deltaNLL)
  plt.scatter(target_aucs, 2*np.asarray(deltaNLL))
  plt.xlabel("AUC")
  plt.ylabel("$-2\Delta\ln{L}$")
  plt.show()
  return target_aucs, delta_aucs, L, c1, c2, NLL
