import matplotlib.pyplot as plt, numpy as np, scipy.integrate, scipy.stats
import roc_auc

h = 0.3

def X(t):
  return (scipy.stats.norm.cdf(t, loc=0, scale=h) + 2*scipy.stats.norm.cdf(t, loc=2, scale=h)) / 3
def Y(t):
  return (scipy.stats.norm.cdf(t, loc=1, scale=h) + 2*scipy.stats.norm.cdf(t, loc=-2, scale=h)) / 3
def Xdot(t):
  return (scipy.stats.norm.pdf(t, loc=0, scale=h) + 2*scipy.stats.norm.pdf(t, loc=2, scale=h)) / 3
def Ydot(t):
  return (scipy.stats.norm.pdf(t, loc=1, scale=h) + 2*scipy.stats.norm.pdf(t, loc=-2, scale=h)) / 3

t_plot = np.linspace(-10, 10, 1001)
dt_plot = t_plot[1] - t_plot[0]
AUC = np.sum(Y(t_plot) * Xdot(t_plot)) * dt_plot

def run(target_AUC):
  xy_guess = roc_auc.xy_guess(X=X, Y=Y, t_plot=t_plot, AUC=target_AUC)
  optimize_result = roc_auc.optimize(X=X, Y=Y, Xdot=Xdot, Ydot=Ydot, AUC=target_AUC, Lambda_scaling=1, Lambda_guess=1)
  x, y = xy = optimize_result.y
  Lambda, c1, c2 = params = optimize_result.p
  
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
  
  return optimize_result
