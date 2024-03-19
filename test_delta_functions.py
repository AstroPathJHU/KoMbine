import matplotlib.pyplot as plt, numpy as np
import delta_functions

responders = [-5, -4, -3, -2, -1, 0, 1, 2, 3]
nonresponders = [-0.5, 0.5, 1.5, 1.7, 2.5, 2.7, 2.8, 3.5, 4, 5, 6, 7]

def plot_params(*, skip_aucs=[]):
  target_aucs = []
  aucs = []
  delta_aucs = []
  L = []
  c1 = []
  c5 = []

  t = np.asarray(sorted(set(responders) | set(nonresponders) | {-np.inf, np.inf}))

  AUC = 0.5
  linspaces = [
    [AUC] + [_ for _ in np.linspace(0, 1, 21) if _ >= AUC],
    [AUC] + [_ for _ in np.linspace(1, 0, 21) if _ <= AUC],
  ]

  for linspace in linspaces:
    last_failed = False
    for target_auc in linspace:
      print(target_auc)
      if target_auc in skip_aucs: continue
      x, y = delta_functions.findxy(responders, nonresponders, AUC=target_auc, c1_guess=1, c5_guess=1, Lambda_guess=1)
      xx = x(t)
      yy = y(t)
      auc = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1]))
      delta_auc = auc - target_auc
      if abs(delta_auc) > 1e-2:
        print("failed", target_auc)
        if last_failed:
          break
        else:
          last_failed = True
          continue
      last_failed = False
      target_aucs.append(target_auc)
      delta_aucs.append(delta_auc)
  plt.figure(figsize=(5, 5))
  plt.scatter(target_aucs, delta_aucs)
  plt.show()
