import matplotlib.pyplot as plt, numpy as np
import roc_picker.delta_functions

responders = np.linspace(-10, 10, 3)
nonresponders = responders+2.5

def plot_params(responders, nonresponders, *, skip_aucs=[], flip_sign=False, yupperlim=None):
  target_aucs = []
  delta_aucs = []
  L = []
  c1 = []
  c5 = []
  NLL = []

  optimizer = roc_picker.delta_functions.DeltaFunctions(responders=responders, nonresponders=nonresponders, flip_sign=flip_sign)

  t = optimizer.ts
  sign = 1
  if flip_sign:
    t = t[::-1]
    sign = -1

  @np.vectorize
  def X(t): return sum(1 for n in nonresponders if n < t)
  @np.vectorize
  def Y(t): return sum(1 for r in responders if r < t)

  xx = X(t) / len(nonresponders)
  yy = Y(t) / len(responders)
  AUC = 1/2 * np.sum((yy[1:]+yy[:-1]) * (xx[1:] - xx[:-1])) * sign

  linspaces = [
    [AUC] + [_ for _ in np.linspace(0, 1, 21) if _ >= AUC],
    [AUC] + [_ for _ in np.linspace(1, 0, 21) if _ <= AUC],
  ]

  plt.figure(figsize=(5, 5))
  for linspace in linspaces:
    last_failed = False
    for target_auc in linspace:
      print(target_auc)
      if target_auc in skip_aucs: continue
      result = optimizer.optimize(AUC=target_auc, c1_guess=1, c5_guess=1, Lambda_guess=1)
      xx = result.x
      yy = result.y
      auc = result.AUC
      delta_auc = auc - target_auc
      if not result.success:
        print("failed", target_auc)
        if last_failed:
          break
        else:
          last_failed = True
          continue
      last_failed = False
      plt.scatter(xx, yy, label=f"{target_auc:.3g}")
      target_aucs.append(target_auc)
      delta_aucs.append(delta_auc)
      c1.append(result.c1)
      c5.append(result.c5)
      L.append(result.Lambda)
      NLL.append(result.NLL)
  plt.legend()
  plt.show()
  plt.figure(figsize=(5, 5))
  plt.scatter(target_aucs, c1, label="$c_{1}$")
  plt.scatter(target_aucs, c5, label="$c_{5}$")
  plt.scatter(target_aucs, L, label=r"$\Lambda$")
  plt.ylim(-10, 100)
  plt.legend()
  plt.show()

  target_aucs = np.asarray(target_aucs)
  deltaNLL = np.asarray(NLL)
  deltaNLL -= np.nanmin(deltaNLL)
  plt.figure(figsize=(5,5))
  plt.scatter(target_aucs, 2*deltaNLL, label=r"$-2\Delta\ln{L}$")
  slc = np.isclose(deltaNLL, np.nanmin(deltaNLL))
  plt.scatter(target_aucs[slc], 2*deltaNLL[slc], label="best fit")
  xlow, xhigh = plt.xlim()
  plt.plot([xlow, xhigh], [1, 1], label="68% CL")
  plt.plot([xlow, xhigh], [3.84, 3.84], label="95% CL")
  plt.legend()
  plt.xlabel("AUC")
  plt.ylabel(r"$-2\Delta\ln{L}$")
  plt.xlim(0, 1)
  plt.ylim(0, yupperlim)
  plt.show()
