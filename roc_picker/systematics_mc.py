import abc, collections, functools, matplotlib.pyplot as plt, numbers, numpy as np, scipy.special

class DistributionBase(abc.ABC):
  @abc.abstractmethod
  def rvs(self, size=None, random_state=None): pass
  @property
  @abc.abstractmethod
  def nominal(self): pass
  def __add__(self, other):
    return AddDistributions(self, other)
  def __radd__(self, other):
    return self + other
  def __sub__(self, other):
    return self + -other
  def __neg__(self):
    return -1 * self
  def __mul__(self, other):
    return MultiplyDistributions(self, other)
  def __rmul__(self, other):
    return self * other
  def __truediv__(self, other):
    return DivideDistributions(self, other)
  def __rtruediv__(self, other):
    return DivideDistributions(other, self)
  def __pow__(self, other):
    return PowerDistributions(self, other)
  def __rpow__(self, other):
    return PowerDistributions(other, self)

class ScipyDistribution(DistributionBase):
  __ids = {}

  def __init__(self, nominal, scipydistribution, id):
    self.__scipydistribution = scipydistribution
    self.__nominal = nominal
    self.__id = id
    if id in self.__ids:
      raise KeyError(f"Created scipy distributions with duplicate id: {id}")
    self.__ids[id] = self

  def rvs(self, size=None, random_state=None):
    if random_state is None: raise TypeError("Need a random state")
    if random_state is not None: random_state += self.__id
    return self.__scipydistribution.rvs(size=size, random_state=random_state)

  @property
  def nominal(self): return self.__nominal

class AddDistributions(DistributionBase):
  def __init__(self, *distributions):
    self.__distributions = distributions
  def rvs(self, *args, **kwargs):
    return sum(d if isinstance(d, numbers.Number) else d.rvs(*args, **kwargs) for d in self.__distributions)
  @property
  def nominal(self): return sum(d if isinstance(d, numbers.Number) else d.nominal for d in self.__distributions)

class MultiplyDistributions(DistributionBase):
  def __init__(self, *distributions):
    self.__distributions = distributions
  def rvs(self, *args, **kwargs):
    result = 1.
    for d in self.__distributions:
      if isinstance(d, numbers.Number):
        result *= d
      else:
        result *= d.rvs(*args, **kwargs)
    return result
  @property
  def nominal(self):
    result = 1.
    for d in self.__distributions:
      if isinstance(d, numbers.Number):
        result *= d
      else:
        result *= d.nominal
    return result

class DivideDistributions(DistributionBase):
  def __init__(self, num, denom):
    self.__num = num
    self.__denom = denom

  def rvs(self, *args, **kwargs):
    num = self.__num
    if not isinstance(self.__num, numbers.Number):
      num = num.rvs(*args, **kwargs)

    denom = self.__denom
    if not isinstance(self.__denom, numbers.Number):
      denom = denom.rvs(*args, **kwargs)

    return num / denom

  @property
  def nominal(self):
    num = self.__num
    if not isinstance(self.__num, numbers.Number):
      num = num.nominal

    denom = self.__denom
    if not isinstance(self.__denom, numbers.Number):
      denom = denom.nominal

    return num / denom

class PowerDistributions(DistributionBase):
  def __init__(self, base, exp):
    self.__base = base
    self.__exp = exp

  def rvs(self, *args, **kwargs):
    base = self.__base
    if not isinstance(self.__base, numbers.Number):
      base = base.rvs(*args, **kwargs)

    exp = self.__exp
    if not isinstance(self.__exp, numbers.Number):
      exp = exp.rvs(*args, **kwargs)

    return base ** exp

  @property
  def nominal(self):
    base = self.__base
    if not isinstance(self.__base, numbers.Number):
      base = base.nominal

    exp = self.__exp
    if not isinstance(self.__exp, numbers.Number):
      exp = exp.nominal

    return base ** exp

class ROCDistributions:
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.__responders = responders
    self.__nonresponders = nonresponders
    self.__flip_sign = flip_sign

  @property
  def responders(self): return self.__responders
  @property
  def nonresponders(self): return self.__nonresponders
  @property
  def flip_sign(self): return self.__flip_sign

  @property
  def nominal(self):
    return ROCInstance(responders=[r.nominal for r in self.responders], nonresponders=[n.nominal for n in self.nonresponders], flip_sign=self.flip_sign)

  def generate(self, size, random_state):
    responders = np.array([r.rvs(size=size, random_state=random_state) for r in self.responders])
    nonresponders = np.array([n.rvs(size=size, random_state=random_state) for n in self.nonresponders])
    return ROCCollection([ROCInstance(responders=responders[:, i], nonresponders=nonresponders[:, i], flip_sign=self.flip_sign) for i in range(size)], nominalroc=self.nominal)

class ROCInstance:
  #using the same convention as DiscreteROC, but with x = X and y = Y
  #(i.e. not handling statistical error on number of patients yet)
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.__responders = responders
    self.__nonresponders = nonresponders
    self.__flip_sign = flip_sign

  @property
  def responders(self): return self.__responders
  @property
  def nonresponders(self): return self.__nonresponders
  @property
  def flip_sign(self): return self.__flip_sign

  @functools.cached_property
  def ts(self):
    return sorted(set(self.responders) | set(self.nonresponders))
  @functools.cached_property
  def Xscr(self):
    return collections.Counter(self.nonresponders)
  @functools.cached_property
  def Yscr(self):
    return collections.Counter(self.responders)

  @functools.cached_property
  def roc(self):
    xscr = self.Xscr
    yscr = self.Yscr
    x = np.zeros(shape=len(self.ts)+2)
    y = np.zeros(shape=len(self.ts)+2)
    sign = 1
    ts = [-np.inf] + self.ts + [np.inf]
    if self.flip_sign:
      sign = -1
      ts = ts[::-1]
    for i, t in enumerate(ts):
      x[i] = sum(v for k, v in xscr.items() if k*sign < t*sign)
      y[i] = sum(v for k, v in yscr.items() if k*sign < t*sign)
      if x[-1]:
        x /= x[-1]
      if y[-1]:
        y /= y[-1]
    return x, y

  @functools.cached_property
  def x(self):
    return self.roc[0]
  @functools.cached_property
  def y(self):
    return self.roc[1]
  @functools.cached_property
  def xplusy(self):
    return self.x + self.y
  @functools.cached_property
  def xminusy(self):
    return self.x - self.y
  xplusy_interp = np.linspace(0, 2, 1001)
  @functools.cached_property
  def xminusy_interp(self):
    return np.interp(self.xplusy_interp, self.xplusy, self.xminusy)

  @functools.cached_property
  def AUC(self):
    xx, yy = self.roc
    return AUC(xx, yy)

def AUC(xx, yy):
  return np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]))

class ROCCollection:
  def __init__(self, rocinstances, nominalroc):
    self.__rocinstances = rocinstances
    self.__nominalroc = nominalroc

  @property
  def rocinstances(self):
    return self.__rocinstances
  @property
  def nominalroc(self):
    return self.__nominalroc

  xplusy_interp = ROCInstance.xplusy_interp
  @functools.cached_property
  def xminusy_interp(self):
    return np.array([roc.xminusy_interp for roc in self.__rocinstances])

  def xminusy_quantiles(self, quantiles):
    return np.quantile(self.xminusy_interp, quantiles, axis=0)
  def roc_quantiles(self, quantiles):
    xminusy_quantiles = self.xminusy_quantiles(quantiles)
    x = (self.xplusy_interp + xminusy_quantiles) / 2
    y = (self.xplusy_interp - xminusy_quantiles) / 2
    return x, y

  def plot(self, saveas=None):
    sigmas = [-2, -1, 0, 1, 2]
    quantiles = [(1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 for nsigma in sigmas]

    (x_m95, x_m68, _, x_p68, x_p95), (y_m95, y_m68, _, y_p68, y_p95) = self.roc_quantiles(quantiles)

    y_p68_interp_to_m68 = np.interp(x_m68, x_p68, y_p68)
    y_p95_interp_to_m95 = np.interp(x_m95, x_p95, y_p95)

    AUC_m95 = AUC(x_m95, y_m95)
    AUC_m68 = AUC(x_m68, y_m68)
    AUC_nominal = self.nominalroc.AUC
    AUC_p68 = AUC(x_p68, y_p68)
    AUC_p95 = AUC(x_p95, y_p95)

    AUC_68_low, AUC_68_high = sorted([AUC_m68, AUC_p68])
    AUC_95_low, AUC_95_high = sorted([AUC_m95, AUC_p95])

    fig, ax = plt.subplots(figsize=(7, 7))

    plt.plot(self.nominalroc.x, self.nominalroc.y, label=f"nominal\nAUC={AUC_nominal:.2f}", color="blue")
    plt.fill_between(x_m68, y_m68, y_p68_interp_to_m68, alpha=0.5, label=f"68% CL\nAUC$\\in$({AUC_68_low:.2f}, {AUC_68_high:.2f})", color="dodgerblue")
    plt.fill_between(x_m95, y_m95, y_p95_interp_to_m95, alpha=0.5, label=f"95% CL\nAUC$\\in$({AUC_95_low:.2f}, {AUC_95_high:.2f})", color="skyblue")

    plt.legend(fontsize=16)
    plt.xlabel("X (Fraction of non-responders)", fontsize=16)
    plt.ylabel("Y (Fraction of responders)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if saveas is None:
      plt.show()
    else:
      plt.savefig(saveas)
