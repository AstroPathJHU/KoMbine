"""
Optimize a ROC curve for sample-wise uncertainties, including systematics.
See docs/02_rocpicker.tex for the math details and
docs/03_examples.md for usage examples.
"""

import abc
import collections
import functools
import numbers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.special

class DistributionBase(abc.ABC):
  """
  Base class for probability distributions with basic math functionality.
  """
  @abc.abstractmethod
  def rvs(self, *, size, random_state=None) -> npt.NDArray[np.floating]:
    """
    Generate random variates from the distribution.
    """
  @property
  @abc.abstractmethod
  def nominal(self) -> float:
    """
    The nominal value of the distribution.
    """
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

class DummyDistribution(DistributionBase):
  """
  A dummy distribution that returns a constant value.
  This is used when doing math with a distribution and a number.
  """
  def __init__(self, value):
    self.__value = value
  def rvs(self, *, size, random_state=None):
    return np.full(size, self.__value)
  @property
  def nominal(self):
    return self.__value
  def __float__(self):
    return float(self.__value)

class ScipyDistribution(DistributionBase):
  """
  Wrapper for scipy distributions to allow a unique id
  and mathematical operations defined in DistributionBase.
  """
  __ids = set()

  def __init__(self, nominal, scipydistribution, unique_id):
    self.__scipydistribution = scipydistribution
    self.__nominal = nominal
    self.__id = unique_id
    if unique_id in self.__ids:
      raise KeyError(f"Created scipy distributions with duplicate id: {unique_id}")
    self.__ids.add(unique_id)

  def __del__(self):
    self.__ids.remove(self.__id)

  def rvs(self, *, size, random_state=None):
    if random_state is None:
      raise TypeError("Need a random state")
    if random_state is not None:
      random_state += self.__id
    return self.__scipydistribution.rvs(size=size, random_state=random_state)

  @property
  def nominal(self):
    """
    The nominal value of the distribution.
    """
    return self.__nominal

  @property
  def unique_id(self):
    """
    Return the unique id of the distribution.
    This is used for the random seed
    """
    return self.__id

class AddDistributions(DistributionBase):
  """
  Two probability distributions added together.
  You can also add a number to a distribution.
  """
  def __init__(self, *distributions: numbers.Number | DistributionBase):
    self.__distributions : list[DistributionBase] = []
    for d in distributions:
      if isinstance(d, numbers.Number):
        self.__distributions.append(DummyDistribution(d))
      elif isinstance(d, DistributionBase):
        self.__distributions.append(d)
      else:
        raise TypeError(f"Invalid type for distribution: {type(d)}")
  def rvs(self, *, size, random_state=None):
    return sum(
      (
        d.rvs(size=size, random_state=random_state)
        for d in self.__distributions
      ),
      start=np.zeros(size, dtype=np.float64)
    )
  @property
  def nominal(self):
    return sum(
      d.nominal
      for d in self.__distributions
    )

class MultiplyDistributions(DistributionBase):
  """
  Two probability distributions multiplied together.
  You can also multiply a distribution by a number.
  """
  def __init__(self, *distributions: numbers.Number | DistributionBase):
    self.__distributions : list[DistributionBase] = []
    for d in distributions:
      if isinstance(d, numbers.Number):
        self.__distributions.append(DummyDistribution(d))
      elif isinstance(d, DistributionBase):
        self.__distributions.append(d)
      else:
        raise TypeError(f"Invalid type for distribution: {type(d)}")
  def rvs(self, *, size, random_state=None):
    result = np.ones(size, dtype=np.float64)
    for d in self.__distributions:
      if isinstance(d, numbers.Number):
        result *= d
      else:
        result *= d.rvs(size=size, random_state=random_state)
    return result
  @property
  def nominal(self):
    result = 1.
    for d in self.__distributions:
      result *= d.nominal
    return result

class DivideDistributions(DistributionBase):
  """
  Two probability distributions divided.
  You can also divide a distribution by a number or vice versa.
  """
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
  """
  A probability distribution raised to the power of another.
  Either the base or the exponent can also be a number.
  """
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
  """
  A collection of probability distributions for the observable
  for responders and non-responders.  You can then generate
  a collection of ROC curves using the generate() method.

  This should typically not be created directly, but rather
  through the Datacard interface.
  """
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.__responders = responders
    self.__nonresponders = nonresponders
    self.__flip_sign = flip_sign

  @property
  def responders(self):
    """
    The probability distributions of the observable for the responders.
    """
    return self.__responders
  @property
  def nonresponders(self):
    """
    The probability distributions of the observable for the non-responders.
    """
    return self.__nonresponders
  @property
  def flip_sign(self):
    """
    Whether or not to flip the sign of the parameter values.
    """
    return self.__flip_sign

  @property
  def nominal(self):
    """
    The nominal ROC curve.
    """
    return ROCInstance(
      responders=[r.nominal for r in self.responders],
      nonresponders=[n.nominal for n in self.nonresponders],
      flip_sign=self.flip_sign
    )

  def generate(self, size, random_state):
    """
    Generate a collection of ROC curves using the random seed.

    Parameters
    ----------
    size: int
      The number of ROC curves to generate.
    random_state: np.random.RandomState
      The random seed to use.
    """
    responders = np.array(
      [r.rvs(size=size, random_state=random_state) for r in self.responders]
    )
    nonresponders = np.array(
      [n.rvs(size=size, random_state=random_state) for n in self.nonresponders]
    )
    return ROCCollection(
      [
        ROCInstance(
          responders=responders[:, i],
          nonresponders=nonresponders[:, i],
          flip_sign=self.flip_sign
        )
        for i in range(size)
      ],
      nominalroc=self.nominal
    )

class ROCInstance:
  """
  A single ROC curve generated with a particular value of the
  parameter for each patient.

  The math uses the same convention as DiscreteROC, but with x = X and y = Y
  (i.e. this method can't handle the statistical error on number of patients yet)
  """
  def __init__(self, responders, nonresponders, *, flip_sign=False):
    self.__responders = responders
    self.__nonresponders = nonresponders
    self.__flip_sign = flip_sign

  @property
  def responders(self):
    """
    The parameter values for the responders
    """
    return self.__responders
  @property
  def nonresponders(self):
    """
    The parameter values for the non-responders
    """
    return self.__nonresponders
  @property
  def flip_sign(self):
    """
    Whether or not to flip the sign of the parameter values.
    """
    return self.__flip_sign

  @functools.cached_property
  def ts(self):
    """
    The parameter values for all patients.
    """
    return sorted(set(self.responders) | set(self.nonresponders))
  @functools.cached_property
  def Xscr(self):
    """
    The number of nonresponders at each parameter value.
    """
    return collections.Counter(self.nonresponders)
  @functools.cached_property
  def Yscr(self):
    """
    The number of responders at each parameter value.
    """
    return collections.Counter(self.responders)

  @functools.cached_property
  def roc(self):
    """
    The coordinates of the ROC curve points, constructed
    using the responder and non-responder counts.
    """
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
    """
    The x coordinates of the ROC curve points.
    """
    return self.roc[0]
  @functools.cached_property
  def y(self):
    """
    The y coordinates of the ROC curve points.
    """
    return self.roc[1]
  @functools.cached_property
  def xplusy(self):
    """
    The (x+y) coordinates of the ROC curve points.
    """
    return self.x + self.y
  @functools.cached_property
  def xminusy(self):
    """
    The (x-y) coordinates of the ROC curve points.
    """
    return self.x - self.y
  xplusy_interp = np.linspace(0, 2, 1001)
  @functools.cached_property
  def xminusy_interp(self):
    """
    The interpolated (x-y) coordinates of the ROC curve points
    to get a consistent x+y coordinate across all the MC generated
    ROC curves.
    """
    return np.interp(self.xplusy_interp, self.xplusy, self.xminusy)

  @functools.cached_property
  def AUC(self):
    """
    Calculate the area under the ROC curve.
    """
    xx, yy = self.roc
    return AUC(xx, yy)

def AUC(xx, yy):
  """
  Calculate the area under the ROC curve.
  """
  return np.sum(0.5 * (xx[1:] - xx[:-1]) * (yy[1:] + yy[:-1]))

class ROCCollection:
  """
  A Monte-Carlo-generated collection of ROC curves.

  Parameters
  ----------
  rocinstances: list of ROCInstance
    The generated ROC curves.
  nominalroc: ROCInstance
    The nominal ROC curve.
  """
  def __init__(self, rocinstances, nominalroc):
    self.__rocinstances = rocinstances
    self.__nominalroc = nominalroc

  @property
  def rocinstances(self):
    """
    The generated ROC curves.
    """
    return self.__rocinstances
  @property
  def nominalroc(self):
    """
    The nominal ROC curve.
    """
    return self.__nominalroc

  xplusy_interp = ROCInstance.xplusy_interp
  @functools.cached_property
  def xminusy_interp(self):
    """
    The x-y difference at each x+y value for each generated ROC curve.
    """
    return np.array([roc.xminusy_interp for roc in self.__rocinstances])

  def xminusy_quantiles(self, quantiles):
    """
    The quantiles of the x-y difference at each x+y value
    using the generated ROC curves.
    """
    return np.quantile(self.xminusy_interp, quantiles, axis=0)
  def roc_quantiles(self, quantiles):
    """
    Find the 68% and 95% CL bands for the ROC curve.
    This is done by finding the quantiles of the x-y difference at each x+y value.
    using the Monte Carlo samples.
    """
    xminusy_quantiles = self.xminusy_quantiles(quantiles)
    x = (self.xplusy_interp + xminusy_quantiles) / 2
    y = (self.xplusy_interp - xminusy_quantiles) / 2
    return x, y

  def plot(self, *, saveas=None, show=False): # pylint: disable=too-many-locals
    """
    Plot the ROC curve with the 68% and 95% CL bands.

    Parameters
    ----------
    saveas: os.PathLike, optional
      The filename to save the plot.
    show: bool, optional
      Whether to show the plot.
    """
    sigmas = [-2, -1, 0, 1, 2]
    quantiles = [
      (1 + scipy.special.erf(nsigma/np.sqrt(2))) / 2 # pylint: disable=no-member
      for nsigma in sigmas
    ]

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

    fig, ax = plt.subplots(figsize=(5, 5)) # pylint: disable=unused-variable

    plt.plot(
      self.nominalroc.x, self.nominalroc.y,
      color="blue",
      label=f"nominal\nAUC={AUC_nominal:.2f}",
    )
    plt.fill_between(
      x_m68, y_m68, y_p68_interp_to_m68,
      color="dodgerblue", alpha=0.5,
      label=f"68% CL\nAUC$\\in$({AUC_68_low:.2f}, {AUC_68_high:.2f})"
    )
    plt.fill_between(
      x_m95, y_m95, y_p95_interp_to_m95,
      color="skyblue", alpha=0.5,
      label=f"95% CL\nAUC$\\in$({AUC_95_low:.2f}, {AUC_95_high:.2f})",
    )

    plt.legend()
    plt.xlabel("X (Fraction of non-responders)")
    plt.ylabel("Y (Fraction of responders)")
    ax.tick_params(axis='both', which='major')
    if saveas is not None:
      plt.savefig(saveas)
    if show:
      plt.show()
    plt.close()

    return {
      "nominal": {
        "x": self.nominalroc.x,
        "y": self.nominalroc.y,
        "AUC": AUC_nominal,
      },
      "68%": [
        {
          "x": x_m68,
          "y": y_m68,
          "AUC": AUC_m68
        },
        {
          "x": x_p68,
          "y": y_p68_interp_to_m68,
          "AUC": AUC_p68
        },
      ],
      "95%": [
        {
          "x": x_m95,
          "y": y_m95,
          "AUC": AUC_m95
        },
        {
          "x": x_p95,
          "y": y_p95_interp_to_m95,
          "AUC": AUC_p95
        },
      ],
    }
