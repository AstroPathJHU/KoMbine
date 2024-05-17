import abc, numbers

class DistributionBase(abc.ABC):
  @abc.abstractmethod
  def rvs(self, size=None, random_state=None): pass
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

  def __init__(self, scipydistribution, id):
    self.__scipydistribution = scipydistribution
    self.__id = id
    if id in self.__ids:
      raise KeyError(f"Created scipy distributions with duplicate id: {id}")
    self.__ids[id] = self

  def rvs(self, size=None, random_state=None):
    if random_state is None: raise TypeError("Need a random state")
    if random_state is not None: random_state += self.__id
    return self.__scipydistribution.rvs(size=size, random_state=random_state)

class AddDistributions(DistributionBase):
  def __init__(self, *distributions):
    self.__distributions = distributions
  def rvs(self, *args, **kwargs):
    return sum(d if isinstance(d, numbers.Number) else d.rvs(*args, **kwargs) for d in self.__distributions)

class MultiplyDistributions(DistributionBase):
  def __init__(self, *distributions):
    self.__distributions = distributions
  def rvs(self, *args, **kwargs):
    result = 1
    for d in self.__distributions:
      if isinstance(d, numbers.Number):
        result *= d
      else:
        result *= d.rvs(*args, **kwargs)
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

class ROC:
  def __init__(self, responders, nonresponders):
    self.__responders = responders
    self.__nonresponders = nonresponders
