"""
Miscellaneous utilities for ROC Picker
"""
import typing

T = typing.TypeVar("T")
R = typing.TypeVar("R")

class InspectableCache(typing.Generic[T, R]):
  """
  Cache decorator that allows inspection of cached values.
  """
  def __init__(self, func: typing.Callable[[T], R]):
    self._func = func
    self.cache: dict[tuple[T], R] = {}
    self.__name__ = getattr(func, "__name__", "InspectableCache")
    self.__doc__ = getattr(func, "__doc__", None)

  def __call__(self, arg: T) -> R:
    key = (arg,)
    if key in self.cache:
      return self.cache[key]
    result = self._func(arg)
    self.cache[key] = result
    return result

  def __getattr__(self, name: str) -> typing.Any:
    # Delegate attribute access to the original function
    return getattr(self._func, name)

  def __iter__(self) -> typing.Iterator[tuple[tuple[T], R]]:
    yield from self.cache.items()
