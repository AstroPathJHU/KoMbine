"""
Utility functions for the unit tests.
"""

import typing

def flip_sign_curve(k):
  """
  flip the key k for comparison:
  The nominal curve is compared to itself,
  but when flip_sign is true, the plus and minus confidence
  intervals are flipped.
  """
  return {
    "nominal": "nominal",
    "p68": "m68",
    "p95": "m95",
    "m68": "p68",
    "m95": "p95",
  }[k]

class Tolerance(typing.TypedDict):
  "typed class for atol and rtol to pass to np.testing.assert_allclose"
  rtol: float
  atol: float
