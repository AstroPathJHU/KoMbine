"""Tests for transient Gurobi WLS / token-network retries."""

from unittest import mock

import gurobipy as gp

from kombine.utilities import (
  GurobiOptimizerMixin,
  is_transient_gurobi_wls_error,
)


TOKEN_DNS_ERROR = gp.GurobiError(
  6,
  "Could not resolve host: token.gurobi.com "
  "(code 6, command POST https://token.gurobi.com/api/v1/tokens)",
)


def test_is_transient_gurobi_wls_error_detects_token_dns():
  """Classifier matches token DNS failures, not unrelated errors."""
  assert is_transient_gurobi_wls_error(TOKEN_DNS_ERROR)
  assert not is_transient_gurobi_wls_error(gp.GurobiError(10001, "Out of memory"))
  assert not is_transient_gurobi_wls_error(ValueError("token.gurobi.com"))


def test_optimize_with_wls_retry_recovers_after_transient_errors():
  """Two transient failures then success sleeps with exponential backoff."""
  mixin = GurobiOptimizerMixin()
  model = mock.Mock()
  model.optimize.side_effect = [TOKEN_DNS_ERROR, TOKEN_DNS_ERROR, None]

  with mock.patch("kombine.utilities.time.sleep") as sleep_mock:
    mixin._optimize_with_wls_retry(  # pylint: disable=protected-access
      model, max_attempts=5, initial_delay_s=5.0,
    )

  assert model.optimize.call_count == 3
  assert sleep_mock.call_args_list == [mock.call(5.0), mock.call(10.0)]


def test_optimize_with_wls_retry_gives_up_after_max_attempts():
  """Exhausted retries re-raise the last transient GurobiError."""
  mixin = GurobiOptimizerMixin()
  model = mock.Mock()
  model.optimize.side_effect = TOKEN_DNS_ERROR

  with mock.patch("kombine.utilities.time.sleep"):
    try:
      mixin._optimize_with_wls_retry(  # pylint: disable=protected-access
        model, max_attempts=3, initial_delay_s=1.0,
      )
    except gp.GurobiError as exc:
      assert "token.gurobi.com" in str(exc)
    else:
      raise AssertionError("expected GurobiError") from None

  assert model.optimize.call_count == 3


if __name__ == "__main__":
  test_is_transient_gurobi_wls_error_detects_token_dns()
  test_optimize_with_wls_retry_recovers_after_transient_errors()
  test_optimize_with_wls_retry_gives_up_after_max_attempts()
  print("[PASS]")
