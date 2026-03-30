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

def format_value_for_json(value, precision):
  """
  Recursively formats float values in lists/dictionaries to a specified precision.
  """
  if isinstance(value, float):
    return round(value, precision)
  if isinstance(value, list):
    return [format_value_for_json(item, precision) for item in value]
  if isinstance(value, dict):
    return {k: format_value_for_json(v, precision) for k, v in value.items()}
  return value

class Tolerance(typing.TypedDict):
  "typed class for atol and rtol to pass to np.testing.assert_allclose"
  rtol: float
  atol: float

def compare_dict_keys(
    current: typing.Dict[str, typing.Any],
    reference: typing.Dict[str, typing.Any],
) -> None:
  """
  Compare the keys of two dictionaries.
  Raises an AssertionError if there are missing or extra keys.
  """
  current_keys = set(current.keys())
  reference_keys = set(reference.keys())

  missing_keys = current_keys - reference_keys
  extra_keys = reference_keys - current_keys

  if missing_keys:
    raise AssertionError(f"Keys missing in reference file: {', '.join(sorted(missing_keys))}")
  if extra_keys:
    raise AssertionError(f"Extra keys found in reference file: {', '.join(sorted(extra_keys))}")


def generate_synthetic_datacard(
  n_patients: int,
  mean_survival_time: float,
  censor_rate: float,
  seed: int,
) -> str:
  """
  Generate a synthetic datacard for testing purposes.

  Args:
      n_patients: Number of patients to generate.
      mean_survival_time: Mean survival time (exponential distribution).
      censor_rate: Proportion of patients to censor (0.0 to 1.0).
      seed: Random seed for reproducibility.

  Returns:
      String representation of the datacard in Yi format.
  """
  import numpy as np  # pylint: disable=import-outside-toplevel

  rng = np.random.RandomState(seed)  # pylint: disable=no-member

  # Generate survival times from exponential distribution
  survival_times = rng.exponential(mean_survival_time, n_patients)

  # Generate censoring (1=censored, 0=event)
  censoring = rng.choice([0, 1], size=n_patients, p=[1-censor_rate, censor_rate])

  # Generate synthetic cell counts (for demonstration)
  group1_counts = rng.poisson(50, n_patients)
  group2_counts = rng.poisson(50, n_patients)

  # Build datacard string
  lines = []
  for i in range(n_patients):
    line = f"{survival_times[i]:.6f}\t{censoring[i]}\t{group1_counts[i]}\t{group2_counts[i]}"
    lines.append(line)

  return "\n".join(lines)


def generate_two_group_datacard(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
  n_patients_per_group: int,
  lambda_0: float,
  lambda_1: float,
  threshold: float,
  observable_range_0: tuple[float, float],
  observable_range_1: tuple[float, float],
  censor_rate: float = 0.3,
  seed: int = 42,
) -> str:
  """
  Generate a synthetic two-group datacard for testing hazard ratios.

  Args:
      n_patients_per_group: Number of patients per group.
      lambda_0: Hazard rate for group 0 (below threshold).
      lambda_1: Hazard rate for group 1 (above threshold).
      threshold: Observable threshold separating the groups.
      observable_range_0: (min, max) observable values for group 0.
      observable_range_1: (min, max) observable values for group 1.
      censor_rate: Proportion of patients to censor (default 0.3).
      seed: Random seed for reproducibility.

  Returns:
      String representation of the datacard content.
  """
  import numpy as np  # pylint: disable=import-outside-toplevel

  np.random.seed(seed)

  # Group 0: below threshold
  survival_times_0 = np.random.exponential(1.0 / lambda_0, size=n_patients_per_group)
  observables_0 = np.random.uniform(*observable_range_0, size=n_patients_per_group)

  # Group 1: above threshold
  survival_times_1 = np.random.exponential(1.0 / lambda_1, size=n_patients_per_group)
  observables_1 = np.random.uniform(*observable_range_1, size=n_patients_per_group)

  # Randomly censor
  censored_0 = np.random.random(n_patients_per_group) < censor_rate
  censored_1 = np.random.random(n_patients_per_group) < censor_rate

  # Combine all data
  all_survival_times = np.concatenate([survival_times_0, survival_times_1])
  all_censored = np.concatenate([censored_0, censored_1])
  all_observables = np.concatenate([observables_0, observables_1])

  # Format data
  survival_times_str = '\t'.join(f'{t:.4f}' for t in all_survival_times)
  censored_str = '\t'.join('1' if c else '0' for c in all_censored)
  observables_str = '\t'.join(f'{o:.4f}' for o in all_observables)

  target_hr = lambda_1 / lambda_0

  datacard_content = f"""observable_type fixed
------------
# Synthetic datacard with known hazard ratio = {target_hr:.4f}
# Group 0 (below {threshold}): lambda_0 = {lambda_0}
# Group 1 (above {threshold}): lambda_1 = {lambda_1}
# Expected HR = lambda_1/lambda_0 = {target_hr:.4f}
------------
survival_time\t{survival_times_str}
censored\t{censored_str}
observable\t{observables_str}
"""

  return datacard_content


def generate_two_group_datacard_from_hr(  # pylint: disable=too-many-arguments,too-many-positional-arguments
  n_patients_per_group: int,
  target_hr: float,
  threshold: float,
  observable_range_0: tuple[float, float],
  observable_range_1: tuple[float, float],
  censor_rate: float = 0.3,
  seed: int = 42,
  lambda_0: float = 0.1,
) -> str:
  """
  Generate a synthetic two-group datacard from a target hazard ratio.

  This is a convenience wrapper around generate_two_group_datacard that computes
  lambda_1 from lambda_0 and the target hazard ratio.

  Args:
      n_patients_per_group: Number of patients per group.
      target_hr: Target hazard ratio (lambda_1 / lambda_0).
      threshold: Observable threshold separating the groups.
      observable_range_0: (min, max) observable values for group 0.
      observable_range_1: (min, max) observable values for group 1.
      censor_rate: Proportion of patients to censor (default 0.3).
      seed: Random seed for reproducibility.
      lambda_0: Base hazard rate for group 0 (default 0.1).

  Returns:
      String representation of the datacard content.
  """
  # Compute lambda_1 from target hazard ratio
  lambda_1 = target_hr * lambda_0

  return generate_two_group_datacard(
    n_patients_per_group=n_patients_per_group,
    lambda_0=lambda_0,
    lambda_1=lambda_1,
    threshold=threshold,
    observable_range_0=observable_range_0,
    observable_range_1=observable_range_1,
    censor_rate=censor_rate,
    seed=seed,
  )
