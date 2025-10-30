"""
Test edge cases for xmax functionality.
"""

import pathlib
import numpy as np

import kombine.datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"

def test_xmax_beyond_all_times():
  """
  Test that xmax beyond all death times works correctly.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
  
  # Death times are at 2, 3, 4, 5
  # Use xmax=10 which is beyond all death times
  times_xmax = kml.get_times_for_plot(xmax=10.0)
  print(f"Times with xmax=10.0 (beyond all deaths): {times_xmax}")
  
  # Should include all death times and xmax
  assert 0 in times_xmax
  assert 2 in times_xmax
  assert 3 in times_xmax
  assert 4 in times_xmax
  assert 5 in times_xmax
  assert 10.0 in times_xmax
  
  # Should not have any "extra" time beyond last death since there's no time > xmax
  # The list should be: 0, 2, 3, 4, 5, 10
  assert len(times_xmax) == 6, f"Expected 6 times, got {len(times_xmax)}"
  
  print("✓ test_xmax_beyond_all_times passed")


def test_xmax_at_death_time():
  """
  Test that xmax exactly at a death time works correctly.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Death times are at 2, 3, 4, 5
  # Use xmax=4.0 which is exactly a death time
  times_xmax = kml.get_times_for_plot(xmax=4.0)
  print(f"Times with xmax=4.0 (at death time): {times_xmax}")

  # Should include 0, 2, 3, 4 (all times <= xmax)
  assert 0 in times_xmax
  assert 2 in times_xmax
  assert 3 in times_xmax
  assert 4 in times_xmax

  # Times beyond xmax should not be included
  assert 5 not in times_xmax, "Times beyond xmax should not be included"

  # xmax=4 is already a death time, so it shouldn't be added again
  # The list should be: 0, 2, 3, 4
  assert len(times_xmax) == 4, f"Expected 4 times, got {len(times_xmax)}"

  print("✓ test_xmax_at_death_time passed")


def test_xmax_before_first_death():
  """
  Test that xmax before the first death time works correctly.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Death times are at 2, 3, 4, 5
  # Use xmax=1.5 which is before the first death time
  times_xmax = kml.get_times_for_plot(xmax=1.5)
  print(f"Times with xmax=1.5 (before first death): {times_xmax}")

  # Should include 0 and xmax (1.5)
  assert 0 in times_xmax
  assert 1.5 in times_xmax

  # Should not include any death times since they're all > xmax
  assert 2 not in times_xmax
  assert 3 not in times_xmax
  assert 4 not in times_xmax
  assert 5 not in times_xmax

  print("✓ test_xmax_before_first_death passed")


def test_xmax_plot_with_simple_datacard():
  """
  Test xmax with multiple xmax values to ensure it works consistently.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)
  
  # Test with different xmax values
  for xmax_val in [2.5, 3.5, 4.5]:
    output_file = here / "test_output" / f"test_xmax_edge_{xmax_val}.pdf"
    config = KaplanMeierPlotConfig(
      xmax=xmax_val,
      saveas=output_file,
      show=False,
      print_progress=False,
    )

    kml.plot(config=config)

    # Verify the plot was created
    assert output_file.exists(), f"Plot file should be created for xmax={xmax_val}"
  
  print("✓ test_xmax_plot_with_simple_datacard passed - multiple plots created")


if __name__ == "__main__":
  test_xmax_beyond_all_times()
  test_xmax_at_death_time()
  test_xmax_before_first_death()
  test_xmax_plot_with_simple_datacard()
  print("\n✓ All edge case tests passed!")
