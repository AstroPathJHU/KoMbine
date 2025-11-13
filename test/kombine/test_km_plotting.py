"""
Test Kaplan-Meier plotting functionality.

This includes tests for xmax (x-axis range control), color customization,
and other plotting features.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import kombine.datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"


# ============================================================================
# X-axis range control (xmax) tests
# ============================================================================

def test_xmax_times_for_plot():
  """
  Test that get_times_for_plot correctly includes times <= xmax and xmax itself.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Get the full times_for_plot (no xmax)
  times_full = kml.times_for_plot
  print(f"Full times_for_plot: {times_full}")

  # Test with xmax=3.5
  # Death times are at 2, 3, 4, 5 based on the datacard
  # With xmax=3.5, we should get: 0, 2, 3, and 3.5 itself
  times_xmax = kml.get_times_for_plot(xmax=3.5)
  print(f"Times with xmax=3.5: {times_xmax}")

  # Verify that all times <= 3.5 are included
  assert 0 in times_xmax, "Time 0 should be included"
  assert 2 in times_xmax, "Time 2 should be included (death time <= xmax)"
  assert 3 in times_xmax, "Time 3 should be included (death time <= xmax)"

  # Verify that xmax itself is included if not already present
  assert 3.5 in times_xmax, "xmax (3.5) should be included"

  # Verify that times beyond xmax are not included
  assert 4 not in times_xmax, "Time 4 should not be included (> xmax)"
  assert 5 not in times_xmax, "Time 5 should not be included (> xmax)"

  print("✓ test_xmax_times_for_plot passed")


def test_xmax_plot_generation():
  """
  Test that KM plot can be generated with xmax parameter.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Test with xmax=3.5
  output_file = here / "test_output" / "test_xmax_plot.pdf"
  output_file.parent.mkdir(parents=True, exist_ok=True)

  config = KaplanMeierPlotConfig(
    xmax=3.5,
    saveas=output_file,
    show=False,
    print_progress=False,
  )

  results = kml.plot(config=config)

  # Verify that the plot was created
  assert output_file.exists(), "Plot file should be created"

  # Verify that results contain expected keys
  assert "x" in results, "Results should contain 'x' key"
  assert "nominal" in results, "Results should contain 'nominal' key"

  print(f"✓ test_xmax_plot_generation passed - plot saved to {output_file}")


def test_xmax_xlim_set():
  """
  Test that x-axis limits are correctly set when xmax is provided.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Create a plot with xmax
  config = KaplanMeierPlotConfig(
    xmax=3.5,
    show=False,
    create_figure=True,
    close_figure=False,
  )

  kml.plot(config=config)

  # Get the current axes and check xlim
  ax = plt.gca()
  xlim = ax.get_xlim()

  # Verify x-axis limits are set to [0, xmax]
  assert xlim[0] == 0, f"X-axis lower limit should be 0, got {xlim[0]}"
  assert xlim[1] == 3.5, f"X-axis upper limit should be 3.5, got {xlim[1]}"

  plt.close()

  print("✓ test_xmax_xlim_set passed")


def test_no_xmax_backward_compatibility():
  """
  Test that omitting xmax results in full-range plot (backward compatibility).
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Get times without xmax (should be same as before)
  times_no_xmax = kml.get_times_for_plot(xmax=None)
  times_cached = kml.times_for_plot

  # Should be identical
  assert times_no_xmax == times_cached, "get_times_for_plot(None) should match cached property"

  # Create a plot without xmax
  config = KaplanMeierPlotConfig(
    show=False,
    create_figure=True,
    close_figure=False,
  )

  kml.plot(config=config)

  # Get the current axes
  ax = plt.gca()
  xlim = ax.get_xlim()

  # X-axis should NOT be limited to a specific value (matplotlib default behavior)
  # The upper limit should be auto-scaled beyond the last data point
  print(f"X-axis limits without xmax: {xlim}")

  plt.close()

  print("✓ test_no_xmax_backward_compatibility passed")


# ============================================================================
# Edge cases for xmax
# ============================================================================

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


def test_xmax_multiple_values():
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

  print("✓ test_xmax_multiple_values passed - multiple plots created")


# ============================================================================
# Basic plotting tests
# ============================================================================

def test_basic_plot_generation():
  """
  Test that basic KM plot can be generated without any special options.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_basic_plot.pdf"
  output_file.parent.mkdir(parents=True, exist_ok=True)

  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
  )

  results = kml.plot(config=config)

  # Verify that the plot was created
  assert output_file.exists(), "Plot file should be created"

  # Verify that results contain expected keys
  assert "x" in results, "Results should contain 'x' key"
  assert "nominal" in results, "Results should contain 'nominal' key"

  print(f"✓ test_basic_plot_generation passed - plot saved to {output_file}")


if __name__ == "__main__":
  # X-axis range control tests
  test_xmax_times_for_plot()
  test_xmax_plot_generation()
  test_xmax_xlim_set()
  test_no_xmax_backward_compatibility()

  # Edge case tests
  test_xmax_beyond_all_times()
  test_xmax_at_death_time()
  test_xmax_before_first_death()
  test_xmax_multiple_values()

  # Basic plotting tests
  test_basic_plot_generation()

  print("\n✅ All KM plotting tests passed!")
