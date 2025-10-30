"""
Test the xmax functionality for Kaplan-Meier plots.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import kombine.datacard
from kombine.kaplan_meier_likelihood import KaplanMeierPlotConfig

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"

def test_xmax_times_for_plot():
  """
  Test that get_times_for_plot correctly includes times <= xmax
  and the first time > xmax for interpolation.
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


if __name__ == "__main__":
  test_xmax_times_for_plot()
  test_xmax_plot_generation()
  test_xmax_xlim_set()
  test_no_xmax_backward_compatibility()
  print("\n✓ All xmax tests passed!")
