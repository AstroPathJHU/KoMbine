"""
Test Kaplan-Meier plotting functionality.

This module contains comprehensive tests for KM plotting features including:

1. X-axis range control (xmax):
   - Basic functionality and time point selection
   - Plot generation with xmax parameter
   - X-axis limit setting
   - Backward compatibility (no xmax)
   - Edge cases: xmax beyond all times, at death time, before first death
   - Multiple xmax values

2. Plot configuration options:
   - Custom title and axis labels
   - Custom figure size
   - Legend control (location, no legend)
   - Tight layout control
   - Custom font sizes (title, labels, ticks, legend)

3. Error band options:
   - Binomial-only error bands
   - Patient-wise-only error bands
   - Combining xmax with different error band configurations

4. Basic plotting:
   - Basic plot generation
   - Include/exclude nominal curve
   - Different output formats (PDF, PNG)
   - Plot results structure validation

5. Combined features:
   - xmax with custom plot options
   - xmax with different error bands
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


def test_plot_with_custom_title_labels():
  """
  Test that custom title and axis labels can be set.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_custom_labels.pdf"
  custom_title = "Custom Survival Analysis"
  custom_xlabel = "Time (months)"
  custom_ylabel = "Survival Probability"

  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    create_figure=True,
    close_figure=False,
    title=custom_title,
    xlabel=custom_xlabel,
    ylabel=custom_ylabel,
  )

  kml.plot(config=config)

  # Verify the plot was created
  assert output_file.exists(), "Plot file should be created"

  # Verify the labels are actually set
  ax = plt.gca()
  assert ax.get_title() == custom_title, f"Title should be '{custom_title}'"
  assert ax.get_xlabel() == custom_xlabel, f"X-label should be '{custom_xlabel}'"
  assert ax.get_ylabel() == custom_ylabel, f"Y-label should be '{custom_ylabel}'"

  plt.close()

  print("✓ test_plot_with_custom_title_labels passed")


def test_plot_with_custom_figsize():
  """
  Test that custom figure size can be set.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_custom_figsize.pdf"
  custom_figsize = (10, 6)

  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    create_figure=True,
    close_figure=False,
    figsize=custom_figsize,
  )

  kml.plot(config=config)

  # Verify the plot was created
  assert output_file.exists(), "Plot file should be created"

  # Verify the figure size is correct
  fig = plt.gcf()
  actual_figsize = fig.get_size_inches()
  assert np.allclose(actual_figsize, custom_figsize), \
    f"Figure size should be {custom_figsize}, got {actual_figsize}"

  plt.close()

  print("✓ test_plot_with_custom_figsize passed")


def test_plot_exclude_nominal():
  """
  Test plotting without the nominal line.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_exclude_nominal.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    create_figure=True,
    close_figure=False,
    include_nominal=False,
  )

  results = kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"

  # The nominal curve should still be in results even if not plotted
  assert "nominal" in results, "Results should contain 'nominal' key"

  # Verify that the nominal line is not in the plot
  # Check the number of lines - should have fewer without nominal
  ax = plt.gca()
  lines = ax.get_lines()
  # With include_nominal=False, there should be no line labeled as nominal
  line_labels = [line.get_label().lower() for line in lines]
  has_nominal = "kaplan-meier" in line_labels or any("nominal" in label for label in line_labels)
  assert not has_nominal, "Nominal line should not be plotted when include_nominal=False"

  plt.close()

  print("✓ test_plot_exclude_nominal passed")


def test_plot_include_binomial_only():
  """
  Test plotting with binomial-only error bands.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_binomial_only.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    include_binomial_only=True,
  )

  results = kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  assert "x" in results, "Results should contain 'x' key"
  print("✓ test_plot_include_binomial_only passed")


def test_plot_include_patient_wise_only():
  """
  Test plotting with patient-wise-only error bands.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_patient_wise_only.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    include_patient_wise_only=True,
  )

  results = kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  assert "x" in results, "Results should contain 'x' key"
  print("✓ test_plot_include_patient_wise_only passed")


def test_plot_no_legend():
  """
  Test plotting with legend_loc=None (implementation may vary).
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_no_legend.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    legend_loc=None,  # Request no legend
  )

  # Test that the config is accepted and plot is created
  kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  # Note: The actual legend behavior may depend on the implementation
  # This test verifies that legend_loc=None is accepted as a valid config

  print("✓ test_plot_no_legend passed")


def test_plot_with_xmax_and_custom_options():
  """
  Test combining xmax with other custom plot options.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_xmax_custom_options.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    xmax=4.0,
    title="Survival Analysis (Limited Range)",
    xlabel="Time (months)",
    ylabel="Survival Rate",
    figsize=(8, 6),
  )

  results = kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  assert "x" in results, "Results should contain 'x' key"
  print("✓ test_plot_with_xmax_and_custom_options passed")


def test_plot_different_output_formats():
  """
  Test that plots can be saved in different formats.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Test PDF (already tested elsewhere, but for completeness)
  pdf_file = here / "test_output" / "test_format.pdf"
  config_pdf = KaplanMeierPlotConfig(
    saveas=pdf_file,
    show=False,
    print_progress=False,
  )
  kml.plot(config=config_pdf)
  assert pdf_file.exists(), "PDF file should be created"

  # Test PNG
  png_file = here / "test_output" / "test_format.png"
  config_png = KaplanMeierPlotConfig(
    saveas=png_file,
    show=False,
    print_progress=False,
  )
  kml.plot(config=config_png)
  assert png_file.exists(), "PNG file should be created"

  print("✓ test_plot_different_output_formats passed")


def test_plot_returns_expected_structure():
  """
  Test that plot results contain all expected keys and data structures.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  config = KaplanMeierPlotConfig(
    show=False,
    print_progress=False,
  )

  results = kml.plot(config=config)

  # Check that essential keys are present
  assert "x" in results, "Results should contain 'x' key"
  assert "nominal" in results, "Results should contain 'nominal' key"

  # Check that x and nominal are array-like
  assert len(results["x"]) > 0, "x should have data points"
  assert len(results["nominal"]) > 0, "nominal should have data points"

  # Check that x and nominal have the same length
  assert len(results["x"]) == len(results["nominal"]), \
    "x and nominal should have the same length"

  print("✓ test_plot_returns_expected_structure passed")


def test_plot_with_no_tight_layout():
  """
  Test plotting with tight_layout disabled.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_no_tight_layout.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    tight_layout=False,
  )

  kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  print("✓ test_plot_with_no_tight_layout passed")


def test_xmax_with_different_error_bands():
  """
  Test that xmax works correctly with different error band configurations.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  # Test xmax with binomial-only
  output_file1 = here / "test_output" / "test_xmax_binomial.pdf"
  config1 = KaplanMeierPlotConfig(
    saveas=output_file1,
    show=False,
    print_progress=False,
    xmax=3.5,
    include_binomial_only=True,
  )
  kml.plot(config=config1)
  assert output_file1.exists(), "Plot with xmax and binomial-only should be created"

  # Test xmax with patient-wise-only
  output_file2 = here / "test_output" / "test_xmax_patient_wise.pdf"
  config2 = KaplanMeierPlotConfig(
    saveas=output_file2,
    show=False,
    print_progress=False,
    xmax=3.5,
    include_patient_wise_only=True,
  )
  kml.plot(config=config2)
  assert output_file2.exists(), "Plot with xmax and patient-wise-only should be created"

  print("✓ test_xmax_with_different_error_bands passed")


def test_plot_with_custom_font_sizes():
  """
  Test plotting with custom font sizes.
  """
  dcfile = datacards / "simple_km_few_deaths.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)
  kml = datacard.km_likelihood(parameter_min=-np.inf, parameter_max=np.inf)

  output_file = here / "test_output" / "test_custom_fonts.pdf"
  config = KaplanMeierPlotConfig(
    saveas=output_file,
    show=False,
    print_progress=False,
    title="Test Plot",
    title_fontsize=16,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
  )

  kml.plot(config=config)

  assert output_file.exists(), "Plot file should be created"
  print("✓ test_plot_with_custom_font_sizes passed")


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
  test_plot_with_custom_title_labels()
  test_plot_with_custom_figsize()
  test_plot_exclude_nominal()
  test_plot_include_binomial_only()
  test_plot_include_patient_wise_only()
  test_plot_no_legend()
  test_plot_with_xmax_and_custom_options()
  test_plot_different_output_formats()
  test_plot_returns_expected_structure()
  test_plot_with_no_tight_layout()
  test_xmax_with_different_error_bands()
  test_plot_with_custom_font_sizes()

  print("\n✅ All KM plotting tests passed!")
