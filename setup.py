"""
ROC Picker is a software package for propagating statistical and systematic
uncertainties in a biomedical analysis.

KoMbine provides Kaplan-Meier curve analysis functionality.

Please see the docs/ folder for full documentation.
"""
import setuptools

setuptools.setup(
  name = "roc_picker",
  packages = setuptools.find_packages(include=["roc_picker", "kombine"]),
  author = "Heshy Roskes",
  author_email = "heshyr@gmail.com",
  install_requires = [
    "gurobipy",
    "matplotlib",
    "numpy",
    "scipy>=1.15",
  ],
  entry_points = {
    "console_scripts": [
      "rocpicker_mc=roc_picker.command_line_interface:plot_systematics_mc_roc",
      "rocpicker_discrete=roc_picker.command_line_interface:plot_discrete_roc",
      "rocpicker_delta_functions=roc_picker.command_line_interface:plot_delta_functions_roc",
      "kombine=kombine.command_line_interface:plot_km_likelihood",
      "kombine_twogroups=kombine.command_line_interface:plot_km_likelihood_two_groups",
    ],
  }
)
