"""
ROC Picker is a software package for propagating statistical and systematic
uncertainties in a biomedical analysis.

Please see the docs/ folder for full documentation.
"""
import setuptools

setuptools.setup(
  name = "roc_picker",
  packages = setuptools.find_packages(include=["roc_picker"]),
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
      "rocpicker_mc=roc_picker.datacard:plot_systematics_mc_roc",
      "rocpicker_discrete=roc_picker.datacard:plot_discrete_roc",
      "rocpicker_delta_functions=roc_picker.datacard:plot_delta_functions_roc",
      "kombine=roc_picker.datacard:plot_km_likelihood",
    ],
  }
)
