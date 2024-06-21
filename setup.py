import setuptools

setuptools.setup(
  name = "roc_picker",
  packages = setuptools.find_packages(include=["roc_picker"]),
  author = "Heshy Roskes",
  author_email = "heshyr@gmail.com",
  install_requires = [
    "matplotlib",
    "numpy",
    "scipy",
  ],
  entry_points = {
    "console_scripts": [
      "rocpicker_mc=roc_picker.datacard:plot_systematics_mc",
    ],
  }
)
