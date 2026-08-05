# CropFusion Kaggle experiment packaging.
#
# This setup.py exists so Kaggle notebooks can install the training package as
# an editable dependency: `pip install -e ./training/kaggle`.
#
# NOTE: placeholder only - the Kaggle entrypoint is defined in
# training/kaggle/notebooks/. The canonical package metadata is managed by the
# individual training modules' pyproject.toml files.

from setuptools import setup

setup(
    name="cropfusion-kaggle",
    version="0.0.0",
    description="CropFusion Kaggle experiment scaffolding (placeholder)",
    packages=[],
    python_requires=">=3.12",
)
