# CropFusion Kaggle experiment packaging.
#
# This setup.py exists so Kaggle notebooks can install the Kaggle Training
# Infrastructure as an editable dependency: `pip install -e ./training/kaggle`.
# It declares the `training.kaggle` namespace package (config, environment,
# logging, workspace, checkpoint, cache, validation, reports) as well as the
# sibling training packages used for orchestration.
#
# The canonical package metadata lives in the individual training modules'
# pyproject.toml files.

from setuptools import find_namespace_packages, setup

setup(
    name="cropfusion-kaggle",
    version="0.1.0",
    description="CropFusion Kaggle Training Infrastructure (R2.1)",
    package_dir={"": ".."},
    packages=find_namespace_packages(
        "..",
        include=["training.kaggle", "training.kaggle.*"],
        exclude=["*.tests", "*.tests.*"],
    ),
    install_requires=[
        "pydantic>=2.5",
        "PyYAML>=6.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "torch>=2.3",
        "scikit-learn>=1.5",
        "rasterio>=1.3",
        "psutil>=5.9",
    ],
    python_requires=">=3.10",
)
