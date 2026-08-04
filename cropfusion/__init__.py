"""CropFusion - umbrella package.

CropFusion is a precision-agriculture decision support platform. It bundles:

* :mod:`ai` - multimodal crop-yield models, preprocessing, training, explainability
* :mod:`services` - dataset manager + spatial temporal alignment manager
* :mod:`quality` - drift, fairness, monitoring and optimization tooling
* :mod:`backend` - FastAPI modular monolith (API, enterprise database, MLOps)
* :mod:`frontend` - React + TypeScript single-page application

Installing this umbrella package is optional; each sub-package is installed
independently from source (see Makefile / environment.yml).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
