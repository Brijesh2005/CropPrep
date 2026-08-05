"""CropFusion - umbrella package.

CropFusion is a precision-agriculture decision support platform. The
repository is organised around two future independent platforms plus shared
code:

* ``training`` - CropFusion Training Platform (research, training,
  experiments, model export): dataset manager, STAM, preprocessing,
  models, training engine, explainability, MLOps and quality tooling.
* ``application`` - CropFusion Prediction Platform (prediction, farmer
  application, inference only): FastAPI backend, React frontend, database,
  GIS, monitoring and Docker assets.
* ``shared`` - reusable project-wide assets (schemas, DTOs, enums,
  interfaces, validation models, utilities, exceptions, configuration
  models, constants, serialization).

Installing this umbrella package is optional; each sub-package is installed
independently from source (see Makefile / environment.yml).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
