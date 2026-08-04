# Roadmap

The following are candidate areas of work. Items are not commitments; they
reflect the direction of the project and are prioritised by community demand.

## Near term

- **Model deployment hardening**: integrate a real model artifact registry
  backend (e.g. MLflow) as an optional drop-in for the filesystem registry.
- **Production dataset pipeline**: scheduled dataset refresh + drift-driven
  retraining automation.
- **UI refinements**: multi-language support (localisation), improved map
  interactions, batch prediction workflows.
- **Performance**: optional GPU inference image with TensorRT support.

## Medium term

- **Kubernetes manifests**: Helm chart + Kustomize for managed-K8s
  deployment, mirroring the Docker Compose topology.
- **Federated/aggregated spatial analysis**: region-wide yield forecasting
  with uncertainty quantification.
- **Explainability v2**: counterfactual explanations and causal attributions.
- **Synthetic data generation** for underrepresented regions/season.

## Longer term

- **Multi-tenant enterprise tier** with organisation workspaces and billing.
- **Mobile companion app** (offline-first, low-connectivity regions).
- **IoT integration**: live weather/sensor streams into the temporal encoder.
- **Carbon & sustainability analytics** over the same spatial pipeline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Feature requests are welcome via the
feature-request template.
