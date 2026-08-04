# Contributing

Thanks for contributing to CropFusion! Please read this guide and our
[Code of Conduct](../CODE_OF_CONDUCT.md).

## Getting started

1. Fork the repository and create a branch from `main`.
2. Set up the environment ([INSTALLATION.md](INSTALLATION.md)).
3. Install pre-commit tooling if you use it; otherwise run `make lint`,
   `make typecheck` and the tests locally.

## What we accept

- Bug fixes with a failing-then-passing test.
- New features following the modular-monolith pattern
  ([DEVELOPMENT.md](DEVELOPMENT.md)).
- Documentation improvements and translations.
- Benchmark/performance improvements backed by data.

## Before opening a PR

- [ ] Code passes `make lint`, `make typecheck`, `make security`.
- [ ] Tests pass: `pytest` (or the affected package suites) and `npm test`.
- [ ] Compose files still validate (`make compose`) if deployment touched.
- [ ] Documentation updated where behaviour changed.
- [ ] No secrets, credentials, or `.env` files in the diff.

## Pull requests

Use the [PR template](../.github/PULL_REQUEST_TEMPLATE.md). Reference the
issue your PR closes. CI runs automatically on push/PR.

## Commit messages

Keep them concise and imperative: "Add X", "Fix Y", "Refactor Z".

## Release process

Releases are cut from `main` by tagging `v<semver>`. The `Release` workflow
runs the full test gate, packages the source tarball, and creates a GitHub
release. See `.github/workflows/release.yml`.

## Questions

Open a discussion or ask in an issue. Security reports must go through
[SECURITY.md](../SECURITY.md) - never post vulnerability details publicly.
