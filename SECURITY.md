# Security Policy

## Supported Versions

Security fixes are backported to the latest stable release and the two most
recent minor releases.

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately by emailing the maintainers or opening a
[GitHub Security Advisory](https://github.com/{owner}/cropfusion/security/advisories/new).

Please include:

- The affected component and version.
- A minimal reproduction or proof-of-concept.
- The impact you believe the issue has.

We aim to acknowledge reports within 72 hours and will coordinate disclosure
once a fix is released.

## Security practices

- Dependency scanning runs weekly (Safety, Trivy, `npm audit`) via
  `.github/workflows/security.yml`.
- Static analysis runs on every PR (Bandit, CodeQL, ESLint).
- Secrets are never stored in the repository; use the `secrets/` directory
  (git-ignored) or the platform secret store in production.
- The production stack terminates TLS at the edge (Caddy), enforces
  non-root containers, and pins image versions.
