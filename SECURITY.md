# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly:

1. **Do not** open a public issue.
2. Email the maintainers with a description of the vulnerability, steps to
   reproduce, and any potential impact.
3. You will receive an acknowledgement within 72 hours.
4. We will work with you to understand the issue and coordinate a fix before
   any public disclosure.

## Scope

This is a simulation tool. It does not process personal data, handle
authentication credentials, or communicate over a network in its default
configuration.

Security concerns relevant to this project include:

- **Numerical stability**: Malformed inputs could cause NaN/Inf propagation
  (mitigated by guards in `sim/core/integrator.py`).
- **Dependency vulnerabilities**: Third-party packages (NumPy, SciPy) may
  contain vulnerabilities. Run `pip audit` periodically.
- **Export control awareness**: See `docs/eu-legal-validation.md`, Section 5
  for dual-use regulation analysis. Do not add classified or export-controlled
  parameters to this repository.

## Regulatory Context

This project has been assessed against EU regulations including the Cyber
Resilience Act (EU) 2024/2847 and NIS2 Directive (EU) 2022/2555. As
non-commercial open-source software under MIT license, it is currently exempt
from CRA obligations (Recital 18). See `docs/eu-legal-validation.md` for the
full analysis.
