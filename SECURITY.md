# Security policy: 6dof-ascent-sim

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/rmednitzer/6dof-ascent-sim/security/advisories/new).
Do not post exploit payloads, secrets, personal data or sensitive operational
details in public issues. If private reporting is unavailable, use an existing
maintainer security contact where published; otherwise request a private channel
with a minimal, non-sensitive issue. Do not assume an invented email address works.

Include the affected revision/version and paths, prerequisites and trust boundary,
expected versus actual behavior, a minimal sanitized reproduction, impact and any
workaround. Use synthetic data. A dependency issue affecting this project is in
scope: coordinate upstream as appropriate without delaying notification here.
This policy does not authorize testing third-party or live deployments, service
disruption, credential use, persistence or access to other people's data.

Existing response targets are retained below as maintainer goals, not guaranteed
service levels. Coordinate disclosure and advisory timing with the maintainer;
urgent active exploitation or credential exposure should be identified immediately.

## Contributor security requirements

Follow [AGENTS.md](AGENTS.md) for secure coding, dependency/CI integrity, verification,
STPA and threat-analysis requirements. Preserve deployment trust boundaries and
document exceptions, residual risk and recovery. Never suppress a security finding
merely to pass CI. For fixes, capture a reproducer, add a meaningful regression,
check adjacent paths, assess affected versions and coordinate remediation/advisories.
Code review and CI do not prove deployed configuration or legal compliance.

## Support and response targets

Only main receives security fixes. Existing acknowledgement target: 72 hours.

## Scope and limits

Report unsafe parsing/path handling, resource exhaustion, dependency exposure,
telemetry or export integrity problems, and numerical failures that can invalidate
documented simulation results. Review non-finite state, configuration validation,
parallel jobs and SLURM boundaries as well as the static telemetry viewer.
The simulator is research software; simulated FTS and boundary checks do not
establish suitability for physical flight, operational control or safety certification.
Keep model assumptions and STPA constraints in docs/stpa-analysis.md and
docs/assumptions.md linked to tests. Do not publish restricted or sensitive inputs.
