# Audit 02 — Security and Code-Quality Findings Register

Date: 2026-06-12
Phases: 2 (security, read-only) and 3 (code quality, read-only).
Every finding cites evidence from a command run this session. Findings that were
investigated and found to be non-issues are recorded under "Investigated and
dismissed" so the negative result is not silently lost.

Severity scale: critical / high / medium / low / info.
Effort: S (hours) / M (a day) / L (multi-day or structural).
Prefixes: `S-` security, `Q-` quality, `D-` documentation drift.

## Methodology and tools actually run

| Check | Tool / command | Outcome |
|-------|----------------|---------|
| Dependency CVEs | `pip-audit` 2.10.1 (venv) | **No known vulnerabilities** |
| Secret scan (history) | `gitleaks detect --source .` | **No leaks**, 57 commits scanned |
| Dangerous sinks | `grep` for `eval|exec|pickle|subprocess|os.system|yaml.load|__import__|marshal|input(` | **none present** |
| Crypto/random | `grep` for `random|secrets|hashlib|md5|sha1` | `secrets.randbits(128)` for sensor seeding; SHA-256 for telemetry integrity; no MD5/SHA-1 |
| File I/O surface | `grep` for `open(|json.load|Path(|savefig|write(` | **output-only**; no parsing of untrusted input |
| SAST (semgrep) | not available in environment | substituted with manual OWASP-relevant pass below |
| Type check | `mypy sim/` (not in toolchain) | informational only |

semgrep, trufflehog were not available; a manual review covering the OWASP
categories relevant to a numerical CLI tool (injection, deserialization, SSRF,
path traversal, unsafe defaults, crypto misuse) was performed instead.

## External input surface enumeration

| Surface | Location | Validation / assessment |
|---------|----------|-------------------------|
| CLI flags `--no-flex`, `--no-slosh` | `sim/main.py:728-729` | boolean `store_true`; no injection surface |
| CLI args `--runs/--seed/--workers` | `sim/montecarlo/dispatcher.py:148-150` | `type=int`; argparse coerces/rejects |
| Environment variables | none | the code reads no env vars |
| Network listeners / clients | none | no sockets, no HTTP, no SSRF surface |
| File parsers (read) | none | no untrusted input is read; the sim only **writes** JSON/PNG |
| File writes | `output/`, `output/plots/` (recorder, dispatcher, postflight, statistics) | fixed/config-driven paths; no user-controlled path component, so no path-traversal vector |
| Deserialization | none | no pickle/marshal/yaml.load; JSON is only emitted, never loaded from untrusted source |
| `config_override` dict | `sim/main.py:_apply_overrides` | trusted in-process API (Monte Carlo); only sets attributes that already exist on `config`; keys starting with `_` are skipped |

Conclusion: the attack surface is intrinsically small. There is no network, no
auth, no persistence of untrusted data, and no deserialization. The residual
security findings below are supply-chain / configuration hardening, not
exploitable code defects.

## Findings register

| ID | Sev | Title | File:line | Effort | Disposition |
|----|-----|-------|-----------|--------|-------------|
| S-04 | low | No dependency lockfile; deps pinned only by lower-bound floors | `pyproject.toml:22-26` | M | backlog |
| S-05 | low | CI ruff (0.9.7) and pre-commit ruff (0.15.16) disagree despite a comment claiming they match | `.github/workflows/ci.yml:28` vs `.pre-commit-config.yaml:14` | S | **fixed** (Phase 4) |
| S-06 | info | Renovate `lockFileMaintenance` is a no-op (no lockfile exists) | `renovate.json5` | S | backlog (tie to S-04) |
| S-07 | info | No CodeQL workflow file in-repo despite commit `9a2e67e`; default-setup status unverifiable | `.github/workflows/` | S | backlog |
| S-08 | low | SECURITY.md tells users to run `pip audit` (not a real subcommand; tool is `pip-audit`) | `SECURITY.md` | S | **fixed** (Phase 5, D-05) |
| Q-01 | medium | Telemetry `health_status` is permanently `NOMINAL`: recorder reads a non-existent `.status` attribute | `sim/telemetry/recorder.py:217,276` | S | **fixed** (Phase 4) |
| Q-02 | low | `_save_config()` hardcoded key list duplicates the dispersion set; silent drift risk | `sim/main.py:86-103` | S | backlog |
| Q-03 | low | HealthMonitor `engine_health` / `sensor_status` channels are never fed real data by the main loop | `sim/main.py:438-443` | M | backlog |
| Q-04 | medium | No end-to-end test of `run_simulation`; `recorder` untested (15% / 28% coverage) | `sim/main.py`, `sim/telemetry/recorder.py` | M | partially addressed (Q-01 test); backlog |
| Q-05 | low | PEG uses `try/except UnboundLocalError: pass` as control flow; silently retains stale coefficients | `sim/gnc/guidance.py:463-467` | S | backlog |
| Q-06 | info | Dead `_v_p` (vis-viva) computed then suppressed with `# noqa: F841` | `sim/orbital/decay.py:231` | S | backlog |
| Q-07 | info | Asymmetric yaw sign `cmd_yaw_rad = -(...)` undocumented | `sim/gnc/control.py:177` | S | backlog |
| D-01 | low | FTS docstring documents quaternions as `[w, x, y, z]`; project + code are scalar-last `[x, y, z, w]` | `sim/safety/fts.py:93-94` | S | **fixed** (Phase 5) |
| D-02 | low | Atmosphere comments say "200 km" but exosphere ceiling is 1000 km | `sim/environment/atmosphere.py:205,210` | S | **fixed** (Phase 5) |
| D-03 | low | Gain-schedule docstring formula uses `mass/mass_ref`; code uses `sqrt(mass/mass_ref)` and undocumented q regimes | `sim/gnc/control.py:49` | S | **fixed** (Phase 5) |
| D-04 | low | Runbook lists `TVC_MAX_SLEW_RATE_DEG_S (10.0)`; config is 20.0 | `docs/runbook.md:113` | S | **fixed** (Phase 5) |

### Detail: Q-01 (medium) — health status never reaches telemetry

`sim/telemetry/recorder.py` lines 217 and 276 read
`getattr(health_monitor, "status", "NOMINAL")`. `HealthMonitor`
(`sim/safety/health_monitor.py`) exposes `health` (a `HealthVector`) and the
method `overall_status()`, but **no `status` attribute** — so the `getattr`
always falls through to the `"NOMINAL"` default. The recorder's own docstring
even specifies "health_monitor: Object exposing `status` (str)", an interface
`HealthMonitor` does not satisfy.

Evidence (commands run):
- `grep` shows the only `.status` references are the two `getattr` calls; the
  class has no such attribute and no test references it.
- A full nominal run was inspected:
  `python -c "...set(f['health_status'] for f in telemetry_internal.json)"`
  -> `{'NOMINAL'}` across **48,816 frames**, even though peak dynamic pressure
  reached 92.6% of the structural limit (the health monitor's WARNING threshold
  is 80%).

Impact: the health-monitoring subsystem computes a result every step
(`main.py:438`) that is silently discarded; telemetry consumers and the mission
summary cannot see WARNING/ALERT/CRITICAL states. Not a security vulnerability
(local, output-only), but a real correctness defect in a safety-adjacent signal.

Exploit plausibility: n/a (no external attacker; integrity hash still covers
whatever is written).

Recommended fix (applied in Phase 4): add a `status` property to `HealthMonitor`
returning `self.overall_status().name`, satisfying the recorder's documented
interface with zero changes to call sites; add a regression test.

### Detail: S-04 (low) — no lockfile / unpinned dependencies

`pyproject.toml` declares `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7` and dev
tools with `>=` floors, and there is no `requirements*.txt` / lockfile (verified
by `ls`). Installs therefore resolve to whatever is latest at install time
(this session pulled numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.0). This is a
reproducibility and supply-chain-drift risk: a future transitive release could
change numerical results or introduce a regression with no pinned baseline to
diff against. Renovate is configured but `lockFileMaintenance` has nothing to
maintain (S-06).

Recommended approach (backlog): add a constraints/lock file (e.g.
`requirements.lock` via `pip-compile`, or commit `pip freeze` of the CI matrix)
and have CI install from it; let Renovate bump it. Deferred because it is a
policy decision affecting reproducibility vs. update cadence, not a local fix.

## Investigated and dismissed (negative results — do not re-flag)

These were raised as candidates (some by an automated module sweep) and
**disproved** with commands this session:

- **gravity.py J3 z-coefficient (`21.0`) "should be 30.0".** Disproved. The full
  J2-J6 model was checked against a numerical gradient of the geopotential built
  directly from Legendre polynomials (`scipy.special.eval_legendre`): relative
  error 3e-6 to 6e-6 at latitudes 0/28.5/45/60/81/89 deg — i.e. the
  finite-difference floor. A wrong odd-in-z J3 term would diverge at high
  latitude; it does not. **The coefficient is correct; do not change it.**
- **decay.py King-Hele "missing periapsis velocity `V_p`, off by ~7-8x".**
  Disproved by derivation: for a circular orbit da/dt = -a^2 rho v^3/(mu*BC), and
  integrating over one period T = 2*pi*sqrt(a^3/mu) yields da/rev = -2*pi*a^2*rho/BC
  exactly — the `V_p` dependence is absorbed into the `a^2` coefficient. The code
  at `decay.py:248` matches this for e=0 (I0(0)=1, exp(0)=1). The docstring sketch
  mentions `V_p`, but the implemented per-revolution form is correct. The only
  real issue is the dead `_v_p` variable (Q-06).
- **propagator.py mypy `union-attr` (lines 303-311).** False positive: `_elements`
  is narrowed by the `state_to_elements()` call inside `orbit_summary`; mypy is
  not in the toolchain.

## Positive controls (checked, clean)

- No vulnerable dependencies (`pip-audit`).
- No secrets in working tree or 57-commit history (`gitleaks`).
- No dangerous execution/deserialization sinks anywhere in `sim/`.
- GitHub Actions are SHA-pinned; CI uses least-privilege `permissions: contents: read`.
- No container/IaC files, so no root-user/`latest`-tag/secret-ARG class of issues.
- Integrator guards NaN/Inf and re-normalizes the quaternion every step
  (`sim/core/integrator.py:136-147`), matching the SECURITY.md mitigation claim.
- JSON is emitted with `allow_nan=False` and a NaN->null sanitizer, so a diverging
  run cannot produce unparseable telemetry (`sim/telemetry/recorder.py:26-40`).
