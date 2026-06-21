# Audit 05 — Cross-Repository Lessons (Comparative Deep-Audit)

Date: 2026-06-21
Repository under audit: `rmednitzer/6dof-ascent-sim`
Comparison corpus: `PX4-Autopilot`, `genesis-world`, `quadrants`, `genesis-nyx`

## Scope and question

This is **not** a defect audit of this repo (those are `audit/02` and `audit/04`).
The question here is outward-looking: *what proven patterns can this simulation
adopt from four sibling repositories?* Each sibling is a domain leader:

| Repo | What it is | Why it is a useful teacher |
|------|-----------|----------------------------|
| **PX4-Autopilot** | Production safety-critical flight-control firmware (C/C++) | The real-world reference for everything in `sim/gnc/` and `sim/safety/` — estimation, control allocation, failsafe |
| **genesis-world** | Multi-physics simulation platform (Python) | A Python physics simulator that scaled to GPUs and thousands of parallel environments |
| **quadrants** | Taichi-derived physics compiler (C++/Python) | Exemplary CI/quality automation, coverage, packaging, and contributor discipline |
| **genesis-nyx** | GPU renderer plugin (docs + examples + wheels) | A reference for documentation, examples-as-canonical-docs, and packaging hygiene |

## Method and validation discipline

Four read-only agents each mined one sibling repo for transferable patterns and
validated their notable claims against trusted external sources (PX4/ECL docs and
the Sola 2017 error-state-EKF derivation; the coverage.py/pytest-cov and Numba
docs; Sphinx `literalinclude`/autodoc docs; numerical-methods references). In
parallel, **every recommendation was reconciled against this repo's actual
source** — `sim/gnc/navigation.py`, `control.py`, `sim/core/{integrator,state}.py`,
`sim/main.py`, the full `sim/safety/` module, `sim/config.py`, the CI workflows,
the ADRs, and `docs/assumptions.md`.

That reconciliation matters: a description-only reading of this repo produces
wrong advice. Several "high priority" patterns a teacher repo suggests are
**already implemented here correctly**, and recommending them would repeat the
very anti-pattern `audit/03` warned against ("No correct code was changed on the
strength of an unverified claim"). Sibling-repo claims were also spot-checked
directly (e.g., genesis-world's `Options` base class really is
`pydantic.BaseModel`; quadrants' AI CI workflows and `tests/coverage_report.py`
really exist; PX4's `ekf2/EKF/covariance.cpp`, `commander/failsafe`,
`lib/control_allocation`, 220 `.msg` files, and per-module param YAML all exist).

## Already correct here — do NOT "fix" these

PX4-shaped advice that this repo **already satisfies** (cited so future work does
not regress or waste effort re-doing them):

- **Joseph-form covariance update.** `navigation.py:_apply_update` already uses
  `P = (I−KH)P(I−KH)ᵀ + KRKᵀ` and re-symmetrises. `audit/04` lists this as a
  verified positive control. *Do not "upgrade" to Joseph form — it is here.*
- **FTS latching.** `fts.py` latches irrevocably (`if self.state.fts_triggered:
  return True`; `_trigger` sets a permanent flag). A reset-able abort flag would
  be a regression.
- **Ordered-severity health with worst-wins.** `health_monitor.py` already has
  `HealthStatus(IntEnum)` NOMINAL<WARNING<ALERT<CRITICAL and
  `overall_status() = max(channels)`.
- **Geometric quaternion attitude error.** `control.py` computes
  `q_err = conj(q_est)·q_des`, forces the short path (`if q_err[3]<0: q_err=-q_err`),
  and uses the vector part (`2·q_err[1]`, `2·q_err[2]`) — the small-angle form of
  PX4's `2·canonical(q_err).imag()`. `fts.py` measures thrust-axis error as the
  true inter-axis angle, not an Euler difference.
- **Guarded normalization.** `state.normalize_quaternion` guards `norm > 1e-10`;
  the integrator raises on non-finite states instead of emitting NaN silently.
- **Quaternion convention is documented** (scalar-last `[x,y,z,w]`) in `CLAUDE.md`,
  `state.py`, and ADR-0002.

The genuine gaps are below.

---

# A. PX4-Autopilot — the keystone (GNC, estimation, safety)

PX4 is where a hobby-tier GNC stack has the most to gain. Findings are grouped by
subsystem; each gives PX4's pattern (with a file anchor), the **reconciled** state
of this repo, and the concrete lesson.

## A1. State estimation (`sim/gnc/navigation.py`)

### A1.1 The EKF does not estimate attitude at all — *the headline finding*

PX4's EKF2 is fundamentally an **error-state attitude filter**: attitude lives in
the state as a 3-DOF rotation error and is corrected **multiplicatively**
(`δq = AxisAngle(K·innov); q ← δq·q; normalize`), with Jacobians derived
symbolically (Symforce) in `src/modules/ekf2/EKF/python/ekf_derivation/`.

**This repo:** the "12-state EKF" estimates position/velocity/accel-bias/gyro-bias
**but not attitude** — the *true* quaternion and body rate are injected via
`navigation.set_attitude(true_state.quaternion, true_state.angular_velocity_body)`
every step (`main.py:419`). `docs/assumptions.md` is admirably honest about this:
"Attitude is not estimated by the EKF … attitude determination errors are not
modeled." So the filter sidesteps the hardest and most safety-relevant part of
strapdown INS, and the FTS covariance/health checks never see attitude
uncertainty.

**Lesson (highest value, highest effort):** add an attitude error-state to the
filter (quaternion in the nominal state, 3-DOF error in `P`, multiplicative
update) and feed it the *measured* gyro instead of truth. This converts the EKF
from a position smoother into a real navigation filter and makes the existing
attitude-based FTS/health limits meaningful under sensor error. Jacobians can be
generated symbolically in pure Python with `sympy` (the open-source core of the
Symforce approach PX4 uses). **Priority: HIGH · Effort: High.**

> Note this *supersedes* the agent's generic "switch additive→multiplicative
> quaternion update" advice — there is no quaternion in the state to update yet.

### A1.2 Innovation gating is a diagonal magnitude test, not a χ² (NIS) test

PX4 gates on the normalized innovation squared `innov²/(gate²·S)` and rejects when
the ratio exceeds 1, **and** keeps a low-pass-filtered test ratio per source for
health reporting, and treats any non-finite innovation as a *named* fault.

**This repo:** `_apply_update` rejects the whole measurement if **any single
component** exceeds `threshold·√S[i,i]` — i.e. it uses only the diagonal of `S`
and an absolute per-axis magnitude, not the Mahalanobis distance `yᵀS⁻¹y`. This
ignores cross-covariance and has no persistence/filtering.

**Lesson:** replace the per-component test with a χ² test on `yᵀS⁻¹y` against the
chi-square quantile for the measurement dimension; keep an EMA of the test ratio
per sensor for telemetry; reject non-finite innovations as a counted fault rather
than relying on the integrator's downstream NaN guard. This is small and isolated.
**Priority: HIGH · Effort: Low.**

### A1.3 No covariance bounding or process-noise floors

PX4 never hard-clips a variance (that corrupts correlations); when a diagonal
grows too large it **fuses a synthetic zero-innovation measurement**
(`covariance.cpp`) to shrink it, and floors bias process noise so it cannot vanish.

**This repo:** covariance is only re-symmetrised; there is no upper/lower bound and
no conditioning beyond that. Under sustained GPS loss above 60 km (COCOM) the
position variance grows unbounded toward the FTS covariance limit with no
conditioning.

**Lesson:** if/when bounding is added, use the zero-innovation-fusion technique,
not a diagonal clip. **Priority: MED · Effort: Low.**

### A1.4 Bias states are unbounded and learn during high dynamics

PX4 hard-limits gyro/accel bias to physical ranges every update and **inhibits**
bias learning during high-acceleration / IMU-clipping epochs where biases are
unobservable, spiking process noise instead.

**This repo:** biases are pure random walks with no clamp; there is no clipping
concept. (`audit/04` AD-18 was a *related* symptom — IMU bias *dispersions* could
go negative and crash 52% of MC runs — fixed by truncation, but the in-filter
bias state itself is still unbounded.)

**Lesson:** clamp estimated biases to physical limits; inhibit accel-bias learning
during burns; add an IMU range/clipping flag that inflates process noise. The
inhibit logic also makes ascent estimation more stable (biases are nearly
unobservable under high specific force). **Priority: HIGH · Effort: Low.**

### A1.5 No sensor-delay horizon

PX4 runs the filter at a *delayed* time horizon using per-sensor timestamped ring
buffers, then an `OutputPredictor` forward-propagates to "now." This repo models a
one-step IMU lag (`prev_specific_force_eci`) but fuses GPS/baro **synchronously**,
ignoring their (much larger) latency. For a sim whose purpose is GNC validation,
GPS latency (50–200 ms) materially affects achievable bandwidth/phase margin.

**Lesson:** parameterize per-sensor latency and fuse from a short measurement
buffer. **Priority: MED · Effort: Med.**

### A1.6 GPS realism: quality gating + correlated noise

PX4 refuses to fuse GPS until EPH/EPV/SACC/NSATS/PDOP pass, and models GPS error
as a **Gauss-Markov** correlated process. This repo adds i.i.d. Gaussian GPS noise
and toggles availability purely on altitude — correlated drift is harder for an
estimator and a more honest dispersion. **Lesson:** add a first-order Gauss-Markov
GPS error (`err ← e^(−dt/τ)·err + σ√(1−e^(−2dt/τ))·N(0,1)`, τ≈100–300 s) and a
fix-quality model; disperse τ and σ. **Priority: MED · Effort: Low.**

## A2. Control (`sim/gnc/control.py`)

### A2.1 Anti-windup is bare clamping — no saturation feedback, no rate taper

PX4's rate controller uses **conditional integration**: the control allocator
reports per-axis saturation, and the integrator only accumulates error in the
*non-saturating* direction; it also tapers integral gain quadratically at large
rate errors. This repo clamps the integral term to `±CONTROL_INTEGRATOR_LIMIT_DEG`
but **keeps integrating even when the TVC is saturated** (the `BoundaryEnforcer`
clamps deflection/slew downstream, but the controller never learns it saturated).

**Lesson:** feed the boundary enforcer's `was_clamped`/saturation back into the
controller and stop integrating into the clamp; optionally add the I-factor taper.
This is directly relevant to the documented **AD-19** terminal-phase fragility,
where the PEG command rides the TVC slew/deflection limits near insertion.
**Priority: HIGH · Effort: Low.**

### A2.2 No explicit control-effectiveness/allocation, no rate loop, no roll

PX4 separates **torque setpoint → effectiveness matrix (pseudo-inverse) →
per-actuator command**, applies slew limits *after* allocation, and runs a
cascaded attitude→rate architecture. This repo maps attitude error straight to a
gimbal angle via lumped PID gains (pitch+yaw only; roll is uncontrolled —
`assumptions.md`), and inertia is a single scalar.

**Lesson:** even for a single-nozzle TVC vehicle, expressing the mapping as an
explicit effectiveness relation (moment-arm × deflection → torque, with the
inverse used for allocation) makes the model extensible to multi-engine/roll
configurations and makes the saturation budget explicit. A short rate-loop inner
stage would also harden the terminal transient. **Priority: MED · Effort: Med.**

## A3. Safety / failsafe (`sim/safety/`)

This repo's safety code is solid (latching FTS, ordered health severity, a
single-choke `BoundaryEnforcer`). PX4's `commander` adds four patterns it lacks —
and three of them bear directly on the dominant Monte-Carlo failure mode (AD-19).

### A3.1 Hysteresis-gated detection — *directly addresses AD-19*

PX4 requires a failure condition to **persist** (a `Hysteresis` time, e.g.
`FD_FAIL_P` seconds) before it trips, and adjusts references across EKF resets so a
reset delta is not read as a failure. This repo's FTS and structural checks trip on
a **single timestep** over the limit. `audit/04`/`BACKLOG` AD-19 observes that
~19% of dispersed runs FTS-abort and *every observed abort is a marginal 25.0x°
thrust-axis hit* right at `FTS_ATTITUDE_LIMIT_DEG`. A short hysteresis gate
(require N consecutive 100 Hz frames, or a debounce time) on the *attitude* FTS
criterion is a textbook mitigation for exactly that marginal-single-sample
population — without weakening the genuine loss-of-control case (a real divergence
persists for many frames). **Priority: HIGH · Effort: Low.** *(Pairs with AD-19;
must regenerate dispersed-MC stats and any telemetry hashes.)*

### A3.2 Per-event clear conditions (sticky vs auto-recover)

PX4 tags each event with a clear policy (`WhenConditionClears` /
`OnModeChangeOrDisarm` / `OnDisarm` / `Never`). This repo's health channels are
recomputed from scratch every step, so a WARNING/ALERT silently auto-clears the
instant the instantaneous condition passes — momentary excursions leave no trace.
**Lesson:** make non-FTS safety events latch until explicitly acknowledged (or at
least record peak/last-seen), so a transient overshoot during max-Q is visible
post-flight. **Priority: MED · Effort: Low.**

### A3.3 Phase-aware deferral of soft limits

PX4 can defer deferrable conditions during critical phases while hard limits
(geofence/structural) always bypass. This repo applies all checks uniformly. A
`defer_soft_safety` window around staging / max-Q / GT→PEG handover would stop
nominal high-dynamics transients from tripping advisory limits in MC — again
relevant to AD-19. **Priority: MED · Effort: Low.**

### A3.4 Pre-launch / arming gate on estimator health

PX4 will not arm with a diverged estimator (`estimatorCheck.cpp` gates on innovation
consistency). This repo has **no arming/preflight concept** — every run commits to
flight regardless of initial filter state. A T-0 "launch commit" predicate (EKF NIS
within a χ² envelope over a pre-launch window; sensors present) would catch bad
initializations in MC before they become aborts, and is a realistic range-safety
analogue. **Priority: MED · Effort: Low.**

### A3.5 Per-component health as (present, degraded, failed) + mode-relative commit

PX4 reports each component as three independent flags and makes arming
mode-relative (GPS loss blocks position modes, not stabilized). This repo's health
is one severity axis per channel. A `present/degraded/failed` triplet plus a
mode-aware commit (e.g., "nominal LEO" needs all; "ballistic safe" tolerates GPS
loss) is a modest extension. **Priority: LOW–MED · Effort: Med.**

## A4. Numerics, sensors, parameters, testing, architecture

- **A4.1 Quaternion integration via the exponential map.** PX4 integrates attitude
  with `q ← q·expq(½ω·dt)` (with a small-angle Taylor branch) — geometrically exact
  on SO(3). This repo integrates the quaternion ODE inside RK4 and renormalises
  (4th-order but drifts off the unit sphere between renormalisations). For the
  current rates this is fine; revisit only if spin/high-rate cases are added.
  **Priority: LOW · Effort: Low.**
- **A4.2 IMU clipping + baro aero-error.** Add a sensor range/clipping flag and a
  velocity-squared static-pressure bias on the barometer (significant through
  max-Q); disperse both. **Priority: MED · Effort: Low.**
- **A4.3 Parameter schema with units/bounds.** PX4 declares every parameter in
  per-module YAML with `type/default/min/max/unit/description`, generating typed,
  range-checked accessors. This repo's `config.py` is ~100 bare, unvalidated,
  unit-in-comment-only constants. *(Converges with genesis-world B3 and the repo's
  own ADR-0009 — see §E.)* **Priority: HIGH · Effort: Med.**
- **A4.4 EKF as an isolatable, replayable unit.** PX4's EKF core is uORB-free and
  driven by a `SensorSimulator` in ~25 fast unit tests, plus a ulog *replay*
  regression. This repo's `NavigationEKF` is *already* fairly decoupled (typed
  measurement inputs; `tests/test_ekf.py`; 38 isolated repros in `audit/04`) — so
  this is partly done. The genuine additions are (a) a synthetic-trajectory sensor
  driver for the filter, and (b) a **full-rate raw-sensor logging mode** so a
  recorded run can be replayed through the EKF as a regression baseline.
  **Priority: MED · Effort: Med.**
- **A4.5 Typed interfaces between subsystems (the uORB lesson).** PX4 modules share
  only message schemas (220 `.msg`), never function calls. This repo's `main.py` is
  an ~420-line loop wiring subsystems by direct calls and a global `config`. Define
  small typed dataclasses (`StateEstimate`, `GuidanceCommand`, `ControlOutput`,
  `SafetyStatus`) as the contracts. *(Converges with genesis-world B1.)*
  **Priority: MED · Effort: High.**

---

# B. genesis-world — Python physics-sim architecture & performance

genesis-world is the closest analogue (a Python physics simulator) and the best
teacher for *scaling* and *structure*.

### B1. Decompose the monolithic loop into a solver/subsystem list
genesis-world's `Simulator.step()` iterates a list of typed solvers behind one
interface (`base_solver.py`), with coupling handled separately; disabling a physics
domain is just dropping it from the active list. This repo's `main.py` hardcodes
the entire wiring and uses `--no-flex`/`--no-slosh` `if` flags inside the loop.
**Lesson:** extract a `Simulator` holding a list of subsystems with a common
`substep(state, dt)->StateDot` (or contribution) interface; `main.py` becomes a
thin harness. **Priority: HIGH · Effort: Med.** *(Same target as A4.5.)*

### B2. Vectorize Monte Carlo over a batch dimension (kill the global-config mutation)
genesis-world runs thousands of environments by adding one leading `n_envs`
dimension to every state array (`scene.build(n_envs=B)`); no per-run processes.
This repo's MC uses `multiprocessing.Pool` where each worker **mutates the global
`sim.config` module** and restores it (`main.py:_save_config/_apply_overrides`,
ADR-0004) — which is *why* it needs process isolation and a hand-maintained key
list (the latent Q-02 bug). **Lesson:** (1) immediate — make config an instance
passed into `run_simulation` (removing global mutation) so runs can share a process
safely; (2) higher-ceiling — add a `BatchVehicleState` of shape `(N, …)` and run N
trajectories under one vectorized RK4. The branchy control/staging/FTS logic makes
full vectorization non-trivial, so stage it: vectorize the environment/dynamics
hot path first, keep per-trajectory control in a thin loop. **Priority: HIGH ·
Effort: Med (config-as-object) → High (full batch).**

### B3. Validated, structured config (pydantic) — *verified in the sibling*
genesis-world's `Options` base class is literally `pydantic.BaseModel` with
`ConfigDict(strict=True, extra="forbid", validate_default=True)` and a custom
error formatter (`genesis/options/options.py`, confirmed by direct read). That
single setting (`extra="forbid"`) would have caught this repo's MC `config_override`
typos at dispatch time. **Lesson:** replace flat `config.py` with grouped,
typed config models (pydantic or dataclasses) carrying units/bounds; validate MC
overrides against the schema before launching workers. **Priority: HIGH · Effort:
Med.** *(This is the concrete "how" for the repo's own ADR-0009.)*

### B4. Numba-JIT the RK4 hot path + reuse scratch buffers
The compile-once/run-many mindset, applied honestly to a pure-NumPy sim: the
integrator allocates ~5 small arrays per sub-stage × 4 × ~60k steps/run. `@numba.njit
(cache=True)` on the derivative + step (state flattened to a float64 vector), plus
pre-allocated `k1..k4` scratch buffers, is a 10–50× single-run win with no
architecture change. *(Independently recommended by the quadrants audit — see C4.)*
**Priority: MED–HIGH (if MC runtime hurts) · Effort: Low–Med.**

### B5. Integrator-as-strategy + substepping
genesis-world exposes named integrators and a `dt`/`substeps` split. This repo's
RK4 is fixed at 100 Hz with **zero-order-hold forces across sub-stages** (ADR-0003;
honest first-order-in-force). **Lesson:** wrap `rk4_step` behind an `Integrator`
protocol; optionally evaluate forces per sub-stage (true RK4) or substep stiff
transients (staging/max-Q) while keeping GNC at 100 Hz. **Priority: MED · Effort:
Med.**

### B6. Testing patterns: golden trajectory + cross-validation oracle + tolerances
genesis-world validates step-by-step against **MuJoCo** as a reference oracle, uses
centralized tolerance constants, and an autouse init/reset fixture so global state
never leaks between tests. **Lessons for this repo:** (a) a committed **golden
nominal-trajectory** regression (`.npz`) asserted with `np.allclose` — the single
highest-ROI test for a numerical sim; (b) a **cross-check oracle** running a
nominal point-mass trajectory through `scipy.integrate.solve_ivp` and asserting the
6-DOF core agrees within tolerance; (c) centralized `TOL_*` constants; (d) an
autouse fixture that snapshots/restores `config` around every test (today's global
module can leak overrides between tests). *(Golden test independently recommended
by quadrants — C3.)* **Priority: HIGH · Effort: Med.**

### B7. A `SIM_FLOAT` dtype alias
One config-level dtype alias used in every `np.zeros(..., dtype=SIM_FLOAT)` makes a
future float32/GPU batch path a one-line switch. **Priority: LOW · Effort: Low.**

---

# C. quadrants — CI/quality automation, coverage, packaging, hygiene

quadrants will never be a dependency here, but its engineering process is the most
directly copyable of any sibling. (All workflow/coverage/AGENTS claims below were
verified by direct read.)

### C1. AI-driven PR quality gates
Six workflows run an LLM agent on each PR with a tight rubric and a hard
PASS/FAIL gate (`check_deleted_comments`, `check_feature_factorization`,
`check_test_coverage`, `check_wrapping`, `check_markup_links`, `pr_change_report`),
each batching with a "save AI cost" delay. The **deleted-comment** check is
especially apt for safety-critical numerical code, where comments encode *why* a
formula/convention was chosen (this repo's history is full of such rationale —
e.g., the AD-04 notch notes, the yaw-sign and aliasing comments). **Lesson:** adopt
the deleted-comment and AI test-coverage checks first (copy workflow, edit the file
globs/repo-layout in the prompt); they need a Claude/Cursor API key as a repo
secret. **Priority: HIGH (deleted-comment, AI coverage) / MED (factorization, PR
report) · Effort: 1–3 h each.**

### C2. Diff coverage + branch coverage + a real gate
quadrants computes **diff coverage** (only PR-changed lines) via
`tests/coverage_report.py`, posts it as a PR comment, and **fails under 80%**; it
collects with `--cov-branch`. This repo uploads `coverage.xml` as an artifact with
**no gate** (line coverage sits at ~55%, per `audit/03`), and does not measure
branch coverage — yet its riskiest code is branchy (staging FSM, FTS, health,
boundary enforcer). **Lessons:** add `--cov-branch`; port the diff-coverage script
and gate PRs on changed-line coverage (more honest than a frozen project %); add
`[tool.coverage.paths]` remapping. **Priority: HIGH · Effort: ~2 h + 30 m.**

### C3. Testing strategy
Beyond B6's golden test: parametrize physics edge-cases (e.g., atmosphere at every
ICAO layer boundary 0/11k/20k/32k/47k/86k m), add a numerical-derivative cross-check
on the aero/force Jacobian (finite-difference vs analytic), split slow
(full-trajectory, MC) tests behind a `slow` marker with `-m "not slow"` by default,
and add a precision-aware `sim_approx` helper. **Priority: HIGH (edge-cases) / MED
(rest) · Effort: ~½ day each.**

### C4. Performance harness
A tiny `BenchmarkPlan`→JSON harness keyed by commit hash gives per-commit
regression visibility for the RK4 loop and MC dispatcher; pair with the C-level
Numba advice (= B4). **Priority: LOW–MED · Effort: ~2 h.**

### C5. Type checking (resolves the repo's open T-03)
quadrants runs **pyright** in its own CI job via `pyrightconfig.json`, reading
stubs so it needs no compiled build. This repo's `BACKLOG` T-03 explicitly leaves
mypy "undecided." A minimal `pyrightconfig.json` over `sim/`+`tests/` plus a CI job
closes that item with a tool that handles NumPy better than mypy in practice
(catches wrong array kwargs, shape/return drift). **Priority: HIGH · Effort: ~2 h.**

### C6. Packaging & hygiene
- `setuptools_scm` for git-tag-derived `version` (so `sim.__version__` is real and
  can stamp telemetry/`MissionSummary`) — replaces the hand-bumped `0.1.0`.
- Split `[dev]` into `[test]`/`[dev]`/`[docs]` dependency groups.
- An **`AGENTS.md`** with machine-checkable rules in quadrants' exact "Code-review
  agents: flag PRs that…" style — e.g., "new physics constants go in the config
  models, not inline"; "new GNC/safety logic goes in its own module, not `main.py`";
  "changes to unit/quaternion conventions must update `CLAUDE.md`."
- `.git-blame-ignore-revs` (for the eventual mass-format commit); a scoped
  `pylint` allowlist for logic smells ruff misses.
**Priority: LOW–MED · Effort: minutes to ~1 h each.**

> Not transferable: kernel-level GPU coverage and the CFFI hard-timeout watchdog —
> noted only to be explicitly out of scope.

---

# D. genesis-nyx — documentation, examples, packaging

This repo already has strong reference-grade docstrings, an `examples/` dir, a
`docs/` tree, ADRs, and a Pages site — nyx shows how to make them *self-consistent
and auto-generated*.

### D1. `literalinclude`: examples ARE the docs
nyx's docs embed each runnable example verbatim via Sphinx `literalinclude`, so a
page literally *is* the script and cannot drift; deleting an example **breaks the
docs build**. This repo's one 596-line `examples/run_and_visualize.py` is not
embedded anywhere, and the Pages `site/index.html` is hand-authored. **Lesson:**
stand up `docs/source/` and embed examples via `literalinclude`. **Priority: HIGH ·
Effort: Low** (once Sphinx exists).

### D2. Sphinx autodoc/autosummary auto-API
nyx auto-generates its entire API reference from docstrings/stubs (autodoc +
autosummary + napoleon + `autodoc_typehints="description"` + intersphinx +
pydata-theme), mocking heavy imports. This repo's docstrings (Savage refs in
`navigation.py`, Greensite/Frosch in `control.py`, per-field `Attributes:` blocks)
are *ready* for this today. Add `autodoc_mock_imports=["matplotlib"]` to keep docs
CI light. **Priority: HIGH · Effort: Med** (mostly scaffolding; prose already
written).

### D3. Numbered, progressive examples
Split the monolithic example into `01_nominal_ascent` → `05_no_flex_slosh`
(summary dict → dashboard → small MC → custom guidance → ablation), each embedded
via D1. **Priority: HIGH · Effort: Low–Med.**

### D4. Swappable components via `Protocol` + options objects
nyx plugs in as a typed sensor selected by its options object. This repo wires
concrete classes directly in `main.py`. Define `AtmosphereModel`/`GravityModel`/
`WindModel`/`GuidanceLaw` `Protocol`s with the current implementations as defaults,
selectable via options. *(Dovetails with B1/B3/A4.5 — same decoupling, from the
docs/API angle.)* **Priority: MED · Effort: Med.**

### D5. Smaller polish
API sidecar examples with a CI `exec()` snippet-checker; version-aware docs hosting
(`latest/stable/vX.Y.Z` + switcher); a `Documentation` URL in `pyproject.toml`;
`sphinx-copybutton`. **Priority: LOW · Effort: Low.**

---

# E. Convergent themes (multiple teachers point the same way)

The strongest signals are where independent audits of different repos converged:

1. **Structured, validated, unit-bearing config** — PX4 (YAML param metadata),
   genesis-world (pydantic `Options`), *and* this repo's own **ADR-0009** all point
   away from the flat mutable `config.py`. This is the single most-reinforced
   change and it also dissolves ADR-0004's global-mutation fragility (Q-02). It
   does require revisiting **ADR-0001**.
2. **A committed golden-trajectory regression test** — genesis-world (MuJoCo
   oracle) and quadrants (image/snapshot regression) both. Highest test ROI here;
   complements the repo's own **Q-04** end-to-end test item.
3. **Numba-JIT + scratch reuse for the hot loop** — genesis-world and quadrants both.
4. **Decouple subsystems behind typed interfaces / a subsystem list** — PX4 (uORB)
   and genesis-world (solver list); also the substrate for swappable components
   (nyx D4) and isolatable EKF tests (A4.4).
5. **Type checking + a coverage gate** — quadrants directly; resolves this repo's
   open **T-03** and the ~55% ungated coverage from `audit/03`.

# F. Validation observations (compare-and-validate, this repo)

- **Doc/code drift in `assumptions.md`:** the *Aerodynamics* section still states a
  "drag-only model … no lift, side force, or aerodynamic moments," but `config.py`
  defines `CN_ALPHA_TABLE_*`/`CMQ_PITCH_DAMPING` and `main.py` consumes
  `aero_result.normal_force_body` and `aero_result.aero_moment_body`. The aero
  model now produces a normal force and a pitch-damping moment; the assumptions doc
  should be updated to match (a `D-`class doc fix, in the style of `audit/03`).
- **CI lints on Python 3.14** while the support matrix is 3.11–3.13 (already
  tracked as T-02) — worth folding into the C5/pyright CI pass.

---

# G. Consolidated, prioritized roadmap

Priorities reconcile each teacher's advice with this repo's actual state; effort is
S (hours) / M (~a day) / L (multi-day/structural). Items that change nominal
physics/telemetry must regenerate committed example artifacts and hashes (per the
repo's standing discipline).

### Tier 1 — high value, low/medium effort (do first)
| ID | Lesson | Source | Pri | Eff |
|----|--------|--------|-----|-----|
| X-01 | χ²/NIS innovation gating + per-sensor filtered test ratio + non-finite fault | PX4 A1.2 | High | S |
| X-02 | Clamp EKF biases to physical limits; inhibit bias learning under high-g/clipping | PX4 A1.4 | High | S |
| X-03 | Anti-windup: stop integrating into TVC saturation (use enforcer feedback) | PX4 A2.1 | High | S |
| X-04 | Hysteresis-gate the attitude FTS criterion (debounce N frames) — mitigates AD-19 | PX4 A3.1 | High | S–M |
| X-05 | Golden nominal-trajectory regression (`.npz`) + `scipy.solve_ivp` cross-oracle + `TOL_*` + autouse config reset | g-world B6 / quadrants C3 | High | M |
| X-06 | `--cov-branch` + diff-coverage PR gate (port `coverage_report.py`) | quadrants C2 | High | S |
| X-07 | pyright CI job (`pyrightconfig.json`) — closes T-03 | quadrants C5 | High | S |
| X-08 | AI deleted-comment + test-coverage PR checks | quadrants C1 | High | S each |
| X-09 | Sphinx `docs/source/` with autodoc + `literalinclude` of examples | nyx D1/D2 | High | M |

### Tier 2 — high value, higher/structural effort
| ID | Lesson | Source | Pri | Eff |
|----|--------|--------|-----|-----|
| X-10 | Config as validated typed models (pydantic/dataclass, units+bounds); validate MC overrides; remove global mutation | g-world B3 / PX4 A4.3 / ADR-0009 | High | M–L |
| X-11 | Estimate attitude in the EKF (error-state, multiplicative, measured gyro) | PX4 A1.1 | High | L |
| X-12 | Decompose `main.py` into a subsystem list with typed interfaces | g-world B1 / PX4 A4.5 | High→Med | L |
| X-13 | Numba-JIT RK4 + scratch buffers | g-world B4 / quadrants C4 | Med–High | M |
| X-14 | Config-as-object → process-sharable, then batched/vectorized MC | g-world B2 | High→ceiling | M→L |
| X-15 | Numbered progressive examples + `Protocol` swappable components | nyx D3/D4 | Med | M |

### Tier 3 — solid improvements, schedule opportunistically
Covariance bounding via zero-innovation fusion (A1.3, S); sensor-delay horizon for
GPS/baro (A1.5, M); Gauss-Markov GPS + fix quality (A1.6, S); IMU clipping + baro
v² bias (A4.2, S); explicit control effectiveness/rate loop (A2.2, M); per-event
clear conditions (A3.2, S); phase-aware soft-limit deferral (A3.3, S); pre-launch
estimator/arming gate (A3.4, S); per-component present/degraded/failed health
(A3.5, M); integrator-strategy + substepping (B5, M); EKF SensorSimulator +
raw-sensor replay log (A4.4, M); quaternion exp-map integration (A4.1, S);
`setuptools_scm`/dep-groups/`AGENTS.md`/`.git-blame-ignore-revs` (C6, S);
`SIM_FLOAT` alias (B7, S); benchmark harness (C4, S); docs polish/versioned hosting
(D5, S); fix `assumptions.md` aero drift + T-02 lint matrix (F, S).

## Caveats

- **Respect the existing discipline.** Many items change nominal physics/telemetry
  and must regenerate committed artifacts and SHA-256 hashes; several have ADRs
  that must be superseded explicitly (ADR-0001 by X-10; ADR-0003 by B5; ADR-0004 by
  X-14), not silently contradicted.
- **Fidelity tier is intentional.** `docs/assumptions.md` correctly disclaims this
  is not a certification-grade tool; the PX4 lessons raise fidelity but none of
  them turn this into flight software, and that is fine.
- This document recommends; it changes no code. Each Tier-1 item is a small,
  independently landable PR.
