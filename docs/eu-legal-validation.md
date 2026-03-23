# EU Legal Validation Report

Cross-referencing the 6-DOF ascent simulation against applicable EU regulations.

**Assessment date:** 2026-03-23
**Codebase:** `rmednitzer/6dof-ascent-sim` @ branch `claude/eu-legal-validation-qart3`
**License:** MIT (non-commercial open-source distribution)

---

## 1. Regulatory Applicability Matrix

| Regulation | CELEX | Applies? | Rationale |
|---|---|---|---|
| **AI Act** (EU) 2024/1689 | 32024R1689 | **No** — GNC is not an AI system | EKF + PID are deterministic algorithms, not "machine-based systems that infer with varying levels of autonomy and may exhibit adaptiveness" (Art. 3(1)). No learning, no adaptiveness after deployment. |
| **Cyber Resilience Act** (EU) 2024/2847 | 32024R2847 | **Conditional** | Applies to "products with digital elements made available on the market" (Art. 2(1)). Excluded when: (a) certified under EASA Reg. (EU) 2018/1139 (Art. 2(3)), (b) developed exclusively for national security/defence (Art. 2(7)), or (c) non-monetised FOSS (Recital 18). Current MIT-licensed non-commercial distribution is **exempt**. Becomes applicable if integrated into a commercial product. |
| **Product Liability Directive** 2024/2853 | 32024L2853 | **Conditional** | "Does not apply to free and open-source software that is developed or supplied outside the course of a commercial activity" (Art. 2(2)). BUT: if a manufacturer integrates this FOSS into a commercial product, the manufacturer bears liability (Recital 15). Software is explicitly a "product" (Art. 4(1)). |
| **NIS2 Directive** (EU) 2022/2555 | 32022L2555 | **Operator-level** | Space is in NIS2 Annex I (Recital 37). Applies to medium+ enterprises providing essential services. The simulation code itself is not an "entity" — but an operator deploying it for actual launch operations may be a NIS2-regulated entity (Art. 2(1)). |
| **CER Directive** (EU) 2022/2557 | 32022L2557 | **Operator-level** | Space sector explicitly included (Recital 5). Applies to entities providing essential services with critical infrastructure. Same entity-level scoping as NIS2. |
| **Machinery Regulation** (EU) 2023/1230 | 32023R1230 | **No** — simulation only | Covers physical machinery and safety components. A simulation tool is not machinery, nor a safety component placed on the market independently. Would apply if the GNC algorithms were extracted and deployed on actual flight hardware. |
| **GPSR** (EU) 2023/0988 | 32023R0988 | **No** | General product safety — applies to consumer products. Aerospace simulation software is not a consumer product. |
| **EU Dual-Use Regulation** (EU) 2021/821 | 32021R0821 | **Potential** | Category 9 (Aerospace and Propulsion) of the Dual-Use list covers technology for launch vehicles. This simulation includes trajectory optimisation, guidance algorithms, and propulsion parameters that could be relevant under MTCR (Missile Technology Control Regime). **See Section 5 for export control analysis.** |
| **GDPR** (EU) 2016/679 | 32016R0679 | **No** | No personal data is processed. Telemetry contains only vehicle state vectors, not data relating to identified/identifiable natural persons. |

---

## 2. AI Act Deep Analysis

### 2.1 Is the GNC subsystem an "AI system"?

**Definition** (AI Act Art. 3(1)):
> "a machine-based system that is designed to operate with **varying levels of autonomy** and that **may exhibit adaptiveness after deployment**, and that, for explicit or implicit objectives, **infers**, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments"

**Assessment of each GNC component:**

| Component | File | Autonomous? | Adaptive? | Infers? | AI system? |
|---|---|---|---|---|---|
| Extended Kalman Filter | `sim/gnc/navigation.py` | Partially — runs without human input | No — fixed model, no learning | State estimation from sensor fusion | **No** — classical optimal estimation, deterministic given inputs |
| PID Controller | `sim/gnc/control.py` | Yes — closed-loop | No — fixed gains | No — algebraic computation | **No** — deterministic control law |
| Guidance Law | `sim/gnc/guidance.py` | Yes — phase transitions | No — pre-programmed phases | No — scheduled trajectory | **No** — open-loop schedule + feedback |
| Flight Termination System | `sim/safety/fts.py` | Yes — autonomous abort | No — fixed thresholds | No — threshold comparisons | **No** — deterministic safety monitor |

**Conclusion:** None of the GNC subsystems meet the AI Act definition. They are classical deterministic algorithms without adaptiveness or learning. The AI Act does **not** apply.

### 2.2 Safety component analysis

Even if the GNC were reclassified as AI in a future amendment, the "safety component" definition (Art. 3(14)) — "a component which fulfils a safety function, or the failure of which endangers health and safety" — would classify the FTS and boundary enforcer as safety components, triggering **high-risk** classification under Art. 6(1).

**Recommendation:** If any ML/AI component (e.g., neural guidance, reinforcement-learning controller) is ever added, the entire GNC stack would likely become a high-risk AI system requiring:
- Risk management system (Art. 9)
- Data governance (Art. 10)
- Technical documentation (Art. 11)
- Record-keeping / logging (Art. 12)
- Transparency / human oversight (Art. 13-14)
- Accuracy, robustness, cybersecurity (Art. 15)

---

## 3. Cyber Resilience Act Analysis

### 3.1 Current status: Exempt

The CRA applies to "products with digital elements made available on the market" (Art. 2(1)). Three exclusion paths apply:

1. **FOSS exclusion** (Recital 18): "products with digital elements qualifying as free and open-source software that are not monetised by their manufacturers should not be considered to be a commercial activity."
2. **EASA exclusion** (Art. 2(3)): Products certified under Regulation (EU) 2018/1139 are excluded.
3. **Defence/security exclusion** (Art. 2(7)): Products developed exclusively for national security or defence are excluded.

### 3.2 If commercialised: Full CRA compliance required

Should this software be placed on the market commercially, CRA Annex I essential requirements would apply:

| CRA Requirement (Annex I) | Current codebase status | Gap |
|---|---|---|
| Security by design | Boundary enforcer + FTS provide safety-by-design | No cybersecurity-specific design (input validation for network-facing interfaces, authentication) |
| No known exploitable vulnerabilities | No SBOM, no vulnerability scanning | **Gap**: Need SBOM and CVE tracking |
| Secure default configuration | `sim/config.py` has safe defaults | Adequate |
| Protection against unauthorised access | No access control | **Gap** if deployed as a service |
| Confidentiality of data | SHA-256 telemetry hashing (integrity only) | **Gap**: No encryption for data at rest/transit |
| Data integrity | SHA-256 hash in `sim/telemetry/recorder.py` | Partially met |
| Minimised data collection | Telemetry captures only vehicle state | Met |
| Software update mechanism | No update mechanism | **Gap** for deployed products |
| Vulnerability disclosure process | None | **Gap**: Need coordinated disclosure policy |
| Software Bill of Materials (SBOM) | Not generated | **Gap**: CRA Art. 13(5) requires SBOM |

---

## 4. Product Liability Directive Analysis

### 4.1 Current status: Exempt

PLD Art. 2(2): "This Directive does not apply to free and open-source software that is developed or supplied outside the course of a commercial activity."

### 4.2 Integration liability chain

Per PLD Recital 15: "Where free and open-source software supplied outside the course of a commercial activity is subsequently integrated by a manufacturer as a component into a product in the course of a commercial activity," the **integrating manufacturer** bears liability.

**Defectiveness test** (PLD Art. 7): A product is defective when it does not provide the safety a person is entitled to expect, considering:
- Presentation and labelling
- Reasonably foreseeable use
- The moment of placing on the market
- Regulatory requirements

**Simulation-specific risk:** If this simulation is used to validate flight software and the simulation contains modelling errors (documented in `docs/assumptions.md`), downstream liability could arise if:
1. A manufacturer relies on simulation results for certification
2. The simulation fails to predict a failure mode
3. The failure causes damage covered by PLD Art. 6

**Mitigation (already in codebase):**
- `docs/assumptions.md` explicitly documents 30+ modelling simplifications
- Monte Carlo dispersion analysis quantifies parametric uncertainty
- STPA safety analysis identifies hazard scenarios

**Recommendation:** Add explicit disclaimers about simulation fidelity limits and non-suitability for flight certification without independent V&V.

---

## 5. Export Control Analysis (Dual-Use Regulation)

### 5.1 Relevant control lists

EU Regulation (EU) 2021/821, Annex I, Category 9 (Aerospace and Propulsion):

- **9A004**: Space launch vehicles and "spacecraft"
- **9D004**: Software specially designed for "use" of equipment in 9A004
- **9E003**: Technology for "development" of equipment in 9A004

### 5.2 Assessment of controlled elements

| Codebase element | Potential control category | Assessment |
|---|---|---|
| Propulsion model (`sim/vehicle/propulsion.py`) | 9E003.a.1 — propulsion technology | Parameters are public-domain (Merlin-class approximations from published specs). **Not controlled** — publicly available per Art. 2 Note. |
| Guidance algorithms (`sim/gnc/guidance.py`) | 9E003.a.3 — guidance technology | Gravity-turn + linear tangent steering are textbook algorithms. **Not controlled** — basic scientific research per General Technology Note. |
| EKF navigation (`sim/gnc/navigation.py`) | 9E003.a.3 | Standard 12-state EKF, university-level implementation. **Not controlled**. |
| Trajectory data / insertion criteria | 9E003.a — ascent trajectory tech | Generic LEO insertion targeting ISS orbit. **Not controlled** — publicly available. |
| Monte Carlo dispersion (`sim/montecarlo/`) | 9E003.h — reliability/accuracy | Generic statistical methods. **Not controlled**. |
| Complete integrated simulation | 9D004 | Combined system could be argued as "software specially designed" for launch vehicles. **Borderline** — mitigated by MIT license and public availability. |

### 5.3 MTCR considerations

The Missile Technology Control Regime (MTCR) Category I covers "complete rocket systems (including ballistic missiles, space launch vehicles, and sounding rockets) capable of delivering at least a 500 kg payload to a range of at least 300 km."

This simulation models a vehicle that exceeds MTCR Category I thresholds (400 km altitude, multi-tonne payload capacity). While the **software alone** is not a controlled item (it is publicly available fundamental research), its integration with actual hardware parameters could change the classification.

### 5.4 Recommendations

1. **Maintain public availability**: The General Technology Note and public-domain exclusions apply only while the technology remains publicly available. Restricting access could paradoxically increase export control exposure.
2. **No classified parameters**: Never incorporate actual flight vehicle parameters that are subject to export control (e.g., real engine performance tables, actual Cd curves from wind tunnel testing, classified trajectory data).
3. **Academic/research framing**: The MIT license and educational documentation support classification as fundamental research.

---

## 6. NIS2 / CER — Operator Obligations

### 6.1 When applicable

NIS2 and CER apply at the **entity level**, not the software level. They become relevant when an operator:
- Is a medium+ enterprise (NIS2 Art. 2(1))
- Provides essential services in the space sector (NIS2 Annex I, CER Recital 5)
- Uses this simulation as part of critical infrastructure operations

### 6.2 Operator requirements mapped to codebase

| NIS2 Requirement (Art. 21) | Codebase support | Gap for operators |
|---|---|---|
| Risk analysis and security policies | STPA analysis in `docs/stpa-analysis.md` | Operator must extend to IT/OT context |
| Incident handling | FTS provides autonomous incident response | Operator needs reporting procedures (72h to CSIRT per Art. 23) |
| Business continuity | Monte Carlo quantifies failure probability | Operator needs backup/recovery plans |
| Supply chain security | MIT license, public repo | Operator must assess dependency risks |
| Vulnerability handling and disclosure | No CVE process | **Gap**: Operator needs vulnerability management |
| Cybersecurity training | N/A | Operator obligation |

---

## 7. Cross-Reference: Codebase Safety Features vs EU Requirements

### 7.1 Mapping existing safety architecture

| Codebase feature | EU regulatory alignment |
|---|---|
| `BoundaryEnforcer` — command clamping | Machinery Reg. Annex III §1.2.1 (limits of use); AI Act Art. 9 (risk management) if AI added |
| `FlightTerminationSystem` — autonomous abort | CER Art. 13 (resilience measures); AI Act Art. 14 (human oversight) — FTS is deterministic override |
| `HealthMonitor` — 4-level severity | NIS2 Art. 21(2)(a) (risk analysis); AI Act Art. 9(2)(a) (risk identification) |
| SHA-256 telemetry hash | CRA Annex I Part I §3 (data integrity); GDPR Art. 32 (security of processing — if personal data added) |
| STPA analysis | PLD Art. 7(1)(d) (regulatory requirements); AI Act Art. 9 (risk management system) |
| Monte Carlo dispersion | AI Act Art. 15 (accuracy/robustness) — anticipates validation requirements |
| `docs/assumptions.md` | PLD Recital 14 (presentation/labelling of limitations); CRA Art. 13(15) (instructions for use) |
| NaN/Inf guards in integrator | CRA Annex I Part I §1 (no known exploitable vulnerabilities); AI Act Art. 15(4) (resilience) |
| Innovation gating in EKF | AI Act Art. 15(3) (robustness against errors); CRA Annex I Part I §2 (secure by default) |

### 7.2 Gap summary

| # | Gap | Applicable if | Priority | Recommendation |
|---|---|---|---|---|
| G-1 | No SBOM (Software Bill of Materials) | CRA commercialisation | High | Generate SBOM from `pyproject.toml` using CycloneDX or SPDX |
| G-2 | No vulnerability disclosure policy | CRA commercialisation, NIS2 operator | High | Add `SECURITY.md` with coordinated disclosure process |
| G-3 | No access control / authentication | CRA as service, NIS2 operator | Medium | Out of scope for standalone simulation; needed if deployed as service |
| G-4 | No data encryption at rest/transit | CRA, NIS2 | Medium | Encrypt telemetry output files if they contain sensitive trajectory data |
| G-5 | No software update mechanism | CRA commercialisation | Medium | Implement version checking and update notification |
| G-6 | Missing fidelity disclaimer | PLD risk mitigation | High | Add explicit notice that simulation is not suitable for flight certification |
| G-7 | No export control notice | Dual-Use awareness | Medium | Add notice about MTCR/Wassenaar awareness |
| G-8 | Telemetry has integrity but no authentication | CRA Annex I | Low | HMAC-SHA256 would add authentication; current SHA-256 is integrity-only |

---

## 8. Regulatory Timeline

| Date | Regulation | Milestone | Impact |
|---|---|---|---|
| 2024-08-01 | AI Act | Entry into force | No current impact — GNC is not AI |
| 2024-12-10 | CRA | Entry into force | Exempt as non-commercial FOSS |
| 2024-12-08 | PLD | Entry into force | Exempt as non-commercial FOSS |
| 2024-12-13 | GPSR | Applies from this date | Not applicable |
| 2025-02-02 | AI Act | Prohibited practices apply | N/A |
| 2025-08-02 | AI Act | General-purpose AI rules apply | N/A |
| 2026-08-02 | AI Act | High-risk AI rules apply | Relevant if ML/AI components added |
| 2027-01-20 | Machinery Reg. | Full application | N/A for simulation; relevant for flight hardware |
| 2027-12-11 | CRA | Full application (obligations) | Relevant if commercialised by then |

---

## 9. Validation Methodology

This analysis was conducted by:

1. **Querying official EU regulation texts** via structured legal databases covering 61 EU regulations across 10 categories.
2. **Cross-referencing definitions** — particularly AI Act Art. 3(1) (AI system), Art. 3(14) (safety component), CRA Art. 3(1) (product with digital elements), PLD Art. 4(1) (product).
3. **Checking exclusion clauses** — CRA Art. 2(3)/(7), PLD Art. 2(2), AI Act Recital 102-103 (FOSS).
4. **Mapping codebase features** against regulatory requirements article-by-article.
5. **Reviewing Austrian national implementation** context (RIS OGD database, 5,101 federal statutes).

**Disclaimer:** This is a technical compliance assessment, not legal advice. Consult qualified legal counsel for binding opinions on regulatory applicability.
