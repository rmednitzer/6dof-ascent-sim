"""EU regulatory compliance tracker for the 6-DOF ascent simulation.

Maps codebase safety features to applicable EU regulation articles and
generates a compliance status report. This module does NOT make the
simulation compliant — it documents which regulatory requirements are
addressed by existing code and identifies gaps.

Regulatory scope (as of 2026-03-23):
    - AI Act (EU) 2024/1689: Not applicable (GNC uses deterministic algorithms)
    - CRA (EU) 2024/2847: Exempt for non-commercial FOSS; tracked for awareness
    - PLD (EU) 2024/2853: Exempt for non-commercial FOSS
    - NIS2 (EU) 2022/2555: Operator-level only
    - Dual-Use (EU) 2021/821: Publicly available fundamental research
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComplianceStatus(Enum):
    """Assessment status for a regulatory requirement."""

    NOT_APPLICABLE = "not_applicable"
    EXEMPT = "exempt"
    MET = "met"
    PARTIAL = "partial"
    GAP = "gap"


@dataclass(frozen=True)
class RegulatoryRequirement:
    """A single requirement from an EU regulation.

    Attributes:
        regulation:   Short regulation identifier (e.g. ``"CRA"``).
        celex_id:     Official CELEX number for traceability.
        article:      Article or annex reference (e.g. ``"Annex I, Part I, §3"``).
        description:  Human-readable summary of the requirement.
        status:       Current compliance assessment.
        codebase_ref: File path(s) in the codebase that address this requirement.
        notes:        Additional context or gap description.
    """

    regulation: str
    celex_id: str
    article: str
    description: str
    status: ComplianceStatus
    codebase_ref: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# Requirement registry — one entry per mapped obligation
# ---------------------------------------------------------------------------

REQUIREMENTS: tuple[RegulatoryRequirement, ...] = (
    # ---- AI Act ----
    RegulatoryRequirement(
        regulation="AI_ACT",
        celex_id="32024R1689",
        article="Art. 3(1)",
        description="AI system definition — machine-based, autonomous, adaptive, infers",
        status=ComplianceStatus.NOT_APPLICABLE,
        codebase_ref=("sim/gnc/navigation.py", "sim/gnc/control.py", "sim/gnc/guidance.py"),
        notes="EKF, PID, and guidance law are deterministic algorithms without "
        "adaptiveness after deployment. Not AI systems under Art. 3(1).",
    ),
    RegulatoryRequirement(
        regulation="AI_ACT",
        celex_id="32024R1689",
        article="Art. 6(1)",
        description="High-risk classification for safety-component AI",
        status=ComplianceStatus.NOT_APPLICABLE,
        codebase_ref=("sim/safety/fts.py", "sim/safety/boundary_enforcer.py"),
        notes="FTS and BoundaryEnforcer are safety components but are not AI systems. "
        "Would become high-risk if ML/AI guidance is added.",
    ),
    # ---- CRA ----
    RegulatoryRequirement(
        regulation="CRA",
        celex_id="32024R2847",
        article="Art. 2(1)",
        description="Scope — products with digital elements on the market",
        status=ComplianceStatus.EXEMPT,
        notes="Non-commercial FOSS under MIT license. Recital 18 exempts "
        "non-monetised FOSS from commercial activity scope.",
    ),
    RegulatoryRequirement(
        regulation="CRA",
        celex_id="32024R2847",
        article="Annex I, Part I, §1",
        description="No known exploitable vulnerabilities at time of release",
        status=ComplianceStatus.PARTIAL,
        codebase_ref=("sim/core/integrator.py",),
        notes="NaN/Inf guards prevent numerical exploits. No formal vulnerability "
        "scanning or SBOM generation (Gap G-1).",
    ),
    RegulatoryRequirement(
        regulation="CRA",
        celex_id="32024R2847",
        article="Annex I, Part I, §2",
        description="Secure by default configuration",
        status=ComplianceStatus.MET,
        codebase_ref=("sim/config.py",),
        notes="All safety limits have conservative defaults. FTS enabled by default.",
    ),
    RegulatoryRequirement(
        regulation="CRA",
        celex_id="32024R2847",
        article="Annex I, Part I, §3",
        description="Data integrity protection",
        status=ComplianceStatus.PARTIAL,
        codebase_ref=("sim/telemetry/recorder.py",),
        notes="SHA-256 telemetry hash provides integrity. No authentication "
        "(HMAC) or encryption (Gap G-8).",
    ),
    RegulatoryRequirement(
        regulation="CRA",
        celex_id="32024R2847",
        article="Art. 13(5)",
        description="Software Bill of Materials (SBOM)",
        status=ComplianceStatus.GAP,
        codebase_ref=("pyproject.toml",),
        notes="Gap G-1: No SBOM in CycloneDX or SPDX format generated.",
    ),
    # ---- PLD ----
    RegulatoryRequirement(
        regulation="PLD",
        celex_id="32024L2853",
        article="Art. 2(2)",
        description="FOSS exclusion from scope",
        status=ComplianceStatus.EXEMPT,
        notes="Non-commercial FOSS explicitly excluded. Integrating manufacturers "
        "bear liability per Recital 15.",
    ),
    RegulatoryRequirement(
        regulation="PLD",
        celex_id="32024L2853",
        article="Art. 7(1)",
        description="Defectiveness — safety expectations considering presentation",
        status=ComplianceStatus.MET,
        codebase_ref=("docs/assumptions.md", "docs/stpa-analysis.md"),
        notes="Modelling assumptions and safety analysis are documented. "
        "Fidelity limitations are explicitly stated.",
    ),
    # ---- NIS2 ----
    RegulatoryRequirement(
        regulation="NIS2",
        celex_id="32022L2555",
        article="Art. 21(2)(a)",
        description="Risk analysis and information system security policies",
        status=ComplianceStatus.PARTIAL,
        codebase_ref=("docs/stpa-analysis.md",),
        notes="STPA covers physical safety risks. IT/OT cybersecurity risk "
        "analysis is an operator responsibility.",
    ),
    RegulatoryRequirement(
        regulation="NIS2",
        celex_id="32022L2555",
        article="Art. 21(2)(b)",
        description="Incident handling",
        status=ComplianceStatus.PARTIAL,
        codebase_ref=("sim/safety/fts.py", "sim/safety/health_monitor.py"),
        notes="FTS handles flight-safety incidents autonomously. NIS2 incident "
        "reporting (72h to CSIRT, Art. 23) is an operator obligation.",
    ),
    # ---- Dual-Use ----
    RegulatoryRequirement(
        regulation="DUAL_USE",
        celex_id="32021R0821",
        article="Annex I, Cat. 9E003",
        description="Technology for development of space launch vehicles",
        status=ComplianceStatus.EXEMPT,
        notes="Publicly available fundamental research under General Technology Note. "
        "All algorithms are textbook-level; parameters are public-domain approximations.",
    ),
)


@dataclass
class ComplianceReport:
    """Aggregated compliance assessment across all tracked requirements.

    Attributes:
        requirements:  All assessed regulatory requirements.
        summary:       Count of requirements by status.
        gaps:          Requirements with ``GAP`` status that need remediation.
        timestamp_iso: ISO-8601 timestamp of report generation.
    """

    requirements: tuple[RegulatoryRequirement, ...] = ()
    summary: dict[str, int] = field(default_factory=dict)
    gaps: list[RegulatoryRequirement] = field(default_factory=list)
    timestamp_iso: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the report."""
        return {
            "timestamp_iso": self.timestamp_iso,
            "summary": dict(self.summary),
            "total_requirements": len(self.requirements),
            "gaps": [
                {
                    "regulation": r.regulation,
                    "article": r.article,
                    "description": r.description,
                    "notes": r.notes,
                }
                for r in self.gaps
            ],
            "requirements": [
                {
                    "regulation": r.regulation,
                    "celex_id": r.celex_id,
                    "article": r.article,
                    "description": r.description,
                    "status": r.status.value,
                    "codebase_ref": list(r.codebase_ref),
                    "notes": r.notes,
                }
                for r in self.requirements
            ],
        }


def generate_compliance_report() -> ComplianceReport:
    """Generate a compliance report from the requirement registry.

    Returns:
        A :class:`ComplianceReport` summarising the current regulatory posture.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    summary: dict[str, int] = {}
    for req in REQUIREMENTS:
        key = req.status.value
        summary[key] = summary.get(key, 0) + 1

    gaps = [r for r in REQUIREMENTS if r.status == ComplianceStatus.GAP]

    return ComplianceReport(
        requirements=REQUIREMENTS,
        summary=summary,
        gaps=gaps,
        timestamp_iso=now,
    )
