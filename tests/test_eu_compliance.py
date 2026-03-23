"""Tests for the EU regulatory compliance tracker."""

from sim.safety.eu_compliance import (
    ComplianceStatus,
    RegulatoryRequirement,
    generate_compliance_report,
    REQUIREMENTS,
)


class TestRequirementRegistry:
    """Verify the requirement registry is well-formed."""

    def test_all_requirements_have_celex(self) -> None:
        for req in REQUIREMENTS:
            assert req.celex_id, f"{req.regulation} {req.article} missing CELEX ID"

    def test_all_requirements_have_description(self) -> None:
        for req in REQUIREMENTS:
            assert req.description, f"{req.regulation} {req.article} missing description"

    def test_all_statuses_are_valid(self) -> None:
        for req in REQUIREMENTS:
            assert isinstance(req.status, ComplianceStatus)

    def test_no_duplicate_entries(self) -> None:
        keys = [(r.regulation, r.article) for r in REQUIREMENTS]
        assert len(keys) == len(set(keys)), "Duplicate requirement entries found"

    def test_registry_is_immutable(self) -> None:
        assert isinstance(REQUIREMENTS, tuple)


class TestComplianceReport:
    """Verify report generation."""

    def test_report_generation(self) -> None:
        report = generate_compliance_report()
        assert report.timestamp_iso
        assert report.requirements == REQUIREMENTS
        assert sum(report.summary.values()) == len(REQUIREMENTS)

    def test_gaps_are_identified(self) -> None:
        report = generate_compliance_report()
        gap_count = sum(1 for r in REQUIREMENTS if r.status == ComplianceStatus.GAP)
        assert len(report.gaps) == gap_count

    def test_report_to_dict(self) -> None:
        report = generate_compliance_report()
        d = report.to_dict()
        assert "timestamp_iso" in d
        assert "summary" in d
        assert "requirements" in d
        assert "gaps" in d
        assert d["total_requirements"] == len(REQUIREMENTS)

    def test_ai_act_not_applicable(self) -> None:
        """GNC components are deterministic — AI Act should be N/A."""
        ai_reqs = [r for r in REQUIREMENTS if r.regulation == "AI_ACT"]
        assert len(ai_reqs) > 0
        for r in ai_reqs:
            assert r.status == ComplianceStatus.NOT_APPLICABLE

    def test_foss_exemptions(self) -> None:
        """CRA and PLD should show EXEMPT for FOSS scope articles."""
        cra_scope = next(r for r in REQUIREMENTS if r.regulation == "CRA" and "Art. 2" in r.article)
        assert cra_scope.status == ComplianceStatus.EXEMPT

        pld_scope = next(r for r in REQUIREMENTS if r.regulation == "PLD" and "Art. 2" in r.article)
        assert pld_scope.status == ComplianceStatus.EXEMPT

    def test_dual_use_exempt(self) -> None:
        """Publicly available research should be exempt from dual-use controls."""
        du_reqs = [r for r in REQUIREMENTS if r.regulation == "DUAL_USE"]
        assert len(du_reqs) > 0
        for r in du_reqs:
            assert r.status == ComplianceStatus.EXEMPT
