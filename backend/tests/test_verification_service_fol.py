import pytest

from services.verification_service import VerificationService


@pytest.mark.asyncio
async def test_verification_service_uses_fol_supported_and_boosts_confidence(mocker):
    service = VerificationService()

    mocker.patch(
        "services.verification_service.analyze_claim_for_api_plan",
        return_value={
            "claim_normalized": "defense > education (2023)",
            "claim_type": "quantitative_comparison",
            "api_plan": {},
        },
    )
    mocker.patch("services.verification_service.execute_query_plan", return_value=[])
    mocker.patch(
        "services.verification_service.fol_reason_about_claim",
        return_value={
            "verdict": "Supported",
            "summary": "FOL says supported.",
            "evidence_links": [{"finding": "x", "source_url": "https://apps.bea.gov"}],
        },
    )
    mocker.patch(
        "services.verification_service.synthesize_finding_with_llm",
        return_value={
            "verdict": "Inconclusive",
            "summary": "LLM summary",
            "justification": "LLM justification",
            "evidence_links": [],
        },
    )
    mocker.patch.object(
        service.confidence_scorer,
        "compute_confidence",
        return_value={"confidence": 0.65, "R": 0.7, "E": 0.6, "S": 0.6},
    )

    result = await service.verify_claim("Defense spending exceeded education spending in 2023.")

    assert result["verdict"] == "Supported"
    assert result["confidence"] == pytest.approx(0.75, abs=0.001)
    assert result["summary"].startswith("FOL says supported")
    assert result["evidence_links"]


@pytest.mark.asyncio
async def test_verification_service_keeps_llm_verdict_when_fol_inconclusive(mocker):
    service = VerificationService()

    mocker.patch(
        "services.verification_service.analyze_claim_for_api_plan",
        return_value={
            "claim_normalized": "some claim",
            "claim_type": "other",
            "api_plan": {},
        },
    )
    mocker.patch("services.verification_service.execute_query_plan", return_value=[])
    mocker.patch(
        "services.verification_service.fol_reason_about_claim",
        return_value={"verdict": "Inconclusive", "summary": "", "evidence_links": []},
    )
    mocker.patch(
        "services.verification_service.synthesize_finding_with_llm",
        return_value={
            "verdict": "Contradicted",
            "summary": "LLM contradicted.",
            "justification": "",
            "evidence_links": [],
        },
    )
    mocker.patch.object(
        service.confidence_scorer,
        "compute_confidence",
        return_value={"confidence": 0.45, "R": 0.4, "E": 0.5, "S": 0.4},
    )

    result = await service.verify_claim("some claim")
    assert result["verdict"] == "Contradicted"
    assert result["confidence"] == pytest.approx(0.45, abs=0.001)

