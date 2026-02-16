import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import check_api_keys_on_startup, logger
from config.constants import CONFIDENCE_CONFIG
from models import VerifyRequest
from models.verdicts import VerificationResponse
from models.confidence import ConfidenceBreakdown
from domain.confidence import ConfidenceScorer
from services import (
    analyze_claim_for_api_plan,
    execute_query_plan,
    synthesize_finding_with_llm,
)
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"}
    )

@app.post("/verify")
async def verify(req: VerifyRequest) -> VerificationResponse:
    claim = req.claim
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Run startup checks."""
    check_api_keys_on_startup()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Stelthar-API is running :)"}


@app.post("/verify")
async def verify(req: VerifyRequest) -> VerificationResponse:
    claim = (req.claim or "").strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    start_time = asyncio.get_event_loop().time()

    analysis = {}
    all_results = []
    synthesis_result = {}
    confidence_breakdown: ConfidenceBreakdown = {}
    confidence_scorer = ConfidenceScorer()

    try:
        analysis = await analyze_claim_for_api_plan(claim)
        claim_norm = analysis.get("claim_normalized", claim)
        claim_type = analysis.get("claim_type", "Other")
        api_plan = analysis.get("api_plan", {})

        logger.info("Generated API Plan: %s", json.dumps(api_plan, indent=2))

        all_results = await execute_query_plan(api_plan, claim_type)

        sources_results = [r for r in all_results if isinstance(r, dict) and "error" not in r]
        debug_errors = [r for r in all_results if isinstance(r, dict) and "error" in r]

        logger.info(f"Retrieved {len(sources_results)} sources, encountered {len(debug_errors)} errors.")

        synthesis_result = await synthesize_finding_with_llm(claim, analysis, sources_results)
        verdict = synthesis_result.get("verdict", "Inconclusive")
        summary_text = (
            f"{synthesis_result.get('summary','')}. {synthesis_result.get('justification','')}"
            .strip()
            .replace("..", ".")
        )

        confidence_breakdown = await confidence_scorer.compute_confidence(
            sources=sources_results,
            verdict=verdict,
            claim=claim_norm
        )
        confidence_val = confidence_breakdown["confidence"]
        
        confidence_tier = confidence_scorer.get_confidence_tier(confidence_val)

        end_time = asyncio.get_event_loop().time()
        duration = round(end_time - start_time, 2)
        logger.info(f"Verification completed for claim '{claim[:50]}...' in {duration} seconds.")

        return {
            "claim_original": claim,
            "claim_normalized": claim_norm,
            "claim_type": claim_type,
            "verdict": verdict,
            "confidence": confidence_val,
            "confidence_tier": confidence_tier,
            "confidence_breakdown": {
                "source_reliability": confidence_breakdown.get("R", 0.0),
                "evidence_density": confidence_breakdown.get("E", 0.0),
                "semantic_alignment": confidence_breakdown.get("S", 0.0),
            },
            "summary": summary_text,
            "evidence_links": synthesis_result.get("evidence_links", []),
            "sources": sources_results[:20],
            "debug_plan": analysis,
            "debug_log": debug_errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during verification.")
        return {
            "claim_original": claim,
            "claim_normalized": analysis.get("claim_normalized", claim),
            "claim_type": analysis.get("claim_type", "Other"),
            "verdict": "Error",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "confidence_breakdown": {
                "source_reliability": 0.0,
                "evidence_density": 0.0,
                "semantic_alignment": 0.0,
            },
            "summary": f"An error occurred while processing this claim: {str(e)}",
            "evidence_links": [],
            "sources": [r for r in all_results if r and "error" not in r] if all_results else [],
            "debug_plan": analysis,
            "debug_log": [r for r in all_results if r and "error" in r] + [{
                "error": f"Unhandled exception during processing: {str(e)}",
                "source": "internal_verify_endpoint",
                "status": "failed",
            }],
        }


