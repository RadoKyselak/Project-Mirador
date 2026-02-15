import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import check_api_keys_on_startup, logger
from models import VerifyRequest
from services import (
    analyze_claim_for_api_plan,
    execute_query_plan,
    synthesize_finding_with_llm,
    compute_confidence
)

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
async def verify(req: VerifyRequest):
    """
    Verify a claim using government data sources and LLM analysis.
    Args:
        req: VerifyRequest containing the claim to verify
    Returns:
        Verification result with verdict, confidence, sources, and evidence
    """
    claim = (req.claim or "").strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    start_time = asyncio.get_event_loop().time()

    analysis = {}
    all_results = []
    synthesis_result = {}
    confidence_data = {}

    try:
        # Step 1: Analyze the claim and generate a plan for the API
        analysis = await analyze_claim_for_api_plan(claim)
        claim_norm = analysis.get("claim_normalized", claim)
        claim_type = analysis.get("claim_type", "Other")
        api_plan = analysis.get("api_plan", {})

        logger.info("Generated API Plan: %s", json.dumps(api_plan, indent=2))

        # Step 2: Execute queries
        all_results = await execute_query_plan(api_plan, claim_type)

        sources_results = [r for r in all_results if isinstance(r, dict) and "error" not in r]
        debug_errors = [r for r in all_results if isinstance(r, dict) and "error" in r]

        logger.info(f"Retrieved {len(sources_results)} sources, encountered {len(debug_errors)} errors.")

        # Step 3: Synthesize the findings
        synthesis_result = await synthesize_finding_with_llm(claim, analysis, sources_results)
        verdict = synthesis_result.get("verdict", "Inconclusive")
        summary_text = (
            f"{synthesis_result.get('summary','')}. {synthesis_result.get('justification','')}"
            .strip()
            .replace("..", ".")
        )

        # Step 4: Compute confidence result
        confidence_data = await compute_confidence(sources_results, verdict, claim_norm)
        confidence_val = confidence_data.get("confidence", 0.0)

        if confidence_val > 0.75:
            confidence_tier = "High"
        elif confidence_val > 0.5:
            confidence_tier = "Medium"
        else:
            confidence_tier = "Low"

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
                "source_reliability": confidence_data.get("R", 0.0),
                "evidence_density": confidence_data.get("E", 0.0),
                "semantic_alignment": confidence_data.get("S", 0.0),
            },
            "summary": summary_text,
            "evidence_links": synthesis_result.get("evidence_links", []),
            "sources": sources_results,
            "debug_plan": analysis,
            "debug_log": debug_errors,
            "debug_processing_time_seconds": duration,
        }

    except Exception as e:
        logger.exception("Unhandled error during /verify processing for claim: %s", claim)
        return {
            "claim_original": claim,
            "claim_normalized": analysis.get("claim_normalized", claim),
            "claim_type": analysis.get("claim_type", "Other"),
            "verdict": "Error",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "confidence_breakdown": {"R": 0.0, "E": 0.0, "S": 0.0},
            "summary": f"An unexpected internal server error occurred: {type(e).__name__}",
            "evidence_links": [],
            "sources": [r for r in all_results if r and "error" not in r] if all_results else [],
            "debug_plan": analysis,
            "debug_log": [r for r in all_results if r and "error" in r] + [{
                "error": f"Unhandled exception during processing: {str(e)}",
                "source": "internal_verify_endpoint",
                "status": "failed",
            }],
        }
