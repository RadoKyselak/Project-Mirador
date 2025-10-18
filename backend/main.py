from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import VerifyRequest, VerificationResponse, APIPlan
import services
from typing import Optional

app = FastAPI(title="Stelthar-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/verify", response_model=VerificationResponse)
async def verify(req: VerifyRequest):
    if not req.claim or not req.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")
    
    analysis = await services.analyze_claim(req.claim)
    
    sources = await services.gather_evidence(analysis.api_plan, analysis.claim_type)
    
    summary = await services.synthesize_summary(analysis.claim_normalized, sources)

    confidence = services.calculate_confidence(sources, analysis.api_plan)
    
    verdict = "Verifiable" if confidence > 0.6 else "Inconclusive"
    
    return VerificationResponse(
        claim_original=req.claim,
        claim_normalized=analysis.claim_normalized,
        claim_type=analysis.claim_type,
        verdict=verdict,
        confidence=round(confidence, 2),
        summary=summary,
        sources=sources,
        debug_plan=analysis.api_plan
    )
