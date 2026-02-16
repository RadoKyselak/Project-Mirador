import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import check_api_keys_on_startup, logger
from models import VerifyRequest
from models.verdicts import VerificationResponse
from services.verification_service import VerificationService

app = FastAPI(
    title="Stelthar API",
    description="Government data-backed fact-checking API for Project Mirador",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Run startup checks."""
    check_api_keys_on_startup()
    logger.info("Stelthar API started successfully")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

verification_service = VerificationService()

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Stelthar-API is running :)"}


@app.post("/verify", response_model=VerificationResponse)
async def verify(req: VerifyRequest) -> VerificationResponse:
    """
    Verify a user's claim against government data sources.
    
    Args:
        req: Verification request containing the claim
        
    Returns:
        Verification response with verdict, confidence, and evidence
    """
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")
    
    return await verification_service.verify_claim(claim)
