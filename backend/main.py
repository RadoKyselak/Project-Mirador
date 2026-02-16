import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import check_api_keys_on_startup, logger
from models import VerifyRequest
from models.verdicts import VerificationResponse
from services.verification_service import VerificationService
from utils.validation import ValidationError

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

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"}
    )

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running :)"}


@app.post("/verify", response_model=VerificationResponse)
async def verify(req: VerifyRequest) -> VerificationResponse:

    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")
    
    return await verification_service.verify_claim(claim)
