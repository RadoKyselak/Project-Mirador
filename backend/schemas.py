from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class VerifyRequest(BaseModel):
    claim: str

class Source(BaseModel):
    title: str
    url: str
    snippet: str

class APIPlan(BaseModel):
    tier1_params: Dict[str, Any] = {}
    tier2_keywords: List[str] = []

class ClaimAnalysis(BaseModel):
    claim_normalized: str
    claim_type: str
    geographic_entities: List[Dict[str, Any]] = []
    api_plan: APIPlan

class VerificationResponse(BaseModel):
    claim_original: str
    claim_normalized: str
    claim_type: str
    verdict: str
    confidence: float
    summary: str
    sources: List[Source]
    debug_plan: Optional[APIPlan] = None
