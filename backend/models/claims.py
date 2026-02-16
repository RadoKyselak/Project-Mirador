from typing import TypedDict, Literal, List, Dict, Any
from pydantic import BaseModel, validator, Field

ClaimType = Literal[
    "Economic",
    "Demographic",
    "Legislative",
    "Budget",
    "Employment",
    "Other"
]

class APIQueryPlan(TypedDict, total=False):
    """Structure for multi-tier API query plan."""
    tier1_params: Dict[str, Any]
    tier2_keywords: List[str]

class ClaimAnalysis(TypedDict):
    """LLM analysis result for a user claim."""
    claim_normalized: str
    claim_type: ClaimType
    entities: List[str]
    relationship: str
    api_plan: APIQueryPlan

class VerifyRequest(BaseModel):
    """Request body for /verify endpoint with validation."""
    claim: str = Field(..., min_length=3, max_length=5000)
    
    @validator('claim')
    def sanitize_claim(cls, v):
        """Sanitize and validate claim input."""
        from utils.validation import InputValidator

        sanitized = InputValidator.sanitize_claim(v)
        return sanitized
    
    class Config:
        schema_extra = {
            "example": {
                "claim": "The unemployment rate in 2023 was 3.7%"
            }
        }
