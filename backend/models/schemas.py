from pydantic import BaseModel
class VerifyRequest(BaseModel):
    """Request model for the /verify endpoint."""
    claim: str
