from typing import TypedDict, Literal, List, Dict

VerdictType = Literal["Supported", "Contradicted", "Inconclusive"]

class EvidenceLink(TypedDict):
    """Individual evidence citation linking finding to source."""
    finding: str
    source_url: str

class SynthesisResult(TypedDict):
    """LLM synthesis output with verdict and justification."""
    verdict: VerdictType
    summary: str
    justification: str
    evidence_links: List[EvidenceLink]

class VerificationResponse(TypedDict):
    """Complete response from /verify endpoint."""
    claim_original: str
    claim_normalized: str
    claim_type: str
    verdict: VerdictType
    confidence: float
    confidence_tier: Literal["High", "Medium", "Low"]
    confidence_breakdown: Dict[str, float]
    summary: str
    evidence_links: List[EvidenceLink]
    sources: List[Dict]
    debug_plan: Dict
    debug_log: List[Dict]
