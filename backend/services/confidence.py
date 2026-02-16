from typing import Dict, Any, List
from confidence.confidence_scorer import ConfidenceScorer
from models.confidence import ConfidenceBreakdown
_scorer = ConfidenceScorer()

async def compute_confidence(
    sources: List[Dict[str, Any]],
    verdict: str,
    claim: str
) -> Dict[str, Any]:

    breakdown: ConfidenceBreakdown = await _scorer.compute_confidence(
        sources=sources,
        verdict=verdict,
        claim=claim
    )

    return dict(breakdown)
