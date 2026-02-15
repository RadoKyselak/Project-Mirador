from typing import Dict, Any, List
from models.confidence import ConfidenceBreakdown
from domain.confidence import ConfidenceScorer
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
