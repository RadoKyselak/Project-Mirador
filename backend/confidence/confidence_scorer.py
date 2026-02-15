from typing import List
from config import logger
from config.constants import CONFIDENCE_CONFIG
from models.api_responses import SourceData, APIResult
from models.verdicts import VerdictType
from models.confidence import ConfidenceBreakdown
from .reliability_scorer import ReliabilityScorer
from .density_scorer import EvidenceDensityScorer
from .semantic_scorer import SemanticAlignmentScorer


class ConfidenceScorer:
    def __init__(
        self,
        reliability_scorer: ReliabilityScorer = None,
        density_scorer: EvidenceDensityScorer = None,
        semantic_scorer: SemanticAlignmentScorer = None
    ):
        self.reliability_scorer = reliability_scorer or ReliabilityScorer()
        self.density_scorer = density_scorer or EvidenceDensityScorer()
        self.semantic_scorer = semantic_scorer or SemanticAlignmentScorer()
        self.config = CONFIDENCE_CONFIG
    
    async def compute_confidence(
        self,
        sources: List[APIResult],
        verdict: VerdictType,
        claim: str
    ) -> ConfidenceBreakdown:
        valid_sources = [
            s for s in sources 
            if s and "error" not in s
        ]
        
        if not valid_sources:
            logger.warning("No valid sources for confidence calculation.")
            return ConfidenceBreakdown(
                confidence=self.config.DEFAULT_CONFIDENCE,
                R=self.config.DEFAULT_R,
                E=self.config.DEFAULT_E,
                S=self.config.DEFAULT_S,
                S_semantic_sim=self.config.DEFAULT_S_SEMANTIC
            )
        
        R = self.reliability_scorer.score(valid_sources)
        E = self.density_scorer.score(valid_sources)
        S, S_semantic_sim = await self.semantic_scorer.score(valid_sources, verdict, claim)
        
        # Compute final confidence
        confidence = round(
            self.config.R_WEIGHT * R +
            self.config.S_WEIGHT * S +
            self.config.E_WEIGHT * E,
            2
        )
        
        logger.info(
            f"Confidence calculation: R={R} ({len(valid_sources)} sources), "
            f"S={S} (semantic={S_semantic_sim}), E={E} → Total={confidence}"
        )
        
        return ConfidenceBreakdown(
            confidence=confidence,
            R=R,
            E=E,
            S=S,
            S_semantic_sim=S_semantic_sim
        )
    
    @staticmethod
    def get_confidence_tier(confidence: float) -> str:
        if confidence > CONFIDENCE_CONFIG.HIGH_THRESHOLD:
            return "High"
        elif confidence > CONFIDENCE_CONFIG.MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "Low"
