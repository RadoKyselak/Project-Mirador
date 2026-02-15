from typing import List
from config.constants import CONFIDENCE_CONFIG
from models.api_responses import SourceData

class EvidenceDensityScorer:
    """Calculates evidence density (E) based on source count."""
    
    def __init__(self, max_sources: int = None):
        """
        Initialize evidence density scorer.
        Args:
            max_sources: Maximum sources for full density score (default: 5)
        """
        self.max_sources = max_sources or CONFIDENCE_CONFIG.MAX_SOURCES_FOR_FULL_DENSITY
    
    def score(self, sources: List[SourceData]) -> float:
        """
        Calculate evidence density score from 0.0 to 1.0.
        Args:
            sources: List of valid source data  
        Returns:
            Evidence density score (E), capped at 1.0
        """
        if not sources:
            return CONFIDENCE_CONFIG.DEFAULT_E
        
        density = len(sources) / self.max_sources
        return round(min(1.0, density), 2)
