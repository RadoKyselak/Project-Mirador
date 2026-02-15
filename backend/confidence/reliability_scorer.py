from typing import List
from config.constants import SourceReliabilityWeights, CONFIDENCE_CONFIG
from models.api_responses import SourceData


class ReliabilityScorer:
    """Calculates source reliability (R) based on API trustworthiness."""
    
    def __init__(self, weights: SourceReliabilityWeights = None):
        """
        Initialize reliability scorer.
        
        Args:
            weights: Source reliability weights configuration
        """
        self.weights = weights or SourceReliabilityWeights()
    
    def score(self, sources: List[SourceData]) -> float:
        """
        Calculate reliability score from 0.0 to 1.0.
        
        Args:
            sources: List of valid source data (no errors)
            
        Returns:
            Reliability score (R)
        """
        if not sources:
            return CONFIDENCE_CONFIG.DEFAULT_R
        
        total_weight = sum(self._get_weight(source) for source in sources)
        return round(total_weight / len(sources), 2)
    
    def _get_weight(self, source: SourceData) -> float:
        """
        Get reliability weight for a single source.
        
        Args:
            source: Source data dictionary
            
        Returns:
            Weight value based on source URL
        """
        url = source.get("url", "")
        return self.weights.get_weight_for_url(url)
