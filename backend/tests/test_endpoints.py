import pytest
from unittest.mock import AsyncMock, patch


class TestHealthCheckEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, test_client):
        """Test GET / returns healthy status."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Stelthar-API" in data["message"]


class TestVerifyEndpoint:
    """Tests for the /verify endpoint."""
    
    def test_verify_empty_claim(self, test_client):
        """Test /verify with empty claim returns 400."""
        response = test_client.post("/verify", json={"claim": ""})
        
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]
    
    def test_verify_missing_claim(self, test_client):
        """Test /verify without claim field returns 422."""
        response = test_client.post("/verify", json={})
        
        assert response.status_code == 422
    
    @patch("main.analyze_claim_for_api_plan")
    @patch("main.execute_query_plan")
    @patch("main.synthesize_finding_with_llm")
    @patch("main.compute_confidence")
    def test_verify_successful(
        self,
        mock_compute_confidence,
        mock_synthesize,
        mock_execute,
        mock_analyze,
        test_client
    ):
        """Test successful /verify request."""
        mock_analyze.return_value = {
            "claim_normalized": "Test claim normalized",
            "claim_type": "quantitative_value",
            "entities": ["Test"],
            "relationship": "equals",
            "api_plan": {
                "tier1_params": {"bea": None, "census_acs": None, "bls": None},
                "tier2_keywords": ["test"]
            }
        }
        
        mock_execute.return_value = [
            {
                "title": "Test Source",
                "url": "https://example.com",
                "snippet": "Test data",
                "data_value": 100
            }
        ]
        
        mock_synthesize.return_value = {
            "verdict": "Supported",
            "summary": "The claim is supported by data.",
            "justification": "Test data shows value of 100.",
            "evidence_links": [
                {"finding": "Value = 100", "source_url": "https://example.com"}
            ]
        }
        
        mock_compute_confidence.return_value = {
            "confidence": 0.85,
            "R": 0.9,
            "E": 0.8,
            "S": 0.85,
            "S_semantic_sim": 0.75
        }

        mock_analyze.return_value = AsyncMock(return_value=mock_analyze.return_value)()
        mock_execute.return_value = AsyncMock(return_value=mock_execute.return_value)()
        mock_synthesize.return_value = AsyncMock(return_value=mock_synthesize.return_value)()
        mock_compute_confidence.return_value = AsyncMock(return_value=mock_compute_confidence.return_value)()
        
        response = test_client.post("/verify", json={"claim": "Test claim"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["claim_original"] == "Test claim"
        assert data["claim_normalized"] == "Test claim normalized"
        assert data["verdict"] == "Supported"
        assert data["confidence"] == 0.85
        assert data["confidence_tier"] == "High"
        assert len(data["sources"]) == 1
        assert len(data["evidence_links"]) == 1
    
    @patch("main.analyze_claim_for_api_plan")
    def test_verify_handles_exception(self, mock_analyze, test_client):
        """Test /verify handles exceptions gracefully."""
        mock_analyze.side_effect = Exception("Test error")
        response = test_client.post("/verify", json={"claim": "Test claim"})
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "Error"
        assert data["confidence"] == 0.0
        assert "error" in data["summary"].lower()
