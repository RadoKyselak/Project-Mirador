import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException


@pytest.mark.asyncio
class TestCallGemini:
    """Tests for call_gemini function."""
    
    async def test_successful_call(self, sample_gemini_response):
        """Test successful Gemini API call."""
        import main
        
        with patch("main.GEMINI_API_KEY", "test_gemini_key"):
            with patch("main.httpx.AsyncClient") as mock_client_class:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = sample_gemini_response
                mock_response.raise_for_status = MagicMock()
                
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                
                mock_client_class.return_value = mock_client
                
                result = await main.call_gemini("test prompt")
                
                assert "raw" in result
                assert "text" in result
                assert "claim_normalized" in result["text"]
    
    async def test_missing_api_key(self):
        """Test call_gemini with missing API key."""
        import main
        
        with patch("main.GEMINI_API_KEY", None):
            with pytest.raises(HTTPException) as exc_info:
                await main.call_gemini("test")
            
            assert exc_info.value.status_code == 500
            assert "not configured" in exc_info.value.detail
    
    async def test_http_error(self):
        """Test call_gemini with HTTP error."""
        import main
        
        with patch("main.GEMINI_API_KEY", "test_key"):
            with patch("main.httpx.AsyncClient") as mock_client_class:
                mock_client = MagicMock()
                
                from httpx import HTTPStatusError, Request, Response
                mock_request = Request("POST", "http://test.com")
                mock_http_response = Response(500, request=mock_request)
                
                mock_client.post = AsyncMock(
                    side_effect=HTTPStatusError("Error", request=mock_request, response=mock_http_response)
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                
                mock_client_class.return_value = mock_client
                
                with pytest.raises(HTTPException) as exc_info:
                    await main.call_gemini("test")
                
                assert exc_info.value.status_code == 500


@pytest.mark.asyncio
class TestGetEmbeddingsBatchApi:
    """Tests for get_embeddings_batch_api function."""
    
    async def test_successful_embedding(self):
        """Test successful embedding generation."""
        import main
        
        mock_embedding_response = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]}
            ]
        }
        
        with patch("main.GEMINI_API_KEY", "test_gemini_key"):
            with patch("main.httpx.AsyncClient") as mock_client_class:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_embedding_response
                mock_response.raise_for_status = MagicMock()
                
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                
                mock_client_class.return_value = mock_client
                
                result = await main.get_embeddings_batch_api(["text1", "text2"])
                
                assert len(result) == 2
                assert result[0] == [0.1, 0.2, 0.3]
                assert result[1] == [0.4, 0.5, 0.6]
    
    async def test_empty_input(self):
        """Test with empty input list."""
        import main
        
        result = await main.get_embeddings_batch_api([])
        assert result == []
    
    async def test_missing_api_key(self):
        """Test with missing API key."""
        import main
        
        with patch("main.GEMINI_API_KEY", None):
            result = await main.get_embeddings_batch_api(["text1", "text2"])
            assert result == [None, None]
    
    async def test_batch_size_limit(self):
        """Test that batching respects MAX_BATCH_SIZE."""
        import main
        
        texts = [f"text{i}" for i in range(150)]
        
        call_count = 0
        
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 200
            count = 100 if call_count == 1 else 50
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1, 0.2]} for _ in range(count)]
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response
        
        with patch("main.GEMINI_API_KEY", "test_gemini_key"):
            with patch("main.httpx.AsyncClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.post = mock_post
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock()
                
                mock_client_class.return_value = mock_client
                
                result = await main.get_embeddings_batch_api(texts)
                
                assert len(result) == 150
                assert call_count == 2  # Should make 2 API calls
