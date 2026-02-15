import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
class TestQueryBea:
    """Tests for query_bea function."""
    
    async def test_successful_query(self, mock_env_vars, sample_bea_response):
        """Test successful BEA API query."""
        from main import query_bea
        
        params = {
            "DataSetName": "NIPA",
            "TableName": "T31600",
            "Frequency": "A",
            "Year": "2023",
            "LineCode": "2"
        }
        
        with patch("main.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_bea_response
            mock_response.url = "https://apps.bea.gov/api/data?..."
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_client_class.return_value = mock_client
            
            result = await query_bea(params)
            
            assert len(result) == 1
            assert result[0]["title"] == "BEA: NIPA/T31600"
            assert result[0]["data_value"] == 790895.0
            assert result[0]["line_code"] == "2"
    
    async def test_missing_api_key(self, monkeypatch):
        """Test query_bea with missing API key."""
        from main import query_bea
        
        monkeypatch.delenv("BEA_API_KEY", raising=False)
        import importlib
        import main as main_module
        importlib.reload(main_module)
        
        result = await main_module.query_bea({"LineCode": "2"})
        assert result[0]["error"] == "BEA_API_KEY missing"
    
    async def test_missing_line_code(self, mock_env_vars):
        """Test query_bea without LineCode."""
        from main import query_bea
        
        result = await query_bea({})
        assert "error" in result[0]
        assert "missing LineCode" in result[0]["error"]

@pytest.mark.asyncio
class TestQueryCensusAcs:
    """Tests for query_census_acs function."""
    
    async def test_successful_query(self, mock_env_vars, sample_census_response):
        """Test successful Census ACS query."""
        from main import query_census_acs
        
        params = {
            "year": "2021",
            "dataset": "acs/acs1/profile",
            "get": "NAME,DP05_0001E",
            "for": "state:01"
        }
        
        with patch("main.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_census_response
            mock_response.url = "https://api.census.gov/data/..."
            mock_response.headers = {"content-type": "application/json"}
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_client_class.return_value = mock_client
            
            result = await query_census_acs(params)
            
            assert len(result) == 1
            assert "Alabama" in result[0]["title"]
            assert result[0]["data_value"] == 5024279.0
    
    async def test_missing_required_params(self, mock_env_vars):
        """Test query_census_acs with missing required parameters."""
        from main import query_census_acs
        
        result = await query_census_acs({"year": "2021"})
        assert result == []

@pytest.mark.asyncio
class TestQueryBls:
    """Tests for query_bls function."""
    
    async def test_successful_cpi_query(self, mock_env_vars, sample_bls_response):
        """Test successful BLS CPI query."""
        from main import query_bls
        
        params = {"metric": "CPI", "year": "2023"}
        
        with patch("main.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_bls_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_client_class.return_value = mock_client
            
            result = await query_bls(params)
            
            assert len(result) == 1
            assert "CPI" in result[0]["title"]
            assert result[0]["unit"] == "%"
            assert result[0]["data_value"] == pytest.approx(4.1, abs=0.1)
    
    async def test_missing_metric(self, mock_env_vars):
        """Test query_bls with missing metric."""
        from main import query_bls
        
        result = await query_bls({"year": "2023"})
        assert "error" in result[0]
        assert "missing metric" in result[0]["error"]
    
    async def test_unsupported_metric(self, mock_env_vars):
        """Test query_bls with unsupported metric."""
        from main import query_bls
        
        result = await query_bls({"metric": "INVALID", "year": "2023"})
        assert "error" in result[0]
        assert "not supported" in result[0]["error"]

@pytest.mark.asyncio
class TestQueryCongress:
    """Tests for query_congress function."""
    
    async def test_successful_query(self, mock_env_vars):
        """Test successful Congress API query."""
        from main import query_congress
        
        mock_response_data = {
            "bills": [
                {
                    "title": "CHIPS Act",
                    "type": "HR",
                    "number": "1234",
                    "congress": "117",
                    "latestAction": {"text": "Passed Senate"}
                }
            ]
        }
        
        with patch("main.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_client_class.return_value = mock_client
            
            result = await query_congress("CHIPS Act")
            
            assert len(result) == 1
            assert "CHIPS Act" in result[0]["title"]
            assert "Passed Senate" in result[0]["snippet"]
    
    async def test_empty_query(self, mock_env_vars):
        """Test query_congress with empty query."""
        from main import query_congress
        
        result = await query_congress("")
        assert result == []

@pytest.mark.asyncio
class TestQueryDatagov:
    """Tests for query_datagov function."""
    
    async def test_successful_query(self, mock_env_vars):
        """Test successful Data.gov API query."""
        from main import query_datagov
        
        mock_response_data = {
            "result": {
                "results": [
                    {
                        "title": "Federal Budget Dataset",
                        "name": "federal-budget",
                        "notes": "Dataset description",
                        "resources": [
                            {"url": "https://example.com/data.csv", "format": "csv"}
                        ]
                    }
                ]
            }
        }
        
        with patch("main.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            
            mock_client_class.return_value = mock_client
            
            result = await query_datagov("federal budget")
            
            assert len(result) == 1
            assert "Federal Budget Dataset" in result[0]["title"]
            assert result[0]["url"] == "https://example.com/data.csv"
    
    async def test_empty_query(self, mock_env_vars):
        """Test query_datagov with empty query."""
        from main import query_datagov
        
        result = await query_datagov("")
        assert result == []
