import httpx
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from schemas import Source
from config import settings
from datetime import datetime

class DataSource(ABC):
    """Abstract Base Class for all external data source clients."""
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abstractmethod
    async def query(self, params: Optional[Dict] = None, keyword: Optional[str] = None) -> List[Source]:
        pass

class BLSDataSource(DataSource):
    async def query(self, params: Optional[Dict] = None, **kwargs) -> List[Source]:
        if not settings.BLS_API_KEY or not params or not params.get('seriesid'):
            return []
        snippet = f"BLS data for series {params.get('seriesid')}: [data]"
        return [Source(title="BLS Data", url="https://bls.gov", snippet=snippet)]

class FREDDataSource(DataSource):
    async def query(self, params: Optional[Dict] = None, **kwargs) -> List[Source]:
        if not settings.FRED_API_KEY or not params or not params.get('series_id'):
            return []
        snippet = f"FRED data for series {params.get('series_id')}: [data]"
        return [Source(title="FRED Data", url="https://fred.stlouisfed.org", snippet=snippet)]

DATA_SOURCE_MAP = {
    "BLS": BLSDataSource,
    "FRED": FREDDataSource,
}
