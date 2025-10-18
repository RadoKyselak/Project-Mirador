import httpx
import json
from typing import List, Dict, Any, Optional
from config import settings

async def query_bea(params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not settings.BEA_API_KEY or not params: 
        return []
    
    final_params = {
        'UserID': settings.BEA_API_KEY,
        'method': 'GetData',
        'ResultFormat': 'json',
        'DataSetName': params.get('DataSetName'),
        'TableName': params.get('TableName'),
        'Frequency': params.get('Frequency'),
        'Year': params.get('Year')
    }
    url = "https://apps.bea.gov/api/data"
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            data = r.json().get('BEAAPI', {}).get('Results', {})
            results = data.get('Data', [])
            snippets = []
            for item in results:
                desc = item.get('LineDescription', 'Data')
                snippet = f"{desc} for {item.get('TimePeriod')} was ${item.get('DataValue')} billion."
                snippets.append(snippet)
            if not snippets: 
                return []
            
            return [{"title": f"BEA Dataset: {params.get('DataSetName')} - {params.get('TableName')}", "url": str(r.url), "snippet": " ".join(snippets)}]
    except Exception as e:
        print(f"BEA API Error: {e}")
        return []

async def query_census(params: Optional[Dict[str, Any]] = None, keyword_query: Optional[str] = None) -> List[Dict[str, Any]]:
    if not settings.CENSUS_API_KEY: 
        return []
    
    final_params = {'key': settings.CENSUS_API_KEY}
    if params:
        url = f"https://api.census.gov{params.get('endpoint')}"
        final_params.update(params.get('params', {}))
    elif keyword_query:
        url = f"https://api.census.gov/data/2022/acs/acs1"
        final_params.update({'get': 'NAME', 'for': 'us:1', 'q': keyword_query})
    else: 
        return []

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            title = f"Census Data for '{keyword_query or 'parameterized search'}'"
            snippet = str(r.json()[:3])[:700]
            return [{"title": title, "url": str(r.url), "snippet": snippet}]
    except Exception as e:
        print(f"Census API Error: {e}")
        return []

async def query_congress(keyword_query: Optional[str] = None) -> List[Dict[str, Any]]:
    if not settings.CONGRESS_API_KEY or not keyword_query: 
        return []
    
    params = {"api_key": settings.CONGRESS_API_KEY, "q": keyword_query, "limit": 1}
    url = "https://api.congress.gov/v3/bill"
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            return [
                {
                    "title": bill.get('title'), 
                    "url": bill.get('url'), 
                    "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"
                } for bill in bills
            ]
    except Exception as e:
        print(f"Congress API Error: {e}")
        return []

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not settings.DATA_GOV_API_KEY or not keyword_query: 
        return []
    
    params = {"api_key": settings.DATA_GOV_API_KEY, "q": keyword_query, "limit": 1}
    url = "https://api.data.gov/catalog/v1"
    
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return [
                {
                    "title": item.get("title"), 
                    "url": item.get("@id"), 
                    "snippet": item.get("description", "")[:250]
                } for item in r.json().get("results", [])
            ]
    except Exception as e:
        print(f"Data.gov API Error: {e}")
        return []
