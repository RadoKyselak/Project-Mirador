import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from urllib.parse import urlencode
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TO DO: key

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
BEA_API_KEY = os.getenv("BEA_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

async def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def normalize_and_suggest_params(claim: str) -> Dict[str, Any]:
    "You are an expert at deconstructing factual claims into API queries for U.S. government data. "
        "Analyze the user's claim and provide a JSON object with a precise, dynamic plan to verify it. "
        "If a claim does not map well to an API, provide null for that API's parameters.\n\n"
        f"USER CLAIM: '''{claim}'''\n\n"
        "YOUR TASK: Return a single, valid JSON object with the following structure:\n"
        "{\n"
        '  "claim_normalized": "A single, normalized factual claim in one sentence.",\n'
        '  "claim_type": "One of [quantitative, qualitative, causal, factual].",\n'
        '  "search_queries": ["An array of 3 general search queries derived from the claim."],\n'
        '  "api_params": {\n'
        '    "bea": { "DataSetName": "e.g., NIPA", "TableName": "e.g., T10101", "Frequency": "e.g., A", "Year": "e.g., 2023" },\n'
        '    "census": { "endpoint": "e.g., /data/2023/pep/population", "params": { "get": "e.g., NAME,POP", "for": "e.g., state:*" } },\n'
        '    "congress": { "query": "A specific query for a bill or law." }\n'
        '  }\n'
        "}\n\n"
        "EXAMPLE:\n"
        "USER CLAIM: 'The US GDP was around 27 trillion dollars in 2023.'\n"
        "YOUR JSON RESPONSE: {\n"
        '  "claim_normalized": "The United States Gross Domestic Product was approximately $27 trillion in 2023.",\n'
        '  "claim_type": "quantitative",\n'
        '  "search_queries": ["US GDP 2023", "Bureau of Economic Analysis GDP data", "2023 national economic output"],\n'
        '  "api_params": {\n'
        '    "bea": { "DataSetName": "NIPA", "TableName": "T10101", "Frequency": "A", "Year": "2023" },\n'
        '    "census": null,\n'
        '    "congress": null\n'
        '  }\n'
        "}"
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
         return json.loads(text)
    except Exception:
        normalized = claim.strip().replace("\n", " ")
        parsed = {"claim": normalized, "type": "qualitative", "search_queries": [normalized]}
    return parsed
def pick_sources_from_type(claim_type: str) -> List[str]:
    """Selects the best APIs to query based on the claim type."""
    mapping = {
        "quantitative": ["BEA", "CENSUS", "DATA.GOV"],
        "factual": ["CONGRESS", "DATA.GOV"],
        "causal": ["DATA.GOV"],
        "qualitative": ["DATA.GOV", "CENSUS"]
    }
    return mapping.get(claim_type, ["DATA.GOV"])

sync def query_bea(params: Dict[str, Any]) -> List[Dict[str, str]]:

    if not BEA_API_KEY or not params: return []

    base_params = {"UserID": BEA_API_KEY, "method": "GetData", "ResultFormat": "json"}

    final_params = {**base_params, **params}
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            data = r.json()
            results = data.get('BEAAPI', {}).get('Results', {}).get('Data', [])
            snippet = f"Found {len(results)} data points in BEA dataset '{params.get('DataSetName')}'. "
            if results: snippet += f"Latest value for {results[-1].get('TimePeriod')} is {results[-1].get('DataValue')}."
            return [{"title": f"BEA Dataset: {params.get('DataSetName')}", "url": str(r.url), "snippet": snippet}]
    except Exception: return []
async def query_census(params: Dict[str, Any]) -> List[Dict[str, str]]:
    if not CENSUS_API_KEY or not params: return []
    endpoint = params.get("endpoint")
    census_params = params.get("params", {})
    if not endpoint or not census_params: return []
    
    url = f"https://api.census.gov{endpoint}"
    census_params['key'] = CENSUS_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=census_params)
            r.raise_for_status()
            snippet = str(r.json())[:500]
            return [{"title": "US Census Bureau Data", "url": str(r.url), "snippet": snippet}]
    except Exception: return []

async def query_congress(params: Dict[str, Any]) -> List[Dict[str, str]]:
    if not CONGRESS_API_KEY or not params or not params.get('query'): return []
    api_params = {"api_key": CONGRESS_API_KEY, "q": params['query']}
    url = "https://api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=api_params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            results = []
            for bill in bills[:2]:
                results.append({"title": bill.get('title'), "url": bill.get('url'), "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"})
            return results
    except Exception: return []

async def query_datagov(query: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY or not query: return []
    return []


async def query_sources(sources_to_query: List[str], api_params: Dict, fallback_query: str) -> List[Dict[str, str]]:
    tasks = []
    for source in sources_to_query:
        if source == "BEA":
            tasks.append(query_bea(api_params.get("bea")))
        elif source == "CENSUS":
            tasks.append(query_census(api_params.get("census")))
        elif source == "CONGRESS":
            tasks.append(query_congress(api_params.get("congress")))
        elif source == "DATA.GOV":
            tasks.append(query_datagov(fallback_query))

    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist if sublist]

async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> str:
    if not sources: return "No supporting government data could be found to verify this claim."
    return "Could not generate a summary."


@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    
    analysis = await normalize_and_suggest_params(claim)
    
    claim_norm = analysis.get("claim_normalized")
    claim_type = analysis.get("claim_type")
    search_queries = analysis.get("search_queries", [])
    api_params = analysis.get("api_params", {})
    
    sources_to_query = pick_sources_from_type(claim_type)
    
    sources_results = await query_sources(sources_to_query, api_params, search_queries[0] if search_queries else claim_norm)
    
    if not sources_results:
        sources_results.extend(await query_datagov(search_queries[0] if search_queries else claim_norm))

    verdict, confidence = ("Unverifiable", 0.0) if not sources_results else ("Mostly True", 0.8)
    summary = await summarize_with_evidence(claim_norm, sources_results)
    
    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "claim_type": claim_type,
        "search_queries": search_queries,
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "sources": sources_results
    }
