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
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def normalize_and_classify_claim(claim: str) -> Dict[str, Any]:
    prompt = (
        "You are an assistant that extracts a single concise factual claim and classifies it.\n\n"
        f"Input text: '''{claim}'''\n\n"
        "Return a JSON object with keys:\n"
        " - claim: the single, normalized factual claim (one sentence)\n"
        " - type: one of [quantitative, qualitative, causal, factual]\n"
        " - search_queries: an array of 3 short search queries (no more than 8 words each) derived from the claim.\n\n"
        "Output ONLY valid JSON.\n"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(text)
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

async def query_bea(query: str) -> List[Dict[str, str]]:
    if not BEA_API_key:
        return []
    search_term = query.lower()
    dataset_name = "NIPA" 
    if "gdp" in search_term:
        dataset_name = "NIPA"
    elif "investment" in search_term:
        dataset_name = "NIUnderlyingDetail"
    
    params = {
        'UserID': BEA_API_KEY,
        'method': 'GetData',
        'datasetname': dataset_name,
        'TableName': 'T10101',
        'Frequency': 'A',
        'Year': 'ALL',
        'ResultFormat': 'json'
    }
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            results = data.get('BEAAPI', {}).get('Results', {}).get('Data', [])
            snippet = "BEA data found. "
            if results:
                first_point = results[0]
                last_point = results[-1]
                snippet += (f"The series includes data from {first_point.get('TimePeriod')} "
                            f"to {last_point.get('TimePeriod')}, with a final value of "
                            f"{last_point.get('DataValue')}.")
            else:
                snippet = "Could not retrieve specific data points, but the dataset is relevant."

            return [{
                "title": f"BEA Data for Dataset: {dataset_name}",
                "url": str(r.url),
                "snippet": snippet
            }]
    except Exception as e:
        print(f"BEA API Error: {e}")
        return []
async def query_census(query: str) -> List[Dict[str, str]]:
    if not CENSUS_API_KEY: return []
    params = {"get": "NAME,POP", "for": "state:*", "key": CENSUS_API_KEY}
    url = "https://api.census.gov/data/2023/pep/population"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            snippet = str(r.json())[:500]
            return [{"title": "US Census Population Estimates", "url": r.url, "snippet": snippet}]
    except Exception:
        return []

async def query_congress(query: str) -> List[Dict[str, str]]:
    if not CONGRESS_API_KEY: return []
    params = {"api_key": CONGRESS_API_KEY, "q": query}
    url = f"https://api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            results = []
            for bill in bills[:2]:
                results.append({"title": bill.get('title'), "url": bill.get('url'), "snippet": f"Bill Number: {bill.get('number')}, Latest Action: {bill.get('latestAction', {}).get('text')}"})
            return results
    except Exception:
        return []

async def query_datagov(query: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY: return []
    url = "https://api.data.gov/catalog/v1"
    params = {"api_key": DATA_GOV_API_KEY, "q": query}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            j = r.json()
            results = []
            for item in j.get("results", [])[:2]:
                results.append({"title": item.get("title"), "url": item.get("@id"), "snippet": item.get("description", "")[:200]})
            return results
    except Exception:
        return []

async def query_sources(sources_to_query: List[str], search_queries: List[str]) -> List[Dict[str, str]]:
    """Runs queries against the selected APIs in parallel."""
    tasks = []
    primary_query = search_queries[0] if search_queries else ""

    for source in sources_to_query:
        if source == "BEA":
            tasks.append(query_bea(primary_query))
        elif source == "CENSUS":
            tasks.append(query_census(primary_query))
        elif source == "CONGRESS":
            legislative_query = next((q for q in search_queries if "bill" in q or "law" in q), primary_query)
            tasks.append(query_congress(legislative_query))
        elif source == "DATA.GOV":
            tasks.append(query_datagov(primary_query))

    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> str:
    if not sources:
        return "No supporting government data could be found to verify this claim."
    context = "\n".join([f"Source: {s['title']}\nSnippet: {s['snippet']}" for s in sources])
    prompt = (
        "Based on the following snippets from government sources, provide a one-sentence summary assessing the claim's validity.\n\n"
        f"Claim: '''{claim}'''\n\n"
        f"Context:\n{context}\n\n"
        "Summary:"
    )
    res = await call_gemini(prompt)
    return res["text"].strip() if res["text"] else "Could not generate a summary."

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    
    normalized = await normalize_and_classify_claim(claim)
    claim_norm = normalized.get("claim")
    claim_type = normalized.get("type")
    search_queries = normalized.get("search_queries", [])
    
    sources_to_query = pick_sources_from_type(claim_type)
    sources_results = await query_sources(sources_to_query, search_queries)
    
    verdict, confidence = ("Unverifiable", 0.0) if not sources_results else ("Mostly True", 0.75)
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


