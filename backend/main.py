import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    prompt = (
        "You are a world-class research analyst and a U.S. government data expert. Your task is to deconstruct a user's factual claim into a precise, multi-tiered query plan to verify it using specific government APIs. You must act as an expert system, selecting the exact datasets and parameters needed.\n\n"
        "AVAILABLE APIs & DATASETS:\n"
        "1. BEA (Bureau of Economic Analysis): For national, regional, and industry-specific economic data.\n"
        "   - Key Datasets: 'NIPA' (National Income and Product Accounts), 'NIUnderlyingDetail', 'Regional', 'FixedAssets', 'GDPbyIndustry'.\n"
        "   - Key Tables (NIPA): 'T10101' (GDP), 'T20305' (Personal Income), 'T31600' (Govt Spending by Function), 'T70500' (Relation of GDP, GNP, and NNP).\n"
        "   - Required Params: `DataSetName`, `TableName` or `LineCode`, `GeoFips`, `Frequency`, `Year`.\n"
        "2. Census Bureau: For a wide range of demographic, economic, and social data.\n"
        "   - Key Endpoints (Demographic): '/data/2023/pep/population' (Population Estimates), '/data/acs/acs1' (American Community Survey 1-Year), '/data/dec/decennial' (Decennial Census).\n"
        "   - Required Params: `endpoint`, `params` (which includes `get`, `for`, `in`, etc.).\n"
        "3. Congress.gov: For U.S. federal legislative information.\n"
        "   - Required Params: `query` (a keyword search string).\n"
        "4. Data.gov: A comprehensive catalog of U.S. government data, best used for keyword searches on topics not covered by the specialized APIs above.\n"
        "   - Required Params: `query` (a keyword search string).\n\n"
        f"USER CLAIM: '''{claim}'''\n\n"
        "YOUR RESPONSE (Must be a single, valid JSON object):\n"
        "{\n"
        '  "claim_normalized": "Your clear, verifiable statement.",\n'
        '  "claim_type": "Your classification (e.g., quantitative, factual, Other).",\n'
        '  "api_plan": {\n'
        '    "tier1_params": {\n'
        '      "bea": { "DataSetName": "...", "TableName": "...", "Frequency": "...", "Year": "...", "LineCode": "..." } or null,\n'
        '      "census": { "endpoint": "...", "params": { "get": "...", "for": "..." } } or null\n'
        '    },\n'
        '    "tier2_keywords": ["A list of keyword queries for a broader search."]\n'
        '  }\n'
        "}"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"claim_normalized": claim, "claim_type": "Other", "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}}

def pick_sources_from_type(claim_type: str) -> List[str]:
    sources = ["DATA.GOV"]
    if claim_type == "quantitative":
        sources.extend(["BEA", "CENSUS"])
    elif claim_type == "factual":
        sources.append("CONGRESS")
    return sources

async def query_bea(params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    if not BEA_API_KEY:
        raise HTTPException(status_code=500, detail="BEA_API_KEY is not configured on the server.")
    if not params:
        return []
    final_params = {
        'UserID': BEA_API_KEY, 'method': 'GetData', 'ResultFormat': 'json',
        'DataSetName': params.get('DataSetName'), 'TableName': params.get('TableName'),
        'Frequency': params.get('Frequency'), 'Year': params.get('Year'),
        'LineCode': params.get('LineCode')
    }
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params={k: v for k, v in final_params.items() if v is not None})
            r.raise_for_status()
            data = r.json().get('BEAAPI', {}).get('Results', {})
            results = data.get('Data', [])
            snippets = [f"{item.get('LineDescription', 'Data')} for {item.get('TimePeriod')} was ${item.get('DataValue')} billion." for item in results]
            if not snippets: return []
            return [{"title": f"BEA Dataset: {params.get('DataSetName')} - {params.get('TableName')}", "url": str(r.url), "snippet": " ".join(snippets)}]
    except Exception as e:
        print(f"BEA API Error: {e}")
        return []

async def query_census(params: Dict[str, Any] = None, keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CENSUS_API_KEY:
        raise HTTPException(status_code=500, detail="CENSUS_API_KEY is not configured on the server.")
    if not params and not keyword_query:
        return []
    final_params = {'key': CENSUS_API_KEY}
    if params:
        url = f"https://api.census.gov{params.get('endpoint')}"
        final_params.update(params.get('params', {}))
    else:
        url = f"https://api.census.gov/data/2022/acs/acs1"
        final_params.update({'get': 'NAME', 'for': 'us:1', 'q': keyword_query})
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            return [{"title": f"Census Data for '{keyword_query or 'parameterized search'}'", "url": str(r.url), "snippet": str(r.json()[:3])[:700]}]
    except Exception as e:
        print(f"Census API Error: {e}")
        return []

async def query_congress(keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CONGRESS_API_KEY:
        raise HTTPException(status_code=500, detail="CONGRESS_API_KEY is not configured on the server.")
    if not keyword_query:
        return []
    params = {"api_key": CONGRESS_API_KEY, "q": keyword_query, "limit": 1}
    url = "https://api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            return [{"title": bill.get('title'), "url": bill.get('url'), "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"} for bill in bills]
    except Exception as e:
        print(f"Congress API Error: {e}")
        return []

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY:
        raise HTTPException(status_code=500, detail="DATA_GOV_API_KEY is not configured on the server.")
    if not keyword_query:
        return []
    params = {"api_key": DATA_GOV_API_KEY, "q": keyword_query, "limit": 3}
    url = "https://api.data.gov/catalog/v1"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return [{"title": item.get("title"), "url": item.get("@id"), "snippet": item.get("description", "")[:250]} for item in r.json().get("results", [])]
    except Exception as e:
        print(f"Data.gov API Error: {e}")
        return []

async def execute_query_plan(plan: Dict, claim_type: str) -> List[Dict[str, Any]]:
    tier1_params = plan.get('tier1_params', {})
    tier2_keywords = plan.get('tier2_keywords', [])
    sources_to_query = pick_sources_from_type(claim_type)
    tasks = []
    
    # Tier 1 - Parameterized queries
    if "BEA" in sources_to_query and tier1_params.get("bea"):
        tasks.append(query_bea(params=tier1_params.get("bea")))
    if "CENSUS" in sources_to_query and tier1_params.get("census"):
        tasks.append(query_census(params=tier1_params.get("census")))

    # Tier 2 - Keyword-based queries
    for keyword in tier2_keywords:
        if "DATA.GOV" in sources_to_query:
            tasks.append(query_datagov(keyword))
        if "CENSUS" in sources_to_query:
            tasks.append(query_census(keyword_query=keyword))
        if "CONGRESS" in sources_to_query:
            tasks.append(query_congress(keyword_query=keyword))

    if not tasks:
        return []
        
    query_results = await asyncio.gather(*tasks)
    return [item for sublist in query_results for item in sublist if sublist]

async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> str:
    if not sources:
        return "No supporting government data could be found to verify this claim."
    unique_sources = {s['url']: s for s in sources}.values()
    context = "\n---\n".join([f"Source Title: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}" for s in unique_sources])
    prompt = (
        "You are a meticulous and impartial fact-checker. Your sole responsibility is to analyze the provided evidence from U.S. government data sources and synthesize a definitive conclusion about the user's claim. Do not introduce outside information.\n\n"
        "YOUR METHODOLOGY (CHAIN-OF-THOUGHT):\n"
        "1. First, review all evidence snippets. Identify the key data points relevant to the claim.\n"
        "2. Second, compare the data points. Are they consistent? Do they contradict each other? Is there enough information to make a judgment?\n"
        "3. Third, synthesize your findings into a concise, one-sentence summary that directly addresses the claim. State whether the evidence supports, contradicts, or is insufficient to verify the claim. Start your summary with a clear concluding phrase (e.g., 'The data supports...', 'The data contradicts...', 'The available data is insufficient to...').\n"
        "4. Fourth, provide a brief (1-2 sentence) justification for your conclusion, citing the key pieces of evidence from the snippets.\n\n"
        f"USER'S CLAIM: '''{claim}'''\n\n"
        f"AGGREGATED EVIDENCE:\n{context}\n\n"
        "YOUR RESPONSE (Must be a single, valid JSON object):\n"
        "{\n"
        '  "summary": "Your final, synthesized one-sentence conclusion.",\n'
        '  "justification": "Your brief justification citing the evidence."\n'
        "}"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip().replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(text)
        return f"{parsed.get('summary', '')} {parsed.get('justification', '')}"
    except json.JSONDecodeError:
        return "Could not generate a conclusive summary based on the available data."

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    
    analysis = await analyze_claim_for_api_plan(claim)
    
    claim_norm = analysis.get("claim_normalized")
    claim_type = analysis.get("claim_type")
    api_plan = analysis.get("api_plan", {})
    
    sources_results = await execute_query_plan(api_plan, claim_type)
    
    verdict, confidence = ("Inconclusive", 0.5) if not sources_results else ("Verifiable", 0.95)
    summary = await summarize_with_evidence(claim_norm, sources_results)
    
    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "claim_type": claim_type,
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "sources": sources_results,
        "debug_plan": api_plan
    }
