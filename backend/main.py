import os
import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "GEMINI_API_KEY", "DATA_GOV_API_KEY", "BEA_API_KEY", 
    "CENSUS_API_KEY", "CONGRESS_API_KEY"
]

def check_api_keys_on_startup():
    logger.info("Checking for required API keys...")
    missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing_keys:
        logger.critical(f"FATAL: Missing required environment variables: {', '.join(missing_keys)}")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_keys)}")
    logger.info("All required API keys are configured.")

app = FastAPI(on_startup=[check_api_keys_on_startup])

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

BEA_VALID_TABLES = {
    "T10101",  # GDP
    "T20305",  # Personal Income
    "T31600",  # Govnment Spending by Function
    "T70500"   # Relation of GDP, GNP, and NNP
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning(f"No JSON block found in LLM response text.")
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON block: {e}")
        logger.debug(f"Problematic JSON string: {json_str}")
        return None

async def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY is not configured.")
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
        logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"Error communicating with Gemini: {str(e)}")
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
    
    parsed_plan = extract_json_block(res["text"])
    
    if not parsed_plan:
        logger.warning(f"Could not parse API plan JSON for claim. Falling back to keyword search. Claim: '{claim}'")
        return {
            "claim_normalized": claim, 
            "claim_type": "Other", 
            "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}
        }
    
    return parsed_plan

def pick_sources_from_type(claim_type: str) -> List[str]:
    sources = ["DATA.GOV"]
    if claim_type == "quantitative":
        sources.extend(["BEA", "CENSUS"])
    elif claim_type == "factual":
        sources.append("CONGRESS")
    return sources

# --- API Calls ---

async def query_bea(params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    if not BEA_API_KEY: return [{"error": "BEA_API_KEY is not configured", "source": "BEA", "status": "failed"}]
    if not params: return []
        
    final_params = {
        'UserID': BEA_API_KEY, 'method': 'GetData', 'ResultFormat': 'json',
        'DataSetName': params.get('DataSetName'), 'TableName': params.get('TableName'),
        'Frequency': params.get('Frequency'), 'Year': params.get('Year'),
        'LineCode': params.get('LineCode')
    }
    url = "https.apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params={k: v for k, v in final_params.items() if v is not None})
            r.raise_for_status()
            data = r.json().get('BEAAPI', {}).get('Results', {})
            results = data.get('Data', [])
            
            if results:
                item = results[0]
                desc = item.get('LineDescription', 'Data')
                snippet = f"{desc} for {item.get('TimePeriod')} was ${item.get('DataValue')} billion."
                return [{"title": f"BEA Dataset: {params.get('DataSetName')} - {params.get('TableName')}", "url": str(r.url), "snippet": snippet}]
            else:
                return []
    except httpx.HTTPStatusError as e:
        logger.error(f"BEA API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"BEA API error: {e.response.status_code}", "source": "BEA", "status": "failed"}]
    except Exception as e:
        logger.error(f"BEA API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "BEA", "status": "failed"}]

async def query_census(params: Dict[str, Any] = None, keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CENSUS_API_KEY: return [{"error": "CENSUS_API_KEY is not configured", "source": "CENSUS", "status": "failed"}]
    if not params and not keyword_query: return []
    
    final_params = {'key': CENSUS_API_KEY}
    if params:
        url = f"https.api.census.gov{params.get('endpoint')}"
        final_params.update(params.get('params', {}))
    else:
        url = f"https.api.census.gov/data/2022/acs/acs1"
        final_params.update({'get': 'NAME', 'for': 'us:1', 'q': keyword_query})
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            return [{"title": f"Census Data for '{keyword_query or 'parameterized search'}'", "url": str(r.url), "snippet": str(r.json()[:3])[:700]}]
    except httpx.HTTPStatusError as e:
        logger.error(f"Census API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Census API error: {e.response.status_code}", "source": "CENSUS", "status": "failed"}]
    except Exception as e:
        logger.error(f"Census API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "CENSUS", "status": "failed"}]

async def query_congress(keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CONGRESS_API_KEY: return [{"error": "CONGRESS_API_KEY is not configured", "source": "CONGRESS", "status": "failed"}]
    if not keyword_query: return []
    
    params = {"api_key": CONGRESS_API_KEY, "q": keyword_query, "limit": 1}
    url = "https.api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            return [{"title": bill.get('title'), "url": bill.get('url'), "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"} for bill in bills]
    except httpx.HTTPStatusError as e:
        logger.error(f"Congress API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Congress API error: {e.response.status_code}", "source": "CONGRESS", "status": "failed"}]
    except Exception as e:
        logger.error(f"Congress API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY: return [{"error": "DATA_GOV_API_KEY is not configured", "source": "DATA.GOV", "status": "failed"}]
    if not keyword_query: return []
    
    params = {"api_key": DATA_GOV_API_KEY, "q": keyword_query, "limit": 3}
    url = "https.api.data.gov/catalog/v1"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return [{"title": item.get("title"), "url": item.get("@id"), "snippet": item.get("description", "")[:250]} for item in r.json().get("results", [])]
    except httpx.HTTPStatusError as e:
        logger.error(f"Data.gov API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Data.gov API error: {e.response.status_code}", "source": "DATA.GOV", "status": "failed"}]
    except Exception as e:
        logger.error(f"Data.gov API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "DATA.GOV", "status": "failed"}]

async def execute_query_plan(plan: Dict, claim_type: str) -> List[Dict[str, Any]]:
    tier1_params = plan.get('tier1_params', {})
    tier2_keywords = plan.get('tier2_keywords', [])
    sources_to_query = pick_sources_from_type(claim_type)
    tasks = []
    
    if "BEA" in sources_to_query and tier1_params.get("bea"):
        bea_params = tier1_params.get("bea")
        table_name = bea_params.get("TableName")
        if table_name and table_name in BEA_VALID_TABLES:
            tasks.append(query_bea(params=bea_params))
        elif table_name:
            logger.warning(f"Invalid/Hallucinated BEA TableName: '{table_name}'. Skipping BEA Tier 1 call.")
        else:
            logger.debug("No BEA TableName provided in plan, skipping.")
            
    if "CENSUS" in sources_to_query and tier1_params.get("census"):
        tasks.append(query_census(params=tier1_params.get("census")))

    for keyword in tier2_keywords:
        if "DATA.GOV" in sources_to_query:
            tasks.append(query_datagov(keyword))
        if "CENSUS" in sources_to_query and not tier1_params.get("census"):
            tasks.append(query_census(keyword_query=keyword))
        if "CONGRESS" in sources_to_query:
            tasks.append(query_congress(keyword_query=keyword))

    if not tasks:
        logger.info(f"No API calls to execute for claim type: {claim_type}")
        return []
        
    query_results = await asyncio.gather(*tasks)
    return [item for sublist in query_results for item in sublist if sublist]

async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Calls Gemini to summarize evidence and *requires* it to link findings
    to specific source URLs.
    
    Returns a dictionary with summary, justification, and evidence_links.
    """
    default_summary = {
        "summary": "The available data is insufficient to verify the claim.",
        "justification": "No supporting government data could be found or parsed to verify this claim.",
        "evidence_links": []
    }

    if not sources:
        return default_summary
    
    valid_sources = [s for s in sources if "error" not in s]
    if not valid_sources:
         return default_summary
         
    unique_sources = {s['url']: s for s in valid_sources}.values()
    context = "\n---\n".join([f"Source Title: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}" for s in unique_sources])
    
   prompt = (
        "You are a meticulous and impartial fact-checker. Your sole responsibility is to analyze the provided evidence from U.S. government data sources and synthesize a definitive conclusion about the user's claim. Do not introduce outside information.\n\n"
        "YOUR METHODOLOGY (CHAIN-OF-THOUGHT):\n"
        "1. First, review all evidence snippets. Identify the key data points relevant to the claim.\n"
        "2. Second, compare the data points to the user's claim.\n"
        "3. Third, synthesize your findings into a concise, one-sentence summary that directly addresses the claim. Start with a clear concluding phrase (e.g., 'The data supports...', 'The data contradicts...', 'The available data is insufficient to...').\n"
        "4. Fourth, provide a brief (1-2 sentence) justification for your conclusion.\n"
        "5. Fifth, and most importantly, create a list of 'evidence_links'. For each key finding that supports your justification, you MUST link it directly to the 'URL' of the source snippet it came from.\n\n"
        f"USER'S CLAIM: '''{claim}'''\n\n"
        f"AGGREGATED EVIDENCE:\n{context}\n\n"
        "YOUR RESPONSE (Must be a single, valid JSON object):\n"
        "{\n"
        '  "summary": "Your final, synthesized one-sentence conclusion.",\n'
        '  "justification": "Your brief justification for the conclusion.",\n'
        '  "evidence_links": [\n'
        '    { "finding": "The specific data point or fact used.", "source_url": "The URL of the source it came from." },\n'
        '    { "finding": "Another key data point.", "source_url": "https... (must match a URL from the evidence)" }\n'
        '  ]\n'
        "}"
    )
    
    res = await call_gemini(prompt)
    
    parsed_summary_data = extract_json_block(res["text"])
    
    if not parsed_summary_data:
        logger.error(f"Could not parse summary JSON for claim. Claim: '{claim}'")
        return default_summary
    
    if not all(k in parsed_summary_data for k in ["summary", "justification", "evidence_links"]):
        logger.warning(f"Parsed summary JSON is missing required keys. Claim: '{claim}'")
        return default_summary

    return parsed_summary_data

def compute_confidence(sources: List[Dict[str, Any]], summary_text: str) -> Dict[str, Any]:
    if not sources:
        R, E, S = 0.5, 0.0, 0.3 
        confidence = round((0.4 * R + 0.3 * E + 0.3 * S), 2)
        return {"confidence": confidence, "R": R, "E": E, "S": S}

    # Source reliability
    total_weight = 0
    for s in sources:
        url_lower = s.get("url", "").lower()
        if "apps.bea.gov" in url_lower: weight = 1.0
        elif "api.census.gov" in url_lower: weight = 0.9
        elif "api.congress.gov" in url_lower: weight = 0.8
        elif "api.data.gov" in url_lower: weight = 0.7
        else: weight = 0.6
        total_weight += weight
    R = round(total_weight / len(sources), 2)

    # Evidence density
    n = len(sources)
    E = round(min(1.0, n / 5), 2)

    # Semantic alignment
    summary_lower = summary_text.lower()
    if "data supports" in summary_lower or "data confirms" in summary_lower: S = 0.95
    elif "data contradicts" in summary_lower: S = 0.9
    elif "insufficient" in summary_lower: S = 0.5
    else: S = 0.6
    
    confidence = round((0.4 * R + 0.3 * E + 0.3 * S), 2)
    return {"confidence": confidence, "R": R, "E": E, "S": S}

# --- Main Verification ---

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    
    logger.info(f"Received claim for verification: '{claim}'")
    
    analysis = await analyze_claim_for_api_plan(claim)
    claim_norm = analysis.get("claim_normalized")
    claim_type = analysis.get("claim_type")
    api_plan = analysis.get("api_plan", {})
    
    all_results = await execute_query_plan(api_plan, claim_type)
    sources_results = [r for r in all_results if "error" not in r]
    debug_errors = [r for r in all_results if "error" in r]
    
    if debug_errors:
        logger.warning(f"Encountered {len(debug_errors)} errors during API fetch for claim: '{claim}'")

    summary_data = await summarize_with_evidence(claim_norm, sources_results)
    
    summary_text = f"{summary_data.get('summary', '')} {summary_data.get('justification', '')}"
    
    confidence_data = compute_confidence(sources_results, summary_text)
    confidence_val = confidence_data["confidence"]
    
   summary_lower = summary_text.lower()
    if "data supports" in summary_lower or "data confirms" in summary_lower:
        verdict = "Supported"
    elif "data contradicts" in summary_lower:
        verdict = "Contradicted"
    else:
        verdict = "Inconclusive"
        
    confidence_tier = "Low"
    if confidence_val > 0.75:
        confidence_tier = "High"
    elif confidence_val > 0.5:
        confidence_tier = "Medium"

    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "claim_type": claim_type,
        "verdict": verdict,
        "confidence": confidence_val,
        "confidence_tier": confidence_tier,
        "confidence_breakdown": {
            "source_reliability": confidence_data["R"],
            "evidence_density": confidence_data["E"],
            "semantic_alignment": confidence_data["S"]
        },
        "summary": summary_text, 
        "evidence_links": summary_data.get("evidence_links", []),
        "sources": sources_results,
        "debug_plan": api_plan,
        "debug_log": debug_errors
    }
