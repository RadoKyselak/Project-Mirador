import os
import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from sentence_transformers import SentenceTransformer, util
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "GEMINI_API_KEY", "BEA_API_KEY", "CENSUS_API_KEY", "CONGRESS_API_KEY"
]

def check_api_keys_on_startup():
    logger.info("Startup: checking for required API keys...")
    missing = [k for k in REQUIRED_KEYS if k != "DATA_GOV_API_KEY" and not os.getenv(k)]
    if missing:
        logger.warning("Missing critical env vars: %s. Corresponding API calls will fail.", ", ".join(missing))
    else:
        logger.info("All critical API keys seem present.")

app = FastAPI(on_startup=[check_api_keys_on_startup])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEA_API_KEY = os.getenv("BEA_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

try:
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded.")
except Exception as e:
    logger.error("Failed to load SentenceTransformer model: %s. Semantic scoring will be disabled.", e)
    EMBEDDING_MODEL = None

BEA_VALID_TABLES = {
    "T10101", "T20305", "T31600", "T70500"
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running :)"}

class VerifyRequest(BaseModel):
    claim: str

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract first balanced JSON object from text."""
    if not text: return None
    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        cleaned = re.sub(r"[\x00-\x1f]", "", candidate)
                        return json.loads(cleaned)
                    except Exception: return None
    return None

def _parse_numeric_value(val: Any) -> Optional[float]:
    """Robustly parse numeric values from strings."""
    if val is None: return None
    try:
        s = str(val).strip().replace(",", "").replace("$", "")
        if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
        m = re.match(r"^(-?[\d\.eE+-]+)", s)
        return float(m.group(1)) if m else float(s)
    except Exception: return None

def _apply_multiplier(value: Optional[float], multiplier: Optional[Any]) -> Optional[float]:
    """Apply BEA unit multiplier if present and valid."""
    if value is None: return None
    if multiplier is None: return value
    try: return float(value) * float(multiplier)
    except Exception: return value

async def call_gemini(prompt: str) -> Dict[str, Any]:
    """Calls Gemini API, handles common errors, returns raw response dict."""
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not configured.")
        raise HTTPException(status_code=500, detail="LLM API key not configured on server.")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        logger.error("Gemini HTTP error %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=500, detail=f"LLM API error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error("Gemini request error: %s", str(e))
        raise HTTPException(status_code=500, detail="Error communicating with LLM.")

    text = ""
    if isinstance(data, dict):
        candidates = data.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if isinstance(parts, list) and parts:
                text = parts[0].get("text", "")
        if not text:
            text = data.get("output", "") or data.get("text", "")
    return {"raw": data, "text": text or json.dumps(data)}

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    """Uses LLM to generate an API query plan based on the claim."""
    prompt_template = """
    You are a research analyst expert in U.S. government data APIs (BEA, Census, Data.gov).
    Analyze the user's claim and generate a plan to verify it using these APIs.
    
    Identify:
    1.  `claim_normalized`: A clear, verifiable statement.
    2.  `claim_type`: Classification (e.g., quantitative_comparison, quantitative_value, factual, legislative, other).
    3.  `entities`: Key concepts/metrics mentioned (e.g., ["GDP", "Inflation Rate"], ["Defense Spending", "Education Spending"]).
    4.  `relationship`: The asserted link (e.g., "greater than", "less than", "correlation", "existence", "value is X").
    5.  `api_plan`: JSON with `tier1_params` (for specific BEA/Census calls) and `tier2_keywords` (for Data.gov/Congress search).
    
    AVAILABLE APIs:
    - BEA: NIPA dataset (T31600 for spending by function). Use LineCodes (e.g., 2 for Defense, 14 for Education).
    - Census ACS (American Community Survey): Use `census_acs` key. Provide `year`, `dataset` (e.g., "acs/acs1/profile"), `get` (variable codes, e.g., "NAME,DP05_0001E"), and `for` (geography, e.g., "state:01" for Alabama or "state:*" for all).
    - Data.gov (CKAN): Keyword search via catalog.data.gov.
    - Congress.gov: Keyword search for legislative info.
    
    USER CLAIM: '''{claim}'''
    
    Return ONLY a single valid JSON object.
    
    Example for a claim about defense spending:
    {{
      "claim_normalized": "Federal spending on defense exceeded education spending in 2023.",
      "claim_type": "quantitative_comparison",
      "entities": ["Federal Defense Spending", "Federal Education Spending"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{
          "bea": {{"DataSetName":"NIPA","TableName":"T31600","Frequency":"A","Year":"2023","LineCode":["2","14"]}},
          "census_acs": null
        }},
        "tier2_keywords": ["federal budget appropriations 2023 defense education", "OMB historical tables spending"]
      }}
    }}
    
    Example for a claim about population:
    {{
      "claim_normalized": "The total population of Alabama in 2022 was over 5 million.",
      "claim_type": "quantitative_value",
      "entities": ["Alabama Population", "2022"],
      "relationship": "value is X",
      "api_plan": {{
        "tier1_params": {{
          "bea": null,
          "census_acs": {{
            "year": "2022",
            "dataset": "acs/acs1/profile",
            "get": "NAME,DP05_0001E",
            "for": "state:01"
          }}
        }},
        "tier2_keywords": ["alabama population 2022"]
      }}
    }}
    """
    prompt = prompt_template.format(claim=claim)
    fallback_plan = {"claim_normalized": claim, "claim_type": "Other", "entities": [], "relationship": "unknown",
                     "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}}
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))
        if not parsed or "api_plan" not in parsed:
            logger.warning("Could not parse valid plan JSON from LLM. Falling back.")
            return fallback_plan
        parsed.setdefault("claim_normalized", claim)
        parsed.setdefault("claim_type", "Other")
        parsed.setdefault("entities", [])
        parsed.setdefault("relationship", "unknown")
        parsed.setdefault("api_plan", {"tier1_params": {}, "tier2_keywords": [claim]})
        return parsed
    except HTTPException as e:
        logger.error("LLM failed generating plan: %s", getattr(e, "detail", str(e)))
        return fallback_plan
    except Exception:
        logger.exception("Unexpected error generating plan. Falling back.")
        return fallback_plan

def pick_sources_from_type(claim_type: str) -> List[str]:
    """Selects APIs based on claim type (more robust)."""
    sources = {"DATA.GOV"}
    ct_lower = (claim_type or "").lower()
    if "quantitative" in ct_lower: sources.update({"BEA", "CENSUS"})
    if "factual" in ct_lower: sources.add("CONGRESS")
    if "legislative" in ct_lower: sources.add("CONGRESS")
    return list(sources)
    
async def query_bea(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not BEA_API_KEY: return [{"error": "BEA_API_KEY missing", "source": "BEA", "status": "failed"}]
    final_params = {
        "UserID": BEA_API_KEY,"method": "GetData","ResultFormat": "json",
        "DataSetName": params.get("DataSetName"), "TableName": params.get("TableName"),
        "Frequency": params.get("Frequency"), "Year": params.get("Year"),
        "LineCode": params.get("LineCode"), "GeoFips": params.get("GeoFips"),
    }
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params={k: v for k, v in final_params.items() if v is not None})
            r.raise_for_status()
            payload = r.json()
    except httpx.HTTPStatusError as e:
        logger.error("BEA HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"BEA API error: {e.response.status_code}", "source": "BEA", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("BEA request error: %s", str(e))
        return [{"error": str(e), "source": "BEA", "status": "failed"}]

    results = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    out: List[Dict[str, Any]] = []
    if not results:
        logger.info("BEA returned no rows for params: %s", {k:v for k,v in final_params.items() if v is not None})
        return []

    for item in results:
        line_code_resp = item.get("LineCode") or item.get("SeriesCode") or final_params.get("LineCode")
        desc = item.get("LineDescription") or item.get("SeriesDescription") or "Data"
        data_value_raw = item.get("DataValue") or ""
        numeric = _parse_numeric_value(data_value_raw)
        unit = item.get("Unit") or ""
        unit_multiplier = item.get("UnitMultiplier")
        time_period = item.get("TimePeriod") or final_params.get("Year")
        snippet = f"{desc} ({line_code_resp}) for {time_period}: {data_value_raw}{' '+unit if unit else ''}."

        out.append({
            "title": f"BEA: {params.get('DataSetName')}/{params.get('TableName')}",
            "url": str(r.url),
            "snippet": snippet,
            "data_value": numeric,
            "raw_data_value": data_value_raw,
            "unit": unit,
            "unit_multiplier": unit_multiplier,
            "line_description": desc,
            "line_code": str(line_code_resp) if line_code_resp is not None else None,
            "raw_year": item.get("TimePeriod"),
            "raw_geo": item.get("GeoFips"),
        })
    return out


async def query_census(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not CENSUS_API_KEY: return [{"error": "CENSUS_API_KEY missing", "source": "CENSUS", "status": "failed"}]
    if not params: return []
    final_params = {"key": CENSUS_API_KEY}
    endpoint = params.get("endpoint")
    if not endpoint: return [{"error": "Census endpoint missing", "source": "CENSUS", "status": "failed"}]
    url = f"https://api.census.gov{endpoint}"
    final_params.update(params.get("params", {}))
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            try: snippet = str(r.json()[:5])[:1000]
            except Exception: snippet = str(r.text)[:1000]
            return [{"title": f"Census: {endpoint}", "url": str(r.url), "snippet": snippet}]
    except httpx.HTTPStatusError as e:
        logger.error("Census HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Census API error: {e.response.status_code}", "source": "CENSUS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Census request error: %s", str(e))
        return [{"error": str(e), "source": "CENSUS", "status": "failed"}]


async def query_congress(keyword_query: str) -> List[Dict[str, Any]]:
    if not CONGRESS_API_KEY: return [{"error": "CONGRESS_API_KEY missing", "source": "CONGRESS", "status": "failed"}]
    if not keyword_query: return []
    params = {"api_key": CONGRESS_API_KEY, "q": keyword_query, "limit": 3}
    url = "https://api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            results = []
            for bill in bills:
                url_obj = bill.get("url")
                url_str = url_obj.get("url") if isinstance(url_obj, dict) else str(url_obj or "")
                results.append({
                    "title": f"Congress Bill: {bill.get('title')}",
                    "url": url_str,
                    "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"
                })
            return results
    except httpx.HTTPStatusError as e:
        logger.error("Congress HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Congress API error: {e.response.status_code}", "source": "CONGRESS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Congress request error: %s", str(e))
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]


async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not keyword_query: return []
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": keyword_query, "rows": 5}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            results = r.json().get("result", {}).get("results", [])
            out = []
            for item in results:
                resources = item.get("resources") or []
                url_obj = resources[0].get("url") if resources else None
                resource_url = None
                if isinstance(url_obj, dict): resource_url = url_obj.get("url")
                elif url_obj: resource_url = str(url_obj)
                dataset_page = f"https://catalog.data.gov/dataset/{item.get('name')}" if item.get("name") else None
                final_url = resource_url if resource_url and resource_url.startswith("http") else dataset_page
                out.append({
                    "title": f"Data.gov: {item.get('title')}",
                    "url": final_url,
                    "snippet": (item.get("notes") or "")[:300]
                })
            return out
    except httpx.HTTPStatusError as e:
        logger.error("Data.gov HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Data.gov API error: {e.response.status_code}", "source": "DATA.GOV", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Data.gov request error: %s", str(e))
        return [{"error": str(e), "source": "DATA.GOV", "status": "failed"}]

async def execute_query_plan(plan: Dict[str, Any], claim_type: str) -> List[Dict[str, Any]]:
    """Executes API calls based on the plan."""
    tier1 = plan.get("tier1_params", {}) or {}
    tier2_kws = plan.get("tier2_keywords", []) or []
    sources = pick_sources_from_type(claim_type)
    tasks = []

    if "BEA" in sources and tier1.get("bea"):
        bea_base = tier1["bea"]
        table = bea_base.get("TableName")
        if table and table in BEA_VALID_TABLES:
            line_codes = bea_base.get("LineCode")
            codes_to_run = []
            if isinstance(line_codes, list): codes_to_run = line_codes
            elif line_codes: codes_to_run = [line_codes]

            for code in codes_to_run:
                digits = re.findall(r"\d+", str(code))
                if digits:
                    params = bea_base.copy()
                    params["LineCode"] = digits[0]
                    tasks.append(query_bea(params))
                else: logger.warning("Skipping invalid BEA LineCode: %s", code)
        else: logger.warning("BEA table invalid/missing: %s", table)

    if "CENSUS" in sources and tier1.get("census"):
        tasks.append(query_census(params=tier1["census"]))

    for kw in tier2_kws:
        if "DATA.GOV" in sources: tasks.append(query_datagov(kw))
        if "CONGRESS" in sources: tasks.append(query_congress(keyword_query=kw))

    if not tasks: return []
    results = await asyncio.gather(*tasks)
    return [item for sublist in results if sublist for item in sublist]

async def synthesize_finding_with_llm(
    claim: str, claim_analysis: Dict[str, Any], sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Uses LLM to analyze evidence against the claim and determine verdict."""
    default_response = {
        "verdict": "Inconclusive",
        "summary": "Could not determine outcome based on available data.",
        "justification": "No supporting government data was found or the analysis failed.",
        "evidence_links": [],
    }

    valid_sources = [s for s in sources if s and "error" not in s]
    if not valid_sources:
        return default_response

    context_parts = []
    for s in valid_sources:
        part = f"Source Title: {s.get('title', 'N/A')}\nURL: {s.get('url', 'N/A')}\n"
        # If it's a BEA source with data, add structured info
        if "apps.bea.gov" in s.get("url", "") and s.get("data_value") is not None:
             part += (f"Data Point: {s.get('line_description')} ({s.get('line_code')}) "
                      f"= {s.get('raw_data_value')} {s.get('unit') or ''} "
                      f"(Multiplier: {s.get('unit_multiplier')})\n")
        part += f"Snippet: {s.get('snippet', 'N/A')}"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)

    prompt = f"""
You are an objective fact-checker. Analyze the provided evidence from U.S. government sources against the user's claim.

USER'S CLAIM: '''{claim}'''
Claim Analysis:
- Normalized: {claim_analysis.get('claim_normalized', claim)}
- Type: {claim_analysis.get('claim_type', 'Unknown')}
- Entities: {claim_analysis.get('entities', [])}
- Asserted Relationship: {claim_analysis.get('relationship', 'Unknown')}

AVAILABLE EVIDENCE:
{context}

INSTRUCTIONS:
1.  Carefully review the user's claim and its asserted relationship between entities.
2.  Examine ALL evidence provided. Look for data points directly relevant to the claim's entities and timeframe (if specified). Pay attention to structured data values from BEA. Apply multipliers if needed.
3.  Compare the relevant findings from the evidence to the claim's assertion.
4.  Determine the final `verdict`:
    - "Supported": If the evidence *clearly and directly* supports the claim's assertion.
    - "Contradicted": If the evidence *clearly and directly* contradicts the claim's assertion.
    - "Inconclusive": If the evidence is missing, insufficient, ambiguous, or irrelevant to make a clear judgment.
5.  Write a concise `summary` (1 sentence) stating the final conclusion based on the evidence. Start with "The available data suggests...", "The data supports...", "The data contradicts...", or "The data is insufficient...".
6.  Write a brief `justification` (1-2 sentences) explaining *why* you reached that verdict, citing specific data points or lack thereof from the evidence.
7.  Create `evidence_links` (list of {{"finding": "...", "source_url": "..."}}) linking key pieces of data used in your justification back to their source URLs from the evidence context.

Return ONLY a single valid JSON object with keys: "verdict", "summary", "justification", "evidence_links".
Example Response:
{{
  "verdict": "Contradicted",
  "summary": "The data contradicts the claim that federal defense spending was less than education spending in 2023.",
  "justification": "BEA data for 2023 shows National Defense spending (LineCode 2) was $XXX billion, while Education spending (LineCode 14) was $YYY billion. Since XXX > YYY, the claim is contradicted.",
  "evidence_links": [
    {{"finding": "National Defense Spending 2023 = $XXX billion", "source_url": "https://apps.bea.gov/api/data?..."}},
    {{"finding": "Education Spending 2023 = $YYY billion", "source_url": "https://apps.bea.gov/api/data?..."}}
  ]
}}
"""
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))
        if parsed and all(k in parsed for k in ["verdict", "summary", "justification", "evidence_links"]):
            if parsed["verdict"] not in ["Supported", "Contradicted", "Inconclusive"]:
                logger.warning("LLM returned invalid verdict: %s. Defaulting to Inconclusive.", parsed["verdict"])
                parsed["verdict"] = "Inconclusive"
            return parsed
        else:
            logger.error("Failed to parse valid synthesis JSON from LLM.")
            return default_response
    except HTTPException as e:
        logger.error("LLM failed during synthesis: %s", getattr(e, "detail", str(e)))
        default_response["justification"] += " (LLM call failed.)"
        return default_response
    except Exception:
        logger.exception("Unexpected error during LLM synthesis.")
        default_response["justification"] += " (Unexpected analysis error.)"
        return default_response

def compute_confidence(sources: List[Dict[str, Any]], verdict: str) -> Dict[str, Any]:
    """Calculates confidence score based on source reliability, density, and LLM verdict."""
    valid_sources = [s for s in sources if s and "error" not in s]
    if not valid_sources:
        return {"confidence": 0.3, "R": 0.5, "E": 0.0, "S": 0.3}
    total_weight = 0.0
    for s in valid_sources:
        url = (s.get("url") or "").lower()
        if "apps.bea.gov" in url: weight = 1.0
        elif "api.census.gov" in url: weight = 0.9
        elif "api.congress.gov" in url: weight = 0.8
        elif "catalog.data.gov" in url: weight = 0.7
        else: weight = 0.6
        total_weight += weight
    R = round(total_weight / len(valid_sources), 2)

    E = round(min(1.0, len(valid_sources) / 5.0), 2)

    if verdict == "Supported": S = 0.95
    elif verdict == "Contradicted": S = 0.90
    else: S = 0.5

    confidence = round((0.4 * R + 0.3 * E + 0.3 * S), 2)
    return {"confidence": confidence, "R": R, "E": E, "S": S}

@app.post("/verify")
async def verify(req: VerifyRequest):
    """Main endpoint to verify a claim."""
    claim = (req.claim or "").strip()
    if not claim: raise HTTPException(status_code=400, detail="Empty claim.")

    analysis = {}
    all_results = []
    synthesis_result = {}
    try:
        analysis = await analyze_claim_for_api_plan(claim)
        claim_norm = analysis.get("claim_normalized", claim)
        claim_type = analysis.get("claim_type", "Other")
        api_plan = analysis.get("api_plan", {})

        all_results = await execute_query_plan(api_plan, claim_type)
        sources_results = [r for r in all_results if r and "error" not in r]
        debug_errors = [r for r in all_results if r and "error" in r]

        synthesis_result = await synthesize_finding_with_llm(claim, analysis, sources_results)
        verdict = synthesis_result.get("verdict", "Inconclusive")
        summary_text = f"{synthesis_result.get('summary','')} {synthesis_result.get('justification','')}".strip()

        confidence_data = compute_confidence(sources_results, verdict)
        confidence_val = confidence_data["confidence"]

        if confidence_val > 0.75: confidence_tier = "High"
        elif confidence_val > 0.5: confidence_tier = "Medium"
        else: confidence_tier = "Low"

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
                "semantic_alignment": confidence_data["S"],
            },
            "summary": summary_text,
            "evidence_links": synthesis_result.get("evidence_links", []),
            "sources": sources_results,
            "debug_plan": api_plan,
            "debug_log": debug_errors,
        }

    except Exception as e:
        logger.exception("Unhandled error during /verify processing for claim: %s", claim)
        return {
            "claim_original": claim,"claim_normalized": claim,"claim_type": "Other",
            "verdict": "Error", "confidence": 0.0, "confidence_tier": "Low",
            "confidence_breakdown": {"R": 0.0, "E": 0.0, "S": 0.0},
            "summary": "An unexpected internal error occurred.",
            "evidence_links": [], "sources": [],
            "debug_plan": analysis.get("api_plan", {}), 
            "debug_log": [{"error": f"Unhandled exception: {str(e)}", "source": "internal", "status": "failed"}],
        }


