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
from math import sqrt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "GEMINI_API_KEY", "BEA_API_KEY", "CENSUS_API_KEY", "BLS_API_KEY", "CONGRESS_API_KEY"
]

def check_api_keys_on_startup():
    logger.info("Startup: checking for required API keys...")
    missing = [k for k in REQUIRED_KEYS if k != "DATA_GOV_API_KEY" and not os.getenv(k)]
    if missing:
        logger.warning("Missing critical env vars: %s. Corresponding API calls will fail.", ", ".join(missing))
    else:
        logger.info("All critical API keys seem present.")

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    check_api_keys_on_startup()

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
BLS_API_KEY = os.getenv("BLS_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_EMBED_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL_NAME}:embedContent"

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

async def get_embedding_api(text: str) -> Optional[List[float]]:
    """Calls the Google AI API to get text embeddings."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured. Cannot get embeddings.")
        return None
    
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"model": f"models/{EMBEDDING_MODEL_NAME}", "content": {"parts": [{"text": text}]}}
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(GEMINI_EMBED_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            embedding = data.get("embedding", {}).get("values", [])
            return embedding if embedding else None
    except httpx.HTTPStatusError as e:
        logger.error("Gemini Embed API HTTP error %s: %s", e.response.status_code, e.response.text)
        return None
    except Exception as e:
        logger.error("Error calling Gemini Embed API: %s", str(e))
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity for two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    
    mag_vec1 = sqrt(sum(v**2 for v in vec1))
    mag_vec2 = sqrt(sum(v**2 for v in vec2))
    
    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0.0
        
    return dot_product / (mag_vec1 * mag_vec2)

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    """Uses LLM to generate an API query plan based on the claim."""
    prompt_template = """
    You are a research analyst expert in U.S. government data APIs (BEA, Census, BLS, Data.gov, Congress.gov).
    Your goal is to analyze the user's claim and generate the *best possible single plan* to verify it using these APIs.

    **Analysis Steps:**
    1.  **Normalize Claim:** Rephrase the claim into a clear, verifiable statement (`claim_normalized`).
    2.  **Identify Entities:** List the specific concepts, agencies, metrics, locations, and timeframes mentioned (`entities`).
    3.  **Determine Claim Type:** Classify the claim (`claim_type`: quantitative_comparison, quantitative_value, factual, legislative, economic_indicator, other).
    4.  **Identify Relationship:** State the link asserted (e.g., "greater than", "value is X") (`relationship`).
    5.  **CRITICAL - Select API Strategy:** Based on the entities, choose the *most appropriate* API(s):
        * **Specific Agency/Department Budget/Spending?** (e.g., "Dept of Education budget", "NASA funding"): Primarily use `tier2_keywords` for `Data.gov`. **Do NOT use BEA.**
        * **Broad Economic Function Spending?** (e.g., "total spending on national defense function", "total education spending"): Use `BEA` (T31600).
        * **Demographic Data?** (e.g., "population of Alabama", "median income"): Use `Census ACS`.
        * **Inflation (CPI) or Unemployment Rate?**: Use `BLS`.
        * **Legislation/Bills?** (e.g., "CHIPS Act funding"): Primarily use `tier2_keywords` for `Congress.gov`.
        * **Other Specific Reports/Data/Topics?**: Use `tier2_keywords` for `Data.gov`.
    6.  **Generate API Plan:** Create the `api_plan` JSON. Fill `tier1_params` for BEA/Census/BLS if chosen. Fill `tier2_keywords` for Data.gov/Congress searches. If a specific API is chosen for Tier 1, still include relevant Tier 2 keywords as a backup or for context.

    **AVAILABLE APIs & Parameters:**
    -   `bea`: For broad *functions*. Needs `DataSetName` ("NIPA"), `TableName` ("T31600"), `Frequency` ("A"), `Year`, `LineCode` (e.g., "2" for Defense, "14" for Education).
    -   `census_acs`: For demographics. Needs `year`, `dataset` (e.g., "acs/acs1/profile"), `get` (vars, e.g., "NAME,DP05_0001E"), `for` (geo, e.g., "state:01").
    -   `bls`: For CPI/Unemployment. Needs `metric` ("CPI" or "unemployment") and `year`.
    -   `tier2_keywords`: List of search strings for Data.gov (budgets, reports) and Congress.gov (bills).

    **USER CLAIM:** '''{claim}'''

    **Return ONLY a single valid JSON object containing `claim_normalized`, `claim_type`, `entities`, `relationship`, and `api_plan`.**

    **Example (Agency Budget - Use Tier 2):**
    Claim: "The Department of Defense budget was over $800 billion in 2023."
    {{
      "claim_normalized": "The Department of Defense budget exceeded $800 billion in 2023.",
      "claim_type": "quantitative_value",
      "entities": ["Department of Defense Budget", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{ "bea": null, "census_acs": null, "bls": null }},
        "tier2_keywords": ["Department of Defense budget 2023", "FY2023 defense appropriations summary"]
      }}
    }}

    **Example (Broad Function - Use BEA):**
    Claim: "Total federal spending on the function of defense was more than education in 2023."
    {{
      "claim_normalized": "Total federal spending on the function of defense exceeded total federal spending on the function of education in 2023.",
      "claim_type": "quantitative_comparison",
      "entities": ["Federal Defense Function Spending", "Federal Education Function Spending", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{
          "bea": {{"DataSetName":"NIPA","TableName":"T31600","Frequency":"A","Year":"2023","LineCode":["2", "14"]}},
          "census_acs": null, "bls": null
        }},
        "tier2_keywords": ["federal spending by function 2023"]
      }}
    }}

    **Example (Inflation - Use BLS):**
    Claim: "Inflation was over 3% in 2023."
     {{
      "claim_normalized": "The US CPI inflation rate exceeded 3% in 2023.",
      "claim_type": "economic_indicator",
      "entities": ["US CPI Inflation", "2023"],
      "relationship": "greater than",
      "api_plan": {{
        "tier1_params": {{
          "bea": null, "census_acs": null,
          "bls": {{"metric": "CPI", "year": "2023"}}
        }},
        "tier2_keywords": ["US inflation rate 2023 annual average"]
      }}
    }}
    """
    prompt = prompt_template.format(claim=claim)
    fallback_plan = {
        "claim_normalized": claim, "claim_type": "Other", "entities": [], "relationship": "unknown",
        "api_plan": {"tier1_params": {}, "tier2_keywords": [claim, "general government data"]}
    }
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))

        if not parsed or "api_plan" not in parsed:
            logger.warning("Could not parse valid plan JSON from LLM. Falling back.")
            fallback_plan["debug_raw_llm_response"] = res.get("text", "No text found.")
            return fallback_plan

        parsed.setdefault("claim_normalized", claim)
        parsed.setdefault("claim_type", "Other")
        parsed.setdefault("entities", [])
        parsed.setdefault("relationship", "unknown")
        parsed.setdefault("api_plan", {"tier1_params": {}, "tier2_keywords": [claim]})
        parsed["api_plan"].setdefault("tier1_params", {})
        parsed["api_plan"]["tier1_params"].setdefault("bea", None)
        parsed["api_plan"]["tier1_params"].setdefault("census_acs", None)
        parsed["api_plan"]["tier1_params"].setdefault("bls", None)
        parsed["api_plan"].setdefault("tier2_keywords", [claim] if not parsed["api_plan"].get("tier2_keywords") else parsed["api_plan"].get("tier2_keywords"))

        return parsed
    except HTTPException as e:
        logger.error("LLM failed generating plan: %s", getattr(e, "detail", str(e)))
        fallback_plan["debug_exception"] = str(e)
        return fallback_plan
    except Exception as e:
        logger.exception("Unexpected error generating plan. Falling back.")
        fallback_plan["debug_exception"] = str(e)
        return fallback_plan
        
def pick_sources_from_type(claim_type: str) -> List[str]:
    """Selects APIs based on claim type (more robust)."""
    sources = {"DATA.GOV"}
    ct_lower = (claim_type or "").lower()
    if "quantitative" in ct_lower: sources.update({"BEA", "CENSUS", "BLS"})
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

async def query_bls(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Queries the BLS API for key metrics like CPI (inflation) or Unemployment.
    Params expected: {"metric": "CPI" | "unemployment", "year": "YYYY"}
    """
    if not BLS_API_KEY:
        return [{"error": "BLS_API_KEY missing", "source": "BLS", "status": "failed"}]

    metric = params.get("metric")
    year_str = params.get("year")
    if not metric or not year_str:
        return [{"error": "BLS query missing metric or year", "source": "BLS", "status": "failed"}]

    try:
        year_int = int(year_str)
    except ValueError:
        return [{"error": "BLS query invalid year", "source": "BLS", "status": "failed"}]

    series_map = {
        "CPI": "CUSR0000SA0",
        "unemployment": "LNS14000000"
    }

    series_id = series_map.get(metric)
    if not series_id:
        return [{"error": f"BLS metric '{metric}' not supported", "source": "BLS", "status": "failed"}]

    start_year = str(year_int - 1) if metric == "CPI" else year_str
    end_year = year_str

    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = json.dumps({
        "seriesid": [series_id],
        "startyear": start_year,
        "endyear": end_year,
        "registrationKey": BLS_API_KEY,
        "annualaverage": True
    })
    headers = {'Content-Type': 'application/json'}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, content=payload)
            r.raise_for_status()
            data = r.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
             message = data.get("message", ["Unknown BLS error."])[0]
             logger.error(f"BLS API error: {message}")
             return [{"error": f"BLS API error: {message}", "source": "BLS", "status": "failed"}]

    except httpx.HTTPStatusError as e:
        logger.error("BLS HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"BLS API error: {e.response.status_code}", "source": "BLS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("BLS request error: %s", str(e))
        return [{"error": str(e), "source": "BLS", "status": "failed"}]
    except json.JSONDecodeError:
        logger.error("BLS returned non-JSON response: %s", r.text[:200])
        return [{"error": "BLS API returned invalid JSON", "source": "BLS", "status": "failed"}]


    try:
        series_data = data.get("Results", {}).get("series", [])
        if not series_data:
            return [{"error": "BLS returned no data for series", "source": "BLS", "status": "no_data"}]
        annual_data = series_data[0].get("data", [])
        if not annual_data:
             return [{"error": "BLS returned no annual data points", "source": "BLS", "status": "no_data"}]

        year_values = {}
        for item in annual_data:
            if item.get("period") == "M13" and item.get("year") in [year_str, str(year_int -1)]:
                 value = _parse_numeric_value(item.get("value"))
                 if value is not None:
                     year_values[item.get("year")] = value


        if metric == "CPI":
            current_val = year_values.get(year_str)
            prev_val = year_values.get(str(year_int - 1))

            if current_val is None or prev_val is None:
                 logger.warning(f"BLS missing annual average CPI data for {year_str} or {year_int-1}. Available: {year_values}")
                 return [{"error": f"BLS missing annual average CPI data for {year_str} or {year_int-1}", "source": "BLS", "status": "missing_data"}]
            if prev_val == 0:
                 return [{"error": f"BLS CPI data for {year_int-1} is zero, cannot calculate change.", "source": "BLS", "status": "calculation_error"}]

            percent_change = ((current_val - prev_val) / prev_val) * 100
            data_value = round(percent_change, 1)
            snippet = f"Annual average CPI inflation rate for {year_str} was {data_value}%."
            title = f"BLS: CPI Inflation Rate {year_str}"

        elif metric == "unemployment":
            data_value = year_values.get(year_str)
            if data_value is None:
                logger.warning(f"BLS missing annual average unemployment data for {year_str}. Available: {year_values}")
                return [{"error": f"BLS missing annual average unemployment data for {year_str}", "source": "BLS", "status": "missing_data"}]

            snippet = f"Annual average unemployment rate for {year_str} was {data_value}%."
            title = f"BLS: Unemployment Rate {year_str}"
        else:
             return [{"error": "Invalid BLS metric processing.", "source": "BLS", "status":"failed"}]


        return [{
            "title": title,
            "url": f"https://data.bls.gov/timeseries/{series_id}",
            "snippet": snippet,
            "data_value": data_value,
            "raw_data_value": str(data_value),
            "unit": "%"
        }]

    except Exception as e:
        logger.exception("Failed to parse BLS response")
        return [{"error": f"BLS parsing error: {str(e)}", "source": "BLS", "status": "failed"}]

async def query_census_acs(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Queries the Census ACS API (e.g., /data/2022/acs/acs1/profile).
    Params expected: {"year": "YYYY", "dataset": "path", "get": "VARS", "for": "GEO"}
    """
    if not CENSUS_API_KEY:
        return [{"error": "CENSUS_API_KEY missing", "source": "CENSUS", "status": "failed"}]
    if not all(k in params for k in ["year", "dataset", "get", "for"]):
        logger.warning("Census ACS query missing required params: %s", params)
        return []

    year = params["year"]
    dataset = params["dataset"].strip("/")
    
    url = f"https://api.census.gov/data/{year}/{dataset}"
    final_params = {
        "key": CENSUS_API_KEY,
        "get": params["get"],
        "for": params["for"]
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            data = r.json()

            if not data or len(data) < 2:
                logger.info("Census query returned no data.")
                return []

            headers = data[0]
            rows = data[1:]
            results = []

            for row in rows:
                row_data = dict(zip(headers, row))
                
                snippet_parts = []
                for k, v in row_data.items():
                    if k.upper() not in ["KEY", "FOR", "IN", "STATE", "COUNTY"]:
                        snippet_parts.append(f"{k}: {v}")
                
                snippet = f"Data for {row_data.get('NAME', params['for'])}: " + ", ".join(snippet_parts)
                
                primary_var = params["get"].split(",")[1] if "," in params["get"] else None
                data_value_raw = row_data.get(primary_var)
                numeric_val = _parse_numeric_value(data_value_raw)

                results.append({
                    "title": f"Census ACS: {params['get']} for {row_data.get('NAME', params['for'])}",
                    "url": str(r.url),
                    "snippet": snippet,
                    "data_value": numeric_val,
                    "raw_data_value": data_value_raw,
                    "raw_census_row": row_data
                })
            return results

    except httpx.HTTPStatusError as e:
        logger.error("Census HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Census API error: {e.response.status_code}", "source": "CENSUS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Census request error: %s", str(e))
        return [{"error": str(e), "source": "CENSUS", "status": "failed"}]
    except json.JSONDecodeError:
        logger.error("Census returned non-JSON response: %s", r.text[:200])
        return [{"error": "Census API returned invalid JSON", "source": "CENSUS", "status": "failed"}]


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
    tier1 = plan.get("tier1_params", {}) or {} 
    tier2_kws = plan.get("tier2_keywords", []) or []    

    tasks = []

    if bea_params := tier1.get("bea"):
        table = bea_params.get("TableName")
        if table and table in BEA_VALID_TABLES:
            line_codes = bea_params.get("LineCode")
            codes_to_run = []
            if isinstance(line_codes, list):
                codes_to_run = line_codes
            elif line_codes:
                 codes_to_run = [line_codes]
            for code in codes_to_run:
                digits = re.findall(r"\d+", str(code))
                if digits:
                    params = bea_params.copy()
                    params["LineCode"] = digits[0]
                    tasks.append(query_bea(params))
                else:
                    logger.warning("Skipping invalid BEA LineCode format: %s", code)
        elif table:
             logger.warning("BEA table specified but invalid/not in supported list: %s", table)

    if census_params := tier1.get("census_acs"):
        if all(k in census_params for k in ["year", "dataset", "get", "for"]):
             tasks.append(query_census_acs(params=census_params))
        else:
             logger.warning("Census ACS plan missing required parameters: %s", census_params)

    if bls_params := tier1.get("bls"):
        if all(k in bls_params for k in ["metric", "year"]):
             tasks.append(query_bls(params=bls_params))
        else:
             logger.warning("BLS plan missing required parameters: %s", bls_params)

    unique_kws = sorted(list(set(kw for kw in tier2_kws if kw)))
    for kw in unique_kws:
        tasks.append(query_datagov(kw))
        if "bill" in kw.lower() or "act" in kw.lower() or "law" in kw.lower() or claim_type == "legislative":
             tasks.append(query_congress(keyword_query=kw))

    if not tasks:
        logger.warning("No API calls generated for the plan.")
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Error during API call task {i}: {res}")
            processed_results.append({"error": f"Task execution failed: {res}", "source": "internal", "status": "failed"})
        elif isinstance(res, list):
            processed_results.extend(res)
        elif res is not None:
            processed_results.append(res)
    return processed_results

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
        
        if "apps.bea.gov" in s.get("url", "") and s.get("data_value") is not None:
            part += (f"Data Point: {s.get('line_description')} ({s.get('line_code')}) "
                     f"= {s.get('raw_data_value')} {s.get('unit') or ''} "
                     f"(Multiplier: {s.get('unit_multiplier')})\n")
        elif "api.census.gov" in s.get("url", "") and s.get("data_value") is not None:
             part += (f"Data Point: {s.get('title', 'Census Data')} "
                      f"= {s.get('raw_data_value')}\n")
            
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
    2.  Examine ALL evidence provided. Look for data points (like from BEA, Census, or BLS) directly relevant to the claim's entities and timeframe.
    3.  For BEA data, apply the 'Multiplier' (e.g., a 'DataValue' of 1000 and 'UnitMultiplier' of 1000000 means 100,000,000,000).
    4.  For Census data, use the provided 'Data Point' values.
    **5.  For BLS data, use the 'Data Point' which represents a calculated percentage (e.g., 3.5 for 3.5%). The snippet will clarify the metric (e.g., "CPI inflation rate" or "unemployment rate").**
    6.  Compare the relevant findings from the evidence to the claim's assertion.
    7.  Determine the final `verdict`:
        - "Supported": If the evidence *clearly and directly* supports the claim's assertion.
        - "Contradicted": If the evidence *clearly and directly* contradicts the claim's assertion.
        - "Inconclusive": If the evidence is missing, insufficient, ambiguous, or irrelevant to make a clear judgment.
    8.  Write a concise `summary` (1 sentence) stating the final conclusion based on the evidence.
    9.  Write a brief `justification` (1-2 sentences) explaining *why* you reached that verdict, citing specific data points.
    10. Create `evidence_links` (list of {{"finding": "...", "source_url": "..."}}) linking key data to their source URLs. **You must be extremely careful to match each finding to the *exact* source_url it came from in the evidence context.**
    
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

async def compute_confidence(sources: List[Dict[str, Any]], verdict: str, claim: str) -> Dict[str, Any]:
    """
    Calculates confidence score based on source reliability (R), 
    evidence density (E), and semantic alignment (S) using API embeddings.
    """
    valid_sources = [s for s in sources if s and "error" not in s]
    
    if not valid_sources:
        return {"confidence": 0.3, "R": 0.5, "E": 0.0, "S": 0.3} 
        
    total_weight = 0.0
    for s in valid_sources:
        url = (s.get("url") or "").lower()
        if "apps.bea.gov" in url: weight = 1.0
        elif "api.census.gov" in url: weight = 1.0
        elif "api.bls.gov" in url: weight = 1.0
        elif "api.congress.gov" in url: weight = 0.8
        elif "catalog.data.gov" in url: weight = 0.7
        else: weight = 0.6
        total_weight += weight
    
    R = round(total_weight / len(valid_sources), 2)
    E = round(min(1.0, len(valid_sources) / 5.0), 2)
    S_llm_verdict = 0.5 
    if verdict == "Supported": S_llm_verdict = 0.95
    elif verdict == "Contradicted": S_llm_verdict = 0.90
    S_semantic_sim = 0.0
    
    try:
        evidence_texts = []
        for s in valid_sources:
            evidence_texts.append(s.get('snippet', ''))
            evidence_texts.append(s.get('title', ''))
            if s.get('data_value') is not None:
                evidence_texts.append(f"{s.get('line_description', '')} is {s.get('raw_data_value')}")
        
        evidence_texts = [t for t in evidence_texts if t and isinstance(t, str)]

        if evidence_texts:
            claim_embedding = await get_embedding_api(claim)
            
            evidence_tasks = [get_embedding_api(text) for text in evidence_texts]
            evidence_embeddings = await asyncio.gather(*evidence_tasks)
            
            valid_embeddings = [emb for emb in evidence_embeddings if emb]
            
            if claim_embedding and valid_embeddings:
                similarities = [cosine_similarity(claim_embedding, emb) for emb in valid_embeddings]
                S_semantic_sim = round(float(max(similarities)), 2)
            else:
                S_semantic_sim = 0.3

    except Exception as e:
        logger.error("Error during semantic similarity API calculation: %s", e)
        S_semantic_sim = 0.3 
    
    S = round((S_llm_verdict * 0.7) + (S_semantic_sim * 0.3), 2)

    confidence = round((0.5 * R + 0.2 * E + 0.3 * S), 2)
    
    return {"confidence": confidence, "R": R, "E": E, "S": S, "S_semantic_sim": S_semantic_sim}

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

        confidence_data = await compute_confidence(sources_results, verdict, claim_norm)
        
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
            "claim_original": claim,
            "claim_normalized": claim,
            "claim_type": "Other",
            "verdict": "Error",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "confidence_breakdown": {
                "R": 0.0,
                "E": 0.0,
                "S": 0.0,
            },
            "summary": "An unexpected internal error occurred.",
            "evidence_links": [],
            "sources": [],
            "debug_plan": analysis.get("api_plan", {}),
            "debug_log": [
                {
                    "error": f"Unhandled exception: {str(e)}",
                    "source": "internal",
                    "status": "failed",
                }
            ],
        }


