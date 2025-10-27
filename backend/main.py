import os
import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
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
GEMINI_BATCH_EMBED_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL_NAME}:batchEmbedContents"


BEA_VALID_TABLES = {
    "T10101", "T20305", "T31600", "T70500"
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running :)"}

class VerifyRequest(BaseModel):
    claim: str

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
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
    if val is None: return None
    try:
        s = str(val).strip().replace(",", "").replace("$", "")
        if s.startswith("(") and s.endswith(")"): s = "-" + s[1:-1]
        m = re.match(r"^(-?[\d\.eE+-]+)", s)
        return float(m.group(1)) if m else float(s)
    except (ValueError, TypeError):
        return None

def _apply_multiplier(value: Optional[float], multiplier: Optional[Any]) -> Optional[float]:
    if value is None: return None
    if multiplier is None: return value
    try:
        return float(value) * float(multiplier)
    except (ValueError, TypeError):
        return value

async def call_gemini(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not configured.")
        raise HTTPException(status_code=500, detail="LLM API key not configured on server.")

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Gemini HTTP error %s for URL %s: %s", e.response.status_code, e.request.url, e.response.text)
        raise HTTPException(status_code=500, detail=f"LLM API error: Status {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error("Gemini request error for URL %s: %s", e.request.url, str(e))
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error calling Gemini API.")
        raise HTTPException(status_code=500, detail="Unexpected server error during LLM call.")


    text = ""
    try:
        if isinstance(data, dict):
            candidates = data.get("candidates", [])
            if isinstance(candidates, list) and candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if isinstance(parts, list) and parts:
                    text = parts[0].get("text", "")
            if not text:
                 text = data.get("output", "") or data.get("text", "")
    except (AttributeError, IndexError, TypeError) as e:
        logger.error("Error parsing Gemini response structure: %s. Response: %s", e, data)
        text = json.dumps(data)


    return {"raw": data, "text": text or json.dumps(data)}

async def get_embeddings_batch_api(texts: List[str]) -> List[Optional[List[float]]]:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured. Cannot get embeddings.")
        return [None] * len(texts)
    if not texts:
        return []

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    MAX_BATCH_SIZE = 100
    all_embeddings: List[Optional[List[float]]] = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch_texts = texts[i:i + MAX_BATCH_SIZE]
        requests_body = []
        for text in batch_texts:
            processed_text = text if text and text.strip() else " "
            requests_body.append({
                "model": f"models/{EMBEDDING_MODEL_NAME}",
                "content": {"parts": [{"text": processed_text}]}
            })

        body = {"requests": requests_body}
        batch_results = [None] * len(batch_texts)

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(GEMINI_BATCH_EMBED_ENDPOINT, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                embeddings_list = data.get("embeddings", [])

                if len(embeddings_list) == len(batch_texts):
                    for j, emb_data in enumerate(embeddings_list):
                        values = emb_data.get("values")
                        if values and isinstance(values, list):
                            batch_results[j] = values
                        else:
                            logger.warning(f"Received invalid or missing embedding values for text index {i+j} in batch.")
                else:
                    logger.error(f"Batch embedding response length mismatch in batch starting at index {i}: got {len(embeddings_list)}, expected {len(batch_texts)}")

        except httpx.HTTPStatusError as e:
            logger.error("Gemini Batch Embed API HTTP error %s in batch starting at index %i: %s", e.response.status_code, i, e.response.text)
        except httpx.RequestError as e:
            logger.error("Gemini Batch Embed API request error in batch starting at index %i: %s", i, str(e))
        except Exception as e:
            logger.error("Error calling/processing Gemini Batch Embed API for batch starting at index %i: %s", i, str(e))

        all_embeddings.extend(batch_results)

    return all_embeddings

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = 0.0
    mag_vec1_sq = 0.0
    mag_vec2_sq = 0.0

    for v1, v2 in zip(vec1, vec2):
        dot_product += v1 * v2
        mag_vec1_sq += v1**2
        mag_vec2_sq += v2**2

    mag_vec1 = sqrt(mag_vec1_sq)
    mag_vec2 = sqrt(mag_vec2_sq)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0.0

    return dot_product / (mag_vec1 * mag_vec2)


async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
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
        kw = parsed["api_plan"].get("tier2_keywords")
        parsed["api_plan"]["tier2_keywords"] = kw if isinstance(kw, list) and kw else [claim]

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
    sources = {"DATA.GOV"}
    ct_lower = (claim_type or "").lower()
    if "quantitative" in ct_lower or "economic" in ct_lower:
          sources.update({"BEA", "CENSUS", "BLS"})
    if "factual" in ct_lower: sources.add("CONGRESS")
    if "legislative" in ct_lower: sources.add("CONGRESS")
    return list(sources)

async def query_bea(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not BEA_API_KEY:
        return [{"error": "BEA_API_KEY missing", "source": "BEA", "status": "failed"}]

    requested_line_code_str = str(params.get("LineCode", "")).strip()
    if not requested_line_code_str:
          logger.warning("BEA query called without a specific LineCode in params.")
          return [{"error": "BEA query missing LineCode", "source": "BEA", "status": "failed"}]
    is_requested_code_numeric = requested_line_code_str.isdigit()

    final_params = {
        "UserID": BEA_API_KEY, "method": "GetData", "ResultFormat": "json",
        "DataSetName": params.get("DataSetName"), "TableName": params.get("TableName"),
        "Frequency": params.get("Frequency"), "Year": params.get("Year"),
        "LineCode": requested_line_code_str,
        "GeoFips": params.get("GeoFips"),
    }
    api_params = {k: v for k, v in final_params.items() if v is not None}
    url = "https://apps.bea.gov/api/data"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=api_params)
            r.raise_for_status()
            payload = r.json()
            request_url = str(r.url)
    except httpx.HTTPStatusError as e:
        logger.error("BEA HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"BEA API error: {e.response.status_code}", "source": "BEA", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("BEA request error: %s", str(e))
        return [{"error": str(e), "source": "BEA", "status": "failed"}]
    except json.JSONDecodeError:
          logger.error("BEA returned non-JSON response: %s", r.text[:200])
          return [{"error": "BEA API returned invalid JSON", "source": "BEA", "status": "failed"}]

    results_data = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    out: List[Dict[str, Any]] = []
    if not results_data:
        logger.info("BEA returned no data rows for params: %s", api_params)
        return []

    found_match = False
    for item in results_data:
        returned_line_code_obj = item.get("LineCode") or item.get("SeriesCode")
        if returned_line_code_obj is None:
            continue

        returned_line_code_str = str(returned_line_code_obj).strip()

        match = False
        if returned_line_code_str == requested_line_code_str:
            match = True
        elif is_requested_code_numeric:
            known_mappings_t31600 = {
                "2": ["G16007", "G16046"],
                "14": ["G16029", "G16068", "G16107"]
            }
            if params.get("TableName") == "T31600" and requested_line_code_str in known_mappings_t31600:
                  if returned_line_code_str in known_mappings_t31600[requested_line_code_str]:
                       match = True

        if not match:
            continue

        found_match = True
        desc = item.get("LineDescription") or item.get("SeriesDescription") or "Data"
        data_value_raw = item.get("DataValue") or ""
        numeric = _parse_numeric_value(data_value_raw)
        unit = item.get("Unit") or ""
        unit_multiplier = item.get("UnitMultiplier")
        time_period = item.get("TimePeriod") or final_params.get("Year")
        snippet = f"{desc} ({returned_line_code_str}) for {time_period}: {data_value_raw}{' '+unit if unit else ''}."

        out.append({
            "title": f"BEA: {params.get('DataSetName')}/{params.get('TableName')}",
            "url": request_url,
            "snippet": snippet,
            "data_value": numeric,
            "raw_data_value": data_value_raw,
            "unit": unit,
            "unit_multiplier": unit_multiplier,
            "line_description": desc,
            "line_code": returned_line_code_str,
            "raw_year": item.get("TimePeriod"),
            "raw_geo": item.get("GeoFips"),
        })
        if params.get("TableName") == "T31600":
    for item_out in out:
        if item_out.get("unit_multiplier") is None:
            item_out["unit_multiplier"] = 1000000
        if not item_out.get("unit"):
            item_out["unit"] = "Millions of Dollars"

    if not found_match and results_data:
          logger.warning("BEA returned data for params %s, but no rows matched filter for requested LineCode %s", api_params, requested_line_code_str)

    return out


async def query_bls(params: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            logger.warning("BLS returned no data for series %s, years %s-%s", series_id, start_year, end_year)
            return [{"error": "BLS returned no data for series", "source": "BLS", "status": "no_data"}]

        annual_data = series_data[0].get("data", [])
        if not annual_data:
             logger.warning("BLS returned no annual data points for series %s, years %s-%s", series_id, start_year, end_year)
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
                 logger.error(f"BLS CPI data for previous year {year_int-1} is zero, cannot calculate change.")
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

            data_value = round(data_value, 1)
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
    if not CENSUS_API_KEY:
        return [{"error": "CENSUS_API_KEY missing", "source": "CENSUS", "status": "failed"}]
    required_keys = ["year", "dataset", "get", "for"]
    if not all(k in params and params[k] for k in required_keys):
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
            if r.status_code == 204:
                logger.info("Census query returned status 204 No Content for params: %s", final_params)
                return []
            r.raise_for_status()
            if "application/json" not in r.headers.get("content-type", "").lower():
                 logger.error("Census returned non-JSON content-type: %s. Response: %s", r.headers.get("content-type"), r.text[:200])
                 return [{"error": f"Census API returned non-JSON content-type: {r.headers.get('content-type')}", "source": "CENSUS", "status": "failed"}]

            data = r.json()

            if not isinstance(data, list) or len(data) < 2:
                logger.info("Census query returned no data rows or invalid format for params: %s. Response: %s", final_params, data)
                return []

            headers = data[0]
            rows = data[1:]
            results = []

            for row in rows:
                try:
                    row_data = dict(zip(headers, row))
                except TypeError:
                    logger.warning("Failed to zip Census headers and row. Headers: %s, Row: %s", headers, row)
                    continue

                snippet_parts = []
                for k, v in row_data.items():
                    if k and k.upper() not in ["KEY", "FOR", "IN", "STATE", "COUNTY", "US"]:
                        snippet_parts.append(f"{k}: {v}")

                geo_name = row_data.get('NAME', params['for'])
                snippet = f"Data for {geo_name}: " + ", ".join(snippet_parts)

                get_vars = params["get"].split(',')
                primary_var = get_vars[1] if len(get_vars) > 1 else get_vars[0]
                primary_var = primary_var.strip()

                data_value_raw = row_data.get(primary_var)
                numeric_val = _parse_numeric_value(data_value_raw)

                results.append({
                    "title": f"Census ACS: {primary_var} for {geo_name}",
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
        logger.error("Census returned invalid JSON: %s", r.text[:200])
        return [{"error": "Census API returned invalid JSON", "source": "CENSUS", "status": "failed"}]
    except Exception as e:
          logger.exception("Unexpected error during Census query")
          return [{"error": f"Unexpected error processing Census data: {str(e)}", "source": "CENSUS", "status": "failed"}]


async def query_congress(keyword_query: str) -> List[Dict[str, Any]]:
    if not CONGRESS_API_KEY:
        return [{"error": "CONGRESS_API_KEY missing", "source": "CONGRESS", "status": "failed"}]
    if not keyword_query or not keyword_query.strip():
        return []

    params = {"api_key": CONGRESS_API_KEY, "q": keyword_query, "limit": 3}
    url = "https://api.congress.gov/v3/bill"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            bills = data.get("bills", [])
            results = []
            for bill in bills:
                title = bill.get('title', 'N/A')
                bill_number = f"{bill.get('type', '')}{bill.get('number', '')}"
                congress_num = bill.get('congress', '')
                latest_action_text = bill.get('latestAction', {}).get('text', 'No recent action text.')
                bill_url = f"https://www.congress.gov/bill/{congress_num}th-congress/{bill.get('type','').lower()}-bill/{bill.get('number','')}" if congress_num and bill.get('type') and bill.get('number') else None

                results.append({
                    "title": f"Congress Bill: {title} ({bill_number})",
                    "url": bill_url,
                    "snippet": f"Latest Action: {latest_action_text}"
                })
            return results
    except httpx.HTTPStatusError as e:
        logger.error("Congress API HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Congress API error: {e.response.status_code}", "source": "CONGRESS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Congress API request error: %s", str(e))
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]
    except Exception as e:
          logger.exception("Unexpected error during Congress query")
          return [{"error": f"Unexpected error processing Congress data: {str(e)}", "source": "CONGRESS", "status": "failed"}]


async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not keyword_query or not keyword_query.strip():
        return []

    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": keyword_query, "rows": 5}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            results_list = data.get("result", {}).get("results", [])
            out = []
            for item in results_list:
                title = item.get('title', 'N/A')
                notes = (item.get("notes") or "")[:300]
                resources = item.get("resources") or []
                resource_url = None
                if resources and isinstance(resources, list):
                    preferred_formats = ['csv', 'json', 'xls', 'xlsx', 'zip', 'pdf']
                    found_url = None
                    for res in resources:
                        res_url = res.get('url')
                        res_format = (res.get('format') or '').lower()
                        if res_url and any(fmt in res_format for fmt in preferred_formats):
                            found_url = res_url
                            break
                    if not found_url and resources:
                          resource_url = resources[0].get('url')
                    else:
                          resource_url = found_url

                dataset_page = f"https://catalog.data.gov/dataset/{item.get('name')}" if item.get("name") else None

                final_url = None
                if isinstance(resource_url, str) and resource_url.startswith("http"):
                    final_url = resource_url
                elif dataset_page:
                    final_url = dataset_page

                out.append({
                    "title": f"Data.gov: {title}",
                    "url": final_url,
                    "snippet": notes
                })
            return out
    except httpx.HTTPStatusError as e:
        logger.error("Data.gov API HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Data.gov API error: {e.response.status_code}", "source": "DATA.GOV", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Data.gov API request error: %s", str(e))
        return [{"error": str(e), "source": "DATA.GOV", "status": "failed"}]
    except Exception as e:
          logger.exception("Unexpected error during Data.gov query")
          return [{"error": f"Unexpected error processing Data.gov data: {str(e)}", "source": "DATA.GOV", "status": "failed"}]

async def execute_query_plan(plan: Dict[str, Any], claim_type: str) -> List[Dict[str, Any]]:
    tier1 = plan.get("tier1_params", {}) or {}
    tier2_kws = plan.get("tier2_keywords", []) or []

    tasks = []

    if bea_params := tier1.get("bea"):
        if isinstance(bea_params, dict):
            table = bea_params.get("TableName")
            if table and table in BEA_VALID_TABLES:
                line_codes = bea_params.get("LineCode")
                codes_to_run = []
                if isinstance(line_codes, list):
                    codes_to_run = [str(c).strip() for c in line_codes if str(c).strip()]
                elif line_codes:
                    codes_to_run = [str(line_codes).strip()]

                for code in codes_to_run:
                    if re.match(r"^[A-Z]?\d+[A-Z]?\d*$", code) or code.isdigit():
                        params_copy = bea_params.copy()
                        params_copy["LineCode"] = code
                        tasks.append(query_bea(params_copy))
                    else:
                        logger.warning("Skipping invalid BEA LineCode format in plan: %s", code)
            elif table:
                logger.warning("BEA table specified in plan but not in supported list: %s", table)
            else:
                logger.warning("BEA parameters in plan are not a dictionary: %s", bea_params)


    if census_params := tier1.get("census_acs"):
          if isinstance(census_params, dict) and all(k in census_params for k in ["year", "dataset", "get", "for"]):
               tasks.append(query_census_acs(params=census_params))
          elif isinstance(census_params, dict):
               logger.warning("Census ACS plan missing required parameters: %s", census_params)


    if bls_params := tier1.get("bls"):
          if isinstance(bls_params, dict) and all(k in bls_params for k in ["metric", "year"]):
               tasks.append(query_bls(params=bls_params))
          elif isinstance(bls_params, dict):
               logger.warning("BLS plan missing required parameters: %s", bls_params)


    unique_kws = sorted(list(set(kw for kw in tier2_kws if isinstance(kw, str) and kw.strip())))

    for kw in unique_kws:
        tasks.append(query_datagov(kw))
        if "bill" in kw.lower() or "act" in kw.lower() or "law" in kw.lower() or "congress" in kw.lower() or claim_type == "legislative":
            tasks.append(query_congress(keyword_query=kw))

    if not tasks:
        logger.warning("No API calls generated for the plan.")
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Error during API call task index {i}: {res}", exc_info=True)
            processed_results.append({"error": f"Task execution failed: {type(res).__name__}", "source": "internal", "status": "failed"})
        elif isinstance(res, list):
            processed_results.extend(res)
        elif isinstance(res, dict) and "error" in res:
             processed_results.append(res)
        elif res is not None:
             logger.warning(f"Unexpected result type from task index {i}: {type(res)}. Result: {res}")

    return processed_results


async def synthesize_finding_with_llm(
    claim: str, claim_analysis: Dict[str, Any], sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    default_response = {
        "verdict": "Inconclusive",
        "summary": "Could not determine outcome based on available data.",
        "justification": "No supporting government data was found or the analysis failed.",
        "evidence_links": [],
    }

    valid_sources = [s for s in sources if s and "error" not in s]
    if not valid_sources:
        error_sources = [s for s in sources if s and "error" in s]
        if error_sources:
             default_response["justification"] += f" (API errors encountered: {len(error_sources)})."
        else:
             default_response["justification"] += " (No relevant data sources found)."
        return default_response

    context_parts = []
    source_map_for_linking = {}

    for idx, s in enumerate(valid_sources):
        source_id = f"Source_{idx+1}"
        url = s.get('url', 'N/A')
        title = s.get('title', 'N/A')
        part = f"<{source_id}>\nSource Title: {title}\nURL: {url}\n"

        data_point_text = None
        snippet = s.get('snippet', 'N/A').strip()

        if "apps.bea.gov" in url and s.get("data_value") is not None:
            raw_val = s.get('raw_data_value', 'N/A')
            unit = s.get('unit', '')
            mult = s.get('unit_multiplier')
            line_desc = s.get('line_description', 'BEA Data')
            line_code = s.get('line_code', '')
            data_point_text = f"{line_desc} ({line_code}) = {raw_val}{' '+unit if unit else ''}"
            part += f"Data Point: {data_point_text} (Multiplier: {mult})\n"
        elif "api.census.gov" in url and s.get("data_value") is not None:
             raw_val = s.get('raw_data_value', 'N/A')
             data_point_text = f"{title} = {raw_val}"
             part += f"Data Point: {data_point_text}\n"
        elif "bls.gov" in url and s.get("data_value") is not None:
             raw_val = s.get('raw_data_value', 'N/A')
             data_point_text = snippet
             part += f"Data Point: {data_point_text}\n"

        part += f"Snippet: {snippet}\n</{source_id}>\n"
        context_parts.append(part)

        if data_point_text:
             source_map_for_linking[data_point_text] = url
        source_map_for_linking[snippet] = url


    context = "\n---\n".join(context_parts)
    MAX_CONTEXT_LENGTH = 30000
    if len(context) > MAX_CONTEXT_LENGTH:
          context = context[:MAX_CONTEXT_LENGTH] + "\n... [Context Truncated]"

    prompt = f"""
    You are an objective fact-checker. Analyze the provided evidence from U.S. government sources against the user's claim.

    USER'S CLAIM: '''{claim}'''
    Claim Analysis:
    - Normalized: {claim_analysis.get('claim_normalized', claim)}
    - Type: {claim_analysis.get('claim_type', 'Unknown')}
    - Entities: {claim_analysis.get('entities', [])}
    - Asserted Relationship: {claim_analysis.get('relationship', 'Unknown')}

    AVAILABLE EVIDENCE (Each source is tagged with <Source_N>):
    {context}

    INSTRUCTIONS:
    1.  Carefully review the user's claim and its asserted relationship between entities.
    2.  Examine ALL evidence provided within the <Source_N> tags. Focus on data points (BEA, Census, BLS) directly relevant to the claim's entities and timeframe.
    3.  **BEA Data:** Apply the 'Multiplier' if provided (e.g., a 'DataValue' of 1000 and 'Multiplier' of 1000000 means 1,000,000,000). Assume "Millions of dollars" (multiplier 1,000,000) for BEA NIPA tables like T31600 if multiplier is null/missing but units aren't specified.
    4.  **Census Data:** Use the provided 'Data Point' values directly.
    5.  **BLS Data:** Use the 'Data Point' which represents a calculated percentage (e.g., 3.5 for 3.5% or a raw index value). The Snippet/Title clarifies the metric.
    6.  Compare relevant findings from the evidence to the claim's assertion. Perform calculations if necessary (e.g., comparisons).
    7.  Determine the final `verdict`:
        - "Supported": If evidence *clearly and directly* supports the claim's assertion.
        - "Contradicted": If evidence *clearly and directly* contradicts the claim's assertion.
        - "Inconclusive": If evidence is missing, insufficient, ambiguous, irrelevant, or requires assumptions beyond the data to make a clear judgment.
    8.  Write a concise `summary` (1-2 sentences) stating the final conclusion and the key evidence. **Format large numbers clearly using 'billion' or 'trillion' where appropriate (e.g., '$790.2 billion').**
    9.  Write a brief `justification` (1-2 sentences) explaining *why* you reached that verdict, referencing specific data points or lack thereof.
    10. Create `evidence_links` (list of {{"finding": "...", "source_url": "..."}}) linking **only the most crucial data points** or findings cited in the justification back to their source URLs. Use the exact data point text (e.g., "National defense (G16046) = 790,895") or a concise summary of the finding as the "finding" value. **Match the finding precisely to the source URL provided in the evidence context.** Limit to 2-3 key links.

    Return ONLY a single valid JSON object with keys: "verdict", "summary", "justification", "evidence_links".

    Example Response (BEA Comparison):
    {{
      "verdict": "Contradicted",
      "summary": "The data contradicts the claim; federal spending on national defense ($790.9 billion) significantly exceeded spending on education ($178.6 billion) in 2023.",
      "justification": "BEA NIPA T31600 data for 2023 shows Federal National Defense (G16046) spending was $790,895 million, while Federal Education (G16068) spending was $178,621 million.",
      "evidence_links": [
        {{"finding": "National defense (G16046) = 790,895 (Multiplier: null)", "source_url": "https://apps.bea.gov/api/data?..."}},
        {{"finding": "Education (G16068) = 178,621 (Multiplier: null)", "source_url": "https://apps.bea.gov/api/data?..."}}
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

            corrected_links = []
            if isinstance(parsed.get("evidence_links"), list):
                for link in parsed["evidence_links"]:
                     if isinstance(link, dict) and "finding" in link and "source_url" in link:
                          best_match_url = source_map_for_linking.get(link["finding"])
                          if not best_match_url:
                              for text, url in source_map_for_linking.items():
                                   if link["finding"] in text or text in link["finding"]:
                                        best_match_url = url
                                        break
                          link["source_url"] = best_match_url if best_match_url else link["source_url"]
                          corrected_links.append(link)
                     else:
                          corrected_links.append(link)
                parsed["evidence_links"] = corrected_links

            return parsed
        else:
            logger.error("Failed to parse valid synthesis JSON from LLM response: %s", res.get("text", ""))
            default_response["justification"] += " (LLM response parsing failed.)"
            return default_response
    except HTTPException as e:
        logger.error("LLM failed during synthesis: %s", getattr(e, "detail", str(e)))
        default_response["justification"] += f" (LLM call failed: {e.detail})."
        return default_response
    except Exception as e:
        logger.exception("Unexpected error during LLM synthesis.")
        default_response["justification"] += " (Unexpected analysis error.)"
        return default_response


async def compute_confidence(sources: List[Dict[str, Any]], verdict: str, claim: str) -> Dict[str, Any]:
    valid_sources = [s for s in sources if s and "error" not in s]

    DEFAULT_CONFIDENCE_DATA = {"confidence": 0.3, "R": 0.5, "E": 0.0, "S": 0.3, "S_semantic_sim": 0.0}

    if not valid_sources:
        return DEFAULT_CONFIDENCE_DATA

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

    R = round(total_weight / len(valid_sources), 2) if valid_sources else 0.5
    E = round(min(1.0, len(valid_sources) / 5.0), 2)

    S_llm_verdict = 0.5
    if verdict == "Supported": S_llm_verdict = 0.95
    elif verdict == "Contradicted": S_llm_verdict = 0.90
    S_semantic_sim = 0.0

    try:
        texts_to_embed = [claim] if claim and claim.strip() else []
        if not texts_to_embed:
             raise ValueError("Claim text is empty, cannot compute semantic similarity.")

        source_texts_indices: List[Tuple[int, str]] = []
        current_index = 1

        for s in valid_sources:
            snippet = s.get('snippet', '').strip()
            title = s.get('title', '').strip()
            data_text = None
            if s.get('data_value') is not None:
                data_text = f"{s.get('line_description', s.get('title', 'Data'))}: {s.get('raw_data_value')}"
                data_text = data_text.strip()

            if snippet: texts_to_embed.append(snippet)
            if title: texts_to_embed.append(title)
            if data_text: texts_to_embed.append(data_text)

        if len(texts_to_embed) > 1:
            all_embeddings = await get_embeddings_batch_api(texts_to_embed)

            if not all_embeddings or all_embeddings[0] is None:
                 logger.warning("Claim embedding failed, cannot calculate semantic similarity.")
                 S_semantic_sim = 0.3
            else:
                claim_embedding = all_embeddings[0]
                source_embeddings = all_embeddings[1:]
                valid_source_embeddings = [emb for emb in source_embeddings if emb]

                if valid_source_embeddings:
                    similarities = [cosine_similarity(claim_embedding, emb) for emb in valid_source_embeddings]
                    if similarities:
                        S_semantic_sim = round(float(max(similarities)), 2)
                    else:
                        S_semantic_sim = 0.3
                else:
                    logger.warning("No valid source embeddings returned from batch API call.")
                    S_semantic_sim = 0.3

    except ValueError as ve:
          logger.error(f"Cannot compute semantic similarity: {ve}")
          S_semantic_sim = 0.1
    except Exception as e:
        logger.error("Error during batch semantic similarity calculation: %s", e, exc_info=True)
        S_semantic_sim = 0.3

    S = round((S_llm_verdict * 0.7) + (S_semantic_sim * 0.3), 2)
    confidence = round((0.5 * R + 0.3 * S + 0.2 * E), 2)

    return {"confidence": confidence, "R": R, "E": E, "S": S, "S_semantic_sim": S_semantic_sim}


@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = (req.claim or "").strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    start_time = asyncio.get_event_loop().time()

    analysis = {}
    all_results = []
    synthesis_result = {}
    confidence_data = {}

    try:
        analysis = await analyze_claim_for_api_plan(claim)
        claim_norm = analysis.get("claim_normalized", claim)
        claim_type = analysis.get("claim_type", "Other")
        api_plan = analysis.get("api_plan", {})

        logger.info("Generated API Plan: %s", json.dumps(api_plan, indent=2))

        all_results = await execute_query_plan(api_plan, claim_type)

        sources_results = [r for r in all_results if isinstance(r, dict) and "error" not in r]
        debug_errors = [r for r in all_results if isinstance(r, dict) and "error" in r]

        logger.info(f"Retrieved {len(sources_results)} sources, encountered {len(debug_errors)} errors.")

        synthesis_result = await synthesize_finding_with_llm(claim, analysis, sources_results)
        verdict = synthesis_result.get("verdict", "Inconclusive")
        summary_text = f"{synthesis_result.get('summary','')}. {synthesis_result.get('justification','')}".strip().replace("..", ".")

        confidence_data = await compute_confidence(sources_results, verdict, claim_norm)

        confidence_val = confidence_data.get("confidence", 0.0)

        if confidence_val > 0.75: confidence_tier = "High"
        elif confidence_val > 0.5: confidence_tier = "Medium"
        else: confidence_tier = "Low"

        end_time = asyncio.get_event_loop().time()
        duration = round(end_time - start_time, 2)
        logger.info(f"Verification completed for claim '{claim[:50]}...' in {duration} seconds.")

        return {
            "claim_original": claim,
            "claim_normalized": claim_norm,
            "claim_type": claim_type,
            "verdict": verdict,
            "confidence": confidence_val,
            "confidence_tier": confidence_tier,
            "confidence_breakdown": {
                "source_reliability": confidence_data.get("R", 0.0),
                "evidence_density": confidence_data.get("E", 0.0),
                "semantic_alignment": confidence_data.get("S", 0.0),
            },
            "summary": summary_text,
            "evidence_links": synthesis_result.get("evidence_links", []),
            "sources": sources_results,
            "debug_plan": analysis,
            "debug_log": debug_errors,
            "debug_processing_time_seconds": duration,
        }

    except Exception as e:
        logger.exception("Unhandled error during /verify processing for claim: %s", claim)
        return {
            "claim_original": claim,
            "claim_normalized": analysis.get("claim_normalized", claim),
            "claim_type": analysis.get("claim_type", "Other"),
            "verdict": "Error",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "confidence_breakdown": { "R": 0.0, "E": 0.0, "S": 0.0 },
            "summary": f"An unexpected internal server error occurred: {type(e).__name__}",
            "evidence_links": [],
            "sources": [r for r in all_results if r and "error" not in r] if all_results else [],
            "debug_plan": analysis,
            "debug_log": [r for r in all_results if r and "error" in r] + [{
                "error": f"Unhandled exception during processing: {str(e)}",
                "source": "internal_verify_endpoint",
                "status": "failed",
            }],
        }



