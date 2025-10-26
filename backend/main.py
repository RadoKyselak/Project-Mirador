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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "GEMINI_API_KEY", "DATA_GOV_API_KEY", "BEA_API_KEY",
    "CENSUS_API_KEY", "CONGRESS_API_KEY"
]

def check_api_keys_on_startup():
    logger.info("Startup: checking for required API keys (non-fatal)...")
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        logger.warning("Missing env vars: %s. App will start, but some endpoints will be disabled.", ", ".join(missing))
    else:
        logger.info("All required API keys present.")

app = FastAPI(on_startup=[check_api_keys_on_startup])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")  # not required for catalog.data.gov
BEA_API_KEY = os.getenv("BEA_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

BEA_VALID_TABLES = {
    "T10101",  # GDP
    "T20305",  # Personal Income
    "T31600",  # Government Spending by Function
    "T70500"   # Relation of GDP, GNP, and NNP
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

# -------------------------
# Helpers
# -------------------------
def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract first balanced JSON object from text. Returns None if not found/parseable."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try removing control chars and retry
                    try:
                        cleaned = re.sub(r"[\x00-\x1f]", "", candidate)
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None

def _parse_numeric_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip()
        # remove commas and dollar signs
        s = s.replace(",", "").replace("$", "")
        # parentheses as negative
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        # parse leading numeric token
        m = re.match(r"^(-?[\d\.eE+-]+)", s)
        if m:
            return float(m.group(1))
        return float(s)
    except Exception:
        return None

def _apply_multiplier(value: Optional[float], multiplier: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    if multiplier is None:
        return value
    try:
        # multiplier may be string "1", "1000", etc.
        return float(value) * float(multiplier)
    except Exception:
        # if multiplier unparsable, return original
        return value

# -------------------------
# LLM call
# -------------------------
async def call_gemini(prompt: str) -> Dict[str, Any]:
    """
    Call Gemini. This function raises HTTPException on obvious misconfiguration or HTTP failure.
    Callers should catch exceptions and provide fallbacks to avoid 500s at the endpoint level.
    """
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not configured when calling Gemini.")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            logger.error("Gemini HTTP error %s: %s", e.response.status_code, e.response.text)
            raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error("Gemini request error: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")
    # try to extract text in various shapes
    text = ""
    if isinstance(data, dict):
        cand_list = data.get("candidates") or []
        if isinstance(cand_list, list) and len(cand_list) > 0:
            cand = cand_list[0]
            text = cand.get("content", {}).get("parts", [{}])[0].get("text", "") or cand.get("output", "") or ""
        if not text:
            text = data.get("output", "") or data.get("text", "") or json.dumps(data)
    else:
        text = json.dumps(data)
    return {"raw": data, "text": text}

# -------------------------
# Plan generation
# -------------------------
async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    # Use a triple-quoted f-string so embedded JSON examples don't break Python quoting.
    prompt = f"""
You are a world-class research analyst and a U.S. government data expert. Deconstruct the user's factual claim into a precise API plan.

AVAILABLE APIs & DATASETS:
- BEA: dataset 'NIPA', tables like 'T31600' (Government Spending by Function).
- Census: various endpoints (use only when you can produce endpoint+params).
- Data.gov: for keyword searches.

USER CLAIM: '''{claim}'''

Return a single JSON object like:
{{ 
  "claim_normalized": "...", 
  "claim_type": "quantitative", 
  "api_plan": {{ 
    "tier1_params": {{ 
      "bea": { {{"DataSetName":"NIPA","TableName":"T31600","Frequency":"A","Year":"2023","LineCode":["2","14"]}} }, 
      "census": null 
    }}, 
    "tier2_keywords": ["...","..."] 
  }} 
}}
"""
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))
    except HTTPException as e:
        logger.error("Gemini failed generating plan: %s", getattr(e, "detail", str(e)))
        # Fallback to a conservative keyword-only plan
        return {"claim_normalized": claim, "claim_type": "Other", "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}}
    except Exception:
        logger.exception("Unexpected error generating plan. Falling back.")
        return {"claim_normalized": claim, "claim_type": "Other", "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}}

    if not parsed:
        logger.warning("Could not parse plan JSON from Gemini. Falling back to keyword plan.")
        return {"claim_normalized": claim, "claim_type": "Other", "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}}
    return parsed

# -------------------------
# Source picking
# -------------------------
def pick_sources_from_type(claim_type: str) -> List[str]:
    sources = ["DATA.GOV"]
    if claim_type and "quantitative" in claim_type.lower():
        sources.extend(["BEA", "CENSUS"])
    if claim_type and "factual" in claim_type.lower():
        sources.append("CONGRESS")
    return sources

# -------------------------
# API queries
# -------------------------
async def query_bea(params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    if not BEA_API_KEY:
        return [{"error": "BEA_API_KEY is not configured", "source": "BEA", "status": "failed"}]
    if not params:
        return []

    final_params = {
        "UserID": BEA_API_KEY,
        "method": "GetData",
        "ResultFormat": "json",
        "DataSetName": params.get("DataSetName"),
        "TableName": params.get("TableName"),
        "Frequency": params.get("Frequency"),
        "Year": params.get("Year"),
        "LineCode": params.get("LineCode"),
        "GeoFips": params.get("GeoFips"),
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

    # log (truncated) payload for debugging when rows present
    try:
        logger.debug("BEA payload (truncated): %s", json.dumps(payload)[:2000])
    except Exception:
        pass

    results = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    out: List[Dict[str, Any]] = []
    if not results:
        logger.info("BEA returned no rows for params: %s", {k: v for k, v in final_params.items() if v is not None})
        return []

    logger.debug("BEA returned %d rows for params %s", len(results), {k: v for k, v in final_params.items() if v is not None})
    for item in results:
        # Prefer BEA-provided identifiers when present
        line_code_resp = item.get("LineCode") or item.get("SeriesCode") or final_params.get("LineCode")
        desc = item.get("LineDescription") or item.get("SeriesDescription") or item.get("TableName") or "Data"
        data_value_raw = item.get("DataValue") or item.get("Value") or ""
        numeric = _parse_numeric_value(data_value_raw)
        # unit and multiplier fields (BEA sometimes uses UnitMultiplier)
        unit = item.get("Unit") or item.get("Units") or item.get("UnitOfMeasure") or ""
        unit_multiplier = None
        if "UnitMultiplier" in item and item.get("UnitMultiplier") not in (None, ""):
            try:
                unit_multiplier = float(item.get("UnitMultiplier"))
            except Exception:
                unit_multiplier = None
        time_period = item.get("TimePeriod") or item.get("Year") or final_params.get("Year")
        snippet = f"{desc} for {time_period} was {data_value_raw}"
        if unit:
            snippet += f" {unit}."
        else:
            snippet += "."

        out.append({
            "title": f"BEA Dataset: {params.get('DataSetName')} - {params.get('TableName')}",
            "url": str(r.url),
            "snippet": snippet,
            "data_value": numeric,
            "raw_data_value": data_value_raw,
            "unit": unit,
            "unit_multiplier": unit_multiplier,
            "line_description": desc,
            "line_code": str(line_code_resp) if line_code_resp is not None else None,
            "raw_item": item,
        })
    return out

async def query_census(params: Dict[str, Any] = None, keyword_query: str = None) -> List[Dict[str, Any]]:
    # Only allow structured, param-driven Census queries. Keyword free-text against ACS returned 400 previously.
    if not CENSUS_API_KEY:
        return [{"error": "CENSUS_API_KEY is not configured", "source": "CENSUS", "status": "failed"}]
    if not params:
        logger.warning("Keyword-based Census queries are disabled; use tier1 census params.")
        return []
    final_params = {"key": CENSUS_API_KEY}
    url = f"https://api.census.gov{params.get('endpoint')}"
    final_params.update(params.get("params", {}))
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=final_params)
            r.raise_for_status()
            try:
                snippet = str(r.json()[:3])[:700]
            except Exception:
                snippet = str(r.text)[:700]
            return [{"title": f"Census Data for '{params.get('endpoint')}'", "url": str(r.url), "snippet": snippet}]
    except httpx.HTTPStatusError as e:
        logger.error("Census HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Census API error: {e.response.status_code}", "source": "CENSUS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Census request error: %s", str(e))
        return [{"error": str(e), "source": "CENSUS", "status": "failed"}]

async def query_congress(keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CONGRESS_API_KEY:
        return [{"error": "CONGRESS_API_KEY is not configured", "source": "CONGRESS", "status": "failed"}]
    if not keyword_query:
        return []
    params = {"api_key": CONGRESS_API_KEY, "q": keyword_query, "limit": 1}
    url = "https://api.congress.gov/v3/bill"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            bills = r.json().get("bills", [])
            return [
                {"title": bill.get("title"), "url": bill.get("url"), "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"}
                for bill in bills
            ]
    except httpx.HTTPStatusError as e:
        logger.error("Congress HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Congress API error: {e.response.status_code}", "source": "CONGRESS", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Congress request error: %s", str(e))
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not keyword_query:
        return []
    # Use catalog.data.gov CKAN endpoint (no API key required for search)
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
                resource_url = resources[0].get("url") if resources else None
                dataset_page = f"https://catalog.data.gov/dataset/{item.get('name')}" if item.get("name") else None
                out.append({"title": item.get("title"), "url": resource_url or dataset_page, "snippet": (item.get("notes") or "")[:250]})
            return out
    except httpx.HTTPStatusError as e:
        logger.error("Data.gov HTTP error %s: %s", e.response.status_code, e.response.text)
        return [{"error": f"Data.gov API error: {e.response.status_code}", "source": "DATA.GOV", "status": "failed"}]
    except httpx.RequestError as e:
        logger.error("Data.gov request error: %s", str(e))
        return [{"error": str(e), "source": "DATA.GOV", "status": "failed"}]

# -------------------------
# Plan execution
# -------------------------
async def execute_query_plan(plan: Dict[str, Any], claim_type: str) -> List[Dict[str, Any]]:
    tier1_params = plan.get("tier1_params", {}) or {}
    tier2_keywords = plan.get("tier2_keywords", []) or []
    sources_to_query = pick_sources_from_type(claim_type)
    tasks = []

    # BEA tier1: support list of LineCodes or single value
    if "BEA" in sources_to_query and tier1_params.get("bea"):
        bea_base = tier1_params.get("bea").copy()
        table_name = bea_base.get("TableName")
        if not table_name or table_name not in BEA_VALID_TABLES:
            logger.warning("BEA table missing or not in whitelist: %s", table_name)
        else:
            lc = bea_base.get("LineCode")
            if isinstance(lc, list):
                for code in lc:
                    sub = bea_base.copy()
                    digits = re.findall(r"\d+", str(code))
                    if digits:
                        sub["LineCode"] = digits[0]
                        tasks.append(query_bea(params=sub))
                    else:
                        logger.warning("Skipping non-numeric BEA LineCode: %s", code)
            else:
                if bea_base.get("LineCode"):
                    digits = re.findall(r"\d+", str(bea_base.get("LineCode")))
                    if digits:
                        bea_base["LineCode"] = digits[0]
                    else:
                        bea_base.pop("LineCode", None)
                tasks.append(query_bea(params=bea_base))

    # Census only for structured tier1 params
    if "CENSUS" in sources_to_query and tier1_params.get("census"):
        tasks.append(query_census(params=tier1_params.get("census")))

    # Tier2 keywords: Data.gov and Congress (avoid free-text ACS calls)
    for kw in tier2_keywords:
        if "DATA.GOV" in sources_to_query:
            tasks.append(query_datagov(kw))
        if "CONGRESS" in sources_to_query:
            tasks.append(query_congress(keyword_query=kw))

    if not tasks:
        logger.info("No tasks to execute for plan.")
        return []

    # gather results concurrently; individual tasks return lists (including error dicts)
    results = await asyncio.gather(*tasks)
    flattened: List[Dict[str, Any]] = []
    for sub in results:
        if sub:
            for it in sub:
                flattened.append(it)
    return flattened

# -------------------------
# Summarization & deterministic compare
# -------------------------
async def summarize_with_evidence(claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    default_summary = {
        "summary": "The available data is insufficient to verify the claim.",
        "justification": "No supporting government data could be found or parsed to verify this claim.",
        "evidence_links": [],
    }

    if not sources:
        return default_summary

    valid_sources = [s for s in sources if "error" not in s]
    if not valid_sources:
        return default_summary

    # Try to detect BEA numeric rows and perform deterministic comparison
    def is_defense_row(s: Dict[str, Any]) -> bool:
        txt = " ".join(filter(None, [str(s.get("line_description", "")).lower(), (s.get("title") or "").lower(), (s.get("snippet") or "").lower()]))
        return any(k in txt for k in ["defense", "national defense", "military", "national security"])

    def is_education_row(s: Dict[str, Any]) -> bool:
        txt = " ".join(filter(None, [str(s.get("line_description", "")).lower(), (s.get("title") or "").lower(), (s.get("snippet") or "").lower()]))
        return any(k in txt for k in ["education", "education and training", "higher education", "elementary", "secondary"])

    bea_rows = [s for s in valid_sources if s.get("data_value") is not None]
    logger.debug("Found %d BEA numeric rows: %s", len(bea_rows), [(r.get("line_code"), r.get("line_description"), r.get("data_value")) for r in bea_rows])

    # Prefer matching by BEA line_code if present
    def match_by_linecodes(codes):
        codes_set = {str(c).strip() for c in codes}
        for r in bea_rows:
            if r.get("line_code") and str(r.get("line_code")).strip() in codes_set:
                return r
        return None

    # Known common codes for T31600 (adjust if BEA uses different codes)
    defense_row = match_by_linecodes(["2", "02"])
    education_row = match_by_linecodes(["14", "014"])

    # If not matched by code, try description matching
    if not defense_row:
        for r in bea_rows:
            if is_defense_row(r):
                defense_row = r
                break
    if not education_row:
        for r in bea_rows:
            if is_education_row(r):
                education_row = r
                break

    logger.debug("Selected defense_row: %s", (defense_row or {}).get("line_description"))
    logger.debug("Selected education_row: %s", (education_row or {}).get("line_description"))

    # If both numeric rows available, do deterministic comparison with multiplier handling
    if defense_row and education_row and defense_row.get("data_value") is not None and education_row.get("data_value") is not None:
        try:
            dv_raw = defense_row["data_value"]
            ev_raw = education_row["data_value"]
            dv = _apply_multiplier(dv_raw, defense_row.get("unit_multiplier"))
            ev = _apply_multiplier(ev_raw, education_row.get("unit_multiplier"))
            logger.info(
                "Comparing BEA rows: defense raw=%s mult=%s -> %s ; education raw=%s mult=%s -> %s",
                dv_raw, defense_row.get("unit_multiplier"), dv, ev_raw, education_row.get("unit_multiplier"), ev,
            )
            unit_def = defense_row.get("unit") or ""
            unit_edu = education_row.get("unit") or ""
            if dv is None or ev is None:
                raise ValueError("Unable to compute numeric BEA values for comparison")

            if dv > ev:
                summary = "The BEA data indicates federal spending on national defense exceeded federal spending on education in 2023."
                verdict = "Supported"
            elif dv < ev:
                summary = "The BEA data indicates federal spending on education exceeded federal spending on national defense in 2023."
                verdict = "Contradicted"
            else:
                summary = "The BEA data indicates federal spending on national defense and education were approximately equal in 2023."
                verdict = "Inconclusive"
            justification = f"BEA values: defense = {dv} {unit_def or ''}, education = {ev} {unit_edu or ''}."
            # ensure evidence links point to the specific BEA request URLs (these were created per-LineCode)
            evidence_links = [
                {"finding": f"Defense: {defense_row.get('line_description') or defense_row.get('title')}", "source_url": defense_row.get("url")},
                {"finding": f"Education: {education_row.get('line_description') or education_row.get('title')}", "source_url": education_row.get("url")},
            ]
            return {"summary": summary, "justification": justification, "evidence_links": evidence_links}
        except Exception:
            logger.exception("Numeric comparison failed; falling back to LLM summarization.")

    # Otherwise, prepare filtered evidence and call Gemini for a narrative summary.
    relevance_kws = ["defense", "national defense", "military", "education", "education and training", "omb", "expenditure", "spending", "appropriations"]
    filtered = []
    for s in valid_sources:
        txt = " ".join(filter(None, [s.get("title", ""), s.get("snippet", "")])).lower()
        # keep likely relevant items or BEA rows
        if any(k in txt for k in relevance_kws) or (s.get("url") or "").lower().startswith("https://apps.bea.gov"):
            filtered.append(s)
    if not filtered:
        filtered = valid_sources

    unique = {s.get("url"): s for s in filtered if s.get("url")}
    context = "\n---\n".join([f"Source Title: {s.get('title')}\nURL: {s.get('url')}\nSnippet: {s.get('snippet')}" for s in unique.values()])

    prompt = f"""
You are a meticulous and impartial fact-checker. Using ONLY the provided evidence, determine whether the user's claim is supported, contradicted, or inconclusive.

Provide a single JSON object with: summary (one sentence), justification (1-2 sentences), evidence_links (list of {{finding, source_url}}).

USER CLAIM: '''{claim}'''

AGGREGATED EVIDENCE:
{context}
"""
    try:
        res = await call_gemini(prompt)
        parsed = extract_json_block(res.get("text", ""))
    except HTTPException as e:
        logger.error("Gemini summarization failed: %s", getattr(e, "detail", str(e)))
        default_summary["justification"] += " (LLM summarization failed or was unavailable.)"
        return default_summary
    except Exception:
        logger.exception("Unexpected error calling Gemini for summarization.")
        default_summary["justification"] += " (LLM summarization failed unexpectedly.)"
        return default_summary

    if not parsed:
        logger.warning("Could not parse Gemini summary JSON; returning default summary.")
        return default_summary
    if not all(k in parsed for k in ["summary", "justification", "evidence_links"]):
        logger.warning("Gemini summary JSON missing required keys; returning default summary.")
        return default_summary
    return parsed

# -------------------------
# Confidence scoring
# -------------------------
def compute_confidence(sources: List[Dict[str, Any]], summary_text: str) -> Dict[str, Any]:
    if not sources:
        R, E, S = 0.5, 0.0, 0.3
        confidence = round((0.4 * R + 0.3 * E + 0.3 * S), 2)
        return {"confidence": confidence, "R": R, "E": E, "S": S}
    total_weight = 0.0
    for s in sources:
        url = (s.get("url") or "").lower()
        if "apps.bea.gov" in url:
            weight = 1.0
        elif "api.census.gov" in url:
            weight = 0.9
        elif "api.congress.gov" in url:
            weight = 0.8
        elif "api.data.gov" in url or "catalog.data.gov" in url:
            weight = 0.7
        else:
            weight = 0.6
        total_weight += weight
    R = round(total_weight / len(sources), 2)
    E = round(min(1.0, len(sources) / 5), 2)
    summary_lower = summary_text.lower()
    if ("data supports" in summary_lower or "data confirms" in summary_lower) or ("indicates" in summary_lower and "exceeded" in summary_lower):
        S = 0.95
    elif "data contradicts" in summary_lower:
        S = 0.9
    elif "insufficient" in summary_lower or "inconclusive" in summary_lower:
        S = 0.5
    else:
        S = 0.6
    confidence = round((0.4 * R + 0.3 * E + 0.3 * S), 2)
    return {"confidence": confidence, "R": R, "E": E, "S": S}

# -------------------------
# Main endpoint
# -------------------------
@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = (req.claim or "").strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")

    try:
        analysis = await analyze_claim_for_api_plan(claim)
        claim_norm = analysis.get("claim_normalized", claim)
        claim_type = analysis.get("claim_type", "Other")
        api_plan = analysis.get("api_plan", {})

        all_results = await execute_query_plan(api_plan, claim_type)
        sources_results = [r for r in all_results if "error" not in r]
        debug_errors = [r for r in all_results if "error" in r]

        # Filter relevant sources for summarization (reduce noise)
        relevant = []
        for s in sources_results:
            url = (s.get("url") or "").lower()
            title = (s.get("title") or "").lower()
            snippet = (s.get("snippet") or "").lower()
            txt = " ".join([url, title, snippet])
            if "defense" in txt or "education" in txt or "apps.bea.gov" in url or "data.ed.gov" in url or "catalog.data.gov" in url:
                relevant.append(s)
        if not relevant:
            relevant = sources_results

        summary_data = await summarize_with_evidence(claim_norm, relevant)
        summary_text = f"{summary_data.get('summary','')} {summary_data.get('justification','')}".strip()

        confidence_data = compute_confidence(relevant, summary_text)
        confidence_val = confidence_data["confidence"]

        summary_lower = summary_text.lower()
        if "exceeded" in summary_lower and "defense" in summary_lower:
            verdict = "Supported"
        elif "exceeded" in summary_lower and "education" in summary_lower and "defense" not in summary_lower:
            verdict = "Contradicted"
        elif "data supports" in summary_lower or "data confirms" in summary_lower:
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
                "semantic_alignment": confidence_data["S"],
            },
            "summary": summary_text,
            "evidence_links": summary_data.get("evidence_links", []),
            "sources": sources_results,
            "debug_plan": api_plan,
            "debug_log": debug_errors,
        }
    except Exception as e:
        # Catch unexpected errors and return a safe JSON response (avoid 500s leaking stack traces)
        logger.exception("Unhandled error in /verify")
        return {
            "claim_original": claim,
            "claim_normalized": claim,
            "claim_type": "Other",
            "verdict": "Error",
            "confidence": 0.0,
            "confidence_tier": "Low",
            "confidence_breakdown": {"source_reliability": 0.0, "evidence_density": 0.0, "semantic_alignment": 0.0},
            "summary": "Internal error while processing the claim.",
            "evidence_links": [],
            "sources": [],
            "debug_plan": {},
            "debug_log": [{"error": str(e), "source": "internal", "status": "failed"}],
        }
