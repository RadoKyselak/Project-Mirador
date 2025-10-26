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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "GEMINI_API_KEY", "DATA_GOV_API_KEY", "BEA_API_KEY",
    "CENSUS_API_KEY", "CONGRESS_API_KEY"
]

def check_api_keys_on_startup():
    # Make startup tolerant: log missing keys but do not crash the process.
    logger.info("Checking for required API keys (startup check)...")
    missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing_keys:
        logger.warning(f"Missing environment variables: {', '.join(missing_keys)}. "
                       "The API will still start; individual calls will fail if they require those keys.")
    else:
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
    "T31600",  # Government Spending by Function
    "T70500"   # Relation of GDP, GNP, and NNP
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try to find the first balanced JSON object in text
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start: i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # try a naive bracket-trim approach if nested comments break it
                    try:
                        cleaned = re.sub(r'[\x00-\x1f]', '', candidate)
                        return json.loads(cleaned)
                    except Exception:
                        return None
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
            text = ""
            # handle different shapes safely
            if isinstance(data, dict):
                if isinstance(data.get("candidates"), list) and len(data["candidates"]) > 0:
                    cand = data["candidates"][0]
                    text = cand.get("content", {}).get("parts", [{}])[0].get("text", "") or cand.get("output", "")
                if not text:
                    text = data.get("output", "") or data.get("text", "") or json.dumps(data)
            else:
                # not a dict as expected, stringify
                text = json.dumps(data)
            return {"raw": data, "text": text}
    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
        # Re-raise HTTPException so callers can handle it specifically if desired
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code} - {e.response.text}")
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"Error communicating with Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini: {str(e)}")

async def analyze_claim_for_api_plan(claim: str) -> Dict[str, Any]:
    # Simplified prompt for determining plan; keep it structured so extract_json_block can parse it
    prompt = (
        "You are a research analyst and U.S. government data expert. Decompose the user's claim into:\n"
        " - claim_normalized (short verifiable statement)\n"
        " - claim_type (e.g., quantitative, factual, quantitative_comparison)\n"
        " - api_plan: {tier1_params: {bea: {...} or null, census: {...} or null }, tier2_keywords: [ ... ] }\n\n"
        f"USER CLAIM: '''{claim}'''\n\n"
        "Return a single valid JSON object. Example:\n"
        '{ "claim_normalized": "...", "claim_type":"quantitative", "api_plan": {"tier1_params": {"bea": {"DataSetName":"NIPA","TableName":"T31600","Frequency":"A","Year":"2023","LineCode":[\"2\",\"14\"]}, "census": null}, "tier2_keywords": ["federal government defense spending 2023", "federal government education spending 2023"] } }\n'
    )
    try:
        res = await call_gemini(prompt)
        parsed_plan = extract_json_block(res.get("text", ""))
    except HTTPException as e:
        # Gemini failed (network/401/other); log and fall back to a safe plan
        logger.error(f"Gemini failed while generating API plan: {getattr(e, 'detail', str(e))}")
        return {
            "claim_normalized": claim,
            "claim_type": "Other",
            "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}
        }
    except Exception as e:
        logger.exception("Unexpected error while calling Gemini to generate API plan.")
        return {
            "claim_normalized": claim,
            "claim_type": "Other",
            "api_plan": {"tier1_params": {}, "tier2_keywords": [claim]}
        }

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
    if claim_type and "quantitative" in claim_type.lower():
        sources.extend(["BEA", "CENSUS"])
    if claim_type and "factual" in claim_type.lower():
        sources.append("CONGRESS")
    return sources

# --- Utility helpers ---
def _parse_numeric_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip()
        s = s.replace(",", "").replace("$", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        m = re.match(r"^(-?[\d\.eE+-]+)", s)
        if m:
            return float(m.group(1))
        return float(s)
    except Exception:
        return None

# --- API Calls ---

async def query_bea(params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    if not BEA_API_KEY:
        return [{"error": "BEA_API_KEY is not configured", "source": "BEA", "status": "failed"}]
    if not params:
        return []

    final_params = {
        'UserID': BEA_API_KEY, 'method': 'GetData', 'ResultFormat': 'json',
        'DataSetName': params.get('DataSetName'), 'TableName': params.get('TableName'),
        'Frequency': params.get('Frequency'), 'Year': params.get('Year'),
        'LineCode': params.get('LineCode'), 'GeoFips': params.get('GeoFips')
    }
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params={k: v for k, v in final_params.items() if v is not None})
            r.raise_for_status()
            payload = r.json()
            data_block = payload.get('BEAAPI', {}).get('Results', {})
            results = data_block.get('Data', [])

            out: List[Dict[str, Any]] = []
            if results:
                for item in results:
                    desc = item.get('LineDescription') or item.get('SeriesDescription') or item.get('TableName') or 'Data'
                    data_value_raw = item.get('DataValue') or item.get('Value') or ""
                    numeric = _parse_numeric_value(data_value_raw)
                    unit = item.get('Unit') or item.get('Units') or item.get('UnitOfMeasure') or ""
                    time_period = item.get('TimePeriod') or item.get('Year') or final_params.get('Year')
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
                        "unit": unit,
                        "line_description": desc,
                        "line_code": final_params.get('LineCode'),
                    })
                return out
            else:
                logger.info(f"BEA returned no Data entries for params: {final_params}")
                return []
    except httpx.HTTPStatusError as e:
        logger.error(f"BEA API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"BEA API error: {e.response.status_code}", "source": "BEA", "status": "failed"}]
    except Exception as e:
        logger.error(f"BEA API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "BEA", "status": "failed"}]

async def query_census(params: Dict[str, Any] = None, keyword_query: str = None) -> List[Dict[str, Any]]:
    if not CENSUS_API_KEY:
        return [{"error": "CENSUS_API_KEY is not configured", "source": "CENSUS", "status": "failed"}]
    if not params and not keyword_query:
        return []

    if params:
        final_params = {'key': CENSUS_API_KEY}
        url = f"https://api.census.gov{params.get('endpoint')}"
        final_params.update(params.get('params', {}))
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(url, params=final_params)
                r.raise_for_status()
                # return a short snippet of the JSON response
                snippet = ""
                try:
                    snippet = str(r.json()[:3])[:700]
                except Exception:
                    snippet = str(r.text)[:700]
                return [{"title": f"Census Data for '{params.get('endpoint')}'", "url": str(r.url), "snippet": snippet}]
        except httpx.HTTPStatusError as e:
            logger.error(f"Census API HTTP Error: {e.response.status_code} - {e.response.text}")
            return [{"error": f"Census API error: {e.response.status_code}", "source": "CENSUS", "status": "failed"}]
        except Exception as e:
            logger.error(f"Census API General Error: {str(e)}", exc_info=True)
            return [{"error": str(e), "source": "CENSUS", "status": "failed"}]
    else:
        # Keyword-only census requests are disabled to avoid 400s on ACS endpoints.
        logger.warning("Keyword-based Census searches are disabled for ACS endpoints. Use tier1 census params or search via Data.gov.")
        return []

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
            return [{"title": bill.get('title'), "url": bill.get('url'), "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"} for bill in bills]
    except httpx.HTTPStatusError as e:
        logger.error(f"Congress API HTTP Error: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Congress API error: {e.response.status_code}", "source": "CONGRESS", "status": "failed"}]
    except Exception as e:
        logger.error(f"Congress API General Error: {str(e)}", exc_info=True)
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]

async def query_datagov(keyword_query: str) -> List[Dict[str, str]]:
    if not keyword_query:
        return []
    # prefer CKAN package_search endpoint; it usually doesn't require an API key and is robust
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": keyword_query, "rows": 5}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            results = r.json().get("result", {}).get("results", [])
            out = []
            for item in results:
                resources = item.get("resources", []) or []
                resource_url = resources[0].get("url") if resources else None
                dataset_page = f"https://catalog.data.gov/dataset/{item.get('name')}"
                out.append({
                    "title": item.get("title"),
                    "url": resource_url or dataset_page,
                    "snippet": (item.get("notes", "") or "")[:250]
                })
            return out
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

    # BEA: support single LineCode or list of LineCodes (create separate BEA tasks for each)
    if "BEA" in sources_to_query and tier1_params.get("bea"):
        bea_params_base = tier1_params.get("bea").copy()
        table_name = bea_params_base.get("TableName")
        lc = bea_params_base.get("LineCode")
        if isinstance(lc, list):
            for code in lc:
                try:
                    sub = bea_params_base.copy()
                    # sanitize code to digits if possible
                    digits = re.findall(r"\d+", str(code))
                    if digits:
                        sub["LineCode"] = digits[0]
                    else:
                        logger.warning(f"Skipping non-numeric BEA LineCode item: {code}")
                        continue
                    if table_name and table_name in BEA_VALID_TABLES:
                        tasks.append(query_bea(params=sub))
                except Exception as e:
                    logger.warning(f"Skipping BEA LineCode item due to error: {e}")
        else:
            bea_params = bea_params_base
            if bea_params.get("LineCode"):
                digits = re.findall(r"\d+", str(bea_params.get("LineCode")))
                if digits:
                    bea_params["LineCode"] = digits[0]
                else:
                    bea_params.pop("LineCode", None)
            if table_name and table_name in BEA_VALID_TABLES:
                tasks.append(query_bea(params=bea_params))
            elif table_name:
                logger.warning(f"Invalid/Hallucinated BEA TableName: '{table_name}'. Skipping BEA Tier 1 call.")
            else:
                logger.debug("No BEA TableName provided in plan, skipping.")

    # Only make Census calls if tier1_params.census is present and valid. Do NOT do keyword-driven ACS calls.
    if "CENSUS" in sources_to_query and tier1_params.get("census"):
        tasks.append(query_census(params=tier1_params.get("census")))

    # For tier2 keywords: query Data.gov and Congress (if applicable). Avoid ACS keyword queries.
    for keyword in tier2_keywords:
        if "DATA.GOV" in sources_to_query:
            tasks.append(query_datagov(keyword))
        if "CONGRESS" in sources_to_query:
            tasks.append(query_congress(keyword_query=keyword))

    if not tasks:
        logger.info(f"No API calls to execute for claim type: {claim_type}")
        return []

    query_results = await asyncio.gather(*tasks)
    # flatten results; keep only non-empty sublists
    flattened = []
    for sublist in query_results:
        if sublist:
            for item in sublist:
                flattened.append(item)
    return flattened

async def summarize_with_evidence(claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If numeric BEA rows for defense and education exist, deterministically compare and return
    a programmatic summary + evidence_links. Otherwise call Gemini to synthesize a summary
    but with filtered (relevant) evidence to reduce noise.
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

    # heuristics to detect defense and education rows
    def _is_defense(s: Dict[str, Any]) -> bool:
        txt = " ".join(filter(None, [str(s.get("line_description", "")).lower(), s.get("title", "").lower(), s.get("snippet", "").lower()]))
        return any(k in txt for k in ["defense", "national defense", "military", "national security"])

    def _is_education(s: Dict[str, Any]) -> bool:
        txt = " ".join(filter(None, [str(s.get("line_description", "")).lower(), s.get("title", "").lower(), s.get("snippet", "").lower()]))
        return any(k in txt for k in ["education", "education and training", "elementary", "secondary", "higher education", "education & training"])

    bea_rows = [s for s in valid_sources if s.get("data_value") is not None]
    defense_row = None
    education_row = None
    if bea_rows:
        for r in bea_rows:
            if not defense_row and _is_defense(r):
                defense_row = r
            if not education_row and _is_education(r):
                education_row = r
        # fallback matching by common line_code hints
        if not defense_row:
            for r in bea_rows:
                if str(r.get("line_code", "")).strip() in ("2", "02"):
                    defense_row = r
        if not education_row:
            for r in bea_rows:
                if str(r.get("line_code", "")).strip() in ("14", "014"):
                    education_row = r

    # If both numeric rows available, do deterministic comparison
    if defense_row and education_row and defense_row.get("data_value") is not None and education_row.get("data_value") is not None:
        try:
            dv = float(defense_row["data_value"])
            ev = float(education_row["data_value"])
            unit_def = defense_row.get("unit") or ""
            unit_edu = education_row.get("unit") or ""
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
            evidence_links = [
                {"finding": f"Defense: {defense_row.get('line_description') or defense_row.get('title')}", "source_url": defense_row.get("url")},
                {"finding": f"Education: {education_row.get('line_description') or education_row.get('title')}", "source_url": education_row.get("url")}
            ]
            return {"summary": summary, "justification": justification, "evidence_links": evidence_links}
        except Exception:
            logger.exception("Failed numeric comparison despite available BEA rows; will fallback to Gemini summarization.")

    # Build filtered context for Gemini to avoid noise from unrelated sources
    relevance_kws = ["defense", "national defense", "military", "education", "education and training", "omb", "expenditure", "spending", "appropriations"]
    filtered = []
    for s in valid_sources:
        txt = " ".join(filter(None, [s.get("title", ""), s.get("snippet", "")])).lower()
        if any(k in txt for k in relevance_kws) or s.get("url", "").lower().startswith("https://apps.bea.gov"):
            filtered.append(s)
    if not filtered:
        filtered = valid_sources

    # use s.get('url') to avoid KeyError when url missing
    unique_sources = {s.get('url'): s for s in filtered if s.get("url")}
    context = "\n---\n".join([f"Source Title: {s.get('title')}\nURL: {s.get('url')}\nSnippet: {s.get('snippet')}" for s in unique_sources.values()])

    prompt = (
        "You are a meticulous and impartial fact-checker. Analyze the provided evidence from U.S. government data sources and synthesize a definitive conclusion about the claim.\n"
        "1) Review evidence snippets and identify key data points.\n"
        "2) Compare them to the user's claim.\n"
        "3) Provide a concise one-sentence conclusion starting with 'The data supports...', 'The data contradicts...', or 'The data is inconclusive...'.\n"
        "4) Provide a 1-2 sentence justification.\n"
        "5) Provide evidence_links mapping findings to source URLs.\n\n"
        f"USER'S CLAIM: '''{claim}'''\n\n"
        f"AGGREGATED EVIDENCE:\n{context}\n\n"
        "Return a single valid JSON object with keys: summary, justification, evidence_links."
    )

    try:
        res = await call_gemini(prompt)
        parsed_summary_data = extract_json_block(res.get("text", ""))
    except HTTPException as e:
        logger.error(f"Gemini failed while summarizing evidence: {getattr(e,'detail',str(e))}")
        # Include a hint in the default summary so it's easier to debug when the external API failed.
        default_summary["justification"] = default_summary["justification"] + " (Gemini summarization failed or was unavailable.)"
        return default_summary
    except Exception:
        logger.exception("Unexpected error while calling Gemini for summarization.")
        default_summary["justification"] = default_summary["justification"] + " (Gemini summarization failed unexpectedly.)"
        return default_summary

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

    total_weight = 0
    for s in sources:
        url_lower = s.get("url", "").lower()
        if "apps.bea.gov" in url_lower: weight = 1.0
        elif "api.census.gov" in url_lower: weight = 0.9
        elif "api.congress.gov" in url_lower: weight = 0.8
        elif "api.data.gov" in url_lower or "catalog.data.gov" in url_lower: weight = 0.7
        else: weight = 0.6
        total_weight += weight
    R = round(total_weight / len(sources), 2)

    n = len(sources)
    E = round(min(1.0, n / 5), 2)

    summary_lower = summary_text.lower()
    # make boolean precedence explicit
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

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")

    logger.info(f"Received claim for verification: '{claim}'")

    analysis = await analyze_claim_for_api_plan(claim)
    claim_norm = analysis.get("claim_normalized", claim)
    claim_type = analysis.get("claim_type", "Other")
    api_plan = analysis.get("api_plan", {})

    all_results = await execute_query_plan(api_plan, claim_type)
    sources_results = [r for r in all_results if "error" not in r]
    debug_errors = [r for r in all_results if "error" in r]

    if debug_errors:
        logger.warning(f"Encountered {len(debug_errors)} errors during API fetch for claim: '{claim}'")

    # reduce noise: keep relevant sources for summarization and confidence calc
    relevant_sources = []
    for s in sources_results:
        url = (s.get("url") or "").lower()
        title = (s.get("title") or "").lower()
        snippet = (s.get("snippet") or "").lower()
        txt = " ".join([url, title, snippet])
        if "defense" in txt or "education" in txt or "apps.bea.gov" in url or "data.ed.gov" in url or "catalog.data.gov" in url:
            relevant_sources.append(s)
    if not relevant_sources:
        relevant_sources = sources_results

    summary_data = await summarize_with_evidence(claim_norm, relevant_sources)

    summary_text = f"{summary_data.get('summary', '')} {summary_data.get('justification', '')}".strip()

    confidence_data = compute_confidence(relevant_sources, summary_text)
    confidence_val = confidence_data["confidence"]

    summary_lower = summary_text.lower()
    # make decision heuristics explicit and safe
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
            "semantic_alignment": confidence_data["S"]
        },
        "summary": summary_text,
        "evidence_links": summary_data.get("evidence_links", []),
        "sources": sources_results,
        "debug_plan": api_plan,
        "debug_log": debug_errors
    }
