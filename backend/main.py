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
    "GEMINI_API_KEY", "BEA_API_KEY", "CENSUS_API_KEY", "CONGRESS_API_KEY"
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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

BEA_VALID_TABLES = {
    "T10101", "T20305", "T31600", "T70500"
}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stelthar-API is running."}

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
    except Exception: return None

def _apply_multiplier(value: Optional[float], multiplier: Optional[Any]) -> Optional[float]:
    if value is None: return None
    if multiplier is None: return value
    try: return float(value) * float(multiplier)
    except Exception: return value

async def call_gemini(prompt: str, tools: List[Dict] = None) -> Dict[str, Any]:
    """Generic Gemini API caller, now with tool support."""
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not configured.")
        raise HTTPException(status_code=500, detail="LLM API key not configured on server.")
    
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    if tools:
        body["tools"] = [{"function_declarations": tools}]
        
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

    response = {"raw": data, "text": None, "tool_calls": []}
    if candidates := data.get("candidates", []):
        content = candidates[0].get("content", {})
        if parts := content.get("parts", []):
            for part in parts:
                if text := part.get("text"):
                    response["text"] = text
                if func_call := part.get("functionCall"):
                    response["tool_calls"].append(func_call)
    
    return response

async def query_bea(
    TableName: str, 
    Year: str, 
    LineCode: str, 
    Frequency: str = "A", 
    DataSetName: str = "NIPA"
) -> List[Dict[str, Any]]:
    """
    Queries the BEA NIPA API. Use for broad *economic functions* (e.g., 'total defense spending'), 
    not specific agency budgets.
    """
    if TableName not in BEA_VALID_TABLES:
        return [{"error": f"Invalid TableName. Must be one of: {BEA_VALID_TABLES}", "source": "BEA", "status": "failed"}]
    if not BEA_API_KEY: 
        return [{"error": "BEA_API_KEY missing", "source": "BEA", "status": "failed"}]
    
    params = {
        "UserID": BEA_API_KEY, "method": "GetData", "ResultFormat": "json",
        "DataSetName": DataSetName, "TableName": TableName,
        "Frequency": Frequency, "Year": Year, "LineCode": LineCode
    }
    url = "https://apps.bea.gov/api/data"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        logger.error(f"BEA request error: {e}")
        return [{"error": str(e), "source": "BEA", "status": "failed"}]

    results = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    if not results:
        return [{"error": "No data found for parameters.", "source": "BEA", "status": "no_data"}]

    out = []
    for item in results:
        data_value_raw = item.get("DataValue")
        numeric = _parse_numeric_value(data_value_raw)
        unit_multiplier = item.get("UnitMultiplier")
        final_value = _apply_multiplier(numeric, unit_multiplier)
        
        out.append({
            "title": f"BEA: {item.get('LineDescription')}",
            "url": str(r.url),
            "snippet": f"{item.get('LineDescription')} ({item.get('LineCode')}) for {item.get('TimePeriod')}: {data_value_raw} (Unit: {item.get('Unit')}, Multiplier: {unit_multiplier}). Final Value: {final_value}",
            "data_value": final_value,
            "raw_data_value": data_value_raw,
            "line_description": item.get('LineDescription'),
            "line_code": item.get('LineCode'),
            "year": item.get('TimePeriod')
        })
    return out

async def query_census_acs(
    year: str, 
    dataset: str, 
    get_vars: str, 
    for_geo: str
) -> List[Dict[str, Any]]:
    """
    Queries the Census ACS API. Use for demographic data (e.g., population, income).
    Example: year="2022", dataset="acs/acs1/profile", get_vars="NAME,DP05_0001E", for_geo="state:01"
    """
    if not CENSUS_API_KEY:
        return [{"error": "CENSUS_API_KEY missing", "source": "CENSUS", "status": "failed"}]
    
    url = f"https://api.census.gov/data/{year}/{dataset}"
    params = {"key": CENSUS_API_KEY, "get": get_vars, "for": for_geo}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        
        if not data or len(data) < 2:
            return [{"error": "No data found for parameters.", "source": "CENSUS", "status": "no_data"}]

        headers = data[0]
        rows = data[1:]
        results = []
        for row in rows:
            row_data = dict(zip(headers, row))
            snippet = ", ".join(f"{k}: {v}" for k, v in row_data.items() if k not in ["key", "for", "in", "state", "county"])
            results.append({
                "title": f"Census ACS: {row_data.get('NAME', for_geo)}",
                "url": str(r.url),
                "snippet": snippet,
                "data": row_data
            })
        return results
    except Exception as e:
        logger.error(f"Census request error: {e}")
        return [{"error": str(e), "source": "CENSUS", "status": "failed"}]

async def keyword_search_datagov(keyword_query: str) -> List[Dict[str, str]]:
    """
    Searches Data.gov for datasets, reports, and articles. 
    Use for specific agency budgets, reports, or general topics.
    """
    if not keyword_query: return []
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": keyword_query, "rows": 3}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            results = r.json().get("result", {}).get("results", [])
            out = []
            for item in results:
                dataset_page = f"https://catalog.data.gov/dataset/{item.get('name')}"
                snippet = (item.get("notes") or "")[:300]
                org = item.get("organization", {}).get("title", "Unknown Organization")
                out.append({
                    "title": f"Data.gov: {item.get('title')} ({org})",
                    "url": dataset_page,
                    "snippet": snippet
                })
            return out
    except Exception as e:
        logger.error(f"Data.gov request error: {e}")
        return [{"error": str(e), "source": "DATA.GOV", "status": "failed"}]

async def keyword_search_congress(keyword_query: str) -> List[Dict[str, Any]]:
    """
    Searches Congress.gov for bills. Use for claims about legislation.
    """
    if not CONGRESS_API_KEY: 
        return [{"error": "CONGRESS_API_KEY missing", "source": "CONGRESS", "status": "failed"}]
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
                results.append({
                    "title": f"Congress Bill: {bill.get('title')}",
                    "url": bill.get("url", {}).get("url", ""),
                    "snippet": f"Latest Action: {bill.get('latestAction', {}).get('text')}"
                })
            return results
    except Exception as e:
        logger.error(f"Congress request error: {e}")
        return [{"error": str(e), "source": "CONGRESS", "status": "failed"}]

TOOL_DEFINITIONS = [
    {
        "name": "query_bea",
        "description": "Queries BEA NIPA tables. Use for broad *economic functions* (e.g., 'total defense spending'), not specific *agency budgets*.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "TableName": {"type": "STRING", "description": "BEA Table ID, e.g., 'T31600' for federal spending by function."},
                "Year": {"type": "STRING", "description": "The year, e.g., '2023'"},
                "LineCode": {"type": "STRING", "description": "The specific LineCode for the data, e.g., '2' for National Defense in T31600."},
            },
            "required": ["TableName", "Year", "LineCode"]
        }
    },
    {
        "name": "query_census_acs",
        "description": "Queries Census ACS data. Use for demographic data (population, income, etc.).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "year": {"type": "STRING", "description": "The data year, e.g., '2022'"},
                "dataset": {"type": "STRING", "description": "The dataset path, e.g., 'acs/acs1/profile'"},
                "get_vars": {"type": "STRING", "description": "Comma-separated variables, e.g., 'NAME,DP05_0001E'"},
                "for_geo": {"type": "STRING", "description": "The geography, e.g., 'state:01' for Alabama."},
            },
            "required": ["year", "dataset", "get_vars", "for_geo"]
        }
    },
    {
        "name": "keyword_search_datagov",
        "description": "Searches Data.gov. Use for specific *agency budgets*, reports, press releases, or topics not in BEA/Census.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "keyword_query": {"type": "STRING", "description": "The search query, e.g., 'Department of Education budget 2023'"}
            },
            "required": ["keyword_query"]
        }
    },
    {
        "name": "keyword_search_congress",
        "description": "Searches Congress.gov for bills. Use only for claims about legislation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "keyword_query": {"type": "STRING", "description": "The search query, e.g., 'CHIPS and Science Act 2022'"}
            },
            "required": ["keyword_query"]
        }
    }
]

AVAILABLE_TOOLS = {
    "query_bea": query_bea,
    "query_census_acs": query_census_acs,
    "keyword_search_datagov": keyword_search_datagov,
    "keyword_search_congress": keyword_search_congress,
}

async def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Executes a tool call from the LLM and returns a JSON string of the result."""
    func_name = tool_call.get("name")
    func_args = tool_call.get("args", {})
    
    if func_name not in AVAILABLE_TOOLS:
        return json.dumps({"error": f"Unknown tool: {func_name}"})
    
    try:
        func_to_call = AVAILABLE_TOOLS[func_name]
        
        result = await func_to_call(**func_args)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error executing tool {func_name}: {e}")
        return json.dumps({"error": str(e)})


@app.post("/verify")
async def verify(req: VerifyRequest):
    """Main endpoint to verify a claim using a reasoning agent."""
    claim = (req.claim or "").strip()
    if not claim: 
        raise HTTPException(status_code=400, detail="Empty claim.")

    conversation_history = [{"role": "user", "parts": [{"text": f"Begin task: Verify the following claim: '''{claim}'''"}]}]

    MAX_TURNS = 5
    
    try:
        for turn in range(MAX_TURNS):
            prompt = build_agent_prompt(claim, conversation_history)
            agent_response = await call_gemini(prompt, TOOL_DEFINITIONS)

            if agent_text := agent_response.get("text"):
                conversation_history.append({"role": "model", "parts": [{"text": agent_text}]})

            if tool_calls := agent_response.get("tool_calls"):
                tool_call = tool_calls[0] 
                tool_name = tool_call.get("name")
                tool_result_str = await execute_tool_call(tool_call)
                
                conversation_history.append({"role": "model", "parts": [{"functionCall": tool_call}]})
                conversation_history.append({
                    "role": "function",
                    "parts": [{"functionResponse": {"name": tool_name, "response": {"content": tool_result_str}}}]
                })
                logger.info(f"Agent turn {turn}: Called tool {tool_name}, got result: {tool_result_str[:200]}...")

            elif agent_text:
                logger.info(f"Agent turn {turn}: Finished with final answer.")
                final_json = extract_json_block(agent_text)
                
                if final_json and "verdict" in final_json:

                    return {
                        "claim_original": claim,
                        "claim_normalized": final_json.get("claim_normalized", claim),
                        "claim_type": final_json.get("claim_type", "Unknown"),
                        "verdict": final_json.get("verdict", "Inconclusive"),
                        "confidence": final_json.get("confidence", 0.5),
                        "confidence_tier": final_json.get("confidence_tier", "Medium"),
                        "summary": final_json.get("summary", "No summary provided."),
                        "evidence_links": final_json.get("evidence_links", []),
                        "debug_log": conversation_history
                    }
                else:
                    # The agent stopped but didn't provide valid JSON.
                    logger.error("Agent stopped but failed to provide valid JSON.")
                    raise HTTPException(status_code=500, detail="Agent finished but provided a malformed response.")

        logger.warning(f"Max turns ({MAX_TURNS}) reached for claim: {claim}")
        return {
            "claim_original": claim, "verdict": "Inconclusive", "confidence": 0.2,
            "confidence_tier": "Low", "summary": "Could not reach a conclusion in the allotted time.",
            "evidence_links": [], "debug_log": conversation_history
        }

    except Exception as e:
        logger.exception(f"Unhandled error during agent loop for claim: {claim}")
        return {
            "claim_original": claim, "verdict": "Error", "confidence": 0.0,
            "confidence_tier": "Low", "summary": f"An unexpected internal error occurred: {str(e)}",
            "evidence_links": [], "debug_log": conversation_history
        }


def build_agent_prompt(claim: str, history: List[Dict]) -> str:
    """Builds the main prompt for the reasoning agent."""
    
    history_str = ""
    for msg in history:
        role = msg["role"]
        part = msg["parts"][0]
        if role == "user":
            history_str += f"USER:\n{part['text']}\n\n"
        elif role == "model" and "text" in part:
            history_str += f"ASSISTANT (Thought):\n{part['text']}\n\n"
        elif role == "model" and "functionCall" in part:
            func_call = part['functionCall']
            history_str += f"ASSISTANT (Action):\nI will call the tool `{func_call['name']}` with arguments: {json.dumps(func_call['args'])}\n\n"
        elif role == "function":
            func_resp = part['functionResponse']
            history_str += f"TOOL (Observation):\nResult from `{func_resp['name']}`:\n{json.loads(func_resp['response']['content'])}\n\n"

    return f"""
You are a objective, multi-step fact-checker. Your goal is to verify the user's claim using a loop of Thought, Action, and Observation.

**Claim to Verify:** "{claim}"

**Your Process:**
1.  **Thought:** Analyze the claim and the conversation history. Decide if you have enough information, or if you need to use a tool.
2.  **Action:** If you need more data, choose *one* tool from the available list to find the *most specific* piece of missing information.
3.  **Observation:** You will be given the result from the tool.
4.  **Repeat:** Go back to **Thought**. Analyze the new data. If it's not what you need (e.g., it's a broad category, not a specific agency), *think* about why and choose a *different* tool.
5.  **Final Answer:** Once you have gathered enough relevant evidence to make a judgment, stop calling tools and provide your final answer *only* in the specified JSON format.

**CRITICAL RULES:**
-   **BEA vs. Data.gov:** `query_bea` is for *broad economic functions* (e.g., "total spending on defense"). `keyword_search_datagov` is for *specific agency budgets* (e.g., "Department of Defense budget"). Do not confuse them.
-   **One Tool at a Time:** Call only one tool per turn.
-   **Analyze Evidence:** Do not just accept data. If `query_bea` gives you data for "National Defense Function" but the claim is for "Department of Defense," you must recognize this mismatch and try `keyword_search_datagov` instead.

**Conversation History:**
{history_str}

**Your Task:**
Based on the history, provide your next **Thought** and, if necessary, a **Tool Call**.
If you have enough information, provide *only* your **Final Answer** in this exact JSON format:
{{
  "claim_normalized": "A clear, verifiable statement of the claim.",
  "claim_type": "quantitative_comparison / quantitative_value / factual / legislative / other",
  "verdict": "Supported / Contradicted / Inconclusive",
  "confidence": 0.0,
  "confidence_tier": "Low / Medium / High",
  "summary": "A concise summary (1-2 sentences) of the final conclusion.",
  "evidence_links": [
    {{"finding": "The key piece of data found.", "source_url": "The URL of the source."}}
  ]
}}
"""
