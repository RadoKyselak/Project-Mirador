import os
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from urllib.parse import urlencode

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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# === THIS IS THE NEW HEALTH CHECK ROUTE ===
@app.get("/")
async def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok", "message": "Stelthar-API is running."}

class VerifyRequest(BaseModel):
    claim: str

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str

async def call_gemini(prompt: str, system: str = "") -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(GEMINI_ENDPOINT, headers=headers, json=body)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {r.status_code} {r.text}")
        data = r.json()
        text = ""
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = "" # Or handle the error as you see fit
        return {"raw": data, "text": text}

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
    import json
    try:
        parsed = json.loads(text)
    except Exception:
        # Fallback if Gemini fails to produce valid JSON
        normalized = claim.strip().replace("\n", " ")
        parsed = {"claim": normalized, "type": "qualitative", "search_queries": [normalized]}
    return parsed

def pick_sources_from_type(claim_type: str) -> List[str]:
    return ["DATA.GOV"]

async def query_datagov(claim: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY:
        return []
    url = "https://api.data.gov/catalog/v1"
    params = {"api_key": DATA_GOV_API_KEY, "q": claim}
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status() # Will raise an exception for 4XX/5XX responses
            j = r.json()
            results = []
            for item in j.get("results", [])[:3]:
                results.append({
                    "title": item.get("title", "No Title"),
                    "url": item.get("@id", ""),
                    "snippet": item.get("description", "No description available.")[:200]
                })
            return results
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
            return []
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            return []


async def query_sources(sources: List[str], claim: str) -> List[Dict[str, str]]:
    tasks = [query_datagov(claim)]
    results = await asyncio.gather(*tasks)
    return [item for sub in results for item in sub]

def assess_confidence(sources: List[Dict[str, str]], claim_type: str) -> tuple[str, float]:
    if not sources:
        return ("Unverifiable", 0.0)
    score = min(0.5 + (len(sources) * 0.15), 0.95)
    verdict = "Mostly True" if score > 0.6 else "Unclear"
    return (verdict, round(score, 2))

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
    
    sources_results = await query_sources(pick_sources_from_type(claim_type), claim_norm)
    verdict, confidence = assess_confidence(sources_results, claim_type)
    summary = await summarize_with_evidence(claim_norm, sources_results)
    
    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "claim_type": claim_type,
        "search_queries": normalized.get("search_queries", []),
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "sources": sources_results
    }
