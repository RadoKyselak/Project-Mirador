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

class VerifyRequest(BaseModel):
    claim: str

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str

async def call_gemini(prompt: str, system: str = "") -> Dict[str, Any]:
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
            candidates = data.get("candidates") or []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts") or candidates[0].get("parts")
                if parts:
                    text = "".join(parts)
            if not text:
                def find_text(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            res = find_text(v)
                            if res:
                                return res
                    if isinstance(obj, list):
                        for v in obj:
                            res = find_text(v)
                            if res:
                                return res
                    return None
                text = find_text(data) or ""
        except Exception:
            text = ""
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
    text = res["text"].strip()
    import json
    try:
        parsed = json.loads(text)
    except Exception:
        normalized = claim.strip().replace("\n", " ")
        cl = normalized.lower()
        if any(w in cl for w in ["%","percent","increase","decrease","grow","fell","rate"]):
            ctype = "quantitative"
            queries = [normalized, "gdp 2024 value", "official statistics"]
        elif any(w in cl for w in ["bill","passed","law","legislation","congress"]):
            ctype = "factual"
            queries = [normalized, "congress.gov " + normalized, "official bill text"]
        else:
            ctype = "qualitative"
            queries = [normalized, normalized + " government data", "official report"]
        parsed = {"claim": normalized, "type": ctype, "search_queries": queries}
    return parsed

def pick_sources_from_type(claim_type: str) -> List[str]:
    return ["DATA.GOV"]

async def query_datagov(claim: str) -> List[Dict[str, str]]:
    if not DATA_GOV_API_KEY:
        return []
    url = "https://api.data.gov/catalog/v1"
    params = {"api_key": DATA_GOV_API_KEY, "q": claim}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            return []
        j = r.json()
        results = []
        for item in j.get("results", [])[:3]:
            results.append({
                "title": item.get("title", "No Title"),
                "url": item.get("@id", ""),
                "snippet": item.get("description", "No description available.")
            })
        return results

async def query_sources(sources: List[str>, claim: str) -> List[Dict[str, str]]:
    tasks = []
    for s in sources:
        if s == "DATA.GOV":
            tasks.append(query_datagov(claim))
        else:
            tasks.append(asyncio.sleep(0, result=[]))
    results = await asyncio.gather(*tasks)
    flat = [item for sub in results for item in sub]
    return flat

def assess_confidence(sources: List[Dict[str, str]], claim_type: str) -> (str, float):
    if not sources:
        return ("Unverifiable", 0.0)
    
    score = 0.5 
    verdict = "Unclear"
    if len(sources) > 0:
        score = 0.75
        verdict = "Mostly True"

    return (verdict, round(score, 2))

async def summarize_with_evidence(claim: str, sources: List[Dict[str, str]]) -> str:
    context_parts = []
    for s in sources[:6]:
        context_parts.append(f"Source: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}\n---\n")
    context = "\n".join(context_parts)
    prompt = (
        "You are a concise assistant that reads official-source snippets and returns:\n"
        "1) a one-sentence plain-English conclusion about the claim\n"
        "2) a short justification (1-2 sentences) citing which sources support or contradict\n\n"
        f"Claim: '''{claim}'''\n\n"
        f"Context from official sources:\n{context}\n\n"
        "Return only JSON: {\"summary\": ..., \"justification\": ...}\n"
    )
    res = await call_gemini(prompt)
    text = res["text"].strip()
    import json
    try:
        parsed = json.loads(text)
        return parsed.get("summary", "") + " " + parsed.get("justification", "")
    except Exception:
        if sources:
            return f"According to {sources[0]['title']}, the claim appears to be supported. See linked sources."
        else:
            return "No supporting government data found."

@app.post("/verify")
async def verify(req: VerifyRequest):
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Empty claim.")
    normalized = await normalize_and_classify_claim(claim)
    claim_norm = normalized.get("claim")
    claim_type = normalized.get("type")
    queries = normalized.get("search_queries", [])
    sources_to_query = pick_sources_from_type(claim_type)
    sources_results = await query_sources(sources_to_query, claim_norm)
    verdict, confidence = assess_confidence(sources_results, claim_type)
    summary = await summarize_with_evidence(claim_norm, sources_results)
    return {
        "claim_original": claim,
        "claim_normalized": claim_norm,
        "claim_type": claim_type,
        "search_queries": queries,
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "sources": sources_results
    }
