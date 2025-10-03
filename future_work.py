#!/usr/bin/env python3
# future_work_with_verification.py (LLM-as-a-Judge, normalized scoring)

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jsonschema import Draft202012Validator, ValidationError
from litellm import acompletion
import requests

# -------------------- CONFIG --------------------
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = 0
TIMEOUT = 120
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
MAX_TOKENS = 400
ENABLE_REPAIR_PASS = True
FREQ_AWARE = False  # set True to reward frequency of occurrence across multiple papers

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SCHOLAR_API_KEY", None)

last_request_time = 0
def rate_limit():
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < 3.1:
        sleep_time = 3.1 - time_since_last
        print(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    last_request_time = time.time()

# -------------------- Forecast Schema --------------------
SYSTEM_PROMPT = """You are an expert research analyst.
Given a professor’s name and affiliation, forecast their most likely research direction in 2025.
"""

JSON_SCHEMA = {
    "name": "future_research_forecast",
    "schema": {
        "type": "object",
        "properties": {
            "professor": {"type": "string"},
            "affiliation": {"type": "string"},
            "predicted_keywords": {"type": "array", "items": {"type": "string"}},
            "predicted_subfields": {"type": "array", "items": {"type": "string"}},
            "predicted_modalities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["professor", "affiliation", "predicted_keywords",
                     "predicted_subfields", "predicted_modalities"],
        "additionalProperties": False
    }
}
JSON_VALIDATOR = Draft202012Validator(JSON_SCHEMA["schema"])

# -------------------- LLM Prediction --------------------
class TemporaryModelError(Exception):
    pass

def validate_or_raise(payload: Dict[str, Any]) -> None:
    JSON_VALIDATOR.validate(payload)

async def llm_once(name: str, affiliation: str, repair_note: str = "") -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROFESSOR: {name}\nAFFILIATION: {affiliation}"}
    ]
    if repair_note:
        messages.append({
            "role": "system",
            "content": f"Previous response violated schema ({repair_note}). Return ONLY valid JSON."
        })

    resp = await acompletion(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
    )
    text = resp.choices[0].message["content"]
    data = json.loads(text)
    validate_or_raise(data)
    return data

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type(TemporaryModelError)
)
async def call_model_with_repairs(name: str, affiliation: str) -> Dict[str, Any]:
    return await llm_once(name, affiliation)

# -------------------- Semantic Scholar Fetch --------------------
def get_author_id_by_name(name: str, affiliation: str = "") -> Optional[str]:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    rate_limit()
    url = f"{SEMANTIC_SCHOLAR_API}/author/search"
    params = {"query": name, "fields": "name,affiliations,url,citationCount"}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        return None
    results = resp.json().get("data", [])
    if not results:
        return None
    if affiliation:
        for a in results:
            if affiliation.lower() in str(a.get("affiliations", "")).lower():
                return a["authorId"]
    sorted_results = sorted(results, key=lambda x: x.get("citationCount", 0), reverse=True)
    return sorted_results[0]["authorId"]

def get_all_2025_papers(author_id: str, author_name: str = "") -> List[Dict[str, Any]]:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    rate_limit()
    url = f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers"
    params = {"fields": "title,abstract,year,venue,url", "limit": 1000}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        return []
    return [p for p in resp.json().get("data", []) if p.get("year") == 2025]

# -------------------- Paper Metadata Extraction --------------------
async def extract_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    title = paper.get("title", "No title")
    abstract = paper.get("abstract", "No abstract available")
    venue = paper.get("venue", "Unknown venue")

    extraction_prompt = f"""Extract structured metadata from this 2025 research paper.

PAPER:
- Title: {title}
- Abstract: {abstract}
- Venue: {venue}

Return JSON with arrays for actual_keywords, actual_subfields, actual_modalities.
"""
    schema = {
        "name": "paper_metadata",
        "schema": {
            "type": "object",
            "properties": {
                "actual_keywords": {"type": "array", "items": {"type": "string"}},
                "actual_subfields": {"type": "array", "items": {"type": "string"}},
                "actual_modalities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["actual_keywords", "actual_subfields", "actual_modalities"],
            "additionalProperties": False
        }
    }

    try:
        resp = await acompletion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert extracting metadata."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_schema", "json_schema": schema}
        )
        return json.loads(resp.choices[0].message["content"]) | {"title": title}
    except Exception:
        return {
            "actual_keywords": title.lower().split()[:5],
            "actual_subfields": ["unknown"],
            "actual_modalities": ["unknown"],
            "title": title
        }

# -------------------- LLM Judge --------------------
async def llm_judge_prediction_vs_actual(pred: Dict[str, Any], paper_metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Collapse all actuals into one set
    actual_keywords = {kw.lower() for p in paper_metas for kw in p.get("actual_keywords", [])}
    actual_subfields = {sf.lower() for p in paper_metas for sf in p.get("actual_subfields", [])}
    actual_modalities = {md.lower() for p in paper_metas for md in p.get("actual_modalities", [])}

    papers_str = "\n".join(
        [f"- {p['title']}: keywords={p.get('actual_keywords', [])}, "
         f"subfields={p.get('actual_subfields', [])}, "
         f"modalities={p.get('actual_modalities', [])}" for p in paper_metas]
    )

    prompt = f"""You are an expert judge of AI research forecasts.

Forecast:
- Keywords: {pred.get("predicted_keywords", [])}
- Subfields: {pred.get("predicted_subfields", [])}
- Modalities: {pred.get("predicted_modalities", [])}

Actual 2025 publications:
{papers_str}

For each forecasted keyword, subfield, and modality:
- Return 1 if it appears in the actual research (allow synonyms/close topics), else 0.
If FREQ_AWARE = True, return a fractional value 0–1 representing its frequency across papers.

Return strict JSON with arrays matching forecast order.
"""
    schema = {
        "name": "prediction_match",
        "schema": {
            "type": "object",
            "properties": {
                "keyword_matches": {"type": "array", "items": {"type": "number"}},
                "subfield_matches": {"type": "array", "items": {"type": "number"}},
                "modality_matches": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["keyword_matches", "subfield_matches", "modality_matches"]
        }
    }

    resp = await acompletion(
        model=MODEL,
        messages=[{"role": "system", "content": "You are an expert evaluator."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
        response_format={"type": "json_schema", "json_schema": schema}
    )
    matches = json.loads(resp.choices[0].message["content"])

    # compute scores
    def compute_scores(pred_list, match_list, actual_set):
        tp = sum(1 for m in match_list if m > 0)
        precision = tp / len(pred_list) if pred_list else 0
        recall = tp / len(actual_set) if actual_set else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision+recall) > 0 else 0
        return precision, recall, f1, tp

    kw_p, kw_r, kw_f1, _ = compute_scores(pred["predicted_keywords"], matches["keyword_matches"], actual_keywords)
    sf_p, sf_r, sf_f1, _ = compute_scores(pred["predicted_subfields"], matches["subfield_matches"], actual_subfields)
    md_p, md_r, md_f1, _ = compute_scores(pred["predicted_modalities"], matches["modality_matches"], actual_modalities)

    return {
        "keyword_precision": round(kw_p, 3),
        "keyword_recall": round(kw_r, 3),
        "subfield_precision": round(sf_p, 3),
        "subfield_recall": round(sf_r, 3),
        "modality_precision": round(md_p, 3),
        "modality_recall": round(md_r, 3),
        "overall_score": round((kw_f1 + sf_f1 + md_f1)/3, 3),
        "matches": matches,
        "n_actual_items": len(paper_metas),
        "author_prediction": pred
    }

# -------------------- Verification --------------------
async def verify_prediction(name: str, affiliation: str, pred: Dict[str, Any]) -> Dict[str, Any]:
    try:
        author_id = await asyncio.get_event_loop().run_in_executor(None, get_author_id_by_name, name, affiliation)
        if not author_id:
            return {"prediction": pred, "verification": None, "error": "author_id not found"}
        papers_2025 = await asyncio.get_event_loop().run_in_executor(None, get_all_2025_papers, author_id, name)
        if not papers_2025:
            return {"prediction": pred, "verification": None, "error": "no 2025 papers found"}

        paper_metas = [await extract_paper_metadata(p) for p in papers_2025]
        assessment = await llm_judge_prediction_vs_actual(pred, paper_metas)
        return {"prediction": pred, "verification": assessment}
    except Exception as e:
        return {"prediction": pred, "verification": None, "error": str(e)}

# -------------------- Batch --------------------
async def process_and_verify(professors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async def worker(entry):
        async with sem:
            pred = await call_model_with_repairs(entry["professor"], entry["affiliation"])
            return await verify_prediction(entry["professor"], entry["affiliation"], pred)
    tasks = [worker(entry) for entry in professors]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(professors), desc="Predict+Verify"):
        results.append(await fut)
    return results

def load_professors_from_json(file_path="data/influential_ai_ppl_list.json") -> List[Dict[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    professors = []
    if "TIME100_AI_Professors" in data:
        for year_data in data["TIME100_AI_Professors"].values():
            for person in year_data.get("people", []):
                professors.append({"professor": person["name"], "affiliation": person["affiliation"]})
    if "AI2050_Fellows" in data:
        for year_data in data["AI2050_Fellows"].values():
            for year, level_data in year_data.items():
                for person in level_data:
                    professors.append({"professor": person, "affiliation": ""})
    return professors

# -------------------- Main --------------------
async def main():
    professors = load_professors_from_json()
    print(f"Loaded {len(professors)} professors from data file")
    results = await process_and_verify(professors)
    with open("predictions_with_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Finished. Results saved to predictions_with_verification.json")

if __name__ == "__main__":
    asyncio.run(main())
