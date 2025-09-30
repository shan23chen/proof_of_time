# emnlp_topics_benchmarks_litellm.py
# Run: python emnlp_topics_benchmarks_litellm.py

import os, re, json, asyncio
from threading import Thread
from typing import Dict, Any, Tuple, List
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jsonschema import Draft202012Validator, ValidationError

from litellm import acompletion  # Async OpenAI-compatible API

# -------------------- CONFIG --------------------
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")  # any LiteLLM-supported model
TEMPERATURE = 0
TIMEOUT = 120
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
SPLIT_WORKERS = max(1, int(os.getenv("SPLIT_WORKERS", "1")))
MAX_TOKENS = 400  # plenty for our JSON
ENABLE_REPAIR_PASS = True

# -------------------- PROMPT & SCHEMA --------------------
SYSTEM_PROMPT = """You are an expert research curator for NLP/ML conferences.
Extract structured metadata from each paper using ONLY the title and abstract.

Return STRICT JSON matching the provided schema.

Tagging rules:
- topics (coarse): 1–5 concise themes (e.g., "evaluation/benchmarks", "machine translation", "retrieval augmentation").
- primary_topic: exactly one fine-grained label from FINE_TOPICS (below) that best characterizes the paper. If none fits, craft a new concise label prefixed with "other:" and also include it in fine_topics.
- fine_topics: up to 8 labels from FINE_TOPICS; if nothing fits, use "other:<short label>" entries that you invent to best describe the work.
- domain_tags: optional domains (e.g., "biomedical", "legal", "finance", "multilingual", "low-resource", "social media"). Return an empty list if none apply.
- Benchmark detection:
  * is_benchmark = true iff the paper RELEASES a new dataset/benchmark/leaderboard or a major standardized evaluation suite.
  * If true, fill benchmark_name (short) and up to 5 benchmark_tasks (e.g., "NER", "VQA", "MMLU-style multitask"). If false, return benchmark_name as an empty string and benchmark_tasks as an empty list.
- Be conservative; do not infer beyond evidence.

FINE_TOPICS (controlled vocabulary):
Core NLP Tasks
- "tokenization/subwording", "morphology/POS", "parsing/dependency", "parsing/constituency",
  "coreference", "anaphora/cataphora", "WSD/lexical semantics",
  "NER", "RE/relation extraction", "event extraction", "temporal IE",
  "slot filling", "fact verification", "stance detection",
  "QA/extractive", "QA/abstractive", "open-domain QA", "MCQ reasoning",
  "machine translation", "summarization", "simplification",
  "dialogue/task-oriented", "dialogue/open-domain",
  "style transfer/controllable text", "toxicity detection", "hate/abuse detection"

LLMs & Methods
- "pretraining/objectives", "continued pretraining/domain-adapt",
  "instruction tuning/SFT", "preference learning/DPO",
  "RLHF/GRPO/PPO", "tool use/function calling", "agents/planning",
  "retrieval augmentation/dense", "retrieval augmentation/sparse",
  "routing/mixture-of-experts", "speculative decoding/medusa",
  "sampling/decoding control", "long-context/memory",
  "multilingual alignment", "code LMs/generation", "program repair"

Safety, Robustness, Interpretability
- "safety/jailbreak defense", "safety/toxicity/hallucination",
  "bias/fairness", "privacy/federated/DP", "robustness/OOD/adversarial",
  "evaluation methodology", "audit/measurement",
  "interpretability/SAE/features", "mechanistic interpretability",
  "explanations/self-consistency", "uncertainty/calibration"

Multimodality & Speech
- "VLM/vision-language", "image captioning", "text-to-image",
  "video QA/reasoning", "ASR/speech recognition", "TTS/speech synthesis",
  "audio LMs", "multimodal RAG"

Efficiency & Systems
- "distillation", "quantization", "pruning/LoRA/adapters", "KV-cache/attention optimizations",
  "throughput/latency/inference serving", "data filtering/deduplication",
  "dataset synthesis/augmentation", "scaling laws", "theory/generalization"

Knowledge & Structure
- "knowledge graphs/ontologies", "neural-symbolic", "reasoning with tools",
  "factuality/knowledge editing"

Domains (use in domain_tags when applicable)
- "biomedical", "clinical", "legal", "finance", "education", "social media",
  "scientific literature", "code", "multilingual", "low-resource"
"""

JSON_SCHEMA = {
  "name": "paper_metadata",
  "schema": {
    "type": "object",
    "properties": {
      "topics": { "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5 },
      "primary_topic": { "type": "string" },
      "fine_topics": { "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8 },
      "domain_tags": { "type": "array", "items": {"type": "string"}, "maxItems": 5 },
      "is_benchmark": { "type": "boolean" },
      "benchmark_name": { "type": "string" },
      "benchmark_tasks": { "type": "array", "items": {"type": "string"}, "maxItems": 5 }
    },
    "required": [
      "topics",
      "primary_topic",
      "fine_topics",
      "domain_tags",
      "is_benchmark",
      "benchmark_name",
      "benchmark_tasks"
    ],
    "additionalProperties": False
  },
  "strict": True
}

# Build a JSON Schema validator object once
JSON_VALIDATOR = Draft202012Validator(JSON_SCHEMA["schema"])

def build_user_content(title: str, abstract: str) -> str:
    return f"TITLE: {title or ''}\n\nABSTRACT: {abstract or ''}"

class TemporaryModelError(Exception):
    pass

def extract_json_block(text: str) -> str:
    """
    Try to pull the first top-level JSON object from text.
    """
    if not text:
        raise ValueError("Empty response text")
    # Simple bracket matching to find the first {...} block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    # Fallback: last brace
    end = text.rfind("}")
    if end != -1 and end > start:
        return text[start:end+1]
    raise ValueError("Could not extract a full JSON object")

def validate_or_raise(payload: Dict[str, Any]) -> None:
    JSON_VALIDATOR.validate(payload)

def postprocess_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize missing optionals if classed as non-benchmark
    d.setdefault("topics", [])
    d.setdefault("fine_topics", [])
    d.setdefault("domain_tags", [])
    d.setdefault("is_benchmark", False)
    if not d.get("is_benchmark", False):
        d["benchmark_name"] = ""
        d["benchmark_tasks"] = []
    else:
        d.setdefault("benchmark_name", "")
        d.setdefault("benchmark_tasks", [])
    return d

async def llm_once(title: str, abstract: str, repair_note: str = "") -> Dict[str, Any]:
    """
    Single LLM call with strict JSON schema + optional repair instruction.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(title, abstract)}
    ]
    if repair_note:
        messages.append({
            "role": "system",
            "content": f"Previous response violated the schema ({repair_note}). "
                       f"Return ONLY valid JSON for the schema—no prose."
        })

    resp = await acompletion(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": JSON_SCHEMA
        },
        extra_body={
            # Some providers honour 'seed' for extra determinism; safe to include if ignored.
            "seed": 7
        }
    )
    text = resp.choices[0].message["content"]
    if not text:
        raise TemporaryModelError("Empty LLM content")
    # Parse JSON strictly
    try:
        data = json.loads(text)
    except Exception:
        # Try to extract a JSON block if the model leaked extra text
        data = json.loads(extract_json_block(text))
    # Validate against our schema
    try:
        validate_or_raise(data)
    except ValidationError as ve:
        raise TemporaryModelError(f"Schema validation failed: {ve.message}")
    return postprocess_defaults(data)

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type(TemporaryModelError)
)
async def call_model_with_repairs(title: str, abstract: str) -> Dict[str, Any]:
    try:
        return await llm_once(title, abstract)
    except TemporaryModelError as e:
        if ENABLE_REPAIR_PASS:
            # One guided repair attempt
            return await llm_once(title, abstract, repair_note=str(e))
        raise

async def process_split(ds, concurrency: int = MAX_CONCURRENCY):
    sem = asyncio.Semaphore(concurrency)
    n = len(ds)

    topics_col: List[List[str]] = [None]*n
    primary_topic_col: List[str] = [None]*n
    fine_topics_col: List[List[str]] = [None]*n
    domain_tags_col: List[List[str]] = [None]*n
    is_bench_col: List[bool] = [None]*n
    bench_name_col: List[str] = [None]*n
    bench_tasks_col: List[List[str]] = [None]*n

    async def worker(i, row):
        async with sem:
            md = await call_model_with_repairs(row.get("title",""), row.get("abstract",""))
            topics_col[i] = md.get("topics", [])
            primary_topic_col[i] = md.get("primary_topic", "")
            fine_topics_col[i] = md.get("fine_topics", [])
            domain_tags_col[i] = md.get("domain_tags", [])
            is_bench_col[i] = bool(md.get("is_benchmark", False))
            bench_name_col[i] = md.get("benchmark_name", "")
            bench_tasks_col[i] = md.get("benchmark_tasks", [])

    tasks = [worker(i, ds[i]) for i in range(n)]
    for fut in tqdm(asyncio.as_completed(tasks), total=n, desc="LLM infer"):
        await fut

    ds = ds.add_column("topics", topics_col)
    ds = ds.add_column("primary_topic", primary_topic_col)
    ds = ds.add_column("fine_topics", fine_topics_col)
    ds = ds.add_column("domain_tags", domain_tags_col)
    ds = ds.add_column("is_benchmark", is_bench_col)
    ds = ds.add_column("benchmark_name", bench_name_col)
    ds = ds.add_column("benchmark_tasks", bench_tasks_col)
    return ds

async def process_all_splits(dsdict: DatasetDict) -> DatasetDict:
    out: Dict[str, Any] = {}
    items = list(dsdict.items())

    async def run_single(split: str, ds) -> Tuple[str, Any]:
        print(f"Processing split: {split} | rows={len(ds)}")
        processed = await process_split(ds, concurrency=MAX_CONCURRENCY)
        return split, processed

    if SPLIT_WORKERS > 1 and len(items) > 1:
        sem = asyncio.Semaphore(SPLIT_WORKERS)

        async def run_with_limit(split: str, ds):
            async with sem:
                return await run_single(split, ds)

        tasks = [asyncio.create_task(run_with_limit(split, ds)) for split, ds in items]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Split workers"):
            split, processed = await fut
            out[split] = processed
    else:
        for split, ds in items:
            split_name, processed = await run_single(split, ds)
            out[split_name] = processed

    return DatasetDict(out)


async def amain() -> DatasetDict:
    """Asynchronous entry point so notebooks can simply `await amain()`."""
    dsdict: DatasetDict = load_dataset("AIM-Harvard/EMNLP-Accepted-Papers")
    out = await process_all_splits(dsdict)

    out_dir = "emnlp_with_topics_benchmarks"
    out.save_to_disk(out_dir)
    print(f"\nSaved enriched dataset to: {out_dir}")

    # Quick summary
    total = sum(len(ds) for ds in out.values())
    benchmarks = sum(int(x) for ds in out.values() for x in ds["is_benchmark"])
    print(f"Total rows: {total} | Detected benchmark papers: {benchmarks}")
    return out


def main() -> DatasetDict:
    """Synchronous wrapper that works in both scripts and notebooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(amain())

    # If we reach here we are already inside an event loop (e.g. Jupyter).
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def runner():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result_box["value"] = new_loop.run_until_complete(amain())
        except BaseException as exc:  # propagate KeyboardInterrupt too
            error_box["error"] = exc
        finally:
            new_loop.close()

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error_box:
        raise error_box["error"]
    return result_box["value"]


if __name__ == "__main__":
    main()
