#!/usr/bin/env python3
"""Quick-answer API benchmark harness.

The harness intentionally has no third-party runtime dependencies. It reads
API keys from .env, performs one smoke call per provider, and stops on the
first provider failure instead of substituting fallback endpoints.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FRESHQA_README = "https://raw.githubusercontent.com/freshllms/freshqa/main/README.md"
SIMPLEQA_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
USER_AGENT = "liner-quick-answer-benchmark/0.1"
SMOKE_QUERY = "As of today, who is the CEO of OpenAI? Answer briefly and include sources."


class BenchmarkError(Exception):
    """Provider or benchmark execution failure that must stop the run."""


def env_path() -> Path:
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if path.exists():
            return path
    return ROOT / ".env"


@dataclass(frozen=True)
class Question:
    benchmark: str
    query_id: str
    query: str
    reference_answer: str = ""
    category: str = ""


@dataclass(frozen=True)
class Provider:
    key: str
    product: str
    env_var: str
    pricing_note: str


PROVIDERS = [
    Provider("liner", "Liner Quick Answer", "LINER_API_KEY", "$0.003/request published baseline"),
    Provider("perplexity", "Perplexity Sonar", "PERPLEXITY_API_KEY", "Provider usage/pricing; low search context"),
    Provider("exa", "Exa Answer", "EXA_API_KEY", "$0.005/answer published baseline"),
    Provider("parallel", "Parallel Chat", "PARALLEL_API_KEY", "Published pricing snapshot required before external use"),
    Provider("brave", "Brave Answers", "BRAVE_ANSWER_API_KEY", "Brave Answers plan pricing"),
    Provider("tavily", "Tavily Search include_answer", "TAVILY_API_KEY", "Search credits; include_answer enabled"),
]


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        raise BenchmarkError(f"Missing .env at {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    missing = [p.env_var for p in PROVIDERS if not env.get(p.env_var)]
    if missing:
        raise BenchmarkError("Missing required environment variables: " + ", ".join(missing))
    return env


def now_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def http_json(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = {"User-Agent": USER_AGENT}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    req = urllib.request.Request(url, data=encoded, headers=request_headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            first = resp.read(1)
            ttfb_ms = (time.perf_counter() - started) * 1000
            rest = resp.read()
            total_ms = (time.perf_counter() - started) * 1000
            payload_bytes = first + rest
            text = payload_bytes.decode("utf-8", errors="replace")
            try:
                payload = json.loads(text) if text.strip() else {}
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Non-JSON response from {url}: {exc}") from exc
            timing = {"ttfb_ms": round(ttfb_ms, 2), "total_ms": round(total_ms, 2)}
            return payload, timing
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code} from {url}: {redact(text[:500])}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"Network error from {url}: {exc.reason}") from exc


def http_sse(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = 90.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    request_headers.update(headers)
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=request_headers, method="POST")
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    buffer = ""
    first_answer_ms = None
    first_citation_ms = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            first = resp.read(1)
            ttfb_ms = (time.perf_counter() - started) * 1000
            chunks = [first]
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    event = parse_sse_event(event_text)
                    if event is not None:
                        elapsed = (time.perf_counter() - started) * 1000
                        events.append(event)
                        if first_answer_ms is None and extract_answer(event):
                            first_answer_ms = elapsed
                        if first_citation_ms is None and extract_citations(event):
                            first_citation_ms = elapsed
            if buffer.strip():
                event = parse_sse_event(buffer)
                if event is not None:
                    events.append(event)
            total_ms = (time.perf_counter() - started) * 1000
            timing = {
                "ttfb_ms": round(ttfb_ms, 2),
                "first_answer_ms": round(first_answer_ms, 2) if first_answer_ms else None,
                "first_citation_ms": round(first_citation_ms, 2) if first_citation_ms else None,
                "total_ms": round(total_ms, 2),
            }
            return events, timing
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code} from {url}: {redact(text[:500])}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"Network error from {url}: {exc.reason}") from exc


def parse_sse_event(text: str) -> dict[str, Any] | None:
    event_name = "message"
    data_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if not data_lines:
        return None
    data_text = "\n".join(data_lines)
    if data_text == "[DONE]":
        return {"event": event_name, "data": "[DONE]"}
    try:
        data: Any = json.loads(data_text)
    except json.JSONDecodeError:
        data = data_text
    return {"event": event_name, "data": data}


def redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", text)
    return text


def extract_answer(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload if len(payload) > 10 else ""
    if isinstance(payload, list):
        return "\n".join(filter(None, (extract_answer(x) for x in payload)))
    if not isinstance(payload, dict):
        return ""

    candidates = [
        payload.get("answer"),
        payload.get("content"),
        payload.get("text"),
        payload.get("response"),
        payload.get("output_text"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        parts = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}
            for obj in (message, delta):
                if isinstance(obj, dict) and isinstance(obj.get("content"), str):
                    parts.append(obj["content"])
        if parts:
            return "".join(parts).strip()

    for value in payload.values():
        answer = extract_answer(value)
        if answer:
            return answer
    return ""


def extract_citations(payload: Any) -> list[str]:
    urls: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lower = key.lower()
                if lower in {"url", "link", "source_url"} and isinstance(child, str):
                    urls.append(child)
                elif lower in {"citations", "references", "sources", "results", "web_results"}:
                    walk(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            urls.extend(re.findall(r"https?://[^\s)\]}>\"']+", value))

    walk(payload)
    deduped: list[str] = []
    seen = set()
    for url in urls:
        clean = url.rstrip(".,;")
        if clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def call_provider(
    provider: Provider,
    env: dict[str, str],
    question: Question,
    *,
    require_citations: bool = False,
) -> dict[str, Any]:
    api_key = env[provider.env_var]
    query = question.query
    if provider.key == "liner":
        raw, timing = call_liner(api_key, query)
    elif provider.key == "perplexity":
        raw, timing = call_perplexity(api_key, query)
    elif provider.key == "exa":
        raw, timing = call_exa(api_key, query)
    elif provider.key == "parallel":
        raw, timing = call_parallel(api_key, query)
    elif provider.key == "brave":
        raw, timing = call_brave(api_key, query)
    elif provider.key == "tavily":
        raw, timing = call_tavily(api_key, query)
    else:
        raise BenchmarkError(f"Unknown provider {provider.key}")

    answer = extract_answer(raw)
    citations = extract_citations(raw)
    if not answer:
        raise BenchmarkError(f"{provider.product} returned no parseable answer")
    if require_citations and not citations:
        raise BenchmarkError(f"{provider.product} returned no parseable citations/sources")
    return {
        "provider": provider.key,
        "product": provider.product,
        "benchmark": question.benchmark,
        "query_id": question.query_id,
        "query": query,
        "answer": answer,
        "citations": citations,
        "timing": timing,
        "estimated_cost_usd": estimate_cost(provider.key, raw),
        "pricing_basis": provider.pricing_note,
        "raw": raw,
    }


def call_liner(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    events, timing = http_sse(
        "https://platform.liner.com/api/v1/quick-answer",
        headers={"X-API-Key": api_key},
        body={"messages": [{"role": "user", "content": query}]},
    )
    answer_parts = []
    citations = []
    for event in events:
        data = event.get("data") if isinstance(event, dict) else None
        if isinstance(data, dict) and data.get("type") == "text-delta":
            answer_parts.append(str(data.get("delta") or ""))
        citations.extend(extract_citations(event))
    answer = "".join(answer_parts).strip()
    return {"events": events, "answer": answer, "citations": list(dict.fromkeys(citations))}, timing


def call_perplexity(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.perplexity.ai/v1/sonar",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
            "search_context_size": "low",
        },
    )


def call_exa(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.exa.ai/answer",
        headers={"x-api-key": api_key},
        body={"query": query, "text": True},
    )


def call_parallel(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.parallel.ai/v1beta/chat/completions",
        headers={"x-api-key": api_key},
        body={
            "model": "speed",
            "messages": [{"role": "user", "content": query}],
        },
    )


def call_brave(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.search.brave.com/res/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        body={
            "model": "brave",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        },
    )


def call_tavily(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5,
        },
    )


def estimate_cost(provider_key: str, raw: dict[str, Any]) -> float | None:
    if provider_key == "liner":
        return 0.003
    if provider_key == "perplexity":
        usage = raw.get("usage") if isinstance(raw, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if isinstance(cost, dict) and isinstance(cost.get("total_cost"), (int, float)):
            return round(float(cost["total_cost"]), 8)
    if provider_key == "exa":
        return 0.005
    if provider_key == "parallel":
        return 0.005
    if provider_key == "brave":
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            prompt_tokens = float(usage.get("prompt_tokens") or 0)
            completion_tokens = float(usage.get("completion_tokens") or 0)
            searches = float(usage.get("searches") or usage.get("search_count") or 1)
            return round((searches * 4 / 1000) + ((prompt_tokens + completion_tokens) * 5 / 1_000_000), 8)
    if provider_key == "tavily":
        return 0.008
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(usage, dict):
        for key in ("cost", "cost_usd", "total_cost"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return round(float(value), 8)
    return None


def download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def ensure_datasets(data_dir: Path) -> None:
    mkdir(data_dir)
    fresh_path = data_dir / "freshqa.csv"
    simple_path = data_dir / "simpleqa.csv"
    if not fresh_path.exists():
        readme = download_text(FRESHQA_README)
        match = re.search(r"FreshQA [^\]]+\]\(https://docs\.google\.com/spreadsheets/d/([^/]+)/", readme)
        if not match:
            raise BenchmarkError("Could not locate latest FreshQA Google Sheet in README")
        sheet_id = match.group(1)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        fresh_path.write_text(download_text(csv_url), encoding="utf-8")
    if not simple_path.exists():
        simple_path.write_text(download_text(SIMPLEQA_CSV), encoding="utf-8")


def load_questions(data_dir: Path, freshqa_limit: int, simpleqa_limit: int) -> list[Question]:
    ensure_datasets(data_dir)
    questions: list[Question] = []
    questions.extend(load_freshqa(data_dir / "freshqa.csv", freshqa_limit))
    questions.extend(load_simpleqa(data_dir / "simpleqa.csv", simpleqa_limit))
    if not questions:
        raise BenchmarkError("No benchmark questions loaded")
    return questions


def load_freshqa(path: Path, limit: int) -> list[Question]:
    if limit <= 0:
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_idx = 0
    for idx, line in enumerate(lines):
        if line.lower().startswith("id,split,question,"):
            header_idx = idx
            break
    rows = list(csv.DictReader(lines[header_idx:]))
    loaded: list[Question] = []
    for idx, row in enumerate(rows):
        query = first_present(row, ["Question", "question", "Prompt", "prompt"])
        answer = first_present(row, ["answer_0", "Answer", "answer", "Correct answer", "correct_answer"])
        category = first_present(row, ["fact_type", "false_premise", "Type", "Category", "category", "Class"])
        if not query:
            continue
        loaded.append(Question("freshqa", f"freshqa_{idx + 1:04d}", query, answer, category))
        if len(loaded) >= limit:
            break
    return loaded


def load_simpleqa(path: Path, limit: int) -> list[Question]:
    if limit <= 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    loaded: list[Question] = []
    for idx, row in enumerate(rows):
        query = first_present(row, ["problem", "question", "Question", "prompt"])
        answer = first_present(row, ["answer", "Answer", "target"])
        if not query:
            continue
        loaded.append(Question("simpleqa", f"simpleqa_{idx + 1:04d}", query, answer, "stable_short_answer"))
        if len(loaded) >= limit:
            break
    return loaded


def first_present(row: dict[str, str], keys: list[str]) -> str:
    lower = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key in row and row[key].strip():
            return row[key].strip()
        if key.lower() in lower and lower[key.lower()].strip():
            return lower[key.lower()].strip()
    return ""


def score_answer(answer: str, reference: str, citations: list[str]) -> dict[str, Any]:
    if not reference:
        quality = "unscored"
    else:
        quality = "match" if normalize(reference) in normalize(answer) else "needs_review"
    return {
        "quality": quality,
        "has_citations": bool(citations),
        "citation_count": len(citations),
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def run_smoke(env: dict[str, str], run_dir: Path) -> list[dict[str, Any]]:
    smoke_question = Question("smoke", "smoke_0001", SMOKE_QUERY)
    records = []
    for provider in PROVIDERS:
        try:
            record = call_provider(provider, env, smoke_question, require_citations=True)
        except BenchmarkError as exc:
            write_blocking_report(run_dir, provider, exc, records)
            raise
        write_raw(run_dir, record)
        records.append(to_normalized_record(record))
    write_jsonl(run_dir / "normalized" / "smoke_results.jsonl", records)
    return records


def run_benchmark(
    env: dict[str, str],
    run_dir: Path,
    questions: list[Question],
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = list(existing_records or [])
    completed = {
        (row.get("benchmark"), row.get("provider"), row.get("query_id"))
        for row in records
        if row.get("status") == "ok"
    }
    for question in questions:
        for provider in PROVIDERS:
            if (question.benchmark, provider.key, question.query_id) in completed:
                continue
            try:
                record = call_provider(provider, env, question)
            except BenchmarkError as exc:
                write_blocking_report(run_dir, provider, exc, records)
                raise
            write_raw(run_dir, record)
            normalized = to_normalized_record(record)
            normalized["score"] = score_answer(record["answer"], question.reference_answer, record["citations"])
            normalized["reference_answer"] = question.reference_answer
            normalized["category"] = question.category
            records.append(normalized)
            completed.add((question.benchmark, provider.key, question.query_id))
            write_jsonl(run_dir / "normalized" / "results.jsonl", records)
    return records


def write_raw(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "raw" / record["benchmark"] / record["provider"] / f"{record['query_id']}.json"
    mkdir(path.parent)
    raw_payload = {
        "provider": record["provider"],
        "product": record["product"],
        "benchmark": record["benchmark"],
        "query_id": record["query_id"],
        "query": record["query"],
        "timing": record["timing"],
        "raw": record["raw"],
    }
    path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    record["raw_file"] = str(path.relative_to(run_dir))


def to_normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": record["provider"],
        "product": record["product"],
        "benchmark": record["benchmark"],
        "query_id": record["query_id"],
        "query": record["query"],
        "answer": record["answer"],
        "citations": record["citations"],
        "status": "ok",
        "timing": record["timing"],
        "estimated_cost_usd": record["estimated_cost_usd"],
        "pricing_basis": record["pricing_basis"],
        "raw_file": record.get("raw_file"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    mkdir(path.parent)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BenchmarkError(f"Cannot reuse smoke results; missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_blocking_report(
    run_dir: Path,
    provider: Provider,
    error: Exception,
    completed_records: list[dict[str, Any]],
) -> None:
    mkdir(run_dir)
    required_change = required_user_change(provider, str(error))
    report = [
        "# Blocking Report",
        "",
        "Benchmark execution stopped because a configured provider failed smoke/benchmark execution.",
        "",
        f"- Failed provider: {provider.product} (`{provider.key}`)",
        f"- Required env var: `{provider.env_var}`",
        f"- Error summary: `{redact(str(error))}`",
        f"- Completed records before stop: {len(completed_records)}",
        "",
        "## Required User Change",
        "",
        required_change,
        "",
        "No fallback provider was used.",
    ]
    (run_dir / "blocking_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def required_user_change(provider: Provider, error_text: str) -> str:
    if "INSUFFICIENT_CREDITS" in error_text or "HTTP 402" in error_text:
        return (
            f"Add credits to the account behind `{provider.env_var}` or replace `{provider.env_var}` "
            "with a key that has enough credits, then rerun the benchmark."
        )
    if "HTTP 401" in error_text or "HTTP 403" in error_text:
        return f"Verify that `{provider.env_var}` is valid and has access to {provider.product}."
    if "Network error" in error_text or "Connection reset" in error_text:
        return (
            f"Check whether {provider.product} is reachable from this machine/network and whether the provider "
            "is temporarily resetting requests. If it was transient, rerun with `--resume`; completed calls will be skipped."
        )
    if "no parseable answer" in error_text:
        return f"Check the {provider.product} adapter response schema; the API returned no parseable answer."
    return "Fix the provider key, endpoint access, plan/quota, or adapter contract for the failed provider, then rerun the command."


def write_report(run_dir: Path, smoke: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    report_path = run_dir / "report.md"
    lines = [
        "# Quick Answer API Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Providers: {', '.join(p.product for p in PROVIDERS)}",
        f"- Smoke tests: {len(smoke)} passed, 0 failed.",
        f"- Benchmark calls: {len(records)}.",
        "- Primary benchmark: FreshQA.",
        "- Secondary benchmark: SimpleQA sanity slice.",
        "- Scoring is deterministic and conservative: exact normalized reference containment is marked `match`; other answers require human review.",
        "",
        "## Provider Setup",
        "",
        "| Provider | Product | Pricing basis |",
        "| --- | --- | --- |",
    ]
    for provider in PROVIDERS:
        lines.append(f"| {provider.key} | {provider.product} | {provider.pricing_note} |")
    lines.extend([
        "",
        "## Benchmark Methodology",
        "",
        f"- FreshQA limit: {args.freshqa_limit}",
        f"- SimpleQA limit: {args.simpleqa_limit}",
        "- One request per provider per benchmark question.",
        "- No fallback substitution was allowed. Any provider failure stops the run.",
        "- Raw responses were written under `raw/<benchmark>/<provider>/<query_id>.json`.",
        "",
        "## Smoke Test Results",
        "",
        "| Provider | TTFB ms | Total ms | Citations | Raw file |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for record in smoke:
        timing = record["timing"]
        lines.append(
            f"| {record['provider']} | {timing.get('ttfb_ms')} | {timing.get('total_ms')} | "
            f"{len(record['citations'])} | `{record['raw_file']}` |"
        )
    lines.extend(render_metrics_sections(records))
    lines.extend([
        "",
        "## Failures",
        "",
        "No provider failures were encountered in this completed run.",
        "",
        "## Raw Output Index",
        "",
    ])
    for record in smoke + records:
        lines.append(f"- `{record['provider']}` `{record['benchmark']}` `{record['query_id']}`: `{record['raw_file']}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_metrics_sections(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Accuracy",
        "",
        "| Provider | Calls | Match | Needs review | Unscored |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_provider = group_by(records, "provider")
    for provider, rows in by_provider.items():
        qualities = [((row.get("score") or {}).get("quality") or "unscored") for row in rows]
        lines.append(
            f"| {provider} | {len(rows)} | {qualities.count('match')} | "
            f"{qualities.count('needs_review')} | {qualities.count('unscored')} |"
        )
    lines.extend([
        "",
        "## Citation Quality",
        "",
        "| Provider | Calls with citations | Average citations |",
        "| --- | ---: | ---: |",
    ])
    for provider, rows in by_provider.items():
        counts = [len(row["citations"]) for row in rows]
        lines.append(f"| {provider} | {sum(1 for c in counts if c > 0)} | {mean(counts):.2f} |")
    lines.extend([
        "",
        "## Latency",
        "",
        "| Provider | p50 total ms | p90 total ms | p95 total ms | p50 TTFB ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for provider, rows in by_provider.items():
        totals = [row["timing"].get("total_ms") for row in rows if row["timing"].get("total_ms") is not None]
        ttfbs = [row["timing"].get("ttfb_ms") for row in rows if row["timing"].get("ttfb_ms") is not None]
        lines.append(f"| {provider} | {pct(totals, 50)} | {pct(totals, 90)} | {pct(totals, 95)} | {pct(ttfbs, 50)} |")
    lines.extend([
        "",
        "## Cost",
        "",
        "| Provider | Estimated total USD | Estimated cost / 1K calls | Cost basis completeness |",
        "| --- | ---: | ---: | --- |",
    ])
    for provider, rows in by_provider.items():
        costs = [row["estimated_cost_usd"] for row in rows if row["estimated_cost_usd"] is not None]
        total = sum(costs)
        per_1k = (total / len(rows) * 1000) if rows and len(costs) == len(rows) else None
        basis = "complete" if len(costs) == len(rows) else "partial; refresh provider pricing before publication"
        lines.append(f"| {provider} | {total:.6f} | {per_1k:.4f} | {basis} |")
    return lines


def group_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return grouped


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pct(values: list[float], percentile: int) -> str:
    if not values:
        return "n/a"
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return f"{sorted_values[idx]:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick-answer API smoke tests and benchmarks.")
    parser.add_argument("--freshqa-limit", type=int, default=50)
    parser.add_argument("--simpleqa-limit", type=int, default=50)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reuse-smoke", action="store_true", help="Reuse normalized/smoke_results.jsonl in the selected run dir.")
    parser.add_argument("--resume", action="store_true", help="Resume from normalized/results.jsonl in the selected run dir.")
    parser.add_argument("--run-id", default=now_run_id())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "results" / args.run_id
    mkdir(run_dir / "normalized")
    try:
        env = load_env(env_path())
        smoke = read_jsonl(run_dir / "normalized" / "smoke_results.jsonl") if args.reuse_smoke else run_smoke(env, run_dir)
        records: list[dict[str, Any]] = []
        if not args.smoke_only:
            questions = load_questions(ROOT / "data", args.freshqa_limit, args.simpleqa_limit)
            existing = read_jsonl(run_dir / "normalized" / "results.jsonl") if args.resume else []
            records = run_benchmark(env, run_dir, questions, existing)
        write_report(run_dir, smoke, records, args)
    except BenchmarkError as exc:
        print(f"BLOCKED: {redact(str(exc))}", file=sys.stderr)
        print(f"See {run_dir / 'blocking_report.md'}", file=sys.stderr)
        return 2
    print(f"Completed run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
