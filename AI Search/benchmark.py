#!/usr/bin/env python3
"""AI Search API benchmark harness for FreshQA.

The harness reads API keys from the workspace .env, runs one smoke call per
provider, then runs a balanced FreshQA slice. It writes raw provider payloads
and normalized records without substituting fallback providers.
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
import urllib.request
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
FRESHQA_CSV = WORKSPACE / "Quick Answer" / "data" / "freshqa.csv"
USER_AGENT = "liner-ai-search-benchmark/0.1"
SMOKE_QUERY = "As of today, who is the CEO of OpenAI? Answer briefly and include sources."


class BenchmarkError(Exception):
    """Provider or benchmark execution failure that must stop the run."""


@dataclass(frozen=True)
class Question:
    benchmark: str
    query_id: str
    query: str
    reference_answer: str
    category: str
    false_premise: bool
    fact_type: str


@dataclass(frozen=True)
class Provider:
    key: str
    product: str
    env_var: str
    pricing_note: str


PROVIDERS = [
    Provider("liner", "Liner AI Search", "LINER_API_KEY", "$0.010/request published AI Search baseline"),
    Provider("liner_pro", "Liner AI Search Pro", "LINER_API_KEY", "$0.100/request published AI Search Pro baseline"),
    Provider("perplexity", "Perplexity Sonar Pro", "PERPLEXITY_API_KEY", "Sonar Pro low context request fee plus token usage"),
    Provider("exa", "Exa Answer", "EXA_API_KEY", "$0.005/answer published baseline"),
    Provider("exa_deep", "Exa Deep Search", "EXA_API_KEY", "$0.012/request published Deep Search baseline"),
    Provider("parallel", "Parallel Chat", "PARALLEL_API_KEY", "Parallel Chat API speed processor: $5 per 1K requests"),
    Provider("brave", "Brave Answers", "BRAVE_ANSWER_API_KEY", "Brave Answers usage-derived estimate when available"),
    Provider("tavily", "Tavily Search include_answer advanced", "TAVILY_API_KEY", "Advanced search costs 2 Tavily credits; estimated $0.016/request"),
]


def env_path() -> Path:
    for path in (ROOT / ".env", WORKSPACE / ".env"):
        if path.exists():
            return path
    return WORKSPACE / ".env"


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        raise BenchmarkError(f"Missing .env at {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    missing = [p.env_var for p in PROVIDERS if not env.get(p.env_var)]
    if missing:
        raise BenchmarkError("Missing required environment variables: " + ", ".join(missing))
    return env


def now_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", text)
    return text


def http_json(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
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
            text = (first + rest).decode("utf-8", errors="replace")
            payload = json.loads(text) if text.strip() else {}
            return payload, {"ttfb_ms": round(ttfb_ms, 2), "total_ms": round(total_ms, 2)}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code} from {url}: {redact(text[:700])}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"Network error from {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Non-JSON response from {url}: {exc}") from exc


def http_sse(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = 180.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json", **headers},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    buffer = ""
    first_answer_ms = None
    first_citation_ms = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            first = resp.read(1)
            ttfb_ms = (time.perf_counter() - started) * 1000
            buffer += first.decode("utf-8", errors="replace")
            while True:
                try:
                    chunk = resp.read(4096)
                except (TimeoutError, socket.timeout):
                    if events and extract_answer(events):
                        break
                    raise
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    event = parse_sse_event(event_text)
                    if event is None:
                        continue
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
            return events, {
                "ttfb_ms": round(ttfb_ms, 2),
                "first_answer_ms": round(first_answer_ms, 2) if first_answer_ms else None,
                "first_citation_ms": round(first_citation_ms, 2) if first_citation_ms else None,
                "total_ms": round(total_ms, 2),
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {exc.code} from {url}: {redact(text[:700])}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"Network error from {url}: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise BenchmarkError(f"Timeout reading SSE stream from {url}") from exc


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


def extract_answer(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload if len(payload.strip()) > 10 else ""
    if isinstance(payload, list):
        return "\n".join(part for part in (extract_answer(item) for item in payload) if part)
    if not isinstance(payload, dict):
        return ""

    output = payload.get("output")
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, dict):
            answer = content.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        elif isinstance(content, str) and content.strip():
            return content.strip()

    for key in ("answer", "content", "text", "response", "output_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    choices = payload.get("choices")
    if isinstance(choices, list):
        parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for obj in (choice.get("message"), choice.get("delta"), choice.get("data")):
                if isinstance(obj, dict):
                    content = obj.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        parts.extend(extract_answer(item) for item in content)
        if "".join(parts).strip():
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
                if lower in {"url", "link", "source_url", "sourceurl"} and isinstance(child, str):
                    urls.append(child)
                elif lower in {"citations", "references", "sources", "results", "web_results", "referencechunks"}:
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


def call_provider(provider: Provider, env: dict[str, str], question: Question, *, require_citations: bool = False) -> dict[str, Any]:
    api_key = env[provider.env_var]
    query = question.query
    if provider.key == "liner":
        raw, timing = call_liner_ai_search(api_key, query, pro=False)
    elif provider.key == "liner_pro":
        raw, timing = call_liner_ai_search(api_key, query, pro=True)
    elif provider.key == "perplexity":
        raw, timing = call_perplexity_sonar_pro(api_key, query)
    elif provider.key == "exa":
        raw, timing = call_exa_answer(api_key, query)
    elif provider.key == "exa_deep":
        raw, timing = call_exa_deep_search(api_key, query)
    elif provider.key == "parallel":
        raw, timing = call_parallel_chat(api_key, query)
    elif provider.key == "brave":
        raw, timing = call_brave_answers(api_key, query)
    elif provider.key == "tavily":
        raw, timing = call_tavily_answer(api_key, query)
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
        "reference_answer": question.reference_answer,
        "category": question.category,
        "false_premise": question.false_premise,
        "fact_type": question.fact_type,
        "raw": raw,
    }


def call_liner_ai_search(api_key: str, query: str, *, pro: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = "ai-search-pro" if pro else "ai-search"
    events, timing = http_sse(
        f"https://platform.liner.com/api/v1/{endpoint}",
        headers={"x-api-key": api_key},
        body={"messages": [{"role": "user", "content": query}], "mode": "general"},
    )
    answer_parts: list[str] = []
    for event in events:
        data = event.get("data") if isinstance(event, dict) else None
        if isinstance(data, dict):
            if data.get("type") == "text-delta":
                answer_parts.append(str(data.get("delta") or ""))
            choices = data.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    content = extract_answer(choice)
                    if content:
                        answer_parts.append(content)
    answer = "".join(answer_parts).strip() or extract_answer(events)
    return {"events": events, "answer": answer, "citations": extract_citations(events)}, timing


def call_perplexity_sonar_pro(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.perplexity.ai/v1/sonar",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": query}],
            "search_context_size": "low",
        },
    )


def call_exa_answer(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json("https://api.exa.ai/answer", headers={"x-api-key": api_key}, body={"query": query, "text": True})


def call_exa_deep_search(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        body={
            "query": query,
            "type": "deep",
            "numResults": 10,
            "text": True,
            "outputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "A concise answer to the query, correcting false premises when needed.",
                    }
                },
                "required": ["answer"],
            },
        },
        timeout=180.0,
    )


def call_parallel_chat(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.parallel.ai/v1beta/chat/completions",
        headers={"x-api-key": api_key},
        body={"model": "speed", "messages": [{"role": "user", "content": query}]},
    )


def call_brave_answers(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.search.brave.com/res/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        body={"model": "brave", "messages": [{"role": "user", "content": query}], "stream": False},
    )


def call_tavily_answer(api_key: str, query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return http_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "query": query,
            "search_depth": "advanced",
            "include_answer": "advanced",
            "include_raw_content": False,
            "max_results": 8,
        },
    )


def estimate_cost(provider_key: str, raw: dict[str, Any]) -> float | None:
    if provider_key == "liner":
        return 0.010
    if provider_key == "liner_pro":
        return 0.100
    if provider_key == "exa":
        return 0.005
    if provider_key == "exa_deep":
        cost = raw.get("costDollars") if isinstance(raw, dict) else None
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
            return round(float(cost["total"]), 8)
        return 0.012
    if provider_key == "parallel":
        return 0.005
    if provider_key == "tavily":
        return 0.016
    if provider_key == "perplexity":
        usage = raw.get("usage") if isinstance(raw, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if isinstance(cost, dict) and isinstance(cost.get("total_cost"), (int, float)):
            return round(float(cost["total_cost"]), 8)
        if isinstance(usage, dict):
            prompt = float(usage.get("prompt_tokens") or 0)
            completion = float(usage.get("completion_tokens") or 0)
            return round(0.006 + prompt * 3 / 1_000_000 + completion * 15 / 1_000_000, 8)
        return 0.006
    if provider_key == "brave":
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            prompt_tokens = float(usage.get("prompt_tokens") or 0)
            completion_tokens = float(usage.get("completion_tokens") or 0)
            searches = float(usage.get("searches") or usage.get("search_count") or 1)
            return round((searches * 4 / 1000) + ((prompt_tokens + completion_tokens) * 5 / 1_000_000), 8)
        return None
    return None


def load_freshqa_balanced(limit: int) -> list[Question]:
    if not FRESHQA_CSV.exists():
        raise BenchmarkError(f"Missing FreshQA CSV at {FRESHQA_CSV}")
    lines = FRESHQA_CSV.read_text(encoding="utf-8-sig").splitlines()
    header_idx = next((idx for idx, line in enumerate(lines) if line.lower().startswith("id,split,question,")), 0)
    rows = list(csv.DictReader(lines[header_idx:]))
    selected: list[dict[str, str]] = []

    def take(predicate: Any, n: int) -> None:
        for row in rows:
            if len([x for x in selected if x is row]) >= n:
                break
            if row in selected:
                continue
            if predicate(row):
                selected.append(row)
                if sum(1 for x in selected if predicate(x)) >= n:
                    break

    take(lambda r: r.get("false_premise") == "TRUE", 5)
    take(lambda r: r.get("fact_type") == "fast-changing" and r.get("false_premise") != "TRUE", 8)
    take(lambda r: r.get("fact_type") == "slow-changing" and r.get("false_premise") != "TRUE", 4)
    take(lambda r: r.get("fact_type") == "never-changing" and r.get("false_premise") != "TRUE", 3)
    for row in rows:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)
    selected = selected[:limit]
    questions = []
    for row in selected:
        query_id = f"freshqa_{int(row.get('id') or len(questions) + 1):04d}"
        questions.append(
            Question(
                benchmark="freshqa",
                query_id=query_id,
                query=row["question"].strip(),
                reference_answer=first_answer(row),
                category=("false_premise" if row.get("false_premise") == "TRUE" else row.get("fact_type", "")),
                false_premise=row.get("false_premise") == "TRUE",
                fact_type=row.get("fact_type", ""),
            )
        )
    return questions


def first_answer(row: dict[str, str]) -> str:
    for i in range(10):
        value = row.get(f"answer_{i}", "").strip()
        if value:
            return value
    return ""


def run_smoke(env: dict[str, str], run_dir: Path) -> list[dict[str, Any]]:
    smoke_question = Question("smoke", "smoke_0001", SMOKE_QUERY, "", "", False, "")
    records = []
    for provider in PROVIDERS:
        try:
            record = call_provider(provider, env, smoke_question, require_citations=False)
        except BenchmarkError as exc:
            write_blocking_report(run_dir, provider, exc, records)
            raise
        write_raw(run_dir, record)
        records.append(to_normalized_record(record))
    write_jsonl(run_dir / "normalized" / "smoke_results.jsonl", records)
    return records


def run_benchmark(run_dir: Path, env: dict[str, str], questions: list[Question], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(existing)
    completed = {(r.get("provider"), r.get("query_id")) for r in records if r.get("status") == "ok"}
    for question in questions:
        for provider in PROVIDERS:
            if (provider.key, question.query_id) in completed:
                continue
            try:
                record = call_provider(provider, env, question)
            except BenchmarkError as exc:
                write_blocking_report(run_dir, provider, exc, records)
                raise
            write_raw(run_dir, record)
            normalized = to_normalized_record(record)
            records.append(normalized)
            completed.add((provider.key, question.query_id))
            write_jsonl(run_dir / "normalized" / "results_freshqa_20_each.jsonl", records)
    return records


def write_raw(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "raw" / record["benchmark"] / record["provider"] / f"{record['query_id']}.json"
    mkdir(path.parent)
    payload = {key: record[key] for key in ("provider", "product", "benchmark", "query_id", "query", "timing", "raw")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "reference_answer": record.get("reference_answer", ""),
        "category": record.get("category", ""),
        "false_premise": record.get("false_premise", False),
        "fact_type": record.get("fact_type", ""),
        "raw_file": record.get("raw_file"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    mkdir(path.parent)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_blocking_report(run_dir: Path, provider: Provider, error: Exception, completed: list[dict[str, Any]]) -> None:
    mkdir(run_dir)
    lines = [
        "# Blocking Report",
        "",
        "Benchmark execution stopped because a configured provider failed.",
        "",
        f"- Failed provider: {provider.product} (`{provider.key}`)",
        f"- Required env var: `{provider.env_var}`",
        f"- Error summary: `{redact(str(error))}`",
        f"- Completed records before stop: {len(completed)}",
        "",
        "No fallback provider was used.",
    ]
    (run_dir / "blocking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(run_dir: Path, smoke: list[dict[str, Any]], records: list[dict[str, Any]], questions: list[Question]) -> None:
    by_provider = group_by(records, "provider")
    lines = [
        "# AI Search FreshQA 20 Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Scope: FreshQA {len(questions)}-question slice x {len(PROVIDERS)} providers.",
        "- Scoring in this report is unjudged; use the OpenAI judge report for semantic performance.",
        "- No fallback provider substitution was used.",
        "",
        "## Dataset Slice",
        "",
        "| Fact type | Questions |",
        "| --- | ---: |",
    ]
    categories: dict[str, int] = {}
    for question in questions:
        categories[question.fact_type] = categories.get(question.fact_type, 0) + 1
    for category, count in categories.items():
        lines.append(f"| {category or 'uncategorized'} | {count} |")
    false_count = sum(1 for question in questions if question.false_premise)
    lines.extend([
        "",
        "| False-premise flag | Questions |",
        "| --- | ---: |",
        f"| true | {false_count} |",
        f"| false | {len(questions) - false_count} |",
    ])
    lines.extend([
        "",
        "## Provider Setup",
        "",
        "| Provider | Product | Pricing basis |",
        "| --- | --- | --- |",
    ])
    for provider in PROVIDERS:
        lines.append(f"| {provider.key} | {provider.product} | {provider.pricing_note} |")
    lines.extend([
        "",
        "## Smoke Test Results",
        "",
        "| Provider | TTFB ms | First answer ms | Total ms | Citations | Raw file |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for record in smoke:
        timing = record["timing"]
        lines.append(
            f"| {record['provider']} | {timing.get('ttfb_ms')} | {timing.get('first_answer_ms') or 'n/a'} | "
            f"{timing.get('total_ms')} | {len(record['citations'])} | `{record['raw_file']}` |"
        )
    lines.extend([
        "",
        "## Latency",
        "",
        "| Provider | Calls | p50 TTFB | p50 first answer | p50 total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for provider, rows in by_provider.items():
        ttfb = [r["timing"].get("ttfb_ms") for r in rows if r["timing"].get("ttfb_ms") is not None]
        first = [r["timing"].get("first_answer_ms") for r in rows if r["timing"].get("first_answer_ms") is not None]
        total = [r["timing"].get("total_ms") for r in rows if r["timing"].get("total_ms") is not None]
        lines.append(f"| {provider} | {len(rows)} | {pct(ttfb, 50)} | {pct(first, 50)} | {pct(total, 50)} |")
    lines.extend([
        "",
        "## Cost",
        "",
        "| Provider | Est. total USD | Est. / 1K calls | Pricing basis completeness |",
        "| --- | ---: | ---: | --- |",
    ])
    for provider, rows in by_provider.items():
        costs = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
        if len(costs) == len(rows):
            total = sum(costs)
            lines.append(f"| {provider} | ${total:.6f} | ${total / len(rows) * 1000:.2f} | complete in harness |")
        else:
            lines.append(f"| {provider} | n/a | n/a | partial; refresh pricing before publication |")
    lines.extend([
        "",
        "## Citation Extraction",
        "",
        "| Provider | Calls with citations | Avg citations |",
        "| --- | ---: | ---: |",
    ])
    for provider, rows in by_provider.items():
        counts = [len(r["citations"]) for r in rows]
        lines.append(f"| {provider} | {sum(1 for c in counts if c > 0)}/{len(rows)} | {statistics.fmean(counts):.2f} |")
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- Raw responses: `raw/freshqa/<provider>/<query_id>.json`",
        "- Normalized records: `normalized/results_freshqa_20_each.jsonl`",
        "- Smoke records: `normalized/smoke_results.jsonl`",
    ])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_by(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return grouped


def pct(values: list[float], percentile: int) -> str:
    if not values:
        return "n/a"
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    return f"{sorted_values[idx]:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Search API FreshQA benchmark.")
    parser.add_argument("--freshqa-limit", type=int, default=20)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reuse-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", default=now_run_id())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "results" / "freshqa" / f"freshqa{args.freshqa_limit}_{args.run_id}"
    mkdir(run_dir / "normalized")
    try:
        env = load_env(env_path())
        smoke = read_jsonl(run_dir / "normalized" / "smoke_results.jsonl") if args.reuse_smoke else run_smoke(env, run_dir)
        records: list[dict[str, Any]] = []
        questions = load_freshqa_balanced(args.freshqa_limit)
        if not args.smoke_only:
            existing = read_jsonl(run_dir / "normalized" / "results_freshqa_20_each.jsonl") if args.resume else []
            records = run_benchmark(run_dir, env, questions, existing)
        write_report(run_dir, smoke, records, questions)
    except BenchmarkError as exc:
        print(f"BLOCKED: {redact(str(exc))}", file=sys.stderr)
        print(f"See {run_dir / 'blocking_report.md'}", file=sys.stderr)
        return 2
    print(f"Completed run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
