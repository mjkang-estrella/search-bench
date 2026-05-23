#!/usr/bin/env python3
"""Raw Search API benchmark harness.

Runs small evidence-retrieval slices against raw search results only: no
provider-generated answers, no provider LLM summaries.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import os
import random
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
DATA_DIR = ROOT.parent / "Quick Answer" / "data"
SEARCH_DATA_DIR = ROOT / "data"
FRAMES_TSV = "https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv"
SEALQA_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=vtllms%2Fsealqa&config=seal_0&split=test&offset=0&length={limit}"
)
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
USER_AGENT = "liner-search-benchmark/0.1"
SMOKE_QUERY = "Who received the IEEE Frank Rosenblatt Award in 2010?"


class BenchmarkError(Exception):
    pass


@dataclass(frozen=True)
class Question:
    benchmark: str
    query_id: str
    query: str
    reference_answer: str
    gold_urls: list[str]
    category: str
    gold_domain_urls: list[str] | None = None


@dataclass(frozen=True)
class Provider:
    key: str
    product: str
    env_var: str
    pricing_note: str


PROVIDERS = [
    Provider("liner", "Liner Web Search", "LINER_API_KEY", "$1.00 / 1K requests"),
    Provider("perplexity", "Perplexity Search API", "PERPLEXITY_API_KEY", "$5.00 / 1K requests"),
    Provider("exa", "Exa Search", "EXA_API_KEY", "$7.00 / 1K requests up to 10 results"),
    Provider("parallel", "Parallel Search", "PARALLEL_API_KEY", "$5.00 / 1K requests for 10 results"),
    Provider("brave", "Brave Search", "BRAVE_SEARCH_API_KEY", "$5.00 / 1K requests"),
    Provider("tavily", "Tavily Search Basic", "TAVILY_API_KEY", "$0.008 / credit; basic search = 1 credit"),
]


def env_path() -> Path:
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if path.exists():
            return path
    return ROOT.parent / ".env"


def load_env(path: Path, providers: list[Provider]) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        raise BenchmarkError(f"Missing .env at {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    missing = [provider.env_var for provider in providers if not env.get(provider.env_var)]
    if missing:
        raise BenchmarkError("Missing required environment variables: " + ", ".join(missing))
    return env


def now_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def redact(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    text = re.sub(r"pplx-[A-Za-z0-9_-]+", "pplx-<redacted>", text)
    text = re.sub(r"tvly-[A-Za-z0-9_-]+", "tvly-<redacted>", text)
    text = re.sub(r"BSA[0-9A-Za-z_-]+", "BSA<redacted>", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer <redacted>", text)
    return text


def request_json(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if params:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{query}"
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
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


def download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_questions(benchmark: str, limit: int | None) -> list[Question]:
    if benchmark == "simpleqa":
        return load_simpleqa_questions(limit)
    if benchmark == "frames":
        return load_frames_questions(limit)
    if benchmark == "sealqa":
        return load_sealqa_questions(limit)
    if benchmark == "webwalkerqa":
        return load_webwalkerqa_questions(limit)
    raise BenchmarkError(f"Unknown benchmark {benchmark}")


def load_simpleqa_questions(limit: int | None) -> list[Question]:
    path = DATA_DIR / "simpleqa.csv"
    if not path.exists():
        raise BenchmarkError(f"Missing SimpleQA data at {path}")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    questions: list[Question] = []
    for idx, row in enumerate(rows):
        query = row.get("problem", "").strip()
        answer = row.get("answer", "").strip()
        metadata = parse_metadata(row.get("metadata", ""))
        if not query:
            continue
        questions.append(
            Question(
                benchmark="search_evals_simpleqa",
                query_id=f"simpleqa_{idx + 1:04d}",
                query=query,
                reference_answer=answer,
                gold_urls=[str(url) for url in metadata.get("urls", []) if url],
                category=str(metadata.get("topic", "")),
            )
        )
        if limit is not None and len(questions) >= limit:
            break
    if not questions:
        raise BenchmarkError("No benchmark questions loaded")
    return questions


def load_frames_questions(limit: int | None) -> list[Question]:
    mkdir(SEARCH_DATA_DIR)
    path = SEARCH_DATA_DIR / "frames_test.tsv"
    if not path.exists():
        path.write_text(download_text(FRAMES_TSV), encoding="utf-8")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))
    questions: list[Question] = []
    for idx, row in enumerate(rows):
        query = row.get("Prompt", "").strip()
        answer = row.get("Answer", "").strip()
        gold_urls = frames_gold_urls(row)
        if not query:
            continue
        questions.append(
            Question(
                benchmark="frames",
                query_id=f"frames_{idx + 1:04d}",
                query=query,
                reference_answer=answer,
                gold_urls=gold_urls,
                category=row.get("reasoning_types", "").strip(),
            )
        )
        if limit is not None and len(questions) >= limit:
            break
    if not questions:
        raise BenchmarkError("No FRAMES questions loaded")
    return questions


def frames_gold_urls(row: dict[str, str]) -> list[str]:
    urls: list[str] = []
    for key, value in row.items():
        if key.startswith("wikipedia_link_") and value.strip():
            urls.append(value.strip())
    metadata_urls = parse_metadata(row.get("wiki_links", "")).get("urls", [])
    for url in metadata_urls:
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())
    if row.get("wiki_links", "").strip().startswith("["):
        try:
            parsed = ast.literal_eval(row["wiki_links"])
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            urls.extend(str(url).strip() for url in parsed if str(url).strip())
    return list(dict.fromkeys(urls))


def load_sealqa_questions(limit: int | None) -> list[Question]:
    mkdir(SEARCH_DATA_DIR)
    fetch_limit = limit or 1000
    path = SEARCH_DATA_DIR / f"sealqa_seal_0_{fetch_limit}.json"
    if not path.exists():
        path.write_text(json.dumps({"rows": fetch_hf_rows("vtllms/sealqa", "seal_0", "test", fetch_limit)}, ensure_ascii=False), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    questions: list[Question] = []
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            continue
        row = item["row"]
        query = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        urls = [str(url).strip() for url in row.get("urls", []) if str(url).strip()]
        topic = str(row.get("topic", "")).strip()
        freshness = str(row.get("freshness", "")).strip()
        question_types = row.get("question_types", [])
        if isinstance(question_types, list):
            type_text = ", ".join(str(value) for value in question_types)
        else:
            type_text = str(question_types)
        if not query:
            continue
        questions.append(
            Question(
                benchmark="sealqa",
                query_id=f"sealqa_{int(item.get('row_idx', len(questions))) + 1:04d}",
                query=query,
                reference_answer=answer,
                gold_urls=urls,
                category=" | ".join(part for part in [topic, freshness, type_text] if part),
            )
        )
        if limit is not None and len(questions) >= limit:
            break
    if not questions:
        raise BenchmarkError("No SealQA questions loaded")
    return questions


def load_webwalkerqa_questions(limit: int | None) -> list[Question]:
    mkdir(SEARCH_DATA_DIR)
    fetch_limit = limit or 1000
    path = SEARCH_DATA_DIR / f"webwalkerqa_en_{fetch_limit}.json"
    if not path.exists():
        rows = fetch_hf_rows("callanwu/WebWalkerQA", "default", "main", fetch_limit, lang_filter="en")
        path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for item in payload.get("rows", []):
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            continue
        row = item["row"]
        info = row.get("info") if isinstance(row.get("info"), dict) else {}
        query = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        source_urls = [str(url).strip() for url in info.get("source_website", []) if str(url).strip()]
        root_url = str(row.get("root_url", "")).strip()
        if not query or not source_urls:
            continue
        questions.append(
            Question(
                benchmark="webwalkerqa",
                query_id=f"webwalkerqa_{int(item.get('row_idx', len(questions))) + 1:04d}",
                query=query,
                reference_answer=answer,
                gold_urls=source_urls,
                gold_domain_urls=[root_url] if root_url else [],
                category=" | ".join(
                    part
                    for part in [
                        str(info.get("domain", "")).strip(),
                        str(info.get("type", "")).strip(),
                        str(info.get("difficulty_level", "")).strip(),
                    ]
                    if part
                ),
            )
        )
        if limit is not None and len(questions) >= limit:
            break
    if not questions:
        raise BenchmarkError("No WebWalkerQA questions loaded")
    return questions


def fetch_hf_rows(dataset: str, config: str, split: str, desired_rows: int, *, lang_filter: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    encoded_dataset = urllib.parse.quote(dataset, safe="")
    while len(rows) < desired_rows:
        url = HF_ROWS_URL.format(dataset=encoded_dataset, config=config, split=split, offset=offset, length=page_size)
        payload = json.loads(download_text(url))
        page = payload.get("rows", [])
        if not page:
            break
        for item in page:
            if lang_filter:
                row = item.get("row", {}) if isinstance(item, dict) else {}
                info = row.get("info", {}) if isinstance(row.get("info"), dict) else {}
                if str(info.get("lang", "")).lower() != lang_filter.lower():
                    continue
            rows.append(item)
            if len(rows) >= desired_rows:
                break
        offset += len(page)
        total = int(payload.get("num_rows_total") or 0)
        if total and offset >= total:
            break
    return rows


def parse_metadata(text: str) -> dict[str, Any]:
    try:
        value = ast.literal_eval(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def call_provider(provider: Provider, env: dict[str, str], question: Question, max_results: int) -> dict[str, Any]:
    api_key = env[provider.env_var]
    if provider.key == "liner":
        raw, timing = call_liner(api_key, question.query, max_results)
    elif provider.key == "perplexity":
        raw, timing = call_perplexity(api_key, question.query, max_results)
    elif provider.key == "exa":
        raw, timing = call_exa(api_key, question.query, max_results)
    elif provider.key == "parallel":
        raw, timing = call_parallel(api_key, question.query, max_results)
    elif provider.key == "brave":
        raw, timing = call_brave(api_key, question.query, max_results)
    elif provider.key == "tavily":
        raw, timing = call_tavily(api_key, question.query, max_results)
    else:
        raise BenchmarkError(f"Unknown provider {provider.key}")
    results = normalize_results(provider.key, raw)
    return {
        "provider": provider.key,
        "product": provider.product,
        "benchmark": question.benchmark,
        "query_id": question.query_id,
        "query": question.query,
        "reference_answer": question.reference_answer,
        "gold_urls": question.gold_urls,
        "gold_domain_urls": question.gold_domain_urls or [],
        "category": question.category,
        "results": results[:max_results],
        "timing": timing,
        "estimated_cost_usd": estimate_cost(provider.key, raw),
        "pricing_basis": provider.pricing_note,
        "raw": raw,
    }


def call_liner(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://platform.liner.com/api/v1/search/web",
        headers={"x-api-key": api_key},
        body={"query": query, "max_results": max_results},
    )


def call_perplexity(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://api.perplexity.ai/search",
        headers={"Authorization": f"Bearer {api_key}"},
        body={"query": query, "max_results": max_results, "max_tokens_per_page": 1024},
    )


def call_exa(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        body={"query": query, "numResults": max_results, "contents": {"highlights": True}},
    )


def call_parallel(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://api.parallel.ai/v1beta/search",
        headers={"x-api-key": api_key},
        body={
            "mode": "one-shot",
            "objective": f"Find web pages that answer this factual question: {query}",
            "search_queries": [query],
            "max_results": max_results,
            "excerpts": {"max_chars_per_result": 1200, "max_chars_total": 12000},
        },
    )


def call_brave(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://api.search.brave.com/res/v1/web/search",
        method="GET",
        headers={"X-Subscription-Token": api_key},
        params={"q": limit_words(query, 50), "count": max_results, "text_decorations": "false", "result_filter": "web"},
    )


def call_tavily(api_key: str, query: str, max_results: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return request_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "max_results": max_results,
        },
    )


def normalize_results(provider_key: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    if provider_key == "brave":
        items = ((raw.get("web") or {}).get("results") or []) if isinstance(raw.get("web"), dict) else []
    else:
        items = raw.get("results") if isinstance(raw, dict) else []
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for rank, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        excerpts = item.get("excerpts")
        if isinstance(excerpts, list):
            excerpt_text = "\n".join(str(x) for x in excerpts if x)
        else:
            excerpt_text = ""
        highlights = item.get("highlights")
        if isinstance(highlights, list):
            highlight_text = "\n".join(str(x) for x in highlights if x)
        else:
            highlight_text = ""
        snippet = first_text(
            item,
            ["snippet", "description", "text", "content", "summary", "extra_snippets"],
        )
        if not snippet:
            snippet = excerpt_text or highlight_text
        normalized.append(
            {
                "rank": rank,
                "title": first_text(item, ["title", "name"]),
                "url": first_text(item, ["url", "link"]),
                "snippet": compact_text(snippet, 1600),
                "date": first_text(item, ["date", "publishedDate", "publish_date", "last_updated", "age"]),
                "hostname": hostname(first_text(item, ["url", "link"])),
            }
        )
    return [row for row in normalized if row["url"] or row["title"] or row["snippet"]]


def first_text(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = "\n".join(str(x) for x in value if x)
            if text.strip():
                return text.strip()
    return ""


def compact_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit]


def limit_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def hostname(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def canonical_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower()


def gold_hit(results: list[dict[str, Any]], gold_urls: list[str], gold_domain_urls: list[str] | None = None) -> dict[str, Any]:
    gold = [canonical_url(url) for url in gold_urls]
    for result in results:
        result_url = canonical_url(result.get("url", ""))
        for target in gold:
            if target and (result_url == target or result_url.startswith(target) or target.startswith(result_url)):
                return {"hit": True, "rank": result["rank"], "url": result.get("url", "")}
    gold_hosts = {hostname(url) for url in [*gold_urls, *(gold_domain_urls or [])] if hostname(url)}
    for result in results:
        if result.get("hostname") in gold_hosts:
            return {"hit": True, "rank": result["rank"], "url": result.get("url", ""), "domain_only": True}
    return {"hit": False, "rank": None, "url": None}


def estimate_cost(provider_key: str, raw: dict[str, Any]) -> float:
    if provider_key == "liner":
        return 0.001
    if provider_key == "perplexity":
        return 0.005
    if provider_key == "exa":
        cost = raw.get("costDollars") if isinstance(raw, dict) else None
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
            return float(cost["total"])
        return 0.007
    if provider_key == "parallel":
        return 0.005
    if provider_key == "brave":
        return 0.005
    if provider_key == "tavily":
        return 0.008
    return 0.0


def write_raw(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "raw" / record["benchmark"] / record["provider"] / f"{record['query_id']}.json"
    mkdir(path.parent)
    payload = {
        "provider": record["provider"],
        "product": record["product"],
        "benchmark": record["benchmark"],
        "query_id": record["query_id"],
        "query": record["query"],
        "reference_answer": record["reference_answer"],
        "gold_urls": record["gold_urls"],
        "gold_domain_urls": record.get("gold_domain_urls", []),
        "timing": record["timing"],
        "raw": record["raw"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    record["raw_file"] = str(path.relative_to(run_dir))


def to_normalized_record(record: dict[str, Any]) -> dict[str, Any]:
    hit = gold_hit(record["results"], record["gold_urls"], record.get("gold_domain_urls", []))
    return {
        "provider": record["provider"],
        "product": record["product"],
        "benchmark": record["benchmark"],
        "query_id": record["query_id"],
        "query": record["query"],
        "reference_answer": record["reference_answer"],
        "gold_urls": record["gold_urls"],
        "gold_domain_urls": record.get("gold_domain_urls", []),
        "category": record["category"],
        "results": record["results"],
        "gold_hit": hit,
        "status": "ok",
        "timing": record["timing"],
        "estimated_cost_usd": record["estimated_cost_usd"],
        "pricing_basis": record["pricing_basis"],
        "raw_file": record.get("raw_file"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    mkdir(path.parent)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_smoke(env: dict[str, str], run_dir: Path, max_results: int, providers: list[Provider]) -> list[dict[str, Any]]:
    question = Question("smoke", "smoke_0001", SMOKE_QUERY, "Michio Sugeno", [], "smoke")
    rows = []
    for provider in providers:
        record = call_provider(provider, env, question, max_results)
        write_raw(run_dir, record)
        rows.append(to_normalized_record(record))
    write_jsonl(run_dir / "normalized" / "smoke_results.jsonl", rows)
    return rows


def run_benchmark(env: dict[str, str], run_dir: Path, questions: list[Question], max_results: int, resume: bool, providers: list[Provider]) -> list[dict[str, Any]]:
    records = read_jsonl(run_dir / "normalized" / "results.jsonl") if resume else []
    completed = {(row["provider"], row["query_id"]) for row in records if row.get("status") == "ok"}
    for question in questions:
        for provider in providers:
            if (provider.key, question.query_id) in completed:
                continue
            record = call_provider(provider, env, question, max_results)
            write_raw(run_dir, record)
            records.append(to_normalized_record(record))
            completed.add((provider.key, question.query_id))
            write_jsonl(run_dir / "normalized" / "results.jsonl", records)
            time.sleep(0.1)
    return records


def write_blocking_report(run_dir: Path, exc: Exception, records: list[dict[str, Any]]) -> None:
    mkdir(run_dir)
    lines = [
        "# Blocking Report",
        "",
        "Search API benchmark stopped before completion.",
        "",
        f"- Error: `{redact(str(exc))}`",
        f"- Completed benchmark records: {len(records)}",
        "",
        "Rerun with `--resume` after fixing the provider access, quota, or adapter contract.",
    ]
    (run_dir / "blocking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(run_dir: Path, smoke: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_provider = group_by(records, "provider")
    providers = selected_providers(args.providers)
    lines = [
        "# Search API Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Benchmark: `{benchmark_label(args.benchmark)}` evidence retrieval slice.",
        f"- Scope: {question_count_arg(args)} questions x {len(providers)} providers = {len(records)} benchmark calls.",
        f"- Result depth: top {args.max_results}.",
        "- Provider-generated answers and summaries were disabled where the API exposes that control.",
        "- Search quality scoring here is pre-judge only: gold URL/domain hit and result coverage. Use `openai_judge.py` for answerability scoring.",
        "",
        "## Provider Setup",
        "",
        "| Provider | Product | Pricing basis |",
        "| --- | --- | --- |",
    ]
    for provider in providers:
        lines.append(f"| {provider.key} | {provider.product} | {provider.pricing_note} |")
    lines.extend([
        "",
        "## Smoke Test Results",
        "",
        "| Provider | Results | TTFB ms | Total ms | Raw file |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for row in smoke:
        lines.append(
            f"| {row['provider']} | {len(row['results'])} | {row['timing'].get('ttfb_ms')} | "
            f"{row['timing'].get('total_ms')} | `{row['raw_file']}` |"
        )
    lines.extend([
        "",
        "## Search Quality",
        "",
        "| Provider | Calls | Avg results | Gold URL/domain hit@10 | Avg gold-hit rank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for provider in provider_order():
        rows = by_provider.get(provider, [])
        if not rows:
            continue
        hits = [row["gold_hit"] for row in rows if row.get("gold_hit", {}).get("hit")]
        ranks = [hit["rank"] for hit in hits if hit.get("rank")]
        avg_results = statistics.fmean(len(row["results"]) for row in rows)
        avg_rank = statistics.fmean(ranks) if ranks else 0
        lines.append(f"| {provider} | {len(rows)} | {avg_results:.1f} | {len(hits)}/{len(rows)} | {avg_rank:.2f} |")
    lines.extend(render_latency_cost(by_provider))
    lines.extend(["", "## Raw Output Index", ""])
    for row in smoke + records:
        lines.append(f"- `{row['provider']}` `{row['benchmark']}` `{row['query_id']}`: `{row['raw_file']}`")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_latency_cost(by_provider: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "",
        "## Latency",
        "",
        "| Provider | p50 total ms | p90 total ms | p95 total ms | p50 TTFB ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for provider in provider_order():
        rows = by_provider.get(provider, [])
        if not rows:
            continue
        totals = [row["timing"]["total_ms"] for row in rows]
        ttfbs = [row["timing"]["ttfb_ms"] for row in rows]
        lines.append(f"| {provider} | {pct(totals, 50)} | {pct(totals, 90)} | {pct(totals, 95)} | {pct(ttfbs, 50)} |")
    lines.extend([
        "",
        "## Cost",
        "",
        "| Provider | Estimated total USD | Estimated cost / 1K calls | Pricing basis |",
        "| --- | ---: | ---: | --- |",
    ])
    provider_meta = {p.key: p for p in PROVIDERS}
    for provider in provider_order():
        rows = by_provider.get(provider, [])
        if not rows:
            continue
        total = sum(float(row["estimated_cost_usd"]) for row in rows)
        per_1k = total / len(rows) * 1000
        lines.append(f"| {provider} | ${total:.4f} | ${per_1k:.2f} | {provider_meta[provider].pricing_note} |")
    return lines


def provider_order() -> list[str]:
    return [provider.key for provider in PROVIDERS]


def selected_providers(provider_arg: str | None) -> list[Provider]:
    if not provider_arg:
        return PROVIDERS
    requested = [item.strip() for item in provider_arg.split(",") if item.strip()]
    known = {provider.key: provider for provider in PROVIDERS}
    unknown = [key for key in requested if key not in known]
    if unknown:
        raise BenchmarkError("Unknown providers: " + ", ".join(unknown))
    return [known[key] for key in requested]


def question_count_arg(args: argparse.Namespace) -> int:
    return args.sample_size if args.sample_size is not None else args.limit


def benchmark_label(benchmark: str) -> str:
    if benchmark == "simpleqa":
        return "search_evals-style SimpleQA"
    if benchmark == "frames":
        return "FRAMES"
    if benchmark == "sealqa":
        return "SealQA Seal-0"
    if benchmark == "webwalkerqa":
        return "WebWalkerQA"
    return benchmark


def sample_questions(questions: list[Question], sample_size: int | None, seed: int | None) -> list[Question]:
    if sample_size is None:
        return questions
    if len(questions) < sample_size:
        raise BenchmarkError(f"Cannot sample {sample_size} questions; only {len(questions)} available")
    rng = random.Random(seed)
    sampled = rng.sample(questions, sample_size)
    return sorted(sampled, key=lambda question: question.query_id)


def write_sample_manifest(run_dir: Path, benchmark: str, seed: int | None, source_count: int, sampled: list[Question]) -> None:
    payload = {
        "benchmark": benchmark,
        "seed": seed,
        "source_count": source_count,
        "sample_size": len(sampled),
        "query_ids": [question.query_id for question in sampled],
        "questions": [
            {
                "query_id": question.query_id,
                "query": question.query,
                "reference_answer": question.reference_answer,
                "gold_urls": question.gold_urls,
                "gold_domain_urls": question.gold_domain_urls or [],
                "category": question.category,
            }
            for question in sampled
        ],
    }
    (run_dir / "sample_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Run raw Search API benchmarks.")
    parser.add_argument("--benchmark", choices=["simpleqa", "frames", "sealqa", "webwalkerqa"], default="simpleqa")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--providers", default=None)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--run-id", default=now_run_id())
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reuse-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suites = {
        "simpleqa": "search_evals_simpleqa20",
        "frames": "frames20",
        "sealqa": "sealqa20",
        "webwalkerqa": "webwalkerqa20",
    }
    if args.sample_size == 100:
        suites = {
            "simpleqa": "search_evals_simpleqa100",
            "frames": "frames100",
            "sealqa": "sealqa100",
            "webwalkerqa": "webwalkerqa100",
        }
    suite = suites[args.benchmark]
    run_dir = ROOT / "results" / suite / args.run_id
    mkdir(run_dir / "normalized")
    records: list[dict[str, Any]] = []
    try:
        providers = selected_providers(args.providers)
        env = load_env(env_path(), providers)
        smoke = read_jsonl(run_dir / "normalized" / "smoke_results.jsonl") if args.reuse_smoke else run_smoke(env, run_dir, args.max_results, providers)
        if not args.smoke_only:
            source_questions = load_questions(args.benchmark, None if args.sample_size is not None else args.limit)
            questions = sample_questions(source_questions, args.sample_size, args.seed)
            write_sample_manifest(run_dir, args.benchmark, args.seed if args.sample_size is not None else None, len(source_questions), questions)
            records = run_benchmark(env, run_dir, questions, args.max_results, args.resume, providers)
        write_report(run_dir, smoke, records, args)
    except BenchmarkError as exc:
        records = read_jsonl(run_dir / "normalized" / "results.jsonl")
        write_blocking_report(run_dir, exc, records)
        print(f"BLOCKED: {redact(str(exc))}", file=sys.stderr)
        print(f"See {run_dir / 'blocking_report.md'}", file=sys.stderr)
        return 2
    print(f"Completed run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
