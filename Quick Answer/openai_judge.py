#!/usr/bin/env python3
"""OpenAI semantic judge for benchmark records."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESPONSES_URL = "https://api.openai.com/v1/responses"
PROVIDERS = ["liner", "perplexity", "exa", "parallel", "brave", "tavily"]


class OpenAIJudgeError(Exception):
    pass


def env_path() -> Path:
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if path.exists():
            return path
    return ROOT / ".env"


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    if not env.get("OPENAI_API_KEY"):
        raise OpenAIJudgeError("Missing OPENAI_API_KEY")
    return env


def compact_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": row["provider"],
        "query_id": row["query_id"],
        "question": row["query"],
        "reference_answer": row.get("reference_answer", ""),
        "provider_answer": row.get("answer", ""),
        "citations": row.get("citations", [])[:10],
    }


def response_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "provider": {"type": "string"},
            "query_id": {"type": "string"},
            "correctness": {"type": "string", "enum": ["correct", "partially_correct", "incorrect", "ungradable"]},
            "citation_supported": {"type": "boolean"},
            "false_premise_handled": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": ["provider", "query_id", "correctness", "citation_supported", "false_premise_handled", "rationale"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
    }


def judge_batch(api_key: str, model: str, batch: list[dict[str, Any]], batch_index: int) -> list[dict[str, Any]]:
    instructions = (
        "You are grading quick-answer API outputs. Judge semantic correctness against the reference answer. "
        "For SimpleQA, entity aliases and harmless extra context should still be correct. "
        "Mark citation_supported true only when at least one listed citation plausibly supports the answer. "
        "Return concise rationales under 25 words."
    )
    prompt = {
        "rubric": {
            "correct": "Semantically equivalent to the reference answer or directly satisfies it.",
            "partially_correct": "Contains a meaningful correct element but misses or distorts an important condition.",
            "incorrect": "Contradicts the reference or gives the wrong entity/fact.",
            "ungradable": "Cannot be graded from the provided data.",
            "score_mapping": "correct+citation_supported=1.0, correct without support=0.7, partially_correct=0.5, incorrect/ungradable=0.0",
        },
        "items": batch,
    }
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "benchmark_judgments",
                "strict": True,
                "schema": response_schema(),
            }
        },
    }
    req = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise OpenAIJudgeError(f"OpenAI judge HTTP {exc.code} on batch {batch_index}: {text[:700]}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIJudgeError(f"OpenAI judge network error on batch {batch_index}: {exc.reason}") from exc
    text = extract_output_text(payload)
    parsed = json.loads(text)
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != len(batch):
        raise OpenAIJudgeError(f"OpenAI judge returned invalid result count on batch {batch_index}")
    return [normalize_judgment(item, batch[i]) for i, item in enumerate(results)]


def extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    parts.append(content.get("text", ""))
    text = "".join(parts).strip()
    if not text:
        raise OpenAIJudgeError(f"OpenAI response had no output text: {json.dumps(payload)[:500]}")
    return text


def normalize_judgment(item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    correctness = str(item.get("correctness", "ungradable")).lower()
    citation_supported = bool(item.get("citation_supported", False))
    if correctness == "correct" and citation_supported:
        final_score = 1.0
    elif correctness == "correct":
        final_score = 0.7
    elif correctness == "partially_correct":
        final_score = 0.5
    else:
        final_score = 0.0
    return {
        "provider": source["provider"],
        "query_id": source["query_id"],
        "correctness": correctness,
        "final_score": final_score,
        "citation_supported": citation_supported,
        "false_premise_handled": bool(item.get("false_premise_handled", False)),
        "rationale": str(item.get("rationale", ""))[:240],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def render_report(run_dir: Path, judgments: list[dict[str, Any]], records: list[dict[str, Any]], model: str, prefix: str) -> None:
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_by_key = {(row["provider"], row["query_id"]): row for row in records}
    for row in judgments:
        by_provider[row["provider"]].append(row)
    lines = [
        f"# {prefix} OpenAI-Judged Performance Report",
        "",
        f"- Judge model: `{model}`",
        f"- Records: {len(judgments)}",
        "- Score: correct+citation-supported=100%, correct without citation support=70%, partial=50%, incorrect/ungradable=0%.",
        "",
        "## Performance",
        "",
        "| Provider | Avg score | Correct | Partial | Incorrect | Citation-supported |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider in PROVIDERS:
        rows = by_provider[provider]
        if not rows:
            continue
        counts = Counter(row["correctness"] for row in rows)
        avg_score = statistics.fmean(row["final_score"] for row in rows) * 100
        supported = sum(1 for row in rows if row["citation_supported"])
        lines.append(
            f"| {provider} | {avg_score:.1f}% | {counts['correct']} | {counts['partially_correct']} | "
            f"{counts['incorrect']} | {supported}/{len(rows)} |"
        )
    lines.extend(["", "## Price", "", "| Provider | Avg score | Est. / 1K calls | Score points per $1 |", "| --- | ---: | ---: | ---: |"])
    for provider in PROVIDERS:
        rows = by_provider[provider]
        if not rows:
            continue
        source_rows = [record_by_key[(row["provider"], row["query_id"])] for row in rows]
        avg_score = statistics.fmean(row["final_score"] for row in rows) * 100
        total_cost = sum(row["estimated_cost_usd"] for row in source_rows)
        per_1k = total_cost / len(source_rows) * 1000
        score_per_dollar = avg_score / (per_1k / 1000) if per_1k else 0
        lines.append(f"| {provider} | {avg_score:.1f}% | ${per_1k:.2f} | {score_per_dollar:.0f} |")
    lines.extend(["", "## Latency", "", "| Provider | Avg score | p50 TTFB | p50 total latency |", "| --- | ---: | ---: | ---: |"])
    for provider in PROVIDERS:
        rows = by_provider[provider]
        if not rows:
            continue
        source_rows = [record_by_key[(row["provider"], row["query_id"])] for row in rows]
        avg_score = statistics.fmean(row["final_score"] for row in rows) * 100
        ttfb = statistics.median(row["timing"]["ttfb_ms"] for row in source_rows) / 1000
        total = statistics.median(row["timing"]["total_ms"] for row in source_rows) / 1000
        lines.append(f"| {provider} | {avg_score:.1f}% | {ttfb:.3f}s | {total:.3f}s |")
    (run_dir / f"{prefix.lower()}_openai_judge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--prefix", default="benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    try:
        env = load_env(env_path())
        records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
        compact = [compact_record(row) for row in records]
        output = Path(args.output)
        judgments = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
        for start in range(len(judgments), len(compact), args.batch_size):
            batch = compact[start : start + args.batch_size]
            judgments.extend(judge_batch(env["OPENAI_API_KEY"], args.model, batch, start // args.batch_size + 1))
            write_jsonl(output, judgments)
            time.sleep(0.5)
        render_report(run_dir, judgments, records, args.model, args.prefix)
    except Exception as exc:
        print(f"OPENAI JUDGE BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    print(f"Wrote {run_dir / f'{args.prefix.lower()}_openai_judge_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
