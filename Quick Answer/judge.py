#!/usr/bin/env python3
"""Semantic judge for the 40-each FreshQA benchmark slice.

Uses Gemini API JSON output when GEMINI_API_KEY is available. The judge is
separate from benchmark.py so benchmark provider calls remain untouched.
"""

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
DEFAULT_RUN_ID = "20260521T225612Z"
PROVIDERS = ["liner", "perplexity", "exa", "parallel", "brave", "tavily"]
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
USER_AGENT = "liner-quick-answer-benchmark-judge/0.1"


class JudgeError(Exception):
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
    if not env.get("GEMINI_API_KEY"):
        raise JudgeError("Missing GEMINI_API_KEY in shell environment or .env")
    return env


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "normalized" / "results_40_each.jsonl"
    if not path.exists():
        raise JudgeError(f"Missing balanced input file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": row["provider"],
        "query_id": row["query_id"],
        "question": row["query"],
        "reference_answer": row.get("reference_answer", ""),
        "provider_answer": row.get("answer", ""),
        "citations": row.get("citations", [])[:12],
    }


def judge_batch(api_key: str, model: str, batch: list[dict[str, Any]], batch_index: int) -> list[dict[str, Any]]:
    prompt = {
        "task": "Grade quick-answer API outputs against FreshQA references.",
        "rubric": {
            "correctness": {
                "correct": "The answer is semantically equivalent to the reference or directly satisfies it.",
                "partially_correct": "The answer contains a meaningful correct element but misses or distorts an important condition.",
                "incorrect": "The answer contradicts the reference, accepts a false premise, or gives the wrong entity/fact.",
                "ungradable": "The answer cannot be graded from the question/reference/answer alone.",
            },
            "final_score": {
                "1.0": "correct and citation_supported is true",
                "0.7": "correct but citations are missing, irrelevant, or insufficient to verify from the citation list",
                "0.5": "partially_correct",
                "0.0": "incorrect or ungradable",
            },
            "false_premise_handled": "true when the question has a false premise and the answer explicitly corrects it; otherwise false.",
            "citation_supported": "true only if at least one provided citation URL/title appears plausibly able to support the answer; false if no citations or citations are irrelevant.",
        },
        "output_requirements": [
            "Return only JSON.",
            "Return one object per input item.",
            "Preserve provider and query_id exactly.",
            "Keep rationale under 25 words.",
        ],
        "items": batch,
    }
    body = {
        "contents": [{"parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }
    url = GEMINI_ENDPOINT.format(model=model)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise JudgeError(f"Gemini judge HTTP {exc.code} on batch {batch_index}: {text[:500]}") from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"Gemini judge network error on batch {batch_index}: {exc.reason}") from exc

    text = extract_text(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Gemini judge returned non-JSON on batch {batch_index}: {text[:500]}") from exc
    if isinstance(parsed, dict) and "results" in parsed:
        parsed = parsed["results"]
    if not isinstance(parsed, list):
        raise JudgeError(f"Gemini judge returned unexpected JSON shape on batch {batch_index}")
    if len(parsed) != len(batch):
        raise JudgeError(f"Gemini judge returned {len(parsed)} items for {len(batch)} inputs on batch {batch_index}")
    return [normalize_judgment(item, batch[i]) for i, item in enumerate(parsed)]


def extract_text(payload: dict[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError(f"Gemini judge response missing content: {payload}") from exc
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise JudgeError("Gemini judge response was empty")
    return text.strip()


def normalize_judgment(item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    correctness = str(item.get("correctness", "ungradable")).lower()
    if correctness not in {"correct", "partially_correct", "incorrect", "ungradable"}:
        correctness = "ungradable"
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
        "false_premise_handled": bool(item.get("false_premise_handled", False)),
        "citation_supported": citation_supported,
        "rationale": str(item.get("rationale", ""))[:240],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def render_report(run_dir: Path, judged: list[dict[str, Any]], records: list[dict[str, Any]], model: str) -> None:
    record_by_key = {(r["provider"], r["query_id"]): r for r in records}
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged:
        by_provider[row["provider"]].append(row)

    lines = [
        "# Judged Performance Report: 40 FreshQA Questions per Provider",
        "",
        "## Executive Summary",
        "",
        f"- Run ID: `{run_dir.name}`",
        f"- Judge model: `{model}`",
        "- Scope: FreshQA first 40 common questions, 40 calls per provider.",
        "- Performance score: 1.0 for correct+citation-supported, 0.7 for correct but weak/missing citation, 0.5 for partially correct, 0.0 for incorrect/ungradable.",
        "- This replaces exact-match as the main performance view; exact-match should be treated only as a string sanity check.",
        "",
        "## Performance",
        "",
        "| Provider | Avg score | Correct | Partial | Incorrect | Ungradable | Citation-supported | False-premise handled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider in PROVIDERS:
        rows = by_provider[provider]
        counts = Counter(row["correctness"] for row in rows)
        avg_score = statistics.fmean(row["final_score"] for row in rows) if rows else 0.0
        cite = sum(1 for row in rows if row["citation_supported"])
        false_handled = sum(1 for row in rows if row["false_premise_handled"])
        lines.append(
            f"| {provider} | {avg_score:.3f} | {counts['correct']} | {counts['partially_correct']} | "
            f"{counts['incorrect']} | {counts['ungradable']} | {cite}/40 | {false_handled} |"
        )

    lines.extend([
        "",
        "## Price-Normalized Performance",
        "",
        "| Provider | Avg score | Est. cost / 1K calls | Score per $1 | Pricing status |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for provider in PROVIDERS:
        rows = by_provider[provider]
        avg_score = statistics.fmean(row["final_score"] for row in rows) if rows else 0.0
        source_rows = [record_by_key[(row["provider"], row["query_id"])] for row in rows]
        costs = [row.get("estimated_cost_usd") for row in source_rows if row.get("estimated_cost_usd") is not None]
        if len(costs) == len(source_rows) and source_rows:
            per_1k = sum(costs) / len(source_rows) * 1000
            score_per_dollar = avg_score / (per_1k / 1000) if per_1k else 0
            lines.append(f"| {provider} | {avg_score:.3f} | ${per_1k:.2f} | {score_per_dollar:.1f} | complete in harness |")
        else:
            lines.append(f"| {provider} | {avg_score:.3f} | n/a | n/a | needs pricing refresh |")

    lines.extend([
        "",
        "## Caveats",
        "",
        "- The judge reviewed answer semantics and citation plausibility from answer text plus citation URLs; it did not fetch every cited page.",
        "- False-premise handling is judge-derived and should be spot-checked before external publication.",
        "- Provider-specific citation extraction can affect citation-supported counts, especially Brave where no citations were parsed in the benchmark records.",
        "",
        "## Judgment Records",
        "",
        "- JSONL: `normalized/judge_results_40_each.jsonl`",
    ])
    (run_dir / "performance_report_40_each.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge semantic performance for the balanced 40-each benchmark slice.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ROOT / "results" / args.run_id
    try:
        env = load_env(env_path())
        records = load_records(run_dir)
        compact = [compact_record(row) for row in records]
        judgments: list[dict[str, Any]] = []
        for start in range(0, len(compact), args.batch_size):
            batch = compact[start : start + args.batch_size]
            judgments.extend(judge_batch(env["GEMINI_API_KEY"], args.model, batch, start // args.batch_size + 1))
            write_jsonl(run_dir / "normalized" / "judge_results_40_each.jsonl", judgments)
            time.sleep(0.5)
        render_report(run_dir, judgments, records, args.model)
    except JudgeError as exc:
        print(f"JUDGE BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {run_dir / 'normalized' / 'judge_results_40_each.jsonl'}")
    print(f"Wrote {run_dir / 'performance_report_40_each.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
