#!/usr/bin/env python3
"""OpenAI semantic judge and chart generator for AI Search benchmark records."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
RESPONSES_URL = "https://api.openai.com/v1/responses"
PROVIDERS = ["liner", "liner_pro", "perplexity", "exa", "exa_deep", "parallel", "brave", "tavily"]
PROVIDER_LABELS = {
    "liner": "Liner AI Search",
    "liner_pro": "Liner AI Search Pro",
    "perplexity": "Perplexity Sonar Pro",
    "exa": "Exa Answer",
    "exa_deep": "Exa Deep Search",
    "parallel": "Parallel Chat",
    "brave": "Brave Answers",
    "tavily": "Tavily Advanced Answer",
}
COLORS = {
    "liner": "#00A66A",
    "liner_pro": "#047857",
    "perplexity": "#6D5DF6",
    "exa": "#2563EB",
    "exa_deep": "#0EA5E9",
    "parallel": "#7C3AED",
    "brave": "#F97316",
    "tavily": "#F59E0B",
}


class JudgeError(Exception):
    pass


def env_path() -> Path:
    for path in (ROOT / ".env", WORKSPACE / ".env"):
        if path.exists():
            return path
    return WORKSPACE / ".env"


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
        raise JudgeError("Missing OPENAI_API_KEY")
    return env


def compact_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": row["provider"],
        "query_id": row["query_id"],
        "question": row["query"],
        "reference_answer": row.get("reference_answer", ""),
        "false_premise": row.get("false_premise", False),
        "fact_type": row.get("fact_type", ""),
        "provider_answer": row.get("answer", ""),
        "citations": row.get("citations", [])[:12],
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
        "You are grading AI search API answers against FreshQA references. "
        "Judge semantic correctness, not exact string match. For false-premise questions, a correct answer must reject or correct the false premise. "
        "Mark citation_supported true only when at least one listed URL plausibly supports the answer. "
        "Use ungradable only when the supplied question/reference/answer is insufficient. Keep rationales under 25 words."
    )
    prompt = {
        "rubric": {
            "correct": "Semantically equivalent to the reference answer or directly satisfies it.",
            "partially_correct": "Contains a meaningful correct element but misses or distorts an important condition.",
            "incorrect": "Contradicts the reference, accepts a false premise, or gives the wrong entity/fact.",
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
                "name": "ai_search_benchmark_judgments",
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
        raise JudgeError(f"OpenAI judge HTTP {exc.code} on batch {batch_index}: {text[:700]}") from exc
    except urllib.error.URLError as exc:
        raise JudgeError(f"OpenAI judge network error on batch {batch_index}: {exc.reason}") from exc
    parsed = json.loads(extract_output_text(payload))
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != len(batch):
        raise JudgeError(f"OpenAI judge returned invalid result count on batch {batch_index}")
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
        raise JudgeError(f"OpenAI response had no output text: {json.dumps(payload)[:500]}")
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


def metrics(judgments: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    record_by_key = {(row["provider"], row["query_id"]): row for row in records}
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_provider[row["provider"]].append(row)
    out: dict[str, dict[str, Any]] = {}
    for provider in PROVIDERS:
        rows = by_provider.get(provider, [])
        if not rows:
            continue
        source_rows = [record_by_key[(row["provider"], row["query_id"])] for row in rows]
        counts = Counter(row["correctness"] for row in rows)
        costs = [row.get("estimated_cost_usd") for row in source_rows if row.get("estimated_cost_usd") is not None]
        total_cost = sum(costs) if len(costs) == len(source_rows) else None
        per_1k = total_cost / len(source_rows) * 1000 if total_cost is not None else None
        score = statistics.fmean(row["final_score"] for row in rows) * 100
        out[provider] = {
            "provider": provider,
            "label": PROVIDER_LABELS.get(provider, provider),
            "rows": len(rows),
            "score": score,
            "correct": counts["correct"],
            "partial": counts["partially_correct"],
            "incorrect": counts["incorrect"],
            "ungradable": counts["ungradable"],
            "citation_supported": sum(1 for row in rows if row["citation_supported"]),
            "false_premise_handled": sum(1 for row in rows if row["false_premise_handled"]),
            "cost_total": total_cost,
            "cost_1k": per_1k,
            "score_per_dollar": score / (per_1k / 1000) if per_1k else None,
            "p50_ttfb": median_timing(source_rows, "ttfb_ms"),
            "p50_first_answer": median_timing(source_rows, "first_answer_ms"),
            "p50_total": median_timing(source_rows, "total_ms"),
            "citation_avg": statistics.fmean(len(row.get("citations", [])) for row in source_rows),
        }
    return out


def median_timing(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["timing"].get(key) for row in rows if row["timing"].get(key) is not None]
    return statistics.median(values) if values else None


def render_report(run_dir: Path, judgments: list[dict[str, Any]], records: list[dict[str, Any]], model: str, prefix: str) -> None:
    data = metrics(judgments, records)
    lines = [
        "# AI Search FreshQA 20 OpenAI-Judged Report",
        "",
        "## Executive Summary",
        "",
        f"- Judge model: `{model}`",
        f"- Records: {len(judgments)}",
        "- Score: correct+citation-supported=100%, correct without citation support=70%, partial=50%, incorrect/ungradable=0%.",
        "- The judge did not fetch cited pages; citation support is based on URL/title plausibility from the returned citation list.",
        "",
        "## Performance",
        "",
        "| Provider | Avg score | Correct | Partial | Incorrect | Ungradable | Citation-supported | False-premise handled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider in sorted(data.values(), key=lambda item: item["score"], reverse=True):
        lines.append(
            f"| {provider['label']} | {provider['score']:.1f}% | {provider['correct']} | {provider['partial']} | "
            f"{provider['incorrect']} | {provider['ungradable']} | {provider['citation_supported']}/{provider['rows']} | "
            f"{provider['false_premise_handled']} |"
        )
    lines.extend(["", "## Price", "", "| Provider | Avg score | Est. total cost | Est. / 1K calls | Score points per $1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for provider in sorted(data.values(), key=lambda item: item["cost_1k"] if item["cost_1k"] is not None else 999999):
        cost_total = "n/a" if provider["cost_total"] is None else f"${provider['cost_total']:.6f}"
        cost_1k = "n/a" if provider["cost_1k"] is None else f"${provider['cost_1k']:.2f}"
        spd = "n/a" if provider["score_per_dollar"] is None else f"{provider['score_per_dollar']:.0f}"
        lines.append(f"| {provider['label']} | {provider['score']:.1f}% | {cost_total} | {cost_1k} | {spd} |")
    lines.extend(["", "## Latency", "", "| Provider | Avg score | p50 TTFB | p50 first answer | p50 total latency |", "| --- | ---: | ---: | ---: | ---: |"])
    for provider in sorted(data.values(), key=lambda item: item["p50_total"] if item["p50_total"] is not None else 999999):
        lines.append(
            f"| {provider['label']} | {provider['score']:.1f}% | {fmt_ms(provider['p50_ttfb'])} | "
            f"{fmt_ms(provider['p50_first_answer'])} | {fmt_ms(provider['p50_total'])} |"
        )
    lines.extend(["", "## Files", "", "- Judgment records: `normalized/openai_judge_results_freshqa_20_each.jsonl`", "- Charts: `charts/`"])
    (run_dir / f"{prefix}_openai_judge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.3f}s"


def render_charts(run_dir: Path, data: dict[str, dict[str, Any]], prefix: str) -> None:
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{prefix}_plot_data.json").write_text(json.dumps(list(data.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    write_scatter(
        charts / f"{prefix}_price_vs_performance.svg",
        data,
        x_key="cost_1k",
        y_key="score",
        title="FreshQA 20: Price vs. Performance",
        x_label="Estimated cost per 1K calls (USD)",
        y_label="OpenAI-judged score (%)",
        lower_is_better_x=True,
    )
    write_scatter(
        charts / f"{prefix}_ttfb_vs_performance.svg",
        data,
        x_key="p50_ttfb",
        y_key="score",
        title="FreshQA 20: TTFB vs. Performance",
        x_label="p50 TTFB (ms)",
        y_label="OpenAI-judged score (%)",
        lower_is_better_x=True,
    )
    for svg in charts.glob("*.svg"):
        convert_png(svg)


def write_scatter(path: Path, data: dict[str, dict[str, Any]], *, x_key: str, y_key: str, title: str, x_label: str, y_label: str, lower_is_better_x: bool) -> None:
    rows = [row for row in data.values() if row.get(x_key) is not None and row.get(y_key) is not None]
    width, height = 1040, 700
    left, top, plot_w, plot_h = 116, 94, 760, 470
    max_x = max(float(row[x_key]) for row in rows) * 1.12 if rows else 1.0
    min_y = max(0.0, min(float(row[y_key]) for row in rows) - 10) if rows else 0.0
    max_y = min(100.0, max(float(row[y_key]) for row in rows) + 10) if rows else 100.0
    if max_y - min_y < 20:
        min_y = max(0.0, max_y - 20)

    def sx(value: float) -> float:
        return left + (value / max_x) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - min_y) / (max_y - min_y)) * plot_h

    grid = []
    for i in range(6):
        x_value = max_x * i / 5
        x = sx(x_value)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#EAECF0"/>')
        grid.append(f'<text x="{x:.1f}" y="{top + plot_h + 28}" class="tick" text-anchor="middle">{x_value:.0f}</text>')
    for i in range(6):
        y_value = min_y + (max_y - min_y) * i / 5
        y = sy(y_value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#EAECF0"/>')
        grid.append(f'<text x="{left - 16}" y="{y + 5:.1f}" class="tick" text-anchor="end">{y_value:.0f}</text>')

    points = []
    for row in rows:
        x = sx(float(row[x_key]))
        y = sy(float(row[y_key]))
        provider = row["provider"]
        radius = 12 if provider.startswith("liner") else 9
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{COLORS.get(provider, "#667085")}" stroke="#FFFFFF" stroke-width="3"/>'
            f'<text x="{x + 14:.1f}" y="{y - 10:.1f}" class="label">{esc(row["label"])}</text>'
            f'<text x="{x + 14:.1f}" y="{y + 8:.1f}" class="small">{row["score"]:.1f}%</text>'
        )
    legend = []
    for idx, row in enumerate(rows):
        y = 96 + idx * 30
        legend.append(f'<circle cx="910" cy="{y}" r="6" fill="{COLORS.get(row["provider"], "#667085")}"/>')
        legend.append(f'<text x="924" y="{y + 5}" class="small">{esc(row["label"])}</text>')

    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <style>
    .title {{ font: 700 28px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #101828; }}
    .axis {{ font: 700 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #344054; }}
    .tick {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
    .label {{ font: 750 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #101828; }}
    .small {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
  </style>
  <rect width="{width}" height="{height}" fill="#FFFFFF"/>
  <text x="64" y="54" class="title">{esc(title)}</text>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D0D5DD"/>
  {''.join(grid)}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#667085"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#667085"/>
  {''.join(points)}
  {''.join(legend)}
  <text x="{left + plot_w / 2}" y="{top + plot_h + 64}" class="axis" text-anchor="middle">{esc(x_label)}</text>
  <text x="34" y="{top + plot_h / 2}" class="axis" text-anchor="middle" transform="rotate(-90 34 {top + plot_h / 2})">{esc(y_label)}</text>
  <text x="{left}" y="{height - 28}" class="small">Generated from normalized benchmark records and OpenAI judge results.</text>
</svg>
""",
        encoding="utf-8",
    )


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def convert_png(svg: Path) -> None:
    png = svg.with_suffix(".png")
    for command in (["rsvg-convert", str(svg), "-o", str(png)], ["convert", str(svg), str(png)]):
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--batch-size", type=int, default=21)
    parser.add_argument("--prefix", default="freshqa20")
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
        data = metrics(judgments, records)
        render_report(run_dir, judgments, records, args.model, args.prefix)
        render_charts(run_dir, data, args.prefix)
    except Exception as exc:
        print(f"OPENAI JUDGE BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    print(f"Wrote {run_dir / f'{args.prefix}_openai_judge_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
