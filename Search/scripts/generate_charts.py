#!/usr/bin/env python3
"""Generate SVG benchmark charts from judged Search API results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


COLORS = {
    "liner": "#0f766e",
    "perplexity": "#2563eb",
    "exa": "#7c3aed",
    "parallel": "#dc2626",
    "brave": "#ea580c",
    "tavily": "#475569",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--judgments", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    records = read_jsonl(Path(args.results))
    judgments = read_jsonl(Path(args.judgments))
    by_record = {(row["provider"], row["query_id"]): row for row in records}
    providers = ["liner", "perplexity", "exa", "parallel", "brave", "tavily"]

    rows = []
    for provider in providers:
        js = [row for row in judgments if row["provider"] == provider]
        rs = [by_record[(row["provider"], row["query_id"])] for row in js]
        if not js:
            continue
        score = statistics.fmean(row["final_score"] for row in js) * 100
        total_cost = sum(float(row["estimated_cost_usd"]) for row in rs)
        per_1k = total_cost / len(rs) * 1000
        latency = statistics.median(row["timing"]["total_ms"] for row in rs) / 1000
        rows.append({"provider": provider, "score": score, "per_1k": per_1k, "latency": latency})

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    price_svg = scatter_svg(
        rows,
        title="Search API price vs answerability",
        x_key="per_1k",
        y_key="score",
        x_label="Estimated cost per 1K calls (USD)",
        y_label="OpenAI-judged answerability score (%)",
    )
    latency_svg = scatter_svg(
        rows,
        title="Search API latency vs answerability",
        x_key="latency",
        y_key="score",
        x_label="Median total latency (seconds)",
        y_label="OpenAI-judged answerability score (%)",
    )
    (charts_dir / "price_vs_answerability.svg").write_text(price_svg, encoding="utf-8")
    (charts_dir / "latency_vs_answerability.svg").write_text(latency_svg, encoding="utf-8")
    (charts_dir / "price_vs_answerability.html").write_text(wrap_html(price_svg), encoding="utf-8")
    (charts_dir / "latency_vs_answerability.html").write_text(wrap_html(latency_svg), encoding="utf-8")
    print(f"Wrote charts to {charts_dir}")
    return 0


def scatter_svg(rows: list[dict], title: str, x_key: str, y_key: str, x_label: str, y_label: str) -> str:
    width, height = 980, 620
    left, right, top, bottom = 110, 60, 86, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(row[x_key] for row in rows) * 1.12
    min_y = max(0, min(row[y_key] for row in rows) - 10)
    max_y = min(100, max(row[y_key] for row in rows) + 10)
    if max_y <= min_y:
        max_y = min_y + 1

    def sx(value: float) -> float:
        return left + (value / max_x) * plot_w if max_x else left

    def sy(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#0f172a} .muted{fill:#64748b;font-size:13px} .title{font-size:26px;font-weight:700} .label{font-size:14px;font-weight:600} .tick{font-size:12px;fill:#64748b} .grid{stroke:#e2e8f0;stroke-width:1} .axis{stroke:#94a3b8;stroke-width:1.5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="44" class="title">{escape(title)}</text>',
    ]
    for i in range(6):
        y_value = min_y + (max_y - min_y) * i / 5
        y = sy(y_value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_value:.0f}</text>')
    for i in range(6):
        x_value = max_x * i / 5
        x = sx(x_value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{x_value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 34}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    parts.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    for row in rows:
        x = sx(row[x_key])
        y = sy(row[y_key])
        color = COLORS.get(row["provider"], "#334155")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>')
        parts.append(f'<text x="{x + 13:.1f}" y="{y - 10:.1f}" class="label">{escape(row["provider"])}</text>')
        parts.append(f'<text x="{x + 13:.1f}" y="{y + 8:.1f}" class="muted">{row[y_key]:.1f}%</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def wrap_html(svg: str) -> str:
    return "<!doctype html><meta charset=\"utf-8\"><title>Search API chart</title>" + svg


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
