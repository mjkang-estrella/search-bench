#!/usr/bin/env python3
"""Generate raw URL retrieval reports and charts for Search API benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.parse
from pathlib import Path
from typing import Any


DEFAULT_PROVIDERS = ["liner", "perplexity", "exa", "parallel", "brave", "tavily"]
COLORS = {
    "liner": "#0f766e",
    "perplexity": "#2563eb",
    "exa": "#7c3aed",
    "parallel": "#dc2626",
    "brave": "#ea580c",
    "tavily": "#475569",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hostname(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def canonical_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    return f"{host}{path}".lower()


def url_match(result_url: str, gold_url: str) -> bool:
    result = canonical_url(result_url)
    gold = canonical_url(gold_url)
    return bool(result and gold and (result == gold or result.startswith(gold) or gold.startswith(result)))


def retrieval_match(row: dict[str, Any], k: int) -> dict[str, Any]:
    gold_urls = [url for url in row.get("gold_urls", []) if url]
    gold_domain_urls = [url for url in row.get("gold_domain_urls", []) if url]
    gold_hosts = {hostname(url) for url in [*gold_urls, *gold_domain_urls] if hostname(url)}
    exact_rank = None
    domain_rank = None
    exact_url = None
    domain_url = None
    for result in row.get("results", [])[:k]:
        rank = result.get("rank")
        result_url = result.get("url", "")
        if exact_rank is None and any(url_match(result_url, gold_url) for gold_url in gold_urls):
            exact_rank = rank
            exact_url = result_url
        if domain_rank is None and result.get("hostname") in gold_hosts:
            domain_rank = rank
            domain_url = result_url
    return {
        "exact_hit": exact_rank is not None,
        "exact_rank": exact_rank,
        "exact_url": exact_url,
        "domain_hit": domain_rank is not None,
        "domain_rank": domain_rank,
        "domain_url": domain_url,
        "url_or_domain_hit": exact_rank is not None or domain_rank is not None,
        "url_or_domain_rank": min([rank for rank in [exact_rank, domain_rank] if rank is not None], default=None),
    }


def summarize(records: list[dict[str, Any]], k: int, providers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in providers:
        provider_rows = [row for row in records if row["provider"] == provider]
        if not provider_rows:
            continue
        matches = [retrieval_match(row, k) for row in provider_rows]
        exact_ranks = [m["exact_rank"] for m in matches if m["exact_rank"]]
        domain_ranks = [m["domain_rank"] for m in matches if m["domain_rank"]]
        combined_ranks = [m["url_or_domain_rank"] for m in matches if m["url_or_domain_rank"]]
        total_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in provider_rows)
        per_1k = total_cost / len(provider_rows) * 1000
        exact_hits = sum(1 for m in matches if m["exact_hit"])
        domain_hits = sum(1 for m in matches if m["domain_hit"])
        combined_hits = sum(1 for m in matches if m["url_or_domain_hit"])
        rows.append(
            {
                "provider": provider,
                "calls": len(provider_rows),
                "exact_hits": exact_hits,
                "domain_hits": domain_hits,
                "combined_hits": combined_hits,
                "exact_recall": exact_hits / len(provider_rows) * 100,
                "domain_recall": domain_hits / len(provider_rows) * 100,
                "combined_recall": combined_hits / len(provider_rows) * 100,
                "exact_mrr": statistics.fmean(1 / rank for rank in exact_ranks) if exact_ranks else 0.0,
                "combined_mrr": statistics.fmean(1 / rank for rank in combined_ranks) if combined_ranks else 0.0,
                "avg_exact_rank": statistics.fmean(exact_ranks) if exact_ranks else 0.0,
                "avg_combined_rank": statistics.fmean(combined_ranks) if combined_ranks else 0.0,
                "total_cost": total_cost,
                "per_1k": per_1k,
                "exact_hits_per_dollar": exact_hits / total_cost if total_cost else 0.0,
                "combined_hits_per_dollar": combined_hits / total_cost if total_cost else 0.0,
                "p50_latency_ms": statistics.median(row["timing"]["total_ms"] for row in provider_rows),
            }
        )
    return rows


def write_report(run_dir: Path, records: list[dict[str, Any]], summary: list[dict[str, Any]], k: int, title: str) -> None:
    lines = [
        f"# {title} Raw Search Retrieval Report",
        "",
        f"- Records: {len(records)}",
        f"- Metric depth: top {k}",
        "- `exact_url_hit` matches canonicalized benchmark-provided gold URLs.",
        "- `domain_hit` matches benchmark-provided gold URL hostnames.",
        "- This report intentionally ignores snippet answerability and generated-answer accuracy.",
        "",
        "## Retrieval",
        "",
        "| Provider | Calls | Exact URL hit@10 | Domain hit@10 | URL/domain hit@10 | Exact MRR | URL/domain MRR | Avg exact rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['provider']} | {row['calls']} | {row['exact_hits']}/{row['calls']} ({row['exact_recall']:.1f}%) | "
            f"{row['domain_hits']}/{row['calls']} ({row['domain_recall']:.1f}%) | "
            f"{row['combined_hits']}/{row['calls']} ({row['combined_recall']:.1f}%) | "
            f"{row['exact_mrr']:.3f} | {row['combined_mrr']:.3f} | {row['avg_exact_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Price",
            "",
            "| Provider | Est. / 1K calls | Exact hits / $1 | URL/domain hits / $1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['provider']} | ${row['per_1k']:.2f} | {row['exact_hits_per_dollar']:.0f} | "
            f"{row['combined_hits_per_dollar']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Provider | p50 total latency |",
            "| --- | ---: |",
        ]
    )
    for row in summary:
        lines.append(f"| {row['provider']} | {row['p50_latency_ms'] / 1000:.3f}s |")
    (run_dir / "raw_retrieval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def price_vs_hit_rate_svg(summary: list[dict[str, Any]], title: str, y_key: str, y_label: str) -> str:
    return scatter_svg(
        summary,
        title,
        "per_1k",
        y_key,
        "Estimated cost per 1K calls (USD)",
        y_label,
        y_max=100,
        value_suffix="%",
    )


def grouped_bar_svg(summary: list[dict[str, Any]], title: str) -> str:
    width, height = 1060, 640
    left, top, bottom = 92, 88, 110
    plot_w, plot_h = 900, height - top - bottom
    group_w = plot_w / len(summary)
    bar_w = 22
    parts = svg_header(width, height, title)
    for i in range(6):
        y_value = i * 20
        y = top + plot_h - (y_value / 100) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_value}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    for idx, row in enumerate(summary):
        cx = left + idx * group_w + group_w / 2
        exact_h = row["exact_recall"] / 100 * plot_h
        domain_h = row["combined_recall"] / 100 * plot_h
        color = COLORS.get(row["provider"], "#334155")
        parts.append(f'<rect x="{cx - 28:.1f}" y="{top + plot_h - exact_h:.1f}" width="{bar_w}" height="{exact_h:.1f}" fill="{color}"/>')
        parts.append(f'<rect x="{cx + 4:.1f}" y="{top + plot_h - domain_h:.1f}" width="{bar_w}" height="{domain_h:.1f}" fill="{color}" opacity="0.42"/>')
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 28}" text-anchor="middle" class="label">{escape(row["provider"])}</text>')
        parts.append(f'<text x="{cx - 17:.1f}" y="{top + plot_h - exact_h - 8:.1f}" text-anchor="middle" class="tick">{row["exact_recall"]:.0f}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 34}" text-anchor="middle" class="label">Provider</text>')
    parts.append(f'<text x="26" y="{top + plot_h / 2:.1f}" transform="rotate(-90 26 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">Recall@10 (%)</text>')
    parts.append(f'<rect x="{left + plot_w - 230}" y="42" width="16" height="16" fill="#0f172a"/><text x="{left + plot_w - 206}" y="55" class="muted">Exact URL hit@10</text>')
    parts.append(f'<rect x="{left + plot_w - 230}" y="66" width="16" height="16" fill="#0f172a" opacity="0.42"/><text x="{left + plot_w - 206}" y="79" class="muted">URL/domain hit@10</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_svg(
    summary: list[dict[str, Any]],
    title: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    *,
    y_max: float | None = None,
    value_suffix: str = "",
) -> str:
    width, height = 980, 620
    left, right, top, bottom = 110, 64, 88, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max(row[x_key] for row in summary) * 1.15 if summary else 1
    max_y = y_max or (max(row[y_key] for row in summary) * 1.12 if summary else 1)
    max_y = max(max_y, 1)
    parts = svg_header(width, height, title)

    def sx(value: float) -> float:
        return left + (value / max_x) * plot_w if max_x else left

    def sy(value: float) -> float:
        return top + plot_h - (value / max_y) * plot_h

    for i in range(6):
        y_value = max_y * i / 5
        y = sy(y_value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_value:.0f}</text>')
        x_value = max_x * i / 5
        x = sx(x_value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{x_value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    for row in summary:
        x = sx(row[x_key])
        y = sy(row[y_key])
        color = COLORS.get(row["provider"], "#334155")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>')
        parts.append(f'<text x="{x + 13:.1f}" y="{y - 7:.1f}" class="label">{escape(row["provider"])}</text>')
        parts.append(f'<text x="{x + 13:.1f}" y="{y + 11:.1f}" class="muted">{row[y_key]:.0f}{escape(value_suffix)}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 34}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    parts.append(f'<text x="28" y="{top + plot_h / 2:.1f}" transform="rotate(-90 28 {top + plot_h / 2:.1f})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#0f172a} .title{font-size:26px;font-weight:700}.label{font-size:14px;font-weight:600}.muted{fill:#64748b;font-size:13px}.tick{font-size:12px;fill:#64748b}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#94a3b8;stroke-width:1.5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="92" y="48" class="title">{escape(title)}</text>',
    ]


def wrap_html(svg: str) -> str:
    return "<!doctype html><meta charset=\"utf-8\"><title>Raw retrieval chart</title>" + svg


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--providers")
    parser.add_argument("--price-hit-rate-only", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    records = read_jsonl(run_dir / "normalized" / "results.jsonl")
    providers = [item.strip() for item in args.providers.split(",") if item.strip()] if args.providers else provider_order(records)
    summary = summarize(records, args.k, providers)
    write_report(run_dir, records, summary, args.k, args.title)
    charts_dir = run_dir / "charts" / "raw_retrieval"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "price_vs_exact_url_hit_rate": price_vs_hit_rate_svg(
            summary,
            f"{args.title}: price vs exact URL hit rate",
            "exact_recall",
            "Exact URL hit@10 rate (%)",
        ),
        "price_vs_url_domain_hit_rate": price_vs_hit_rate_svg(
            summary,
            f"{args.title}: price vs URL/domain hit rate",
            "combined_recall",
            "URL/domain hit@10 rate (%)",
        ),
    }
    if not args.price_hit_rate_only:
        charts.update(
            {
                "recall_at_10": grouped_bar_svg(summary, f"{args.title}: raw retrieval recall@10"),
                "cost_vs_exact_hits": scatter_svg(
                    summary,
                    f"{args.title}: cost vs exact URL hits",
                    "per_1k",
                    "exact_hits",
                    "Estimated cost per 1K calls (USD)",
                    "Exact URL hits@10 (count)",
                ),
                "latency_vs_exact_hits": scatter_svg(
                    summary,
                    f"{args.title}: latency vs exact URL hits",
                    "p50_latency_ms",
                    "exact_hits",
                    "Median total latency (ms)",
                    "Exact URL hits@10 (count)",
                ),
            }
        )
    for name, svg in charts.items():
        (charts_dir / f"{name}.svg").write_text(svg, encoding="utf-8")
        (charts_dir / f"{name}.html").write_text(wrap_html(svg), encoding="utf-8")
    print(f"Wrote {run_dir / 'raw_retrieval_report.md'}")
    print(f"Wrote charts to {charts_dir}")
    return 0


def provider_order(records: list[dict[str, Any]]) -> list[str]:
    present = {row["provider"] for row in records}
    ordered = [provider for provider in DEFAULT_PROVIDERS if provider in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


if __name__ == "__main__":
    raise SystemExit(main())
