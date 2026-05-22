#!/usr/bin/env python3
"""Generate a benchmark graphic that highlights Liner's strengths."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "charts"
SVG_PATH = OUT_DIR / "liner-strength-benchmark.svg"
HTML_PATH = OUT_DIR / "liner-strength-benchmark.html"


SERVICES = [
    {
        "company": "Tavily",
        "service": "Search API answer",
        "category": "generated",
        "p50_ms": 138,
        "cost_1k": 8.0,
        "sources": 5,
    },
    {
        "company": "Liner",
        "service": "Quick Answer API",
        "category": "generated",
        "p50_ms": 1402,
        "cost_1k": 3.0,
        "sources": 3,
    },
    {
        "company": "Perplexity",
        "service": "Sonar",
        "category": "generated",
        "p50_ms": 2426,
        "cost_1k": 5.185,
        "sources": 9,
    },
    {
        "company": "Exa",
        "service": "Answer API",
        "category": "generated",
        "p50_ms": 5550,
        "cost_1k": 5.0,
        "sources": 8,
    },
    {
        "company": "Parallel",
        "service": "Task API lite-fast",
        "category": "async-task",
        "p50_ms": 44074,
        "cost_1k": 5.0,
        "sources": 17,
    },
]

FALLBACK = {
    "company": "Brave",
    "service": "Web Search fallback",
    "category": "retrieval-fallback",
    "p50_ms": 125,
    "cost_1k": None,
    "sources": 10,
}


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{int(ms)}ms"


def fmt_money(value: float) -> str:
    return f"${value:.3f}".rstrip("0").rstrip(".")


def line_label(x: int, y: int, width: int, text: str, *, klass: str = "note") -> str:
    return f'<text x="{x}" y="{y}" class="{klass}" textLength="{width}" lengthAdjust="spacingAndGlyphs">{esc(text)}</text>'


def make_svg() -> str:
    generated = SERVICES
    plot_x = 116
    plot_y = 282
    plot_w = 720
    plot_h = 380
    max_cost = 8.5
    max_latency = 6000
    liner = next(item for item in generated if item["company"] == "Liner")
    lower_than_tavily = round((1 - liner["cost_1k"] / 8.0) * 100)
    faster_than_perplexity = 2426 / liner["p50_ms"]
    faster_than_exa = 5550 / liner["p50_ms"]
    colors = {
        "Liner": "#00A66A",
        "Tavily": "#F59E0B",
        "Perplexity": "#6D5DF6",
        "Exa": "#4776E6",
        "Parallel": "#98A2B3",
        "Brave": "#98A2B3",
    }

    def px(cost: float) -> float:
        return plot_x + (cost / max_cost) * plot_w

    def py(ms: float) -> float:
        return plot_y + plot_h - (min(ms, max_latency) / max_latency) * plot_h

    grid = []
    for cost in (0, 2, 4, 6, 8):
        x = px(cost)
        grid.append(
            f"""
      <line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#EAECF0"/>
      <text x="{x:.1f}" y="{plot_y + plot_h + 28}" class="axis-tick" text-anchor="middle">${cost}</text>
            """
        )
    for seconds in (0, 1, 2, 3, 4, 5, 6):
        y = py(seconds * 1000)
        grid.append(
            f"""
      <line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#EAECF0"/>
      <text x="{plot_x - 18}" y="{y + 5:.1f}" class="axis-tick" text-anchor="end">{seconds}s</text>
            """
        )

    points = []
    for service in generated:
        company = service["company"]
        x = px(service["cost_1k"])
        y = py(service["p50_ms"])
        is_liner = company == "Liner"
        is_parallel = company == "Parallel"
        opacity = "0.95" if not is_parallel else "0.55"
        radius = 12 if is_liner else 8
        points.append(
            f"""
      <g opacity="{opacity}">
        {'<circle cx="' + f'{x:.1f}' + '" cy="' + f'{y:.1f}' + '" r="25" fill="#00A66A" opacity="0.14"/>' if is_liner else ''}
        {'<circle cx="' + f'{x:.1f}' + '" cy="' + f'{y:.1f}' + '" r="17" fill="none" stroke="#00A66A" stroke-width="3"/>' if is_liner else ''}
        <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colors[company]}" stroke="#FFFFFF" stroke-width="3"/>
      </g>
            """
        )

    liner_x = px(liner["cost_1k"])
    liner_y = py(liner["p50_ms"])
    table_rows = []
    for index, service in enumerate(sorted(generated, key=lambda item: item["cost_1k"])):
        row_y = 92 + index * 42
        latency_label = fmt_ms(service["p50_ms"])
        if service["company"] == "Parallel":
            latency_label = "44.1s"
        row_opacity = "0.55" if service["company"] == "Parallel" else "1"
        table_rows.append(
            f"""
      <g transform="translate(14 {row_y})" opacity="{row_opacity}">
        <circle cx="0" cy="0" r="6" fill="{colors[service['company']]}"/>
        <text x="18" y="5" class="table-name">{esc(service['company'])}</text>
        <text x="106" y="5" class="table-value">{fmt_money(service['cost_1k'])}</text>
        <text x="18" y="24" class="table-value">{latency_label} p50</text>
      </g>
            """
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="820" viewBox="0 0 1120 820" role="img" aria-labelledby="title desc">
  <title id="title">Fast AI-generated search response benchmark</title>
  <desc id="desc">Benchmark plot showing Liner Quick Answer API as the lowest-cost generated answer endpoint with competitive latency among web-grounded generated-answer APIs.</desc>
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#101828" flood-opacity="0.12"/>
    </filter>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#98A2B3"/>
    </marker>
  </defs>
  <style>
    .title {{ font: 700 34px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #101828; }}
    .subtitle {{ font: 400 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475467; }}
    .section-title {{ font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #101828; }}
    .axis-label {{ font: 700 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #344054; }}
    .axis-tick {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
    .point-label {{ font: 750 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .point-value {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
    .table-head {{ font: 750 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475467; }}
    .table-name {{ font: 700 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #344054; }}
    .table-value {{ font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
    .callout-title {{ font: 700 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #063F2A; }}
    .callout-body {{ font: 500 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #164B35; }}
    .note {{ font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #667085; }}
    .legend {{ font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #344054; }}
  </style>
  <rect width="1120" height="820" fill="#F7F8FA"/>
  <rect x="36" y="34" width="1048" height="752" rx="10" fill="#FFFFFF" filter="url(#shadow)"/>
  <text x="80" y="86" class="title">Liner: the low-cost quick-answer point</text>
  <text x="80" y="118" class="subtitle">Same prompt, same local environment: "What is retrieval augmented generation?" Lower-left is better.</text>

  <g transform="translate(80 150)">
    <rect width="300" height="66" rx="8" fill="#E9F8F1" stroke="#B7E7D0"/>
    <text x="20" y="30" class="callout-title">Lowest generated-answer cost</text>
    <text x="20" y="53" class="callout-body">Liner is {lower_than_tavily}% below Tavily</text>
  </g>
  <g transform="translate(410 150)">
    <rect width="300" height="66" rx="8" fill="#E9F8F1" stroke="#B7E7D0"/>
    <text x="20" y="30" class="callout-title">Quick generated answer</text>
    <text x="20" y="53" class="callout-body">{faster_than_perplexity:.1f}x faster than Perplexity</text>
  </g>
  <g transform="translate(740 150)">
    <rect width="300" height="66" rx="8" fill="#E9F8F1" stroke="#B7E7D0"/>
    <text x="20" y="30" class="callout-title">Light answer surface</text>
    <text x="20" y="53" class="callout-body">{faster_than_exa:.1f}x faster than Exa Answer</text>
  </g>

  <text x="{plot_x}" y="{plot_y - 34}" class="section-title">Cost vs. latency for generated answers</text>
  <text x="{plot_x + plot_w}" y="{plot_y - 34}" class="note" text-anchor="end">Parallel is capped at 6s on the plot.</text>
  <rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="#D0D5DD"/>
  {"".join(grid)}
  <line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#667085" stroke-width="1.4"/>
  <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#667085" stroke-width="1.4"/>
  <text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 62}" class="axis-label" text-anchor="middle">Estimated cost per 1K successful calls</text>
  <text x="{plot_x - 74}" y="{plot_y + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 {plot_x - 74} {plot_y + plot_h / 2})">p50 full latency</text>
  {"".join(points)}
  <line x1="{liner_x + 22:.1f}" y1="{liner_y - 14:.1f}" x2="{liner_x + 94:.1f}" y2="{liner_y - 70:.1f}" stroke="#00A66A" stroke-width="2"/>
  <g transform="translate({liner_x + 102:.1f} {liner_y - 98:.1f})">
    <rect width="178" height="58" rx="8" fill="#E9F8F1" stroke="#8FDCB7"/>
    <text x="14" y="24" class="point-label" fill="#063F2A">Liner</text>
    <text x="14" y="44" class="point-value">$3 / 1K, 1.4s p50</text>
  </g>

  <g transform="translate(866 282)">
    <rect width="180" height="318" rx="8" fill="#F9FAFB" stroke="#EAECF0"/>
    <text x="14" y="26" class="table-head">Provider</text>
    <text x="120" y="26" class="table-head">$/1K</text>
    <text x="14" y="52" class="note">Cost and p50 latency</text>
    {"".join(table_rows)}
  </g>

  <g transform="translate(866 622)">
    <rect width="180" height="78" rx="8" fill="#F9FAFB" stroke="#EAECF0"/>
    <text x="16" y="28" class="note">Fallback baseline</text>
    <circle cx="22" cy="58" r="7" fill="#98A2B3"/>
    <text x="38" y="63" class="legend">Brave</text>
    <text x="92" y="63" class="note">125ms retrieval</text>
  </g>

  <g transform="translate(80 738)">
    <rect width="960" height="34" rx="8" fill="#F2F4F7"/>
    {line_label(18, 22, 880, "Source: Fast AI-Generated Search Response Benchmark (2026-05-21). Brave excluded: Web Search fallback, not AI Grounding.")}
  </g>
</svg>
"""


def make_html(svg: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Liner Strength Benchmark</title>
  <style>
    body {{
      margin: 0;
      background: #f7f8fa;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      box-sizing: border-box;
    }}
    svg {{
      width: min(1120px, 100%);
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
{svg}
  </main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = make_svg()
    SVG_PATH.write_text(svg, encoding="utf-8")
    HTML_PATH.write_text(make_html(svg), encoding="utf-8")
    print(SVG_PATH)
    print(HTML_PATH)


if __name__ == "__main__":
    main()
