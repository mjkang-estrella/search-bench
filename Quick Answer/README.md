# Quick Answer Benchmarks

This folder contains the Quick Answer benchmark harness, datasets, raw outputs,
normalized results, judge outputs, reports, and charts.

The root `.env` one directory above this folder is used for API keys. The
scripts also support a local `Quick Answer/.env` if needed.

## Layout

- `benchmark.py`: provider smoke tests and benchmark runner.
- `judge.py`: Gemini semantic judge used for the FreshQA 40-run.
- `openai_judge.py`: OpenAI Responses API judge used for the SimpleQA 20-run.
- `data/`: downloaded benchmark datasets.
- `results/freshqa/freshqa40_20260521/`: FreshQA 40-question balanced analysis.
- `results/simpleqa/simpleqa20_20260521/`: SimpleQA 20-question analysis.
- `results/blocked-runs/`: earlier stopped runs kept for traceability.
- `charts/`: standalone Liner strength chart artifacts.
- `scripts/`: chart-generation utilities.

## FreshQA

Primary files:

- `results/freshqa/freshqa40_20260521/performance_report_40_each.md`
- `results/freshqa/freshqa40_20260521/price_performance_report_40_each.md`
- `results/freshqa/freshqa40_20260521/price_vs_performance_40_each_no_brave.png`
- `results/freshqa/freshqa40_20260521/ttfb_latency_vs_performance_40_each.png`
- `results/freshqa/freshqa40_20260521/normalized/results_40_each_priced.jsonl`
- `results/freshqa/freshqa40_20260521/normalized/judge_results_40_each.jsonl`

Use this benchmark for current/open-domain questions and false-premise handling.

## SimpleQA

Primary files:

- `results/simpleqa/simpleqa20_20260521/simpleqa20_openai_summary.md`
- `results/simpleqa/simpleqa20_20260521/simpleqa20_openai_price_vs_performance.png`
- `results/simpleqa/simpleqa20_20260521/simpleqa20_openai_ttfb_vs_performance.png`
- `results/simpleqa/simpleqa20_20260521/normalized/results_simpleqa_20_each_priced.jsonl`
- `results/simpleqa/simpleqa20_20260521/normalized/openai_judge_results_simpleqa_20_each.jsonl`

Use this benchmark for short factual-answer sanity checks.

## Run

From this folder:

```bash
python3 benchmark.py --smoke-only
python3 benchmark.py --freshqa-limit 50 --simpleqa-limit 50
```

Resume a blocked run:

```bash
python3 benchmark.py --run-id <run_id> --reuse-smoke --resume --freshqa-limit 50 --simpleqa-limit 50
```

Run an OpenAI judge pass:

```bash
python3 openai_judge.py \
  --run-dir results/simpleqa/simpleqa20_20260521 \
  --input results/simpleqa/simpleqa20_20260521/normalized/results_simpleqa_20_each_priced.jsonl \
  --output results/simpleqa/simpleqa20_20260521/normalized/openai_judge_results_simpleqa_20_each.jsonl \
  --model gpt-5.5 \
  --prefix simpleqa20
```

## Notes

- Raw provider responses are kept under each run's `raw/` directory.
- Normalized JSONL records are kept under each run's `normalized/` directory.
- Generated charts are checked in as both SVG and PNG when available.
- No fallback provider substitution is used during benchmark execution.
