# Search API Benchmarks

This folder contains raw Search API benchmarks. Provider-generated answers are
disabled where the API exposes that control; scoring is based on whether top-k
search results contain enough evidence to answer the benchmark question.

## Current Run

- `results/search_evals_simpleqa20/simpleqa20_20260522/`
- `results/frames20/frames20_20260522/`
- `results/sealqa20/sealqa20_20260522/`

Primary files:

- `report.md`: raw search coverage, latency, and cost report.
- `simpleqa20_search_openai_judge_report.md`: OpenAI-judged answerability report.
- `normalized/results.jsonl`: normalized provider results.
- `normalized/openai_judge_results.jsonl`: answerability judgments.
- `raw/`: full provider payloads for smoke and benchmark calls.
- `charts/price_vs_answerability.{svg,png,html}`
- `charts/latency_vs_answerability.{svg,png,html}`

## Run

```bash
python3 Search/benchmark.py --limit 20 --max-results 10 --run-id simpleqa20_YYYYMMDD
python3 Search/benchmark.py --benchmark frames --limit 20 --max-results 10 --run-id frames20_YYYYMMDD
python3 Search/benchmark.py --benchmark sealqa --limit 20 --max-results 10 --run-id sealqa20_YYYYMMDD
python3 Search/openai_judge.py \
  --run-dir Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD \
  --input Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD/normalized/results.jsonl \
  --output Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD/normalized/openai_judge_results.jsonl \
  --model gpt-5.5 \
  --prefix simpleqa20_search
python3 Search/scripts/generate_charts.py \
  --run-dir Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD \
  --results Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD/normalized/results.jsonl \
  --judgments Search/results/search_evals_simpleqa20/simpleqa20_YYYYMMDD/normalized/openai_judge_results.jsonl
```
