# Search API Benchmarks

This folder contains raw Search API benchmarks. Provider-generated answers are
disabled where the API exposes that control; scoring is based on whether top-k
search results contain enough evidence to answer the benchmark question.

## Current Run

- `results/search_evals_simpleqa20/simpleqa20_20260522/`
- `results/frames20/frames20_20260522/`
- `results/sealqa20/sealqa20_20260522/`
- `results/search_evals_simpleqa100/retrieval100_20260523/`
- `results/sealqa100/retrieval100_20260523/`
- `results/frames100/retrieval100_20260523/`
- `results/webwalkerqa100/retrieval100_20260523/`

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
python3 Search/benchmark.py \
  --benchmark simpleqa \
  --sample-size 100 \
  --seed 20260523 \
  --providers liner,perplexity,exa,parallel \
  --max-results 10 \
  --run-id retrieval100_YYYYMMDD
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
python3 Search/scripts/generate_raw_retrieval_report.py \
  --run-dir Search/results/search_evals_simpleqa100/retrieval100_YYYYMMDD \
  --title SimpleQA \
  --providers liner,perplexity,exa,parallel \
  --price-hit-rate-only
```
