# Liner API Benchmarks

This workspace is organized by product surface.

## Folders

- `Quick Answer/`: completed Quick Answer benchmarks, including FreshQA and SimpleQA runs.
- `Search/`: reserved for upcoming raw Search API benchmarks.
- `AI Search/`: reserved for upcoming AI Search / generated-search benchmarks.

The shared `.env` stays at the workspace root so future benchmark folders can
reuse the same provider keys without duplicating secrets.

## Current Quick Answer Outputs

- FreshQA 40-question balanced run:
  `Quick Answer/results/freshqa/freshqa40_20260521/`
- SimpleQA 20-question OpenAI-judged run:
  `Quick Answer/results/simpleqa/simpleqa20_20260521/`

See `Quick Answer/README.md` for commands, report paths, and chart locations.
