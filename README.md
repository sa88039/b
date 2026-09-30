# Sequence Lab

Generic sequence-simulation and backtest runner.

This repository intentionally stores only generic experiment definitions and aggregate metrics.
Domain-specific research notes, model meanings, predictions, and decision logic are kept outside this repository.

## Run

- GitHub Actions: **Sequence Lab**
- Manual: Actions -> Sequence Lab -> Run workflow
- Scheduled: hourly

## Output

Each run writes:
- `results/latest.json`
- `results/history/*.json`
- `results/failure_report.json`

Experiments are identified only by anonymous IDs such as `EXP_A001`.
