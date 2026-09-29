# latentedge — an ML pattern-recognition trading project

## What this is

A research project looking for a trading edge on Uniswap using a model
trained specifically for this task, not a general-purpose one prompted with
market data. It's driven by learning ML properly — but pursued with the
rigor of a genuine attempt at an edge, not a toy exercise: real data
discipline, honest labeling, walk-forward validation, and no claiming
success without out-of-sample evidence.

The core thesis: there are patterns in market data that a model can learn to
recognize even when no human has written down the rule — something that
fits to real-world structure rather than an explicit heuristic. This project
tries to find one.

## Result

On about 200 days of WETH/USDC 0.05% swaps the model finds a faint direction
signal at a 30-minute horizon and no tradable edge. Walk-forward, pooled out
of sample, the correlation between its prediction and the pre-cost price move
is +0.034 (95% interval +0.023 to +0.045, permutation p = 0.001), positive in
every fold and stable across seeds. It fades at 120 minutes and is absent at
240. The signal is about a twelfth of the size needed to cover the round-trip
cost of trading it, and every trading rule loses money on both the validation
and the test window. An earlier model trained on net return looked much
better (0.35 correlation with its label), but that was cost prediction, not
direction.

The paper is available as a PDF: [`docs/paper/paper.pdf`](docs/paper/paper.pdf).
The LaTeX source is [`docs/paper/paper.tex`](docs/paper/paper.tex), and the sweep and study records it cites are in
[`docs/paper/results/`](docs/paper/results/).

## Getting started

Requires Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and an Apple
Silicon Mac (the model uses MLX, which is Apple Silicon–only).

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `LATENTEDGE_RPC_URL` to an archive-capable RPC endpoint
(e.g. an Alchemy or Infura mainnet URL — free public endpoints don't serve
history older than a few hours). `LATENTEDGE_INGEST_DAYS` controls how much
history a plain `ingest` call pulls; it's set low by default so a first run
is quick.

Run the test suite and type checker:

```bash
uv run pytest
uv run mypy --strict src/
```

Try a small real ingest (pulls a few hours of real WETH/USDC swap history
from your configured RPC endpoint into `data/swaps.parquet`, showing a live
progress dashboard):

```bash
uv run latentedge ingest --days 1
```

Ingest is resumable and safe to re-run — it tracks which blocks it already
has and only ever fetches new ones, so running it again (or with a larger
`--days`) never re-downloads existing data. Blocks are fetched newest
first, so whatever has been ingested at any moment is one continuous stretch
ending at the newest block: press `T` in the ingest dashboard at any time to
stop cleanly and train on what's there. `--from-block`/`--to-block` are
available for a specific range instead of a most-recent-N-days window.

Then train a model on whatever's been ingested so far:

```bash
uv run latentedge train --swaps data/swaps.parquet
```

Then backtest it:

```bash
uv run latentedge backtest --swaps data/swaps.parquet --model data/model.safetensors
```

The backtest replays only the model's untouched test window (its start is
recorded when the model is trained, so newer ingested data can only extend
the window forward), applies the safety guard, and writes total return, max
drawdown, win rate, trade count and a daily Sharpe to
`<model>.backtest.json`. Models trained before the window start was
recorded must be retrained.

The sweep replays the model under a grid of trading rules on the validation
and test windows, next to an always-trade baseline, an oracle and shuffled
predictions, and saves every result under `data/sweeps/`. It runs after
training by default (`LATENTEDGE_SWEEP_AFTER_TRAIN`):

```bash
uv run latentedge sweep --swaps data/swaps.parquet --model data/model.safetensors
```

The study is the robustness check behind the paper's numbers. For several
label horizons it trains several seeds on expanding walk-forward folds, each
judged on bars it never saw, and reports the pooled out-of-sample correlation
with a block-bootstrap interval and a permutation p-value that respect the
overlap between labels. Results go to `data/studies/`:

```bash
uv run latentedge study --swaps data/swaps.parquet
```

Run options are environment variables, not flags (see `.env.example`):
`LATENTEDGE_TRAIN_*` for the training target, network size and excluded
features, `LATENTEDGE_LABEL_*` for the horizon and barrier band,
`LATENTEDGE_SWEEP_*` and `LATENTEDGE_STUDY_*` for the two evaluations.

## Decided so far

- **Execution venue: Uniswap.** An AMM, not an order book — price comes from
  pool reserves/tick math, so the signal → decision → execution pipeline is
  built around that, not bid/ask levels.
- **The model is purpose-trained**, not a general-purpose model prompted
  with numbers. Local inference on the Mac (MLX or similar).
- **Architecture shape**: a signal client, an executor, a safety-guard
  (position sizing, daily-loss limits), and a backtest harness that tracks
  wallet/P&L/win-rate/equity curve honestly.
- **Training data: on-chain Uniswap v3 swap history for the target pool
  itself**, resampled into 1-minute time-weighted price + volume bars (price
  from `sqrtPriceX96`, volume from swap amounts). Training on the same
  pool's own data — not a CEX proxy — sidesteps the question of whether a
  signal learned elsewhere transfers to this venue. Sourced via a subgraph
  or archive-node `eth_getLogs`, whichever proves simpler in practice.
- **Label: a regression target, not a binary classification.** Each bar
  carries both the net return (after costs) and the gross return (the price
  move alone). Training defaults to gross, because a net target is dominated
  by trading cost, which is easy to predict, and a net-trained model can look
  skilled without knowing anything about direction. Triple-
  barrier labeling (take-profit / stop-loss / time-limit) over a 30-minute
  horizon by default, with the net outcome computed after the pool's fee
  tier, an estimated slippage from pool depth at trade size, and gas.
  Regression over classification because magnitude is what position sizing
  needs, and P&L — not label accuracy — is the metric that actually matters.
- **Stack: Python throughout, MLX for the model.** No second language, no
  service boundary — everything from ingestion through the backtest
  harness runs in-process. MLX is Python-first and best-tuned for Apple
  Silicon; the project's real goal is learning ML, and Python is where
  that ecosystem actually lives (see the spec for the full rationale).
- **v1 pool: WETH/USDC, 0.05% fee tier.** The most liquid, longest-history,
  most-traded Uniswap v3 pool. Deliberately the hardest market to find an
  edge in, chosen so pipeline correctness (data, labeling, backtest
  accounting) gets proven against clean, abundant data before moving to a
  thinner, less efficient pool where an edge might be easier to find but
  data quality is worse.

## Open questions

- How much a learned signal actually transfers across markets/pairs.
  Deliberately **not** tested in v1 — training natively on the target
  pool's own data sidesteps rather than answers this. Real open question
  for a v2 that considers more than one pool/pair.
- Whether a less liquid pool has a larger signal, or a smaller one that costs
  less to trade. Only the most liquid pool has been measured.

## Non-goals (for now)

- Real money. This stays a research/backtest project until there's real
  out-of-sample evidence of an edge.
- Multi-chain/multi-venue generality. One pair on one venue first.
- Cross-market signal transfer. Not attempted until v1's single-pool
  pipeline is proven.
