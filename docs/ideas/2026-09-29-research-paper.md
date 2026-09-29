# Research paper (idea, not yet drafted)

Goal: write the project up as a scientific paper. The research question is
whether tradable edges still exist in a liquid on-chain market, and whether a
purpose-trained model can find them. A paper needs a full round trip: data,
method, experiments, and an honest conclusion, including a negative one.

Format: LaTeX source under `docs/paper/`, source files only, no new
dependencies. Plain research prose.

## Current state of the evidence (2026-09-29)

Not enough for conclusions yet.

- The pipeline exists: ingest, 1-minute bars, triple-barrier net-return
  labels, a small MLX regressor, chronological train/validate/test split.
- One training run so far on WETH/USDC 0.05%. Correlation with the label:
  train 0.17, validate 0.09, test 0.43. MSE improvement over the mean
  baseline: about 3%, 1% and 18%.
- The test figure is much higher than validate and the test window has a
  higher label variance, so it looks like a regime difference.
- The label is net of gas and slippage and two features are gas price and
  volatility. The model may be predicting trading cost, not direction.
  This is a hypothesis to test, not a finding.
- No backtest P&L exists yet. The only metric that decides whether an edge
  exists is backtest P&L on the untouched test window.

## Experiments the paper needs

1. Backtest on the test window only: total return, max drawdown, win rate,
   trade count, Sharpe-like ratio (in progress).
2. Controls, all through the same backtest:
   - shuffled predictions (a model with no information),
   - an always-trade baseline,
   - a retrain without gas price and volatility, to separate cost
     prediction from direction.
3. More data: ingest is running for 150 days and will be extended. Report
   results as a function of history length.
4. Robustness: several seeds, more than one label horizon, and a walk-forward
   variant instead of one split.
5. Later: a second, less liquid pool, and cross-market transfer (v2).

## Sections

1. Abstract, introduction, related work (market efficiency, AMM
   microstructure, learning-based trading).
2. Prior attempt and why it failed (see below).
3. Data: on-chain Uniswap v3 swap history and how it is ingested (see below).
4. Method: bars, point-in-time features, triple-barrier net-return labels,
   model, chronological splits, backtest accounting.
5. Results, including controls and null findings.
6. Threats to validity: cost-prediction confound, gaps in the data, regime
   shift, single pool, simplified slippage model, capital held for the full
   horizon in the backtest.
7. Limitations and future work.

## Section: the prior attempt

Document the trajectory from the first concept, and why it failed.

- First concept (2026-09-23): a Solana order-book bot. Phoenix order-book
  snapshots in, Jupiter for execution, a safety guard for sizing and daily
  loss limits.
- The model was Laya (`aac6fef/laya-mlx`), a general-purpose typed-decision
  text classifier, served locally. It was asked one question, whether now is a
  good moment to trade, given a text line with best bid, best ask and depth
  counts. It traded when its confidence passed a threshold.
- The backtest used CoinGecko history, which has no order-book data, so the
  model was fed a synthetic bid/ask ladder generated from spot price plus a
  fixed cost in basis points. It was tested on fabricated books that were a
  deterministic function of price.
- Result: 23% win rate, worse than buy-and-hold (recalled, no result file
  survived the rewrite; see open items).
- Root cause: the model had never seen financial time series, had no learned
  mapping from these inputs to price behaviour, and answered "trade"
  almost regardless of input. The inputs also carried no predictive
  information, and the snapshot had no history.
- Restart (2026-09-24, commit `39efa1e`): a purpose-trained model, the
  venue's own on-chain data, net-of-cost labels, a chronological split with
  a test window used once.
- Framing: a negative result with a cause. It shows the failure mode of
  applying a general model to numeric market data and motivates each design
  rule of the second attempt.

Open items:
- Re-run the old backtest from the pre-rewrite commit in a scratch worktree
  for measured numbers, if the model still installs. Planned for later.
  Otherwise cite the recalled figures and say so.
- Record why Laya was chosen in the first place.

## Section: ingestion

Needs a solid write-up of its own, since the data are the foundation.

- Source: `eth_getLogs` for Swap events from an archive RPC endpoint,
  decoded from `sqrtPriceX96`, with tick, in-range liquidity, amounts, and
  the block base fee.
- Disjoint-interval progress tracking and gap-fill, so a block is never
  fetched twice.
- Atomic, resumable writes; flush before raising on failure; the training
  step refuses data with holes.
- Rate limiting: from concurrency slots to a proactive requests-per-second
  limiter that learns a safe rate, backs off to the last known-good rate,
  self-raises its ceiling, and persists what it learned.
- Newest-to-oldest fetch order: any interruption leaves one continuous
  stretch anchored at the newest block (or at data already on disk), so
  the partial dataset is immediately trainable and no run is wasted.
  Training refuses non-continuous data.
- Locating label entry and exit swaps by binary search instead of rescanning.
- Sources: the ingest and rate-limiting design specs and the git history.

## Constraints on the writing

- No tooling or process fingerprints in anything committed.
- Report null and negative results as results.
- No claim of an edge without out-of-sample evidence.
