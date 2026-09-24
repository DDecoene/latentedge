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
- **Label: net-profitable trade outcome, not raw price direction.** Triple-
  barrier labeling (take-profit / stop-loss / time-limit) over a 30-minute
  horizon, with the bracket outcome computed net of the pool's fee tier and
  an estimated slippage from pool depth at trade size. This ties the label
  to what a real trade would actually earn, matching the safety-guard/
  position-sizing already in the architecture.
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

## Non-goals (for now)

- Real money. This stays a research/backtest project until there's real
  out-of-sample evidence of an edge.
- Multi-chain/multi-venue generality. One pair on one venue first.
- Cross-market signal transfer. Not attempted until v1's single-pool
  pipeline is proven.
