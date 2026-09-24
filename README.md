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

## Open questions

- Data source and resolution for training.
- What exactly gets labeled / predicted, and over what horizon.
- How much a learned signal actually transfers across markets/pairs — an
  assumption to test, not take for granted.
- v1 scope: which single pool/pair to start with.

## Non-goals (for now)

- Real money. This stays a research/backtest project until there's real
  out-of-sample evidence of an edge.
- Multi-chain/multi-venue generality. One pair on one venue first.
