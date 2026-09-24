# (unnamed) — an ML pattern-recognition trading project

## What this actually is

The real goal is learning ML — this is a toy project, not a revenue target. But
it's being pursued as if it were a genuine attempt at a live trading edge: no
scoping down the ML rigor (data quality, labeling discipline, walk-forward
validation, honest out-of-sample evaluation) just because it's "only" a
learning exercise.

The thesis: there are signals in market data that only a model can pick up —
patterns that fit to real-world structure rather than an explicit rule a human
wrote down. We're trying to find one, and to build the muscle of doing ML
properly along the way.

## Decided so far

- **Execution venue: Uniswap.** Moving off Solana/Jupiter. This is an AMM, not
  an order book — price comes from pool reserves/tick math, not bid/ask
  levels, so the whole signal → decision → execution pipeline needs
  rethinking around that.
- **Not reusing the previous code.** The prior version (Solana/Jupiter +
  "Laya", a general-purpose text classifier misapplied to numeric market
  data) has been deleted. Its architecture shape is worth repeating —
  signal client / executor / safety-guard / backtest-harness split,
  position-sizing and daily-loss risk limits, wallet/P&L/equity-curve
  accounting in the backtester — but none of the integration code (order-book
  decoding, Solana tx signing, atom/decimal math) survives the move to an
  EVM/AMM target.
- **The model should be genuinely trained for this**, not a general-purpose
  model prompted with market data. Local inference on the Mac (MLX or
  similar) is the deployment target.

## Open questions

- What data source and resolution for training (see discussion — CEX minute
  klines were floated as training data, decoupled from the execution venue
  decision)?
- What exactly gets labeled / predicted, and over what horizon?
- What's the project/repo name (dropping "layatrade" since Laya is gone)?
- How much of "signal transfers across markets" is actually true here, and
  how do we test that assumption rather than assume it?
- What does v1 scope down to — a single Uniswap pool/pair to start?

## Non-goals (for now)

- Real money. This stays a research/backtest project until (if ever) there's
  real out-of-sample evidence of an edge.
- Multi-chain/multi-venue generality. Pick one pair on one venue first.
