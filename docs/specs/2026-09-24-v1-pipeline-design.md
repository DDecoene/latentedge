# latentedge v1 pipeline — design spec

Status: draft, awaiting review.

## 1. Goal and scope

Build the first end-to-end loop of the project: pull real on-chain history
for one Uniswap v3 pool, turn it into labeled training data, train a
purpose-built model, and backtest that model's trading decisions honestly
(wallet, fees, slippage, P&L, win-rate, equity curve). No real money moves
in v1 — see README non-goals. The goal of v1 is to prove the *pipeline* is
correct and honest, on the hardest, cleanest market available (WETH/USDC,
0.05% fee tier), before ever asking whether it's profitable somewhere an
edge is more plausible.

Decisions already locked in (from README, not reopened here):
data source (on-chain Uniswap v3 swaps for the target pool, resampled to
1-minute bars), label (triple-barrier net-of-fees/slippage outcome,
30-minute horizon), pool (WETH/USDC 0.05%), local MLX inference.

## 2. Stack

One language, one process: **Python throughout**, MLX for the model. No
Rust, no second runtime, no HTTP boundary between components — this was a
deliberate reversal of an earlier draft of this spec that split trading
logic into Rust; see the rationale below.

Why single-language, and why Python specifically over single-language
Rust:

- **MLX is Python-first.** It's Apple's own tuned framework for Apple
  Silicon's unified memory architecture — the best-fit choice already
  locked in for local Mac inference. MLX has no mature Rust bindings; an
  all-Rust stack would mean giving up MLX for a less-tuned Metal backend
  (`candle`/`burn`), which cuts against not scoping down ML rigor.
- **The project's real goal is learning ML.** Nearly all ML literature, examples,
  and debugging help assume Python. An all-Rust stack is the path more
  likely to hit an ecosystem wall and force exactly the kind of "scrap and
  rebuild" the user explicitly wants to avoid — Rust's ML training
  ecosystem is real but thin, with a slower iterate-and-inspect loop than
  Python's during the feature-engineering phase, which is where the most
  iteration will happen.
- **The strongest reason for Rust — real money on the line — doesn't
  apply to v1** (non-goal). Rigor for the money-handling logic (safety-
  guard, backtest accounting) is instead maintained through strict typing
  (type hints + mypy in strict mode, or pydantic models for the
  wallet/trade data structures) and the TDD discipline already standard
  on this project. If v1 ever produces real out-of-sample evidence of an
  edge, hardening the by-then well-understood trading-logic slice in Rust
  is a legitimate future step — a small, justified rewrite of known-good
  code, not a rebuild of the unknown parts.

## 3. Components

```
  Uniswap v3 pool  ───▶  ingest ─▶ bars ─▶ labels ─▶ train ─▶ trained model
  (subgraph / RPC)                                                 │
                                                                    ▼
                                                          signal client
                                                                    │
                                                                    ▼
                                                            safety-guard
                                                                    │
                                                                    ▼
                                                              executor
                                                                    │
                                                                    ▼
                                                         backtest harness
                                              (wallet, P&L, win-rate,
                                               equity curve, fills)
```

All of the above runs as ordinary in-process Python function calls —
signal client calling the loaded MLX model directly, no server, no
serialization boundary. The pipeline is a script/CLI (or a small set of
them: `ingest`, `train`, `backtest`), not a set of services.

### 3.1 Ingestion (Python)

Pulls raw `Swap` events for the WETH/USDC 0.05% pool. Two candidate
sources, tried in this order for whichever proves simpler against the real
service — this needs to be proven against the real API, not assumed:

1. A Uniswap v3 subgraph (via The Graph's decentralized network, queried
   over HTTP) — simplest if it has full history for this pool and doesn't
   rate-limit unreasonably.
2. Direct `eth_getLogs` against an archive RPC provider (e.g. Alchemy free
   tier) as a fallback if the subgraph is incomplete, rate-limited, or
   unreliable — decode `Swap` events and reconstruct price from
   `sqrtPriceX96` directly.

Output: raw swap records written to local Parquet files — no database in
v1, flat files are enough for a single pool's history. Each record
captures, at minimum: block, timestamp, `sqrtPriceX96`, `tick`,
**`liquidity`** (the pool's in-range active liquidity, emitted as part of
the same `Swap` event — no separate `Mint`/`Burn` ingestion needed for
v1's slippage estimate), swap amounts, and tx hash. Also capture each
block's **base fee** (from the block header, via the subgraph if exposed
there, otherwise a cheap separate `eth_getBlockByNumber` call) — needed
for the gas-cost accounting in 3.3/3.7.

### 3.2 Bar construction (Python)

Resamples irregular swaps into fixed 1-minute bars: time-weighted average
price over the minute (last known price carried forward through gaps),
swap count, and volume (token0/token1 amounts summed). This is the AMM
analogue of OHLCV — no open/high/low in the CEX-candle sense since a
1-minute window may contain zero or many swaps; document this explicitly
rather than faking OHLC from sparse data.

### 3.3 Labeling (Python)

Triple-barrier labeling per bar `t`: simulate entering a trade at `t`,
track forward until either (a) take-profit hit, (b) stop-loss hit, or (c)
the 30-minute time limit expires, whichever comes first. Label is a
**continuous value: realized net return** (not a binary profitable/not
classification) — entry/exit price adjusted for the pool's 0.05% fee on
both legs, an estimated slippage from trade size vs. pool depth at that
block, and an estimated gas cost. Regression over classification because
P&L is what actually matters: a model that ranks "definitely profitable
but small" below "marginally profitable but large" on a binary label would
look identical to one that gets the magnitude right, and magnitude is
exactly what position sizing (safety-guard) needs to act on.

**Reference trade size:** labels are computed against a fixed reference
notional (**$1,000**), independent of whatever the safety-guard later
sizes a real position at. This breaks an otherwise circular dependency —
labels are baked in during training data prep, before any position-sizing
decision exists. The backtest executor (3.7) recomputes its own slippage
at the safety-guard's actual chosen size using the same liquidity data;
the label and the backtest fill can differ in size, and that's expected —
the label defines what a *standard* trade would have earned, not what the
eventual backtest necessarily trades.

**Slippage estimate:** derived from the `liquidity` value captured in 3.1
(in-range active liquidity at that block), treated as locally constant
across the reference trade size — a concentrated-liquidity approximation,
not exact tick-crossing math (see non-goals). Good enough to distinguish
"deep, liquid moment" from "thin, risky moment"; not a claim of exact
execution-price prediction.

**Gas cost estimate:** the block's base fee (captured in 3.1) times a
fixed gas-used constant for a single-hop Uniswap v3 `exactInput` swap
(**150,000 gas**, an approximate industry-standard figure for a
single-tick-range swap), applied on both entry and exit legs. Ignoring gas
would overstate profitability for a strategy that can trade as often as
every 30 minutes — not acceptable for a project that explicitly commits to
honest evidence before claiming an edge.

**Stale-price fix (entry and exit):** because 3.2's bars carry the last
known price forward through gaps with no swaps, computing a fill —
entry *or* exit — against a *resampled bar* price risks pricing a trade at
a level nothing actually traded at. Both legs are anchored to **real swap
prices only**, never a bar's carried-forward value:
- **Entry**: the next actual swap price at or after `t`. If no swap occurs
  before the 30-minute time limit, there is no valid entry fill and the
  bar is excluded from training (consistent with 3.2/4's handling of
  low-activity gaps).
- **Exit**: barrier checks walk the underlying raw swap events in the
  forward window and compare against real traded prices only — a quiet
  pool can never spuriously hit a take-profit/stop-loss barrier that
  nothing actually traded through.

Take-profit/stop-loss thresholds are a parameter, not hardcoded — start
with a symmetric band sized from the pool's realized volatility (e.g. some
multiple of the trailing N-bar standard deviation of returns) so the label
isn't tuned to one arbitrary number. Exact multiple is a training
hyperparameter, tuned only against the validation split (3.4) — never
against the final held-out test window.

### 3.4 Feature engineering + training (Python/MLX)

v1 feature set is deliberately small and standard, not creative: rolling
returns over a few windows, rolling volatility, volume, and time-since-last-
swap (a real AMM-specific signal — a stale pool behaves differently than
an active one). Every feature is computed **point-in-time**: a bar's
features use only data available as of that bar's timestamp — no centered
windows, no peeking at future bars, no exception. This is what "no
shuffling across time" actually requires at the feature level, not just at
the split level.

A regression model (predicting net return, not a profitable/not class),
trained to minimize error against the continuous label from 3.3.

**Three-way, chronological split** — train / validate / test, each a
strictly later time window than the last, never shuffled:
- **Train**: fit model parameters.
- **Validate**: tune hyperparameters (TP/SL multiple, feature-window
  lengths). This set can be touched repeatedly during development.
- **Test**: touched exactly once, at the very end, for the numbers that
  actually get reported and the backtest in 3.8. If test-set performance
  ever influences a decision (a hyperparameter change, a feature change),
  it has silently become a second validation set and stops meaning
  anything — the project's honest-evidence commitment depends on keeping
  this boundary real, not nominal.

Report out-of-sample metrics honestly from the test set only (regression
error, e.g. MAE/RMSE on held-out net return, and *also* the resulting
backtest P&L — a model can have low error and still lose money after real
costs, and the backtest is the metric that actually matters).

### 3.5 Signal client (Python)

Loads the trained MLX model once, holds it in memory for the run. Given a
bar's feature vector, runs the model's forward pass directly (in-process,
no server, no HTTP) and returns a predicted net return (a scalar, not a
class probability). The safety-guard/executor use this magnitude directly
for position sizing, not just a directional signal. For a full backtest,
predictions are computed as
one batched forward pass over the whole feature matrix rather than one
call per bar — avoids per-call overhead entirely and keeps a 500k+ bar
backtest fast without needing a separate optimization pass later.

### 3.6 Safety-guard (Python)

Carried over from the architecture shape already decided: position sizing
and daily-loss lockout, enforced even in backtest mode. Enforcing it in
backtest is what lets the backtest results reflect what a real deployment
would actually have done, not an idealized unconstrained strategy — and
it's directly exercisable in a test (a previous version of this guard had
a bug where it never reset and silently froze a 90-day backtest at day 54;
this needs an explicit test that runs a backtest long enough to cross a
guard-reset boundary). Implemented with strict type hints / pydantic
models for its state and inputs, since this is the money-handling logic
where the rigor Rust would have bought needs to come from discipline and
tests instead.

### 3.7 Executor (Python)

Backtest mode only in v1: given a signal and the safety-guard's actual
sizing decision (which may differ from the label's $1,000 reference
notional — see 3.3), apply a simulated fill against the pool's fee tier
plus a **slippage and gas cost recomputed at the real trade size**, using
the same liquidity/base-fee data and formulas as 3.3, not the label's
precomputed values. No live tx signing, no real wallet — that's out of
scope until (if ever) there's real out-of-sample evidence of an edge
(README non-goals).

### 3.8 Backtest harness (Python)

Tracks wallet balance, per-trade P&L, running win-rate, and an equity
curve over the full backtest window. Runs against the **held-out test
window only** (3.4) — never the train or validate windows, so the
reported numbers are the same untouched out-of-sample evidence as the
regression metrics, not a second look at data already used for tuning.
Must be able to answer, at minimum: total return, max drawdown, win-rate,
number of trades, and Sharpe-like risk-adjusted return. Output as both a
machine-readable summary (JSON) and a human-readable chart, continuing the
pattern from the prior version's TUI/chart work (same visual intent,
reimplemented in Python — e.g. `plotext`/`rich` for a terminal chart, or a
simple matplotlib output).

## 4. Error handling

- Ingestion: if a data source is unavailable or rate-limited, fail loudly
  and stop — no silent partial-history runs. Partial data silently used as
  if complete is worse than an explicit failure.
- Bar construction / labeling: any bar with insufficient underlying swap
  data (e.g. a multi-hour gap) is flagged and excluded from training, not
  interpolated across silently.
- Signal client: if the model fails to load, or a feature vector is
  malformed/out of expected range, raise rather than return a fallback
  prediction — the caller must treat "no prediction" as "skip this bar,"
  never default to a directional guess.
- Safety-guard: lockouts and sizing limits are hard stops, not
  soft/advisory — this was the whole point of carrying the component over.

## 5. Testing strategy

TDD throughout, and — as important — real integration tests against real
external services where practical, not just mocks:

- Ingestion: an integration test that actually queries the chosen data
  source for a small known block range and checks the decoded output
  against a hand-verified expected swap (catches source-specific quirks
  early, the way the CoinGecko User-Agent 403 was only caught by hitting
  the real API).
- Bar construction / labeling: unit tests with synthetic swap sequences
  covering edge cases (no swaps in a minute, one huge swap, price gap
  across the take-profit/stop-loss band mid-bar).
- Signal client: a test that loads a real trained (small/fixture) model
  and runs an actual forward pass, not a mocked prediction — this project
  has already been bitten once by a gap between "the mock passes" and "the
  real thing runs" (the CoinGecko User-Agent 403, the uvicorn-vs-python
  distinction), and single-process doesn't remove that risk, it just moves
  it to "does the model actually load and predict."
- Safety-guard: a test that runs a backtest long enough to cross a
  daily-loss-lockout reset boundary and asserts trading resumes correctly
  (regression test for the bug already found once).
- Backtest harness: an end-to-end replay test (continuing the existing
  `c2c164c` pattern from before) over a small fixed historical window with
  hand-computable expected P&L.

## 6. Non-goals (v1)

- Real money / live execution (README).
- Multi-pool or multi-chain generality (README).
- Cross-market signal transfer (README, deferred to v2).
- A database — flat Parquet files are enough for one pool's history.
- A model-serving process/API — single in-process pipeline, single user.
- Feature-set sophistication — the small, standard feature set above is
  the v1 baseline; expanding it is future work once the pipeline is
  proven end-to-end.
- Exact tick-crossing slippage math. The concentrated-liquidity
  approximation in 3.3/3.7 (local liquidity treated as constant across the
  trade) is deliberately not exact AMM price-impact math — good enough to
  rank "deep" vs. "thin" liquidity moments for a $1,000-scale reference
  trade; revisit if position sizes grow large enough for tick-crossing to
  matter.

## 7. Open questions carried forward

- Cross-market signal transfer (README) — untouched by this spec,
  deliberately deferred to v2.
- Exact take-profit/stop-loss multiple and feature-window lengths are
  training hyperparameters, tuned empirically during implementation against
  the validation split only (3.4) — never the held-out test window.
