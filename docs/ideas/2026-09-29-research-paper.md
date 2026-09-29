# Research paper (idea, not yet drafted)

Goal: write the project up as a scientific paper. The research question is
whether tradable edges still exist in a liquid on-chain market, and whether a
purpose-trained model can find them. A paper needs a full round trip: data,
method, experiments, and an honest conclusion, including a negative one.

Format: LaTeX source under `docs/paper/`, source files only, no new
dependencies. Plain research prose.

## Current state of the evidence (2026-09-29, evening)

First full round trip on WETH/USDC 0.05%: about 200 days of swaps (blocks
24,648,164 to 26,084,512), 1-minute bars, 30-minute triple-barrier net-return
labels, a small MLX regressor (7 features, effectively linear), chronological
70/15/15 split. Test window is about 30 days (43k bars).

Result so far: no tradable edge found. This is a null result and should be
reported as one.

- Regression fit is small but positive. Correlation with the label: train
  0.25, validate 0.23, test 0.35. MSE improvement over the mean baseline:
  about 7%, 6% and 12%. The test figure is again higher than validate, which
  looks like a regime difference, not skill.
- The model never predicts a positive net return. On the test window every
  prediction is negative (mean -0.126%, max -0.046%), so the plain rule
  "trade when the prediction is above zero" takes no trades and the backtest
  ends at exactly the starting equity.
- Sweep (saved in `data/sweeps/20260929T173520Z.json`, git state recorded in
  the file), every rule replayed on validate and test, $10,000 start, 10%
  max position:
  - Always trade: -45.4% validate, -44.6% test, about -$0.83 per trade
    (5.3k to 5.5k trades, win rate 26% and 34%).
  - Model's top 1% of bars by predicted return: -3.9% validate, -3.8% test,
    -$0.90 and -$1.23 per trade. Top 10%: -$0.85 and -$0.87 per trade. Top
    50%: -$0.76 and -$0.81 per trade.
  - The per-trade loss is the same in every slice. Total loss shrinks only
    because fewer trades are taken. The model's ranking has no per-trade
    value on this data.
  - Oracle (predicts each bar's realized net return): +19.1% validate, +15.7%
    test, only +$0.10 to +$0.14 per trade. Even perfect foresight earns
    little under about 0.13% round-trip cost (roughly 0.10% pool fees, the
    rest gas and slippage) against typical 30-minute moves near 0.4%.
  - Absolute-edge rows (predicted return above a minimum) take zero or two
    trades and carry no information.
- Cost-prediction confound, tested (`data/sweeps/20260929T175549Z.json`, same
  model as above): the label is net of gas and slippage and two features are
  gas price and volatility. Splitting each label into its pre-cost price
  move (gross) and its cost, the model's predictions correlate with gross
  return at +0.03 (validate) and +0.01 (test), and with cost at -0.95 and
  -0.96. The spread of the cost part is about a fifth to two-fifths of the
  gross part's, yet it accounts for nearly all of the correlation, so the
  0.35 test correlation with net return is cost prediction, not direction.
  The model carries no measurable information about the price move.
- Shuffled-predictions control (3 seeds, same trade counts): the model's
  ranked slices average about $0.18 per trade better than shuffled
  predictions on the test window (roughly -$0.85 against -$1.10 for the top
  10%), and every rule still loses. The gap is consistent with the model
  picking bars that are cheap to trade, which fits the finding above.

Pipeline note: training now records the feature set and label settings with
the model, and a backtest or sweep relabels with those, so a run cannot be
replayed under different settings by accident. The sweep, not the plain
backtest, is the pipeline's result (the backtest is one of its rules), and it
writes a plain-language verdict with its numbers.

Methodology note worth a paragraph: an earlier backtest lost 43% because the
signal client returned the model's standardized output (in standard
deviations) while the guard read it as a real return, so "predicted above
zero" meant "above the average training return". It was found because a
suspicious result was investigated, fixed in the signal client, and covered by
a regression test. Any result produced before that fix (including a -43%
backtest) is invalid and should not be cited.

## The narrative: what a learned pattern is, and why this one failed

Part of the story, and probably the paper's most useful teaching thread. The
working idea at the start: give a model data, let its weights adjust until a
pattern emerges, apply that pattern to unseen data to predict where the price
goes in the next window, and trade on the prediction. That describes the
mechanism correctly. What it leaves out, and what the project made concrete:

1. A model can only find a pattern that is in the data. Weights adjust toward
   whatever regularity exists and cannot create one. Shown directly: on
   synthetic swaps with a planted drift the pipeline recovers it (gross
   correlation 0.82 to 0.85 on unseen data); on a pure random walk it finds
   nothing on unseen data (0.02 to 0.03), while its fit on the training
   data alone was 0.19.
2. The pattern the model did learn was the wrong one. Gas price and
   volatility make a trade cheaper or dearer, which is easy to predict, so
   the model learned trading cost (correlation -0.96 with cost, +0.01 with
   the pre-cost price move). It raised the headline correlation with net
   return to 0.35 without saying anything about direction. A good-looking
   fit statistic was cost prediction, found only by splitting the label
   into price move and cost.
3. Unseen data is what separates a real pattern from a fitted one. Any
   flexible model fits noise on its training data; the chronological split
   is the instrument that exposes it (train 0.19, unseen 0.02 on the random
   walk).
4. Absence of a pattern here is the expected result, not a defect. The inputs
   were the last minutes of price and volume in the most watched pool
   on-chain, visible to every participant, and easy patterns in public
   inputs are traded away, which removes them. That is market efficiency in
   miniature.
5. A real pattern must also clear the trading cost. A round trip costs about
   0.13%, and even perfect foresight of every trade earned only +15.5% over
   the test window, so a weak signal is not enough; it has to be a strong
   one.

What follows for the design: the test was validated (the pipeline can find
direction when it exists), so the open question is the information, not the
machinery. Inputs others cannot act on quickly are the next candidates,
starting with signed order flow (direction and size of swaps), which price
bars discard. Every experiment is gated the same way: gross (pre-cost)
correlation must be clearly above zero on both validate and test before
costs matter. A null result under that gate, with the planted-signal check
behind it, is a result the paper can defend.

Wording note: claim "no detectable signal", not "no information". Labels
overlap (one 30-minute label per 1-minute bar), so the effective sample is
about 1/30 of the bar count and correlations under roughly 0.03 cannot be
told apart from zero.

## Experiments the paper needs

Done (2026-09-29):

1. Backtest on the test window only.
2. Always-trade baseline and oracle ceiling, through the same backtest.
3. Rank rule (top share of bars by predicted return, cutoff set on the
   validation window), which works even when no prediction is above zero.
4. More data: 200 days ingested. Not yet reported as a function of history
   length.

5. Gross-return check (done, see above): pre-cost label, and the
   correlation of predictions with gross return and with cost, in training
   metrics and in every sweep.
6. Shuffled-predictions control in the sweep (done, see above).

Still needed:

1. Retrain without gas price and volatility (`LATENTEDGE_TRAIN_EXCLUDE_FEATURES`)
   to remove the cost signal and see whether any direction is left.
2. Longer label horizon (hours; `LATENTEDGE_LABEL_HORIZON_MINUTES`), so a
   fixed cost is small against the move. The take-profit/stop-loss band must
   widen with it (`LATENTEDGE_LABEL_BARRIER_STDS`), since the current 2
   standard deviations of a 1-minute return is hit within minutes.
4. Richer inputs: order-flow imbalance and swap direction, and a non-linear
   model, tested against the same baselines.
5. Robustness: several seeds, more than one horizon, a walk-forward variant
   instead of one split, results as a function of history length.
6. Later: a second, less liquid pool, and cross-market transfer (v2).

Sweep records live under `data/sweeps/` (gitignored). Copy the ones the paper
cites into a versioned place before drafting, with the model hash and git
state each file already records.

## Experiment log

One line per run, appended as they finish. Each run is one trained model
(own `LATENTEDGE_TRAIN_OUT`) and one saved sweep; record the sweep file, the
label settings and the feature set with it.

| Sweep file | Features | Horizon, band | Direction (gross corr, test) | Cost corr (test) | Result |
|---|---|---|---|---|---|
| 20260929T173520Z | all 7 | 30 min, 2 std | not measured | not measured | no edge (first sweep) |
| 20260929T175549Z | all 7 | 30 min, 2 std | +0.01 | -0.96 | no edge; model predicts cost |

Pipeline validation (synthetic data, `tests/test_cli_backtest.py`): on swaps
whose price follows a hidden drift that flips sign every two hours, the same
pipeline (features, labels, net-return target, training, metrics) recovers
the planted signal, with gross correlation 0.82 (validate) and 0.85 (test).
On a pure random walk the same run gives 0.03 and 0.02. The real-data figures
(0.025 and 0.01) sit at the random-walk level. Caveat for the write-up: labels
overlap (a 30-minute label per 1-minute bar), so the effective sample is
roughly 1/30 of the bar count and a correlation must be about 0.03 or more
before it is distinguishable from zero. "No detectable signal" is the claim,
not "no information".

Order-flow features (added after the price-only null): per bar, signed dollar
flow (USDC paid in minus paid out, so net buying of WETH) and the largest
single swap; per window, `flow_imbalance_{5,15,30}` (net flow over volume,
-1 to +1) and `large_swap_share_15` (largest swap over the window's volume).
All use only past bars and are shifted one bar like the other features.
Validated on synthetic swaps where a regime tilts both trade direction and,
weakly, the price: with the lagged returns removed, the pipeline still finds
it (gross correlation 0.18 validate, 0.27 test), and a price-only model finds
about 0.07 and 0.13. Real-data run pending; the price-only baseline is
reproduced with `LATENTEDGE_TRAIN_EXCLUDE_FEATURES=order_flow`.

Planned: without gas and volatility at 30 minutes; all features at 240
minutes with a 6 std band; both together.

## Sections

1. Abstract, introduction, related work (market efficiency, AMM
   microstructure, learning-based trading).
2. Prior attempt and why it failed (see below).
3. Data: on-chain Uniswap v3 swap history and how it is ingested (see below).
4. Method: bars, point-in-time features, triple-barrier net-return labels,
   model, chronological splits, backtest accounting.
5. Results, including controls and null findings.
6. Threats to validity: cost-prediction confound, gaps in the data, regime
   shift, single pool, one 30-day test window, simplified slippage model,
   capital held for the full horizon in the backtest, an earlier unit bug in
   the signal path (fixed and tested).
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

Measured rate limit and ingest time (to be written up as a short subsection):

- Empirically, about 6.2 requests/sec is the sweet spot against the RPC
  provider. At or below it there are no HTTP 429 responses. Above it the
  provider throttles, and each 429 carries a wait time, so the time lost
  waiting costs more than the extra request rate gained. Net blocks per
  minute drops noticeably once 429s start.
- At that rate, ingestion settles at roughly 30 blocks/min. Report how long
  the full history took (and takes for the 150-day window and any extension),
  since it sets the practical cost of adding more data. Fill in the measured
  wall-clock time from the progress file before drafting.
- This is probably not the most efficient way to collect the data. Other
  methods may exist (a provider with bulk or export endpoints, a self-hosted
  node, a public dataset of Uniswap v3 swaps). `eth_getLogs` over an archive
  endpoint was the approach already known, and there was no time pressure,
  so a slow, safe ingest was acceptable. Say this plainly in the paper as a
  limitation of the data collection and not as a recommended method.
- Constraints behind the choice: this was the only truly free option known
  to the author, and running a node was not possible because of hardware
  and budget limits. The free-tier rate limit is therefore what sets the
  ingest speed.

## Constraints on the writing

- No tooling or process fingerprints in anything committed.
- Report null and negative results as results.
- No claim of an edge without out-of-sample evidence.
