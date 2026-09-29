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
backtest, is the pipeline's result (the backtest is one of its rules).

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

## Results summary (2026-09-29): a null result

Setting: WETH/USDC 0.05% on Ethereum mainnet, about 200 days of swaps
(blocks 24,642,884 to 26,084,902), 1-minute bars, 30-minute triple-barrier
labels, chronological 70/15/15 split, test window about 30 days (43k bars).
Cited runs are in `docs/paper/results/` (see the experiment log below).

Finding: no tradable edge, and the cause is measurable.

1. What the first model appeared to learn was trading cost. Trained on net
   return, its predictions correlated 0.35 with the test label but +0.01
   with the pre-cost price move and -0.96 with the cost part of the label.
   Net return mixes direction and cost, cost is the easier part to predict,
   so a model can fit the label without knowing anything about direction.
2. Trained directly on the pre-cost return, the model finds a faint
   direction signal: gross correlation +0.04 (validate) and +0.03 to +0.04
   (test), positive on every split. That is about 0.1% to 0.2% of the
   variance of the move, and inside sampling error (labels overlap, so the
   effective sample is about 1,400 independent points per window and the
   standard error is about 0.03).
3. More information and more capacity did not change it. Adding four
   order-flow features (signed net buying, largest-swap share) and a
   64,64 network with early stopping, in place of one 16-unit layer, gave
   the same gross correlation (+0.04 validate, +0.03 test). Two very
   different models reading the same inputs agree, which points at the
   inputs, not the model.
4. No trading rule made money on either window in any run. The rank rules
   (trade the model's top share of bars) lose about as much per trade as
   trading every bar (about -$0.83 on a $1,000 reference trade) and no
   less than shuffled predictions in the cases that matter; the minimum-edge
   rule (predicted move above the round-trip cost) takes zero to three
   trades because no predicted move approaches the cost.
5. The pipeline can find a signal when one exists. On synthetic swaps with a
   planted drift it recovers gross correlation 0.82 to 0.85 on unseen data,
   with order flow alone it finds a signal that price alone barely shows
   (0.18 to 0.27 against 0.07 to 0.13), and on a pure random walk it finds
   0.02 to 0.03. The real-data figures sit at the random-walk level.

Why a small correlation cannot be traded here (back of envelope, normal
predictions): a trade taken at prediction z-score z has expected gross move
about rho * sigma * z, with rho the correlation and sigma the spread of the
move (0.17% on the test window). Round-trip cost is about 0.13% (roughly
0.10% pool fees, the rest gas and slippage), so a trade needs rho * z of
about 0.76. Selecting only the top 1% of predictions (mean z about 2.7)
still needs rho of about 0.29, against 0.04 observed. Even perfect foresight
of every trade returned only +15.5% to +19% over a window (oracle rows),
because typical moves are small against the cost.

Reading: the result matches market efficiency for a liquid, heavily
arbitraged pool and public inputs. Easy patterns in what every participant
sees are traded away. What the experiments add is the decomposition: the
apparent skill under a net target was cost prediction, the direction signal
under a gross target is small and not distinguishable from noise, and neither
order flow nor model size moves it.

Not tested, so not claimed: horizons beyond 30 minutes (the arithmetic above
says a 4-hour horizon needs rho of about 0.27 at z of 1 and still far above
0.04 for a rank rule), other pools, other bar sizes, other model families
than a small feed-forward network, walk-forward evaluation, several seeds,
data outside these 200 days, and information not in the swap logs (other
venues, mempool).

## Experiments the paper needs

Done (2026-09-29):

1. Backtest on the test window only.
2. Always-trade baseline and oracle ceiling, through the same backtest.
3. Rank rule (top share of bars by predicted return, cutoff set on the
   validation window), which works even when no prediction is above zero.
4. More data: about 200 days ingested. Not yet reported as a function of
   history length.
5. Gross-return check: pre-cost label, and the correlation of predictions
   with gross return and with cost, in training metrics and in every sweep.
6. Shuffled-predictions control in the sweep.
7. Gross (pre-cost) training target, in place of net.
8. Order-flow features (signed flow imbalance, largest-swap share), tested
   against the price-only model.
9. Larger network (64,64) with early stopping, tested against the 16-unit
   model.
10. Planted-signal validation of the pipeline on synthetic data (linear
    drift, order-flow-led drift, combined pattern, pure random walk).

Still needed:

1. Longer label horizon (hours; `LATENTEDGE_LABEL_HORIZON_MINUTES`, with the
   band widened through `LATENTEDGE_LABEL_BARRIER_STDS`). The arithmetic in
   the results summary says it is unlikely to change the conclusion; run it
   so the claim is measured, not argued.
2. Retrain without gas price and volatility. With the gross target this is a
   control, not a route to an edge.
3. Robustness: several seeds, more than one horizon, a walk-forward variant
   instead of one split, results as a function of history length.
4. A second, less liquid pool, and cross-market transfer (v2).

Sweep records live under `data/sweeps/` (gitignored). The ones the paper cites
are copied to `docs/paper/results/sweeps/`, with a README describing each; the
files carry the model hash and git state of the run.

## Experiment log

One line per run, appended as they finish. Each run is one trained model
(own `LATENTEDGE_TRAIN_OUT`) and one saved sweep; record the sweep file, the
label settings and the feature set with it.

| Sweep file | Features | Horizon, band | Direction (gross corr, test) | Cost corr (test) | Result |
|---|---|---|---|---|---|
| 20260929T173520Z | all 7 | 30 min, 2 std | not measured | not measured | no edge (first sweep) |
| 20260929T175549Z | all 7 | 30 min, 2 std | +0.01 | -0.96 | no edge; model predicts cost |
| 20260929T183041Z | all 7 + 4 order flow | 30 min, 2 std | +0.01 | -0.96 | no edge; order flow added nothing measurable (validate gross corr +0.03) |
| 20260929T184128Z | all 7 + 4 order flow, trained on gross | 30 min, 2 std | +0.04 validate, +0.04 test | +0.10, +0.12 | no edge; first positive direction correlation on all splits but tiny (about 0.2% of variance), inside sampling error, and every rule loses |
| 20260929T185051Z | as above, 64,64 network, early stopping | 30 min, 2 std | +0.04 validate, +0.03 test | +0.04, +0.02 | no edge; a bigger network changes nothing (same gross correlation as the 16-unit model), so capacity was not the limit; every rule loses |

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
about 0.07 and 0.13. Real-data run (`20260929T183041Z`, block range 24,642,884 to 26,084,902, about
200 days, 250 requested; the ingest was stopped before the extra 50 days were
fetched): gross correlation 0.006 train, 0.025 validate, 0.011 test, the same
as the price-only model; the fit to net return is still cost prediction
(cost correlation -0.93 to -0.96). Order flow at 1-minute bars and a 30-minute
horizon adds no detectable direction signal. The price-only baseline is
reproduced with `LATENTEDGE_TRAIN_EXCLUDE_FEATURES=order_flow`.

Model capacity (`LATENTEDGE_TRAIN_HIDDEN`, early stopping): the original model
is one hidden layer of 16 units, close to linear. Network size is now
configurable, and training stops when the validate loss stops improving and
restores the best epoch. On a synthetic combined pattern (drift sign is the
product of a flow regime and a volatility regime), a 64,64 network without
early stopping fit training data almost perfectly (gross correlation 0.71) but
did worse on unseen data (0.13) than the 16-unit model (0.19); with early
stopping the sizes 16, 64,64 and 128,64 all reach about 0.21 on validate. The
early-stopping point uses the validate window, so validate results are
slightly optimistic and the test window is the judge.

Planned: all features at 240 minutes with a 6 std band; without gas and
volatility at 30 minutes (a control under the gross target).

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
- At that rate, ingestion settles at roughly 1,600 to 1,800 blocks/min (measured from the ingest log; the earlier "30 blocks/min" figure was wrong). Report how long
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

## Walk-forward study result (2026-09-29, late)

`latentedge study`, 200 days, last half in five walk-forward folds, five seeds,
64,64 network, gross target. Pooled out-of-sample gross correlation: 30 min
+0.034 [+0.023, +0.045], p=0.001, all five folds positive, seeds 0.027 to 0.032;
120 min +0.021 [+0.002, +0.040], p=0.047; 240 min +0.014 [-0.010, +0.036],
p=0.34. Top 1% of bars at 30 min: gross +0.012% against cost 0.148%, net
-0.136%. The paper's claim changed from "no detectable signal" to "a real,
faint signal that is a twelfth of the cost of trading it". Record copied to
`docs/paper/results/studies/`.
