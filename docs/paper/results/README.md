# Cited results

Sweep records the paper cites, copied from `data/sweeps/` (which is
gitignored). Each file holds the scenario rows for both windows, the training
metrics with the split correlations, the block ranges ingested, the model's
hash and the git state the run was made from.

All runs: WETH/USDC 0.05%, blocks 24,642,884 to 26,084,902 (about 200 days),
1-minute bars, chronological 70/15/15 split, $10,000 start, 10% maximum
position.

| File | Model | What it shows |
|---|---|---|
| `20260929T173520Z.json` | 7 price features, 16 units, net target | First sweep: no edge; rank rules lose the same per trade as always trading. Run before the gross/cost diagnostics existed, so it has no shuffled rows. |
| `20260929T175549Z.json` | same model | Adds shuffled control and the gross/cost split: gross correlation +0.03 validate, +0.01 test; cost correlation about -0.96. Git state recorded as dirty (the run was made from uncommitted changes to the commit named in the file). |
| `20260929T183041Z.json` | plus 4 order-flow features, net target | Order flow adds no detectable direction signal (gross correlation +0.03, +0.01). |
| `20260929T184128Z.json` | same features, gross target | Model no longer tracks cost; gross correlation +0.04 validate, +0.04 test; every rule loses. |
| `20260929T185051Z.json` | 64,64 network, gross target, early stopping | Same gross correlation as the 16-unit model (+0.04, +0.03); capacity is not the limit. |

Not cited: `20260929T180148Z` (a re-run of the `20260929T175549Z` model with
identical results).

## Walk-forward study

`studies/20260929T192107Z.json`: the robustness study (`latentedge study`) at
horizons of 30, 120 and 240 minutes, five seeds, five walk-forward test folds
over the last half of the same history, 64,64 network, gross target, all eleven
features, 1,000 bootstrap and 1,000 permutation resamples. Holds every
(horizon, seed, fold) row and each horizon's pooled summary: gross correlation
with block-bootstrap interval and permutation p-value, correlation by seed and
by fold, and the mean gross, cost and net return of the top slices against all
bars. The config and the git state of the run are recorded in the file.
