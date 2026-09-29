# Configurable pool (idea, not yet specced)

Goal: let other users run latentedge against a Uniswap pool other than the
hardcoded WETH/USDC 0.05%, without editing code.

## What is tied to the current pool

- `config.py`: `POOL_ADDRESS`, `TOKEN0/1_SYMBOL`, `TOKEN0/1_DECIMALS`,
  `FEE_TIER_BPS`, `POOL_CREATION_BLOCK`.
- `bars.py`: uses `TOKEN0_DECIMALS`, column named `volume_usdc`.
- `labeling.py`, `executor.py`: fee from `FEE_TIER_BPS`.
- `uniswap_math.py`, `costs.py`: assume token0 is the USD quote and token1
  is WETH; the price inversion depends on it.
- Default data/model paths (`data/swaps.parquet`, `data/model.safetensors`)
  are pool-agnostic, so two pools would overwrite or mix each other's data,
  and a model could be backtested against the wrong pool.

## Proposed change

1. `LATENTEDGE_POOL_ADDRESS` env var (plus matching `--pool` flag). Default
   stays the WETH/USDC pool.
2. Discover token0/token1/fee/decimals/symbols from the pool contract via
   `eth_call` instead of hardcoded constants. Find the creation block with
   the same `eth_getCode` binary search used for the current value, and
   cache it after the first run.
3. Replace the module constants with a runtime `PoolConfig` dataclass built
   once at startup and passed to bars, labeling, costs and the executor.
4. Remove the "token0 = USDC" assumption: choose the quote token via
   `LATENTEDGE_QUOTE_TOKEN` or auto-pick a known stablecoin. Rename
   `volume_usdc` to `volume_quote`.
5. Namespace defaults per pool: `data/<pool-short-address>/swaps.parquet`
   and `.../model.safetensors`. Store the pool address in parquet and model
   metadata, and make `train`/`backtest` refuse a mismatch.

## Open question

Scope of the first version: stablecoin-quoted pools only (WETH/USDC,
WBTC/USDT, ...) or arbitrary pairs? Arbitrary pairs need a USD conversion
for `REFERENCE_NOTIONAL_USD` and gas costs. Leaning stablecoin-quoted only,
since cross-market transfer is already a v2 item.

## Next step

Write a design and an implementation plan before implementing.
