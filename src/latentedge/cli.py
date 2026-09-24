from pathlib import Path

import click
import httpx
import pandas as pd

from latentedge import config
from latentedge.bars import build_bars
from latentedge.features import compute_feature_stats, save_feature_stats, standardize_features
from latentedge.ingest.chunked import DEFAULT_CHUNK_SIZE, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BACKOFF_SECONDS, ingest_range
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as train_model
from latentedge.split import chronological_split
from latentedge.store import read_swaps
from latentedge.training_data import FEATURE_COLUMNS, assemble_training_data


@click.group()
def cli() -> None:
    """latentedge: ingest, train, and backtest the v1 WETH/USDC pipeline."""


DEFAULT_RPC_URL = "https://ethereum.publicnode.com"


@cli.command()
@click.option("--from-block", type=int, required=True)
@click.option("--to-block", type=int, required=True)
@click.option("--rpc-url", type=str, default=DEFAULT_RPC_URL, help="An archive-capable RPC endpoint for ranges reaching back further than a few hours (e.g. an Alchemy/Infura URL) — free public endpoints gate deep history behind a paid token.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Blocks per eth_getLogs call — keep well under your provider's per-call limit.")
@click.option("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
@click.option("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
def ingest(from_block: int, to_block: int, rpc_url: str, out: Path, chunk_size: int, max_retries: int, retry_backoff_seconds: float) -> None:
    # A large range (e.g. a year of history) needs chunking to respect
    # provider limits and resumability to survive a multi-hour run being
    # interrupted — see ingest.chunked for the real logic; fetch_swaps
    # already populates base_fee_wei from the same eth_getBlockByNumber
    # call it makes for each block's timestamp, no separate backfill pass.
    out.parent.mkdir(parents=True, exist_ok=True)

    def report(chunk_start: int, chunk_end: int, count: int) -> None:
        click.echo(f"  blocks {chunk_start}-{chunk_end}: {count} swaps")

    with httpx.Client(timeout=30.0) as client:
        total = ingest_range(
            config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
            chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
            on_progress=report,
        )
    click.echo(f"wrote {total} new swap records to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
@click.option("--epochs", type=int, default=100)
def train(swaps: Path, out: Path, epochs: int) -> None:
    swap_df = read_swaps(swaps)
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)

    # Take-profit/stop-loss band sized from the pool's own realized
    # volatility (spec 3.3), not a fixed guess — a symmetric band at
    # 2x the trailing 1-bar return std-dev.
    bar_return_std = bar_df["price_usdc_per_weth"].pct_change().std()
    tp_sl_fraction = max(bar_return_std * 2, 0.001) if pd.notna(bar_return_std) else 0.01

    assembled = assemble_training_data(
        bar_df, swap_df, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=tp_sl_fraction
    )

    train_split, _validate_split, _test_split = chronological_split(assembled, train_fraction=0.7, validate_fraction=0.15)

    # Standardize using train-split statistics only — computing stats
    # from validate/test data would leak information about those splits
    # into training. The same stats are saved alongside the model so
    # SignalClient applies an identical transform at inference time.
    stats = compute_feature_stats(train_split, FEATURE_COLUMNS)
    train_split = standardize_features(train_split, FEATURE_COLUMNS, stats)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")

    regressor = NetReturnRegressor(input_dim=len(FEATURE_COLUMNS))
    losses = train_model(regressor, x, y, epochs=epochs, learning_rate=0.001)

    out.parent.mkdir(parents=True, exist_ok=True)
    save(regressor, out)
    save_feature_stats(stats, Path(str(out) + ".stats.json"))
    click.echo(f"trained {epochs} epochs, final loss {losses[-1]:.6f}, tp_sl_fraction={tp_sl_fraction:.5f}, saved to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
def backtest(swaps: Path, model: Path) -> None:
    # Intentional stub (see the implementation plan): assembling the real
    # entry/exit swap arrays run_backtest needs is its own piece of glue
    # work. A non-zero exit avoids this reading as a successful backtest.
    raise click.ClickException("backtest command not implemented yet — see backtest.run_backtest for the underlying logic")
