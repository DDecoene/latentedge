import sys
from pathlib import Path

import click
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from latentedge import config
from latentedge.bars import build_bars
from latentedge.features import compute_feature_stats, save_feature_stats, standardize_features
from latentedge.ingest.chunked import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FLUSH_EVERY_N_CHUNKS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    ingest_range,
)
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as train_model
from latentedge.split import chronological_split
from latentedge.store import read_swaps
from latentedge.training_data import FEATURE_COLUMNS, assemble_training_data
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen
from latentedge.tui.train_screen import TrainScreen


@click.group()
def cli() -> None:
    """latentedge: ingest, train, and backtest the v1 WETH/USDC pipeline."""
    # Loaded fresh on every invocation (not just at import) so a .env
    # file in whatever directory the command is run from is picked up.
    # load_dotenv() with no path searches upward from this *file's*
    # location, not the process's cwd — an easy-to-miss python-dotenv
    # default that would silently ignore a .env in the user's actual
    # working directory. Search from cwd explicitly instead.
    load_dotenv(dotenv_path=Path.cwd() / ".env")


DEFAULT_RPC_URL = "https://ethereum.publicnode.com"


@cli.command()
@click.option("--from-block", type=int, required=True)
@click.option("--to-block", type=int, required=True)
@click.option(
    "--rpc-url",
    type=str,
    default=DEFAULT_RPC_URL,
    envvar="LATENTEDGE_RPC_URL",
    help="An archive-capable RPC endpoint for ranges reaching back further than a few hours (e.g. an Alchemy/Infura URL) — free public endpoints gate deep history behind a paid token. Falls back to the LATENTEDGE_RPC_URL env var (or a .env file) if omitted, so a provider key never needs to appear as a bare CLI argument.",
)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Blocks per eth_getLogs call — keep at or under your provider's per-call limit (10 on Alchemy's free tier).")
@click.option("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Concurrent chunk requests — at a small chunk size, a large range needs this to finish in a reasonable time.")
@click.option("--flush-every-n-chunks", type=int, default=DEFAULT_FLUSH_EVERY_N_CHUNKS, help="Batches writes to the output file — writing after every chunk would mean rewriting the whole file hundreds of thousands of times over a large range.")
@click.option("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
@click.option("--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS)
def ingest(
    from_block: int,
    to_block: int,
    rpc_url: str,
    out: Path,
    chunk_size: int,
    max_workers: int,
    flush_every_n_chunks: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    # A large range (e.g. a year of history) needs chunking to respect
    # provider limits, concurrency to finish in a reasonable time, and
    # resumability to survive a multi-hour run being interrupted — see
    # ingest.chunked for the real logic; fetch_swaps already populates
    # base_fee_wei from the same eth_getBlockByNumber call it makes for
    # each block's timestamp, no separate backfill pass.
    out.parent.mkdir(parents=True, exist_ok=True)

    if sys.stdout.isatty():
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=from_block, to_block=to_block,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds, ingest_fn=ingest_range,
            train_assemble_fn=lambda p: _assemble_train_data(p, Path("data/model.safetensors")),
        )
        LatentEdgeApp(start_screen=screen).run()
        return

    def report(chunk_start: int, chunk_end: int, count: int) -> None:
        click.echo(f"  blocks {chunk_start}-{chunk_end}: {count} swaps")

    with httpx.Client(timeout=30.0) as client:
        total = ingest_range(
            config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
            chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers, flush_every_n_chunks=flush_every_n_chunks,
            on_progress=report,
        )
    click.echo(f"wrote {total} new swap records to {out}")


def _assemble_train_data(swaps_path: Path, out_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    swap_df = read_swaps(swaps_path)
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
    save_feature_stats(stats, Path(str(out_path) + ".stats.json"))
    train_split = standardize_features(train_split, FEATURE_COLUMNS, stats)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    return x, y, len(FEATURE_COLUMNS)


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
@click.option("--epochs", type=int, default=100)
def train(swaps: Path, out: Path, epochs: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if sys.stdout.isatty():
        screen = TrainScreen(
            swaps_path=swaps, out_path=out, epochs=epochs,
            assemble_fn=lambda p: _assemble_train_data(p, out),
        )
        LatentEdgeApp(start_screen=screen).run()
        return

    x, y, input_dim = _assemble_train_data(swaps, out)
    regressor = NetReturnRegressor(input_dim=input_dim)
    losses = train_model(regressor, x, y, epochs=epochs, learning_rate=0.001)
    save(regressor, out)
    click.echo(f"trained {epochs} epochs, final loss {losses[-1]:.6f}, saved to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
def backtest(swaps: Path, model: Path) -> None:
    # Intentional stub (see the implementation plan): assembling the real
    # entry/exit swap arrays run_backtest needs is its own piece of glue
    # work. A non-zero exit avoids this reading as a successful backtest.
    raise click.ClickException("backtest command not implemented yet — see backtest.run_backtest for the underlying logic")
