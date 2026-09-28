import sys
import time
from pathlib import Path

import click
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from latentedge import config
from latentedge.bars import build_bars
from latentedge.features import compute_feature_stats, save_feature_stats, standardize_features, standardize_value
from latentedge.metrics import build_training_metrics, save_training_metrics
from latentedge.ingest.chunked import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    DEFAULT_FLUSH_EVERY_N_CHUNKS,
    DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    RATE_LIMIT_BACKOFF_MULTIPLIER,
    ingest_range,
)
from latentedge.ingest.progress import extend_window_for_new_blocks, read_progress
from latentedge.ingest.rpc_logs import RateLimitError, RpcLogsError, describe_error, get_latest_block
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as train_model
from latentedge.split import chronological_split
from latentedge.store import read_swaps
from latentedge.training_data import FEATURE_COLUMNS, AssembledTrainingData, SplitArrays, assemble_training_data
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


def _get_latest_block_with_retries(
    client: httpx.Client, rpc_url: str, max_retries: int, backoff_seconds: float,
) -> int:
    """Same retry-with-backoff behavior as every chunk fetch in
    ingest_range — a single transient network hiccup at startup must not
    kill the whole run before it even begins.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return get_latest_block(client, rpc_url)
        except (httpx.HTTPError, RpcLogsError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                sleep_seconds = backoff_seconds * (2**attempt)
                if isinstance(exc, RateLimitError):
                    sleep_seconds *= RATE_LIMIT_BACKOFF_MULTIPLIER
                click.echo(
                    f"  chain head lookup failed ({describe_error(rpc_url, exc)}) — "
                    f"retry {attempt + 1}/{max_retries}, waiting {sleep_seconds:.1f}s",
                    err=True,
                )
                time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


@cli.command()
@click.option(
    "--from-block", type=int, default=None, envvar="LATENTEDGE_FROM_BLOCK",
    help="Start of the block range. Omit together with --to-block to derive the range from --days instead. Falls back to the LATENTEDGE_FROM_BLOCK env var (or a .env file).",
)
@click.option(
    "--to-block", type=int, default=None, envvar="LATENTEDGE_TO_BLOCK",
    help="End of the block range. Omit together with --from-block to derive the range from --days instead. Falls back to the LATENTEDGE_TO_BLOCK env var (or a .env file).",
)
@click.option(
    "--days",
    type=float,
    default=config.DEFAULT_INGEST_DAYS,
    envvar="LATENTEDGE_INGEST_DAYS",
    help="How many most-recent days of blocks to ingest, ending near the current chain head — used only when --from-block/--to-block are both omitted. Falls back to the LATENTEDGE_INGEST_DAYS env var (or a .env file), letting a small test run (e.g. 1 day) be tried before committing to a full history pull.",
)
@click.option(
    "--rpc-url",
    type=str,
    default=DEFAULT_RPC_URL,
    envvar="LATENTEDGE_RPC_URL",
    help="An archive-capable RPC endpoint for ranges reaching back further than a few hours (e.g. an Alchemy/Infura URL) — free public endpoints gate deep history behind a paid token. Falls back to the LATENTEDGE_RPC_URL env var (or a .env file) if omitted, so a provider key never needs to appear as a bare CLI argument.",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"), envvar="LATENTEDGE_INGEST_OUT",
    help="Where to write/append swap records. Falls back to the LATENTEDGE_INGEST_OUT env var (or a .env file).",
)
@click.option(
    "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, envvar="LATENTEDGE_CHUNK_SIZE",
    help="Blocks per eth_getLogs call — keep at or under your provider's per-call limit (10 on Alchemy's free tier). Falls back to the LATENTEDGE_CHUNK_SIZE env var (or a .env file).",
)
@click.option(
    "--max-workers", type=int, default=DEFAULT_MAX_WORKERS, envvar="LATENTEDGE_MAX_WORKERS",
    help="Ceiling on concurrent chunk requests — the auto-throttle backs off below this under rate limiting. Falls back to the LATENTEDGE_MAX_WORKERS env var (or a .env file).",
)
@click.option(
    "--flush-every-n-chunks", type=int, default=DEFAULT_FLUSH_EVERY_N_CHUNKS, envvar="LATENTEDGE_FLUSH_EVERY_N_CHUNKS",
    help="Batches writes to the output file — writing after every chunk would mean rewriting the whole file hundreds of thousands of times over a large range. Falls back to the LATENTEDGE_FLUSH_EVERY_N_CHUNKS env var (or a .env file).",
)
@click.option(
    "--max-retries", type=int, default=DEFAULT_MAX_RETRIES, envvar="LATENTEDGE_MAX_RETRIES",
    help="Falls back to the LATENTEDGE_MAX_RETRIES env var (or a .env file).",
)
@click.option(
    "--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS, envvar="LATENTEDGE_RETRY_BACKOFF_SECONDS",
    help="Falls back to the LATENTEDGE_RETRY_BACKOFF_SECONDS env var (or a .env file).",
)
@click.option(
    "--concurrency-cooldown-seconds", type=float, default=DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    envvar="LATENTEDGE_CONCURRENCY_COOLDOWN_SECONDS",
    help="After any rate limit, pauses every worker (not just the one that hit it) for this long before any new request goes out — stops the other workers from immediately re-triggering the same limit. Falls back to the LATENTEDGE_CONCURRENCY_COOLDOWN_SECONDS env var (or a .env file).",
)
@click.option(
    "--max-rate-limit-backoff-seconds", type=float, default=DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    envvar="LATENTEDGE_MAX_RATE_LIMIT_BACKOFF_SECONDS",
    help="A rate-limited chunk retries forever (it's expected, temporary provider behavior, not a bug) rather than giving up after --max-retries — this caps how long each wait between retries can grow to. Falls back to the LATENTEDGE_MAX_RATE_LIMIT_BACKOFF_SECONDS env var (or a .env file).",
)
@click.option(
    "--train-after-ingest/--no-train-after-ingest", default=False, envvar="LATENTEDGE_TRAIN_AFTER_INGEST",
    help="Start training immediately once ingestion finishes, instead of pausing for a [T]/[Q] prompt (TTY) or just exiting (non-TTY). Off by default so ingestion still pauses for review first. Falls back to the LATENTEDGE_TRAIN_AFTER_INGEST env var (or a .env file).",
)
@click.option(
    "--train-out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"), envvar="LATENTEDGE_TRAIN_OUT",
    help="Where a chained (--train-after-ingest, or the TTY dashboard's [T]) training run saves its model. Falls back to the LATENTEDGE_TRAIN_OUT env var (or a .env file).",
)
@click.option(
    "--train-epochs", type=int, default=100, envvar="LATENTEDGE_TRAIN_EPOCHS",
    help="Epoch count for a chained (--train-after-ingest, or the TTY dashboard's [T]) training run. Falls back to the LATENTEDGE_TRAIN_EPOCHS env var (or a .env file).",
)
def ingest(
    from_block: int | None,
    to_block: int | None,
    days: float,
    rpc_url: str,
    out: Path,
    chunk_size: int,
    max_workers: int,
    flush_every_n_chunks: int,
    max_retries: int,
    retry_backoff_seconds: float,
    concurrency_cooldown_seconds: float,
    max_rate_limit_backoff_seconds: float,
    train_after_ingest: bool,
    train_out: Path,
    train_epochs: int,
) -> None:
    # A large range (e.g. a year of history) needs chunking to respect
    # provider limits, concurrency to finish in a reasonable time, and
    # resumability to survive a multi-hour run being interrupted — see
    # ingest.chunked for the real logic; fetch_swaps already populates
    # base_fee_wei from the same eth_getBlockByNumber call it makes for
    # each block's timestamp, no separate backfill pass.
    if (from_block is None) != (to_block is None):
        raise click.UsageError("--from-block and --to-block must be given together, or both omitted to use --days instead.")

    if from_block is None:
        with httpx.Client(timeout=30.0) as client:
            try:
                head = _get_latest_block_with_retries(client, rpc_url, max_retries, retry_backoff_seconds)
            except (httpx.HTTPError, RpcLogsError) as exc:
                raise click.ClickException(describe_error(rpc_url, exc)) from None
        naive_to = head - config.HEAD_BLOCK_SAFETY_BUFFER
        blocks_in_range = max(int(days * 86400 / config.AVG_BLOCK_SECONDS), 1)
        naive_from = naive_to - blocks_in_range + 1
        # If the naive most-recent-N-days window is already (partly or
        # fully) ingested, walk further back toward the pool's
        # deployment block until a day's worth of genuinely new blocks
        # is found — never re-request what's already on disk.
        from_block = extend_window_for_new_blocks(
            read_progress(out), naive_from, naive_to, blocks_in_range, config.POOL_CREATION_BLOCK,
        )
        to_block = naive_to
    assert to_block is not None  # guaranteed by the from_block/to_block XOR check above

    out.parent.mkdir(parents=True, exist_ok=True)

    if sys.stdout.isatty():
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=from_block, to_block=to_block,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            concurrency_cooldown_seconds=concurrency_cooldown_seconds,
            max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds, ingest_fn=ingest_range,
            train_assemble_fn=_assemble_train_data,
            model_out_path=train_out, train_epochs=train_epochs, train_after_ingest=train_after_ingest,
        )
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"ingest failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    def report(chunk_start: int, chunk_end: int, count: int) -> None:
        click.echo(f"  blocks {chunk_start}-{chunk_end}: {count} swaps")

    with httpx.Client(timeout=30.0) as client:
        try:
            total = ingest_range(
                config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
                chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
                max_workers=max_workers, flush_every_n_chunks=flush_every_n_chunks,
                concurrency_cooldown_seconds=concurrency_cooldown_seconds,
                max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds,
                on_progress=report,
            )
        except (httpx.HTTPError, RpcLogsError) as exc:
            raise click.ClickException(describe_error(rpc_url, exc)) from None
    click.echo(f"wrote {total} new swap records to {out}")

    if train_after_ingest:
        _run_train_direct(out, train_out, train_epochs)


def _assemble_train_data(swaps_path: Path) -> AssembledTrainingData:
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
    train_split, validate_split, test_split = chronological_split(assembled, train_fraction=0.7, validate_fraction=0.15)

    # Standardize using train-split statistics only — computing stats
    # from validate/test data would leak information about those splits
    # into training. The caller saves these stats alongside the model,
    # after the model itself is safely on disk, so SignalClient never
    # sees a stats file that doesn't match the model next to it.
    #
    # net_return's own (mean, std) rides along in the same stats dict —
    # SplitArrays.y below stays on the raw net_return scale (only
    # FEATURE_COLUMNS get standardized into x), so callers that want a
    # standardized training target must apply these net_return stats
    # themselves via features.standardize_value/unstandardize_value.
    stats = compute_feature_stats(train_split, [*FEATURE_COLUMNS, "net_return"])

    def to_arrays(split: pd.DataFrame) -> SplitArrays:
        standardized = standardize_features(split, FEATURE_COLUMNS, stats)
        x = standardized[FEATURE_COLUMNS].to_numpy(dtype="float32")
        y = standardized["net_return"].to_numpy(dtype="float32")
        return SplitArrays(x=x, y=y)

    return AssembledTrainingData(
        train=to_arrays(train_split),
        validate=to_arrays(validate_split),
        test=to_arrays(test_split),
        input_dim=len(FEATURE_COLUMNS),
        stats=stats,
    )


def _run_train_direct(swaps: Path, out: Path, epochs: int) -> None:
    """Trains without the TUI — used both by `train` when stdout isn't a
    tty and by `ingest --train-after-ingest` in that same non-tty case,
    where there's no ingest TUI screen around to chain into TrainScreen.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    assembled = _assemble_train_data(swaps)
    regressor = NetReturnRegressor(input_dim=assembled.input_dim)
    # net_return's raw scale (~1e-3) makes the MSE loss surface too flat
    # for Adam to make real progress in a practical number of epochs —
    # train on the standardized target and let build_training_metrics
    # unstandardize predictions back for reporting.
    y_train = standardize_value(assembled.train.y, assembled.stats["net_return"])
    losses = train_model(regressor, assembled.train.x, y_train, epochs=epochs, learning_rate=0.001)
    save(regressor, out)
    save_feature_stats(assembled.stats, Path(str(out) + ".stats.json"))
    metrics = build_training_metrics(regressor, assembled, losses)
    save_training_metrics(metrics, Path(str(out) + ".metrics.json"))
    val_corr = metrics["splits"]["validate"]["correlation"]
    click.echo(
        f"trained {epochs} epochs, final loss {losses[-1]:.6f}, val corr {val_corr:.4f}, saved to {out}"
    )


@cli.command()
@click.option(
    "--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"), envvar="LATENTEDGE_TRAIN_SWAPS",
    help="Falls back to the LATENTEDGE_TRAIN_SWAPS env var (or a .env file).",
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"), envvar="LATENTEDGE_TRAIN_OUT",
    help="Falls back to the LATENTEDGE_TRAIN_OUT env var (or a .env file).",
)
@click.option(
    "--epochs", type=int, default=100, envvar="LATENTEDGE_TRAIN_EPOCHS",
    help="Falls back to the LATENTEDGE_TRAIN_EPOCHS env var (or a .env file).",
)
def train(swaps: Path, out: Path, epochs: int) -> None:
    if sys.stdout.isatty():
        out.parent.mkdir(parents=True, exist_ok=True)
        screen = TrainScreen(
            swaps_path=swaps, out_path=out, epochs=epochs,
            assemble_fn=_assemble_train_data,
        )
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"train failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    _run_train_direct(swaps, out, epochs)


@cli.command()
@click.option(
    "--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"), envvar="LATENTEDGE_BACKTEST_SWAPS",
    help="Falls back to the LATENTEDGE_BACKTEST_SWAPS env var (or a .env file).",
)
@click.option(
    "--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"), envvar="LATENTEDGE_BACKTEST_MODEL",
    help="Falls back to the LATENTEDGE_BACKTEST_MODEL env var (or a .env file).",
)
def backtest(swaps: Path, model: Path) -> None:
    # Intentional stub (see the implementation plan): assembling the real
    # entry/exit swap arrays run_backtest needs is its own piece of glue
    # work. A non-zero exit avoids this reading as a successful backtest.
    raise click.ClickException("backtest command not implemented yet — see backtest.run_backtest for the underlying logic")
