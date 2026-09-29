import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from latentedge import config
from latentedge.backtest import BacktestObserver, build_backtest_inputs, daily_sharpe, run_backtest
from latentedge.bars import build_bars
from latentedge.features import compute_feature_stats, save_feature_stats, standardize_features, standardize_value
from latentedge.labeling import LabelSettings
from latentedge.metrics import (
    TrainingMetrics, build_training_metrics, load_training_metrics, prediction_correlations, save_training_metrics,
)
from latentedge.ingest.chunked import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    DEFAULT_FLUSH_EVERY_N_CHUNKS,
    DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_RPS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    RATE_LIMIT_BACKOFF_MULTIPLIER,
    ingest_range,
)
from latentedge.ingest.progress import internal_gaps, read_progress, uncovered_gaps
from latentedge.ingest.rpc_logs import (
    RateLimitError,
    RpcLogsError,
    describe_error,
    get_block_at_or_after_timestamp,
    get_latest_block,
)
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as train_model
from latentedge.safety_guard import SafetyGuard
from latentedge.signal_client import SignalClient
from latentedge.split import chronological_split
from latentedge import sweep as sweeping
from latentedge.store import read_swaps
from latentedge.training_data import FEATURE_COLUMNS, AssembledTrainingData, SplitArrays, assemble_training_data
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.backtest_screen import BacktestScreen, describe_summary
from latentedge.tui.sweep_screen import SweepScreen
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


def _startup_call_with_retries(
    step: str,
    call: Callable[[], int],
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
    max_rate_limit_backoff_seconds: float,
) -> int:
    """Same retry behavior as every chunk fetch in ingest_range — a
    single transient hiccup while resolving the block range must not kill
    the whole run before it even begins. A rate limit is expected,
    temporary provider behavior (especially right after a previous run),
    so it retries until it clears with a capped backoff; any other error
    gives up after max_retries.
    """
    attempt = 0
    while True:
        try:
            return call()
        except (httpx.HTTPError, RpcLogsError) as exc:
            attempt += 1
            rate_limited = isinstance(exc, RateLimitError)
            if not rate_limited and attempt >= max_retries:
                raise
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            if rate_limited:
                sleep_seconds = min(sleep_seconds * RATE_LIMIT_BACKOFF_MULTIPLIER, max_rate_limit_backoff_seconds)
            of = "" if rate_limited else f"/{max_retries}"
            click.echo(
                f"  {step} failed ({describe_error(rpc_url, exc)}) — "
                f"retry {attempt}{of}, waiting {sleep_seconds:.1f}s",
                err=True,
            )
            time.sleep(sleep_seconds)


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
    help="Thread pool size (simultaneous connections) — the actual request pace against the provider is a separate ceiling, set via the LATENTEDGE_INGEST_MAX_RPS env var. Falls back to the LATENTEDGE_MAX_WORKERS env var (or a .env file).",
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
@click.option(
    "--backtest-after-train/--no-backtest-after-train", default=False, envvar="LATENTEDGE_BACKTEST_AFTER_TRAIN",
    help="Start the backtest immediately once training finishes, instead of pausing for a [B]/[Q] prompt (TTY) or just exiting (non-TTY). Applies to the training that follows an ingest too (--train-after-ingest or [T]). Uses the LATENTEDGE_BACKTEST_* env vars for its settings. Falls back to the LATENTEDGE_BACKTEST_AFTER_TRAIN env var (or a .env file).",
)
@click.option(
    "--sweep-after-train/--no-sweep-after-train", default=True, envvar="LATENTEDGE_SWEEP_AFTER_TRAIN",
    help="Run the scenario sweep immediately once training finishes (on by default; the sweep is the pipeline's result), before any chained backtest. Uses the LATENTEDGE_SWEEP_* env vars for its settings. Falls back to the LATENTEDGE_SWEEP_AFTER_TRAIN env var (or a .env file).",
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
    backtest_after_train: bool,
    sweep_after_train: bool,
) -> None:
    # A large range (e.g. a year of history) needs chunking to respect
    # provider limits, concurrency to finish in a reasonable time, and
    # resumability to survive a multi-hour run being interrupted — see
    # ingest.chunked for the real logic; fetch_swaps already populates
    # base_fee_wei from the same eth_getBlockByNumber call it makes for
    # each block's timestamp, no separate backfill pass.
    if (from_block is None) != (to_block is None):
        raise click.UsageError("--from-block and --to-block must be given together, or both omitted to use --days instead.")

    # A bad training setting should fail now, not after hours of ingesting.
    _label_settings_from_env()
    _excluded_features_from_env()

    max_rps = float(os.environ.get("LATENTEDGE_INGEST_MAX_RPS", DEFAULT_MAX_RPS))
    fixed_rps = _resolve_fixed_rps()

    if from_block is None:
        target_timestamp = int(time.time() - days * 86400)
        with httpx.Client(timeout=30.0) as client:
            try:
                def with_retries(step: str, call: Callable[[], int]) -> int:
                    return _startup_call_with_retries(
                        step, call, rpc_url, max_retries, retry_backoff_seconds, max_rate_limit_backoff_seconds,
                    )

                head = with_retries("chain head lookup", lambda: get_latest_block(client, rpc_url))
                # An exact, on-chain-verified anchor for the window's
                # start — never an estimate from a constant average
                # block time, which drifts from the chain's real block
                # times and would make "--days N" mean a slightly
                # different span every time. Missing blocks inside
                # [from_block, to_block] (including any already-known
                # internal gap) are queued and filled below exactly as
                # for an explicit --from-block/--to-block range.
                from_block = with_retries(
                    "start-block lookup",
                    lambda: get_block_at_or_after_timestamp(
                        client, rpc_url, target_timestamp, config.POOL_CREATION_BLOCK, head,
                    ),
                )
            except (httpx.HTTPError, RpcLogsError) as exc:
                raise click.ClickException(describe_error(rpc_url, exc)) from None
        to_block = head - config.HEAD_BLOCK_SAFETY_BUFFER
    assert to_block is not None  # guaranteed by the from_block/to_block XOR check above

    out.parent.mkdir(parents=True, exist_ok=True)

    progress_intervals = read_progress(out)

    if sys.stdout.isatty():
        # The TUI is the only UI in a TTY session — all work happens
        # inside it, never as plain click.echo lines around it.
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=from_block, to_block=to_block,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers, max_rps=max_rps, fixed_rps=fixed_rps,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            concurrency_cooldown_seconds=concurrency_cooldown_seconds,
            max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds, ingest_fn=ingest_range,
            train_assemble_fn=_assemble_train_data,
            model_out_path=train_out, train_epochs=train_epochs, train_after_ingest=train_after_ingest,
            backtest_fn=_chained_backtest_fn(out, train_out), backtest_after_train=backtest_after_train,
            sweep_fn=_chained_sweep_fn(out, train_out), sweep_after_train=sweep_after_train,
            env_path=Path.cwd() / ".env",
        )
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"ingest failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    range_total = max(to_block - from_block + 1, 0)
    range_remaining = sum(end - start + 1 for start, end in uncovered_gaps(progress_intervals, from_block, to_block))
    click.echo(
        f"{out}: {range_total} blocks in requested range, {range_total - range_remaining} already in file, "
        f"{range_remaining} remaining to download"
    )

    def report(chunk_start: int, chunk_end: int, count: int) -> None:
        click.echo(f"  blocks {chunk_start}-{chunk_end}: {count} swaps")

    with httpx.Client(timeout=30.0) as client:
        try:
            total = ingest_range(
                config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
                chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
                max_workers=max_workers, max_rps=max_rps, fixed_rps=fixed_rps, flush_every_n_chunks=flush_every_n_chunks,
                concurrency_cooldown_seconds=concurrency_cooldown_seconds,
                max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds,
                on_progress=report,
            )
        except (httpx.HTTPError, RpcLogsError) as exc:
            raise click.ClickException(describe_error(rpc_url, exc)) from None
    click.echo(f"wrote {total} new swap records to {out}")

    if train_after_ingest:
        _run_train_direct(out, train_out, train_epochs)
        if sweep_after_train:
            _run_sweep_direct(out, train_out)
        if backtest_after_train:
            _run_backtest_direct(out, train_out, BacktestParams.from_env())


def _resolve_fixed_rps() -> float | None:
    """None when auto-throttling is on (the default); otherwise the
    fixed req/s from LATENTEDGE_INGEST_FIXED_RPS, which is required in
    that mode — a fixed mode with no fixed value would silently be
    something else."""
    raw_mode = os.environ.get("LATENTEDGE_INGEST_AUTO_THROTTLE", "true").strip().lower()
    if raw_mode in ("1", "true", "yes", "on"):
        return None
    if raw_mode not in ("0", "false", "no", "off"):
        raise click.UsageError(
            f"LATENTEDGE_INGEST_AUTO_THROTTLE must be true or false, got {raw_mode!r}."
        )
    raw_rps = os.environ.get("LATENTEDGE_INGEST_FIXED_RPS", "").strip()
    try:
        fixed_rps = float(raw_rps)
    except ValueError:
        fixed_rps = 0.0
    if fixed_rps <= 0:
        raise click.UsageError(
            "LATENTEDGE_INGEST_AUTO_THROTTLE is off, so LATENTEDGE_INGEST_FIXED_RPS must be set to a "
            f"positive number of requests/sec (got {raw_rps!r})."
        )
    return fixed_rps


def _require_continuous_data(swaps_path: Path) -> None:
    """Refuses to train on a file whose ingested block coverage has
    holes — bars, rolling features and forward returns computed across a
    gap silently span time that was never observed.
    """
    intervals = read_progress(swaps_path)
    if not intervals:
        raise click.ClickException(
            f"{swaps_path} has no ingest progress record, so its block coverage can't be verified as continuous — re-run ingest."
        )
    gaps = internal_gaps(intervals)
    if gaps:
        shown = ", ".join(f"{start}-{end} ({end - start + 1} blocks)" for start, end in gaps[:5])
        more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
        raise click.ClickException(
            f"{swaps_path} is not continuous: missing block range(s) {shown}{more}. "
            "Ingest a range that spans them, or remove the older data on the far side of the gap."
        )


def _label_settings_from_env() -> LabelSettings:
    """LATENTEDGE_LABEL_HORIZON_MINUTES (how long a labeled trade may run)
    and LATENTEDGE_LABEL_BARRIER_STDS (take-profit/stop-loss band, in
    standard deviations of the 1-bar return). Only training reads these:
    a backtest or sweep relabels with what the model was trained under."""
    default = LabelSettings()
    try:
        minutes = float(os.environ.get("LATENTEDGE_LABEL_HORIZON_MINUTES", default.horizon_seconds / 60))
        stds = float(os.environ.get("LATENTEDGE_LABEL_BARRIER_STDS", default.barrier_stds))
    except ValueError as exc:
        raise click.UsageError(f"invalid LATENTEDGE_LABEL_* value: {exc}") from None
    if minutes * 60 < config.BAR_INTERVAL_SECONDS or stds <= 0:
        raise click.UsageError(
            "LATENTEDGE_LABEL_HORIZON_MINUTES must cover at least one bar and LATENTEDGE_LABEL_BARRIER_STDS must be positive."
        )
    return LabelSettings(horizon_seconds=round(minutes * 60), barrier_stds=stds)


def _excluded_features_from_env() -> list[str]:
    """LATENTEDGE_TRAIN_EXCLUDE_FEATURES: comma-separated feature names to
    leave out of training, e.g. base_fee_gwei,volatility to test whether
    the model predicts trading cost rather than direction."""
    excluded = [name.strip() for name in os.environ.get("LATENTEDGE_TRAIN_EXCLUDE_FEATURES", "").split(",") if name.strip()]
    unknown = [name for name in excluded if name not in FEATURE_COLUMNS]
    if unknown:
        raise click.UsageError(
            f"LATENTEDGE_TRAIN_EXCLUDE_FEATURES names unknown feature(s) {unknown}; choose from {FEATURE_COLUMNS}."
        )
    if len(excluded) >= len(FEATURE_COLUMNS):
        raise click.UsageError("LATENTEDGE_TRAIN_EXCLUDE_FEATURES leaves no features to train on.")
    return excluded


def _model_setup(metrics: TrainingMetrics) -> tuple[list[str], LabelSettings]:
    """The feature columns and label settings a model was trained with (the
    original seven features and 30-minute labels for a model whose metrics
    predate recording them)."""
    default = LabelSettings()
    settings = LabelSettings(
        horizon_seconds=metrics.get("label_horizon_seconds", default.horizon_seconds),
        barrier_stds=metrics.get("label_barrier_stds", default.barrier_stds),
    )
    return metrics.get("feature_columns", list(FEATURE_COLUMNS)), settings


def _prepare_labeled_bars(
    swaps_path: Path, report: Callable[..., None], settings: LabelSettings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest-coverage check, bars, features and labels: everything both
    training and backtesting need before the data is split. Returns the
    labeled bars and the raw swaps their swap indices refer to."""
    report("checking ingest coverage")
    _require_continuous_data(swaps_path)
    report("reading swaps")
    swap_df = read_swaps(swaps_path)
    report("building bars")
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)

    # Take-profit/stop-loss band sized from the pool's own realized
    # volatility (spec 3.3), not a fixed guess — a symmetric band at
    # settings.barrier_stds x the 1-bar return std-dev. The band has to
    # widen with the horizon: a fixed narrow band is hit within minutes and
    # a longer horizon would change nothing.
    bar_return_std = bar_df["price_usdc_per_weth"].pct_change().std()
    tp_sl_fraction = max(bar_return_std * settings.barrier_stds, 0.001) if pd.notna(bar_return_std) else 0.01

    assembled = assemble_training_data(
        bar_df, swap_df, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=tp_sl_fraction,
        on_label_progress=lambda done, total: report("labeling bars", done, total),
        horizon_seconds=settings.horizon_seconds,
    )
    return assembled, swap_df


def _assemble_train_data(
    swaps_path: Path, on_progress: Callable[[str, int, int], None] | None = None
) -> AssembledTrainingData:
    """on_progress(stage, done, total) is called between and within the
    slow stages; a caller may raise from it to abort the assembly."""
    def report(stage: str, done: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    settings = _label_settings_from_env()
    excluded = _excluded_features_from_env()
    feature_columns = [name for name in FEATURE_COLUMNS if name not in excluded]

    assembled, _ = _prepare_labeled_bars(swaps_path, report, settings)
    report("splitting and standardizing")
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
    stats = compute_feature_stats(train_split, [*feature_columns, "net_return"])

    def to_arrays(split: pd.DataFrame) -> SplitArrays:
        standardized = standardize_features(split, feature_columns, stats)
        x = standardized[feature_columns].to_numpy(dtype="float32")
        y = standardized["net_return"].to_numpy(dtype="float32")
        return SplitArrays(x=x, y=y, gross=split["gross_return"].to_numpy(dtype="float64"))

    return AssembledTrainingData(
        train=to_arrays(train_split),
        validate=to_arrays(validate_split),
        test=to_arrays(test_split),
        input_dim=len(feature_columns),
        stats=stats,
        test_start=int(test_split["bar_start"].iloc[0]),
        feature_columns=tuple(feature_columns),
        label_settings=settings,
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
    "--backtest-after-train/--no-backtest-after-train", default=False, envvar="LATENTEDGE_BACKTEST_AFTER_TRAIN",
    help="Start the backtest immediately once training finishes, instead of pausing for a [B]/[Q] prompt (TTY) or just exiting (non-TTY).  Uses the LATENTEDGE_BACKTEST_* env vars for its settings. Falls back to the LATENTEDGE_BACKTEST_AFTER_TRAIN env var (or a .env file).",
)
@click.option(
    "--epochs", type=int, default=100, envvar="LATENTEDGE_TRAIN_EPOCHS",
    help="Falls back to the LATENTEDGE_TRAIN_EPOCHS env var (or a .env file).",
)
@click.option(
    "--sweep-after-train/--no-sweep-after-train", default=True, envvar="LATENTEDGE_SWEEP_AFTER_TRAIN",
    help="Run the scenario sweep immediately once training finishes (on by default; the sweep is the pipeline's result), before any chained backtest. Uses the LATENTEDGE_SWEEP_* env vars for its settings. Falls back to the LATENTEDGE_SWEEP_AFTER_TRAIN env var (or a .env file).",
)
def train(swaps: Path, out: Path, epochs: int, backtest_after_train: bool, sweep_after_train: bool) -> None:
    _label_settings_from_env()
    _excluded_features_from_env()
    if sys.stdout.isatty():
        out.parent.mkdir(parents=True, exist_ok=True)
        screen = TrainScreen(
            swaps_path=swaps, out_path=out, epochs=epochs,
            assemble_fn=_assemble_train_data,
            backtest_fn=_chained_backtest_fn(swaps, out), backtest_after_train=backtest_after_train,
            sweep_fn=_chained_sweep_fn(swaps, out), sweep_after_train=sweep_after_train,
        )
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"train failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    _run_train_direct(swaps, out, epochs)
    if sweep_after_train:
        _run_sweep_direct(swaps, out)
    if backtest_after_train:
        _run_backtest_direct(swaps, out, BacktestParams.from_env())


@cli.command()
@click.option(
    "--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"), envvar="LATENTEDGE_BACKTEST_SWAPS",
    help="Falls back to the LATENTEDGE_BACKTEST_SWAPS env var (or a .env file).",
)
@click.option(
    "--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"), envvar="LATENTEDGE_BACKTEST_MODEL",
    help="Falls back to the LATENTEDGE_BACKTEST_MODEL env var (or a .env file).",
)
@click.option(
    "--initial-equity", type=float, default=10_000.0, envvar="LATENTEDGE_BACKTEST_INITIAL_EQUITY",
    help="Starting equity in USD. Falls back to the LATENTEDGE_BACKTEST_INITIAL_EQUITY env var (or a .env file).",
)
@click.option(
    "--max-position-fraction", type=float, default=0.10, envvar="LATENTEDGE_BACKTEST_MAX_POSITION_FRACTION",
    help="Largest single position as a fraction of equity. Falls back to LATENTEDGE_BACKTEST_MAX_POSITION_FRACTION.",
)
@click.option(
    "--daily-loss-limit-fraction", type=float, default=0.02, envvar="LATENTEDGE_BACKTEST_DAILY_LOSS_LIMIT_FRACTION",
    help="Realized daily loss, as a fraction of equity, that locks trading out for the day. "
    "Falls back to LATENTEDGE_BACKTEST_DAILY_LOSS_LIMIT_FRACTION.",
)
@click.option(
    "--full-size-return", type=float, default=0.002, envvar="LATENTEDGE_BACKTEST_FULL_SIZE_RETURN",
    help="Predicted net return at which a position sizes at the full maximum. "
    "Falls back to LATENTEDGE_BACKTEST_FULL_SIZE_RETURN.",
)
def backtest(
    swaps: Path,
    model: Path,
    initial_equity: float,
    max_position_fraction: float,
    daily_loss_limit_fraction: float,
    full_size_return: float,
) -> None:
    """Replays the model's decisions over its untouched test window only.

    The window's start is the one recorded when the model was trained, so
    data ingested since (which would move a fresh split's boundary into
    the model's own training data) can only extend the window forward.
    """
    params = BacktestParams(initial_equity, max_position_fraction, daily_loss_limit_fraction, full_size_return)
    if sys.stdout.isatty():
        screen = BacktestScreen(backtest_fn=lambda observer: _compute_backtest(swaps, model, params, observer))
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"backtest failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    _run_backtest_direct(swaps, model, params)


@dataclass(frozen=True)
class BacktestParams:
    initial_equity: float = 10_000.0
    max_position_fraction: float = 0.10
    daily_loss_limit_fraction: float = 0.02
    full_size_return: float = 0.002

    @classmethod
    def from_env(cls) -> "BacktestParams":
        """What a chained run (--backtest-after-train) uses: the same env
        vars the backtest command's own options fall back to."""
        d = cls()
        try:
            return cls(
                initial_equity=float(os.environ.get("LATENTEDGE_BACKTEST_INITIAL_EQUITY", d.initial_equity)),
                max_position_fraction=float(
                    os.environ.get("LATENTEDGE_BACKTEST_MAX_POSITION_FRACTION", d.max_position_fraction)
                ),
                daily_loss_limit_fraction=float(
                    os.environ.get("LATENTEDGE_BACKTEST_DAILY_LOSS_LIMIT_FRACTION", d.daily_loss_limit_fraction)
                ),
                full_size_return=float(os.environ.get("LATENTEDGE_BACKTEST_FULL_SIZE_RETURN", d.full_size_return)),
            )
        except ValueError as exc:
            raise click.UsageError(f"invalid LATENTEDGE_BACKTEST_* value: {exc}") from None


def _chained_backtest_fn(swaps: Path, model: Path) -> Callable[[BacktestObserver], dict]:
    """The backtest a TrainScreen chains into, configured from env vars
    (resolved now, so a bad value fails before the TUI starts)."""
    params = BacktestParams.from_env()
    return lambda observer: _compute_backtest(swaps, model, params, observer)


def _compute_backtest(
    swaps: Path, model: Path, params: BacktestParams, observer: BacktestObserver
) -> dict:
    """Runs the backtest and writes <model>.backtest.json; returns the
    summary. The observer is told of each slow stage and then watches the
    replay itself; any of its callbacks may raise to abort. Failures raise
    ClickException."""
    metrics_path = Path(str(model) + ".metrics.json")
    if not metrics_path.exists():
        raise click.ClickException(f"{metrics_path} not found — train the model first.")
    test_start = load_training_metrics(metrics_path).get("test_start")
    if test_start is None:
        raise click.ClickException(
            f"{metrics_path} has no recorded test window (trained before this was tracked), so an out-of-sample "
            "backtest can't be guaranteed — retrain the model."
        )

    feature_columns, settings = _model_setup(load_training_metrics(metrics_path))
    assembled, swap_df = _prepare_labeled_bars(swaps, observer, settings)
    window = assembled[assembled["bar_start"] >= test_start]
    if window.empty:
        raise click.ClickException(f"no labeled bars at or after the model's test window start ({test_start}).")

    observer("replaying test window")
    inputs = build_backtest_inputs(window, swap_df, feature_columns)
    client = SignalClient(model, input_dim=len(feature_columns), feature_columns=feature_columns)
    guard = SafetyGuard(
        max_position_fraction=params.max_position_fraction,
        daily_loss_limit_fraction=params.daily_loss_limit_fraction,
        full_size_return=params.full_size_return,
    )
    result = run_backtest(
        features=inputs.features,
        entry_prices=inputs.entry_prices,
        exit_prices=inputs.exit_prices,
        entry_swaps=inputs.entry_swaps,
        exit_swaps=inputs.exit_swaps,
        signal_client=client,
        guard=guard,
        initial_equity_usd=params.initial_equity,
        timestamps=inputs.timestamps,
        observer=observer,
        horizon_seconds=settings.horizon_seconds,
    )

    summary = {
        "test_start": test_start,
        "bars": len(window),
        "initial_equity_usd": params.initial_equity,
        "max_position_fraction": params.max_position_fraction,
        "daily_loss_limit_fraction": params.daily_loss_limit_fraction,
        "full_size_return": params.full_size_return,
        "total_return_usd": result.total_return_usd,
        "total_return_fraction": result.total_return_usd / params.initial_equity,
        "max_drawdown_usd": result.max_drawdown_usd,
        "win_rate": result.win_rate,
        "num_trades": result.num_trades,
        "sharpe": daily_sharpe(np.array(result.equity_curve), inputs.timestamps),
    }
    Path(str(model) + ".backtest.json").write_text(json.dumps(summary, indent=2))
    return summary


def _run_backtest_direct(swaps: Path, model: Path, params: BacktestParams) -> None:
    """Backtests without the TUI — used by `backtest` when stdout isn't a
    tty and by the non-tty chained (--backtest-after-train) runs."""
    summary = _compute_backtest(swaps, model, params, BacktestObserver())
    click.echo(f"backtest over {describe_summary(summary)}")
    click.echo(f"summary saved to {model}.backtest.json")


DEFAULT_SWEEP_MIN_EDGES = "0,0.0013"
DEFAULT_SWEEP_FULL_SIZE_RETURNS = "0.002"
DEFAULT_SWEEP_TOP_FRACTIONS = "0.01,0.05,0.10,0.25,0.50,1.0"
DEFAULT_SWEEP_SHUFFLE_SEEDS = "0,1,2"


def _parse_float_list(name: str, raw: str) -> list[float]:
    try:
        values = [float(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise click.UsageError(f"{name} must be a comma-separated list of numbers, got {raw!r}.") from None
    if not values:
        raise click.UsageError(f"{name} must list at least one number.")
    return values


def _parse_int_list(name: str, raw: str) -> list[int]:
    """Comma-separated integers; an empty value is an empty list."""
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise click.UsageError(f"{name} must be a comma-separated list of integers, got {raw!r}.") from None


@dataclass(frozen=True)
class SweepGrid:
    min_edges: list[float]
    full_size_returns: list[float]
    top_fractions: list[float]
    shuffle_seeds: list[int]

    @classmethod
    def from_env(cls) -> "SweepGrid":
        """The grid from the LATENTEDGE_SWEEP_* env vars (their defaults if unset)."""
        def raw(name: str, default: str) -> str:
            return os.environ.get(name, default)

        return cls(
            _parse_float_list("LATENTEDGE_SWEEP_MIN_EDGES", raw("LATENTEDGE_SWEEP_MIN_EDGES", DEFAULT_SWEEP_MIN_EDGES)),
            _parse_float_list(
                "LATENTEDGE_SWEEP_FULL_SIZE_RETURNS",
                raw("LATENTEDGE_SWEEP_FULL_SIZE_RETURNS", DEFAULT_SWEEP_FULL_SIZE_RETURNS),
            ),
            _parse_float_list(
                "LATENTEDGE_SWEEP_TOP_FRACTIONS", raw("LATENTEDGE_SWEEP_TOP_FRACTIONS", DEFAULT_SWEEP_TOP_FRACTIONS)
            ),
            _parse_int_list(
                "LATENTEDGE_SWEEP_SHUFFLE_SEEDS", raw("LATENTEDGE_SWEEP_SHUFFLE_SEEDS", DEFAULT_SWEEP_SHUFFLE_SEEDS)
            ),
        )


def _chained_sweep_fn(swaps: Path, model: Path) -> Callable[[sweeping.SweepObserver], dict]:
    """The sweep a TrainScreen chains into, configured from env vars
    (resolved now, so a bad value fails before the TUI starts)."""
    params = BacktestParams.from_env()
    grid = SweepGrid.from_env()
    out_dir = Path(os.environ.get("LATENTEDGE_SWEEP_OUT_DIR", "data/sweeps"))
    return lambda observer: _compute_sweep(swaps, model, params, grid, out_dir, observer)


def _run_sweep_direct(swaps: Path, model: Path) -> None:
    """Sweeps without the TUI — the non-tty chained (--sweep-after-train) runs."""
    result = _chained_sweep_fn(swaps, model)(sweeping.SweepObserver())
    click.echo(sweeping.describe_sweep(result))


def _compute_sweep(
    swaps: Path, model: Path, params: BacktestParams, grid: SweepGrid, out_dir: Path,
    observer: sweeping.SweepObserver,
) -> dict:
    """Runs every scenario of the grid on the validation and test windows
    plus the reference strategies, saves the lot under out_dir and returns
    {"path", "scenarios", "selection"}. Failures raise ClickException."""
    metrics_path = Path(str(model) + ".metrics.json")
    if not metrics_path.exists():
        raise click.ClickException(f"{metrics_path} not found — train the model first.")
    metrics = load_training_metrics(metrics_path)
    test_start = metrics.get("test_start")
    if test_start is None:
        raise click.ClickException(
            f"{metrics_path} has no recorded test window (trained before this was tracked) — retrain the model."
        )
    validate_bars = int(metrics["splits"]["validate"]["n"])

    feature_columns, settings = _model_setup(metrics)
    assembled, swap_df = _prepare_labeled_bars(swaps, observer, settings)
    windows = sweeping.split_windows(assembled, test_start, validate_bars)
    for name, window in windows.items():
        if window.empty:
            raise click.ClickException(f"the {name} window has no labeled bars.")

    observer("computing model predictions")
    client = SignalClient(model, input_dim=len(feature_columns), feature_columns=feature_columns)
    inputs = {name: build_backtest_inputs(window, swap_df, feature_columns) for name, window in windows.items()}
    model_predictions = {name: client.predict_batch(inputs[name].features) for name in windows}
    # Works for any model, old or new: does it predict direction (gross) or
    # what trading costs, on the same windows the rules are replayed on.
    diagnostics = {
        name: prediction_correlations(
            model_predictions[name], windows[name]["net_return"].to_numpy(dtype="float64"),
            windows[name]["gross_return"].to_numpy(dtype="float64"),
        )
        for name in windows
    }

    scenarios = sweeping.build_scenarios(
        grid.min_edges, grid.full_size_returns, grid.top_fractions, grid.shuffle_seeds
    )
    rows: list[dict] = []
    for index, scenario in enumerate(scenarios):
        observer(
            f"{scenario.window} / {scenario.signal} / {sweeping.describe_rule(asdict(scenario))}",
            index, len(scenarios),
        )
        predictions = sweeping.predictions_for(
            scenario, model_predictions[scenario.window], model_predictions[sweeping.WINDOW_VALIDATE],
            windows[scenario.window],
        )
        row = sweeping.run_scenario(
            scenario, inputs[scenario.window], predictions, params.max_position_fraction,
            params.daily_loss_limit_fraction, params.initial_equity, settings.horizon_seconds,
        )
        rows.append(row)
        observer.on_scenario(row)

    meta = {
        "model": str(model),
        "model_sha256": sweeping.file_sha256(model),
        "swaps": str(swaps),
        "ingested_block_ranges": read_progress(swaps),
        "test_start": test_start,
        "validate_bars": validate_bars,
        "initial_equity_usd": params.initial_equity,
        "max_position_fraction": params.max_position_fraction,
        "daily_loss_limit_fraction": params.daily_loss_limit_fraction,
        "min_edges": grid.min_edges,
        "full_size_returns": grid.full_size_returns,
        "top_fractions": grid.top_fractions,
        "shuffle_seeds": grid.shuffle_seeds,
        "bar_interval_seconds": config.BAR_INTERVAL_SECONDS,
        "label_horizon_seconds": settings.horizon_seconds,
        "label_barrier_stds": settings.barrier_stds,
        "feature_columns": feature_columns,
        "prediction_diagnostics": diagnostics,
        "training_metrics": metrics,
    }
    path = sweeping.write_sweep(out_dir, meta, rows)
    return {"path": str(path), "scenarios": rows, "selection": sweeping.select_on_validate(rows)}


@cli.command()
@click.option(
    "--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"), envvar="LATENTEDGE_SWEEP_SWAPS",
    help="Falls back to the LATENTEDGE_SWEEP_SWAPS env var (or a .env file).",
)
@click.option(
    "--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"), envvar="LATENTEDGE_SWEEP_MODEL",
    help="Falls back to the LATENTEDGE_SWEEP_MODEL env var (or a .env file).",
)
@click.option(
    "--out-dir", type=click.Path(path_type=Path), default=Path("data/sweeps"), envvar="LATENTEDGE_SWEEP_OUT_DIR",
    help="Where each sweep is saved (one timestamped .json and .csv per run, never overwritten). "
    "Falls back to the LATENTEDGE_SWEEP_OUT_DIR env var (or a .env file).",
)
@click.option(
    "--min-edges", default=DEFAULT_SWEEP_MIN_EDGES, envvar="LATENTEDGE_SWEEP_MIN_EDGES",
    help="Comma-separated minimum predicted returns a trade must clear. Falls back to LATENTEDGE_SWEEP_MIN_EDGES.",
)
@click.option(
    "--top-fractions", default=DEFAULT_SWEEP_TOP_FRACTIONS, envvar="LATENTEDGE_SWEEP_TOP_FRACTIONS",
    help="Comma-separated shares of bars the model trades, the ones it ranks highest, at full size — a rule "
    "that works even when no prediction is above zero. The cutoff is set on the validation window. "
    "Falls back to LATENTEDGE_SWEEP_TOP_FRACTIONS.",
)
@click.option(
    "--full-size-returns", default=DEFAULT_SWEEP_FULL_SIZE_RETURNS, envvar="LATENTEDGE_SWEEP_FULL_SIZE_RETURNS",
    help="Comma-separated predicted returns at which a position sizes at the full maximum. "
    "Falls back to LATENTEDGE_SWEEP_FULL_SIZE_RETURNS.",
)
@click.option(
    "--shuffle-seeds", default=DEFAULT_SWEEP_SHUFFLE_SEEDS, envvar="LATENTEDGE_SWEEP_SHUFFLE_SEEDS",
    help="Comma-separated seeds; each adds the rank rules run on the model's predictions shuffled among the "
    "bars — the control a real signal must beat. Empty turns the control off. "
    "Falls back to LATENTEDGE_SWEEP_SHUFFLE_SEEDS.",
)
def sweep(
    swaps: Path, model: Path, out_dir: Path, min_edges: str, full_size_returns: str, top_fractions: str,
    shuffle_seeds: str,
) -> None:
    """Replays the model under a grid of trading rules on the validation and
    test windows, next to an always-trade and an oracle reference, and
    keeps every result for comparison. Equity and risk limits come from the
    LATENTEDGE_BACKTEST_* settings."""
    params = BacktestParams.from_env()
    grid = SweepGrid(
        _parse_float_list("--min-edges", min_edges), _parse_float_list("--full-size-returns", full_size_returns),
        _parse_float_list("--top-fractions", top_fractions), _parse_int_list("--shuffle-seeds", shuffle_seeds),
    )
    if sys.stdout.isatty():
        screen = SweepScreen(sweep_fn=lambda observer: _compute_sweep(swaps, model, params, grid, out_dir, observer))
        LatentEdgeApp(start_screen=screen).run()
        if screen.error is not None:
            click.echo(f"sweep failed: {screen.error}", err=True)
            raise SystemExit(1)
        return

    result = _compute_sweep(swaps, model, params, grid, out_dir, sweeping.SweepObserver())
    click.echo(sweeping.describe_sweep(result))
