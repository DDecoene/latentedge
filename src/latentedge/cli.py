from pathlib import Path

import click
import httpx

from latentedge import config
from latentedge.bars import build_bars
from latentedge.features import compute_features
from latentedge.ingest.pool_state import backfill_base_fee
from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.labeling import label_bars
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as train_model
from latentedge.split import chronological_split
from latentedge.store import read_swaps, write_swaps


@click.group()
def cli() -> None:
    """latentedge: ingest, train, and backtest the v1 WETH/USDC pipeline."""


DEFAULT_RPC_URL = "https://ethereum.publicnode.com"


@cli.command()
@click.option("--from-block", type=int, required=True)
@click.option("--to-block", type=int, required=True)
@click.option("--rpc-url", type=str, default=DEFAULT_RPC_URL)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
def ingest(from_block: int, to_block: int, rpc_url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        records = fetch_swaps(config.POOL_ADDRESS, from_block, to_block, client, rpc_url)
        records = backfill_base_fee(records, client, rpc_url)
    write_swaps(records, out)
    click.echo(f"wrote {len(records)} swap records to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
@click.option("--epochs", type=int, default=100)
def train(swaps: Path, out: Path, epochs: int) -> None:
    swap_df = read_swaps(swaps)
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)
    labeled = label_bars(bar_df, swap_df, tp_sl_fraction=0.01)
    labeled = labeled[~labeled["excluded"]].reset_index(drop=True)
    featured = compute_features(labeled, return_windows=[5, 15, 30], volatility_window=15)
    featured = featured.dropna().reset_index(drop=True)

    train_split, _validate_split, _test_split = chronological_split(featured, train_fraction=0.7, validate_fraction=0.15)

    feature_columns = ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap"]
    x = train_split[feature_columns].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")

    regressor = NetReturnRegressor(input_dim=len(feature_columns))
    losses = train_model(regressor, x, y, epochs=epochs, learning_rate=0.001)

    out.parent.mkdir(parents=True, exist_ok=True)
    save(regressor, out)
    click.echo(f"trained {epochs} epochs, final loss {losses[-1]:.6f}, saved to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
def backtest(swaps: Path, model: Path) -> None:
    click.echo("backtest command wiring — see backtest.run_backtest for the underlying logic")
