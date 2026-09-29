from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from latentedge import config
from latentedge.cli import DEFAULT_RPC_URL, cli
from latentedge.ingest.progress import write_progress
from latentedge.training_data import AssembledTrainingData, SplitArrays


def test_cli_exposes_expected_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "train" in result.output
    assert "backtest" in result.output


def test_ingest_rpc_url_falls_back_to_env_var_when_flag_omitted(monkeypatch: pytest.MonkeyPatch):
    # Real-world need: passing a provider API key as a bare CLI argument
    # leaves it in shell history; an env var (from a .env file or a
    # secrets manager on a VPS/CI) is the standard way to avoid that.
    captured: dict[str, str] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["rpc_url"] = rpc_url
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_RPC_URL": "https://example-from-env.invalid"},
    )

    assert captured["rpc_url"] == "https://example-from-env.invalid"


def test_ingest_numeric_and_path_options_fall_back_to_env_vars(monkeypatch: pytest.MonkeyPatch):
    # The user wants every ingest knob settable from .env, not just
    # --rpc-url/--days — nobody should have to remember or retype CLI
    # flags for a run they do the same way every time.
    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["out"] = out
        captured["chunk_size"] = kwargs["chunk_size"]
        captured["max_workers"] = kwargs["max_workers"]
        captured["flush_every_n_chunks"] = kwargs["flush_every_n_chunks"]
        captured["max_retries"] = kwargs["max_retries"]
        captured["retry_backoff_seconds"] = kwargs["retry_backoff_seconds"]
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest"],
        env={
            "LATENTEDGE_FROM_BLOCK": "100",
            "LATENTEDGE_TO_BLOCK": "200",
            "LATENTEDGE_INGEST_OUT": "/tmp/env-swaps.parquet",
            "LATENTEDGE_CHUNK_SIZE": "5",
            "LATENTEDGE_MAX_WORKERS": "2",
            "LATENTEDGE_FLUSH_EVERY_N_CHUNKS": "3",
            "LATENTEDGE_MAX_RETRIES": "7",
            "LATENTEDGE_RETRY_BACKOFF_SECONDS": "1.5",
        },
    )

    assert captured["from_block"] == 100
    assert captured["to_block"] == 200
    assert str(captured["out"]) == "/tmp/env-swaps.parquet"
    assert captured["chunk_size"] == 5
    assert captured["max_workers"] == 2
    assert captured["flush_every_n_chunks"] == 3
    assert captured["max_retries"] == 7
    assert captured["retry_backoff_seconds"] == 1.5


def test_train_options_fall_back_to_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_assemble(swaps_path):
        captured["swaps_path"] = swaps_path
        empty = SplitArrays(x=np.zeros((1, 1), dtype="float32"), y=np.zeros(1, dtype="float32"))
        return AssembledTrainingData(
            train=empty, validate=empty, test=empty, input_dim=1, stats={"net_return": (0.0, 1.0)}
        )

    def fake_train(model, x, y, epochs, learning_rate):
        captured["epochs"] = epochs
        return [0.0]

    monkeypatch.setattr("latentedge.cli._assemble_train_data", fake_assemble)
    monkeypatch.setattr("latentedge.cli.train_model", fake_train)

    out_path = tmp_path / "env-model.safetensors"
    swaps_path = tmp_path / "env-swaps.parquet"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["train"],
        env={
            "LATENTEDGE_TRAIN_SWAPS": str(swaps_path),
            "LATENTEDGE_TRAIN_OUT": str(out_path),
            "LATENTEDGE_TRAIN_EPOCHS": "3",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["swaps_path"] == swaps_path
    assert captured["epochs"] == 3


def test_ingest_train_after_ingest_chains_training_in_non_tty_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Lets an unattended, non-interactive ingest (e.g. a long overnight
    # run) go straight into training without a human pressing [T].
    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        return 3

    def fake_assemble(swaps_path):
        captured["swaps_path"] = swaps_path
        empty = SplitArrays(x=np.zeros((1, 1), dtype="float32"), y=np.zeros(1, dtype="float32"))
        return AssembledTrainingData(train=empty, validate=empty, test=empty, input_dim=1, stats={"net_return": (0.0, 1.0)})

    def fake_train(model, x, y, epochs, learning_rate):
        captured["epochs"] = epochs
        return [0.0]

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)
    monkeypatch.setattr("latentedge.cli._assemble_train_data", fake_assemble)
    monkeypatch.setattr("latentedge.cli.train_model", fake_train)

    out_path = tmp_path / "swaps.parquet"
    train_out_path = tmp_path / "model.safetensors"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(out_path)],
        env={
            "LATENTEDGE_TRAIN_AFTER_INGEST": "true",
            "LATENTEDGE_TRAIN_OUT": str(train_out_path),
            "LATENTEDGE_TRAIN_EPOCHS": "3",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["swaps_path"] == out_path
    assert captured["epochs"] == 3
    assert "trained 3 epochs" in result.output


def test_ingest_train_after_ingest_off_by_default_does_not_train(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        return 0

    def fake_assemble(swaps_path):
        raise AssertionError("training must not run when --train-after-ingest is off")

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)
    monkeypatch.setattr("latentedge.cli._assemble_train_data", fake_assemble)

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")]
    )

    assert result.exit_code == 0, result.output


def test_cli_loads_dotenv_file_from_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Regression test for real .env support: a value only present in a
    # .env file (never exported into the shell) must still reach the CLI.
    env_file = tmp_path / ".env"
    env_file.write_text("LATENTEDGE_RPC_URL=https://example-from-dotenv.invalid\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LATENTEDGE_RPC_URL", raising=False)

    captured: dict[str, str] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["rpc_url"] = rpc_url
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(cli, ["ingest", "--from-block", "1", "--to-block", "2"])

    assert captured["rpc_url"] == "https://example-from-dotenv.invalid"


def test_ingest_launches_dashboard_when_stdout_is_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launched = {"called": False}

    class _FakeApp:
        def __init__(self, start_screen):
            launched["called"] = True
            self.start_screen = start_screen

        def run(self):
            pass

    # CliRunner.invoke() replaces sys.stdout with its own capture stream
    # (click.testing._NamedTextIOWrapper) for the duration of the call,
    # so patching the pre-invoke stdout object's isatty has no effect on
    # what the command actually sees — patch the class instead.
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda self: True)
    monkeypatch.setattr("latentedge.cli.LatentEdgeApp", _FakeApp)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert launched["called"]


def test_ingest_falls_back_to_plain_output_when_stdout_is_not_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # CliRunner's captured stdout is never a real TTY, so this is
    # actually today's default behavior for every other CLI test in
    # this file too — this test makes that fallback explicit.
    # load_dotenv() sets real os.environ entries with no per-test
    # cleanup, and reads a real .env file from the current directory if
    # one exists (e.g. a developer's own RPC provider key) — clear the
    # env var and run from an empty tmp_path so the default is
    # deterministic regardless of test order or the real repo checkout.
    monkeypatch.delenv("LATENTEDGE_RPC_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    captured: dict[str, str] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["rpc_url"] = rpc_url
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code == 0
    assert captured["rpc_url"] == DEFAULT_RPC_URL
    assert "wrote 0 new swap records" in result.output


def test_train_launches_dashboard_when_stdout_is_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launched = {"called": False}

    class _FakeApp:
        def __init__(self, start_screen):
            launched["called"] = True

        def run(self):
            pass

    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda self: True)
    monkeypatch.setattr("latentedge.cli.LatentEdgeApp", _FakeApp)

    runner = CliRunner()
    runner.invoke(cli, ["train", "--swaps", str(tmp_path / "swaps.parquet")])

    assert launched["called"]


def test_ingest_without_block_range_derives_from_block_via_on_chain_timestamp_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
    monkeypatch.setattr("latentedge.cli.time.time", lambda: 1_700_000_000.0)

    captured_anchor_args: dict[str, int] = {}

    def fake_get_block_at_or_after_timestamp(client, rpc_url, target_timestamp, floor_block, head_block):
        captured_anchor_args["target_timestamp"] = target_timestamp
        captured_anchor_args["floor_block"] = floor_block
        captured_anchor_args["head_block"] = head_block
        return 19_000_000

    monkeypatch.setattr("latentedge.cli.get_block_at_or_after_timestamp", fake_get_block_at_or_after_timestamp)

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code == 0, result.output
    # from_block comes straight from the on-chain lookup, anchored to
    # exactly 1 day (in seconds) before the mocked "now".
    assert captured_anchor_args["target_timestamp"] == int(1_700_000_000.0 - 86400)
    assert captured_anchor_args["floor_block"] == config.POOL_CREATION_BLOCK
    assert captured_anchor_args["head_block"] == 20_000_000
    assert captured["from_block"] == 19_000_000
    assert captured["to_block"] == 20_000_000 - 5


def test_ingest_days_falls_back_to_env_var_when_flag_omitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
    monkeypatch.setattr("latentedge.cli.time.time", lambda: 1_700_000_000.0)

    captured_anchor_args: dict[str, int] = {}

    def fake_get_block_at_or_after_timestamp(client, rpc_url, target_timestamp, floor_block, head_block):
        captured_anchor_args["target_timestamp"] = target_timestamp
        return 19_000_000

    monkeypatch.setattr("latentedge.cli.get_block_at_or_after_timestamp", fake_get_block_at_or_after_timestamp)

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--out", str(tmp_path / "swaps.parquet")],
        env={"LATENTEDGE_INGEST_DAYS": "2"},
    )

    assert result.exit_code == 0, result.output
    assert captured_anchor_args["target_timestamp"] == int(1_700_000_000.0 - 2 * 86400)
    assert captured["to_block"] == 20_000_000 - 5
    assert captured["from_block"] == 19_000_000


def test_ingest_explicit_block_range_takes_priority_over_days(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def unexpected_call(client, rpc_url):
        raise AssertionError("get_latest_block should not be called when an explicit range is given")

    monkeypatch.setattr("latentedge.cli.get_latest_block", unexpected_call)
    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "10", "--to-block", "20", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code == 0, result.output
    assert captured["from_block"] == 10
    assert captured["to_block"] == 20


def test_ingest_reports_file_range_and_remaining_block_counts_in_plain_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    # Only blocks inside the requested range count as already in file —
    # the 10 blocks here sit entirely outside it.
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 9), (100, 104)])

    monkeypatch.setattr("latentedge.cli.ingest_range", lambda *args, **kwargs: 0)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "100", "--to-block", "129", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert "30 blocks in requested range" in result.output
    assert "5 already in file" in result.output
    assert "25 remaining to download" in result.output


def test_ingest_rejects_only_one_of_from_block_to_block(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "10", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code != 0
    assert "--from-block and --to-block" in result.output


def test_ingest_shows_plain_language_error_when_days_resolution_cannot_reach_rpc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Regression test: a real connection failure while resolving --days
    # into a block range used to dump a raw Python traceback (see the
    # bug report this fixes). It must instead exit cleanly with a
    # message a person can act on, after exhausting retries (not on the
    # very first failure — see the retry regression test below).
    import httpx as httpx_module

    def unreachable(client, rpc_url):
        raise httpx_module.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    monkeypatch.setattr("latentedge.cli.get_latest_block", unreachable)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet"),
            "--max-retries", "2", "--retry-backoff-seconds", "0.001",
        ],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "internet connection" in result.output


def test_ingest_retries_chain_head_lookup_before_giving_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Regression test: resolving --days into a block range used to make
    # exactly one un-retried get_latest_block call — a single transient
    # network hiccup at startup killed the whole run immediately, even
    # though every other RPC call in the pipeline retries generously.
    import httpx as httpx_module

    attempts = {"count": 0}

    def flaky_then_succeeds(client, rpc_url):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx_module.ConnectError("simulated transient failure")
        return 20_000_000

    monkeypatch.setattr("latentedge.cli.get_latest_block", flaky_then_succeeds)
    monkeypatch.setattr(
        "latentedge.cli.get_block_at_or_after_timestamp",
        lambda client, rpc_url, target_timestamp, floor_block, head_block: floor_block,
    )
    monkeypatch.setattr("latentedge.cli.ingest_range", lambda *args, **kwargs: 0)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet"),
            "--max-retries", "5", "--retry-backoff-seconds", "0.001",
        ],
    )

    assert result.exit_code == 0
    assert attempts["count"] == 3
    assert "retry" in result.output.lower()


def test_ingest_shows_plain_language_error_when_plain_mode_ingest_fails_to_connect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import httpx as httpx_module

    def unreachable(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        raise httpx_module.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    monkeypatch.setattr("latentedge.cli.ingest_range", unreachable)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "internet connection" in result.output


def test_ingest_days_window_stays_anchored_even_when_fully_already_ingested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The --days window is now a fixed, on-chain-verified anchor, not an
    # estimate that walks back to find new work — so a window that's
    # already fully ingested must still be requested as-is (reporting 0
    # new records, via ingest_range's own dedup), never silently shifted
    # to some earlier window instead.
    from latentedge.ingest.progress import write_progress

    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 13_000_000)
    monkeypatch.setattr(
        "latentedge.cli.get_block_at_or_after_timestamp",
        lambda client, rpc_url, target_timestamp, floor_block, head_block: 12_990_000,
    )

    out_path = tmp_path / "swaps.parquet"
    naive_to = 13_000_000 - 5
    write_progress(out_path, [(12_990_000, naive_to)])  # the whole anchored window is already ingested

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--days", "1", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["from_block"] == 12_990_000
    assert captured["to_block"] == naive_to
    assert "wrote 0 new swap records" in result.output


def test_ingest_max_rps_falls_back_to_env_var_with_no_cli_flag(monkeypatch: pytest.MonkeyPatch):
    # LATENTEDGE_INGEST_MAX_RPS is deliberately env-var-only — no --max-rps
    # flag — matching this repo's preference for env vars over new flags
    # for run options that aren't part of every invocation's everyday use.
    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["max_rps"] = kwargs["max_rps"]
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_INGEST_MAX_RPS": "2.5"},
    )

    assert captured["max_rps"] == 2.5


def test_ingest_max_rps_defaults_when_env_var_omitted(monkeypatch: pytest.MonkeyPatch):
    from latentedge.ingest.chunked import DEFAULT_MAX_RPS

    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["max_rps"] = kwargs["max_rps"]
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(cli, ["ingest", "--from-block", "1", "--to-block", "2"])

    assert captured["max_rps"] == DEFAULT_MAX_RPS


def test_train_refuses_data_with_gaps_in_block_coverage(tmp_path: Path):
    swaps = tmp_path / "swaps.parquet"
    write_progress(swaps, [(0, 99), (500, 599)])

    result = CliRunner().invoke(cli, ["train", "--swaps", str(swaps), "--out", str(tmp_path / "m.safetensors")])

    assert result.exit_code != 0
    assert "not continuous" in result.output
    assert "100-499" in result.output


def test_train_refuses_data_with_no_progress_record(tmp_path: Path):
    result = CliRunner().invoke(
        cli, ["train", "--swaps", str(tmp_path / "swaps.parquet"), "--out", str(tmp_path / "m.safetensors")],
    )

    assert result.exit_code != 0
    assert "no ingest progress record" in result.output


def test_continuity_check_passes_for_a_single_contiguous_interval(tmp_path: Path):
    from latentedge.cli import _require_continuous_data

    swaps = tmp_path / "swaps.parquet"
    write_progress(swaps, [(0, 99), (100, 199)])

    _require_continuous_data(swaps)  # must not raise


def _capture_ingest_range(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)
    return captured


def test_ingest_auto_throttle_off_passes_the_fixed_rps(monkeypatch: pytest.MonkeyPatch):
    captured = _capture_ingest_range(monkeypatch)
    result = CliRunner().invoke(
        cli, ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_INGEST_AUTO_THROTTLE": "false", "LATENTEDGE_INGEST_FIXED_RPS": "8.6"},
    )
    assert result.exit_code == 0, result.output
    assert captured["fixed_rps"] == 8.6


def test_ingest_auto_throttle_on_by_default_passes_no_fixed_rps(monkeypatch: pytest.MonkeyPatch):
    captured = _capture_ingest_range(monkeypatch)
    result = CliRunner().invoke(
        cli, ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_INGEST_FIXED_RPS": "8.6"},
    )
    assert result.exit_code == 0, result.output
    assert captured["fixed_rps"] is None


@pytest.mark.parametrize("fixed", [None, "0", "-3", "abc"])
def test_ingest_auto_throttle_off_requires_a_valid_fixed_rps(monkeypatch: pytest.MonkeyPatch, fixed: str | None):
    _capture_ingest_range(monkeypatch)
    env = {"LATENTEDGE_INGEST_AUTO_THROTTLE": "false"}
    if fixed is not None:
        env["LATENTEDGE_INGEST_FIXED_RPS"] = fixed
    result = CliRunner().invoke(cli, ["ingest", "--from-block", "1", "--to-block", "2"], env=env)
    assert result.exit_code != 0
    assert "LATENTEDGE_INGEST_FIXED_RPS" in result.output


def test_ingest_rejects_an_unrecognized_auto_throttle_value(monkeypatch: pytest.MonkeyPatch):
    _capture_ingest_range(monkeypatch)
    result = CliRunner().invoke(
        cli, ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_INGEST_AUTO_THROTTLE": "maybe"},
    )
    assert result.exit_code != 0
    assert "LATENTEDGE_INGEST_AUTO_THROTTLE" in result.output


def test_ingest_start_block_lookup_retries_through_rate_limits_instead_of_aborting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Regression test: only the chain-head lookup used to retry; a single
    # 429 during the start-block binary search aborted the whole start
    # (seen for real against Alchemy right after a previous run). A rate
    # limit must retry until it clears, even past --max-retries.
    from latentedge.ingest.rpc_logs import RateLimitError

    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
    monkeypatch.setattr("latentedge.cli.time.sleep", lambda seconds: None)

    calls = {"count": 0}

    def flaky_anchor(client, rpc_url, target_timestamp, floor_block, head_block):
        calls["count"] += 1
        if calls["count"] <= 4:  # more failures than --max-retries
            raise RateLimitError("429")
        return 19_000_000

    monkeypatch.setattr("latentedge.cli.get_block_at_or_after_timestamp", flaky_anchor)

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    result = CliRunner().invoke(
        cli, ["ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet"), "--max-retries", "2"],
    )

    assert result.exit_code == 0, result.output
    assert calls["count"] == 5
    assert captured["from_block"] == 19_000_000


def test_ingest_start_block_lookup_still_gives_up_on_a_non_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import httpx as httpx_module

    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
    monkeypatch.setattr("latentedge.cli.time.sleep", lambda seconds: None)

    def unreachable(client, rpc_url, target_timestamp, floor_block, head_block):
        raise httpx_module.ConnectError("no route")

    monkeypatch.setattr("latentedge.cli.get_block_at_or_after_timestamp", unreachable)

    result = CliRunner().invoke(
        cli, ["ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet"), "--max-retries", "2"],
    )

    assert result.exit_code != 0
    assert "internet connection" in result.output
