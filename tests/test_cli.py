from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

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


def test_backtest_stub_exits_non_zero_not_silently_succeed():
    # Regression test: the backtest subcommand is an intentional stub
    # (real wiring is separate follow-up work), but it must not exit 0 —
    # that would read as a successful backtest to anyone running it.
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest"])
    assert result.exit_code != 0


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


def test_ingest_without_block_range_derives_it_from_days_and_chain_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # A head well above config.POOL_CREATION_BLOCK (12_376_729) so the
    # naive window stays realistic — a head this low would sit entirely
    # before the pool existed and trigger the floor clamp instead of
    # this test's plain day-arithmetic path.
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
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
    # 1 day of 12s blocks, minus the safety buffer behind the head.
    assert captured["to_block"] == 20_000_000 - 5
    assert captured["from_block"] == 20_000_000 - 5 - 7200 + 1


def test_ingest_days_falls_back_to_env_var_when_flag_omitted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 20_000_000)
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
    assert captured["to_block"] == 20_000_000 - 5
    assert captured["from_block"] == 20_000_000 - 5 - 14400 + 1


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


def test_ingest_backfills_previously_skipped_gaps_before_the_requested_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    # Regression test for a real incident: an earlier run jumped straight
    # to an explicit --from-block/--to-block that left a large stretch of
    # history between two already-ingested ranges never fetched. Nothing
    # else in the codebase ever goes back to look for a gap like this, so
    # it must be closed automatically, before the range this invocation
    # actually asked for, and the user must be told it happened.
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 99), (500, 599)])  # a skipped gap at 100-499

    calls: list[tuple[int, int]] = []

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        calls.append((from_block, to_block))
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "700", "--to-block", "800", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    # The gap is backfilled first, in order, before the requested range.
    assert calls == [(100, 499), (700, 800)]
    assert "previously-skipped" in result.output
    assert "100-499" in result.output


def test_ingest_skips_backfill_entirely_when_progress_has_no_internal_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 99)])  # a single interval — nothing to backfill

    calls: list[tuple[int, int]] = []

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        calls.append((from_block, to_block))
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "200", "--to-block", "300", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [(200, 300)]
    assert "previously-skipped" not in result.output


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


def test_ingest_days_window_walks_back_past_already_ingested_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # A chain head high enough to stay well above the pool's real
    # deployment block (config.POOL_CREATION_BLOCK) so this test
    # exercises the backward walk itself, not the floor clamp.
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 13_000_000)

    out_path = tmp_path / "swaps.parquet"
    naive_to = 13_000_000 - 5
    blocks_in_range = 7200  # 1 day at 12s/block
    naive_from = naive_to - blocks_in_range + 1

    from latentedge.ingest.progress import write_progress

    write_progress(out_path, [(naive_from, naive_to)])  # the whole naive window is already ingested

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
    assert captured["to_block"] == naive_to
    # The whole naive window was already covered, so the request must
    # walk back to an earlier, equally-sized uncovered window instead of
    # silently doing nothing.
    assert captured["from_block"] == naive_from - blocks_in_range


def test_ingest_days_window_reports_zero_when_entire_pool_history_already_ingested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # The floor-reaching case: progress already covers everything back
    # to the pool's deployment block, so there's nothing left anywhere
    # in the pool's history to walk back to. This must be a normal "0
    # new records" outcome, not an error.
    from latentedge import config
    from latentedge.ingest.progress import write_progress

    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 13_000_000)

    out_path = tmp_path / "swaps.parquet"
    naive_to = 13_000_000 - 5
    write_progress(out_path, [(config.POOL_CREATION_BLOCK, naive_to)])

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--days", "1", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["from_block"] == config.POOL_CREATION_BLOCK  # walked all the way to the floor, no further
    assert "wrote 0 new swap records" in result.output
