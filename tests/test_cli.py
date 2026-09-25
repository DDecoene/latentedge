from pathlib import Path

import pytest
from click.testing import CliRunner

from latentedge.cli import DEFAULT_RPC_URL, cli


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
    # message a person can act on.
    import httpx as httpx_module

    def unreachable(client, rpc_url):
        raise httpx_module.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    monkeypatch.setattr("latentedge.cli.get_latest_block", unreachable)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--days", "1", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "internet connection" in result.output


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
