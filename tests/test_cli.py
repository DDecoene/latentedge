from pathlib import Path

import pytest
from click.testing import CliRunner

from latentedge.cli import cli


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
