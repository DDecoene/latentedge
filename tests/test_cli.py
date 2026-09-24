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
