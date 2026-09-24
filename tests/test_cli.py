from click.testing import CliRunner

from latentedge.cli import cli


def test_cli_exposes_expected_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "train" in result.output
    assert "backtest" in result.output
