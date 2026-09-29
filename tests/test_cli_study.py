import json
from pathlib import Path

from click.testing import CliRunner

from latentedge.cli import cli
from tests.test_cli_backtest import _write_trending_swaps

SMALL = {
    "LATENTEDGE_STUDY_HORIZONS_MINUTES": "10", "LATENTEDGE_STUDY_SEEDS": "0,1", "LATENTEDGE_STUDY_FOLDS": "3",
    "LATENTEDGE_STUDY_EPOCHS": "40", "LATENTEDGE_STUDY_BOOTSTRAPS": "30", "LATENTEDGE_STUDY_PERMUTATIONS": "30",
}


def _swaps(tmp_path: Path) -> Path:
    swaps = tmp_path / "swaps.parquet"
    _write_trending_swaps(swaps)
    return swaps


def test_study_runs_every_seed_and_fold_and_saves_a_record(tmp_path: Path):
    swaps = _swaps(tmp_path)
    out_dir = tmp_path / "studies"
    result = CliRunner().invoke(
        cli, ["study", "--swaps", str(swaps), "--out-dir", str(out_dir)], env=SMALL, catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    saved = sorted(out_dir.glob("*.json"))
    assert len(saved) == 1 and saved[0].with_suffix(".csv").exists()
    record = json.loads(saved[0].read_text())
    assert len(record["rows"]) == 2 * 3
    assert record["summaries"][0]["horizon_minutes"] == 10
    assert record["config"]["seeds"] == [0, 1]
    assert "gross corr" in result.output


def test_study_finds_the_planted_trend_with_a_small_p_value(tmp_path: Path):
    swaps = _swaps(tmp_path)
    out_dir = tmp_path / "studies"
    CliRunner().invoke(cli, ["study", "--swaps", str(swaps), "--out-dir", str(out_dir)], env=SMALL, catch_exceptions=False)
    record = json.loads(next(out_dir.glob("*.json")).read_text())
    pooled = record["summaries"][0]["pooled"]
    assert pooled["gross_correlation"] > 0.1
    assert pooled["p_value"] < 0.1


def test_bad_study_settings_are_refused_before_any_work(tmp_path: Path):
    swaps = _swaps(tmp_path)
    result = CliRunner().invoke(
        cli, ["study", "--swaps", str(swaps), "--out-dir", str(tmp_path / "o")],
        env={**SMALL, "LATENTEDGE_STUDY_TEST_SHARE": "1.5"},
    )
    assert result.exit_code != 0
    assert "LATENTEDGE_STUDY" in result.output
    assert not (tmp_path / "o").exists()
