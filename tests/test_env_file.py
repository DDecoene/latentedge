from pathlib import Path

from latentedge.env_file import update_env_value


def test_replaces_existing_value_and_keeps_the_rest(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# comment\nA=1\nLATENTEDGE_INGEST_FIXED_RPS=6.5\nB=2\n")

    update_env_value(env, "LATENTEDGE_INGEST_FIXED_RPS", "6.6")

    assert env.read_text() == "# comment\nA=1\nLATENTEDGE_INGEST_FIXED_RPS=6.6\nB=2\n"


def test_appends_when_missing_and_creates_file(tmp_path: Path):
    env = tmp_path / ".env"

    update_env_value(env, "X", "1")
    update_env_value(env, "Y", "2")

    assert env.read_text() == "X=1\nY=2\n"


def test_ignores_commented_out_assignment(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# X=old\n")

    update_env_value(env, "X", "1")

    assert env.read_text() == "# X=old\nX=1\n"
