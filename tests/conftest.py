import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv_and_shell_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cli.py's `cli()` group calls load_dotenv() against the invoking
    process's cwd on every command, so running the suite from this repo's
    checkout picks up its real .env (RPC key, --days, --train-after-
    ingest, ...) plus any LATENTEDGE_* already exported in the
    developer's shell — either can silently override a test's own
    defaults or its explicit `env={...}` (regression: editing .env's
    real LATENTEDGE_TRAIN_AFTER_INGEST/LATENTEDGE_INGEST_DAYS values
    broke several ingest tests that never mentioned either var). Give
    every test a fresh cwd with no .env file and strip any leaked
    LATENTEDGE_* var so the suite's behavior depends only on what each
    test sets up itself.
    """
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("LATENTEDGE_"):
            monkeypatch.delenv(name, raising=False)
