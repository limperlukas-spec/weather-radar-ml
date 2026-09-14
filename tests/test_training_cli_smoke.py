import os
import shutil
import subprocess
from pathlib import Path


def test_installed_cli_runs_default_pipeline_end_to_end(tmp_path: Path) -> None:
    command = shutil.which("weather-radar-ml")
    assert command is not None

    runs_root = tmp_path / "runs"
    env = os.environ.copy()
    for name in (
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
        "COV_CORE_BRANCH",
        "COVERAGE_PROCESS_START",
        "COVERAGE_FILE",
    ):
        env.pop(name, None)

    completed = subprocess.run(
        [command, f"output.runs_root={runs_root}"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert "run_id=" in completed.stdout
    assert "artifact=" in completed.stdout

    run_dirs = tuple(path for path in runs_root.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    assert sorted(path.name for path in run_dirs[0].iterdir()) == [
        "config.json",
        "history.json",
        "manifest.json",
    ]
