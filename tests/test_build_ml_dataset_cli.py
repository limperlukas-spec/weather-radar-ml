import json
from pathlib import Path

from weather_radar_ml.entrypoints.build_ml_dataset import main


def test_cli_dry_run(tmp_path: Path, capsys) -> None:
    config = tmp_path / "dataset.yaml"
    config.write_text(
        """
name: dry-run
sources:
  - source_id: radar
    artifact_id: prepared-v1
    role: primary
    path: /not-needed.zarr
features:
  dynamic_inputs:
    - {name: rain, source_id: radar, variable: rain, kind: dynamic}
  targets:
    - {name: rain_target, source_id: radar, variable: rain, kind: dynamic}
temporal: {step_minutes: 5, history_minutes: 10, forecast_minutes: 10}
spatial: {mode: full_frame}
splits:
  intervals:
    - name: train
      start: "2023-01-01T00:00:00+00:00"
      end: "2023-01-02T00:00:00+00:00"
    - name: validation
      start: "2023-01-03T00:00:00+00:00"
      end: "2023-01-04T00:00:00+00:00"
    - name: test
      start: "2023-01-05T00:00:00+00:00" 
      end: "2023-01-06T00:00:00+00:00"
""".lstrip(),
        encoding="utf-8",
    )
    checksums = tmp_path / "checksums.json"
    checksums.write_text('{"radar":"abc123"}\n', encoding="utf-8")

    rc = main(
        [
            "--config",
            str(config),
            "--checksums",
            str(checksums),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert output["dry_run"] is True
    assert output["dataset_build_id"]
    assert not (tmp_path / "out").exists()
