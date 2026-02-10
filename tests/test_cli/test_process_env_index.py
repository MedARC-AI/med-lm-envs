from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.cli.process import env_index


def _write_env_index(path: Path, files: dict[str, dict[str, str]]) -> None:
    payload = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {},
        "files": files,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_env_index_inventory_skips_unsafe_keys(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write_env_index(
        processed_dir / "env_index.json",
        {
            "model-a/env-a.parquet": {"env_id": "env-a", "model_id": "model-a"},
            "../escape.parquet": {"env_id": "env-a", "model_id": "model-a"},
            "/abs/path.parquet": {"env_id": "env-a", "model_id": "model-a"},
        },
    )

    inventory = env_index.read_env_index_inventory(processed_dir)
    assert set(inventory.env_paths) == {"env-a"}
    resolved_paths = inventory.env_paths["env-a"]
    assert len(resolved_paths) == 1
    assert resolved_paths[0].as_posix().endswith("model-a/env-a.parquet")

    files = env_index.read_env_index_files(processed_dir)
    assert set(files) == {"model-a/env-a.parquet"}

    models = env_index.read_env_index_models(processed_dir)
    assert models == {"model-a"}
