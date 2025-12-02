"""Helpers for computing win rates from processed parquet outputs."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from medarc_verifiers.cli_new.process import winrate as _win
from medarc_verifiers.cli_new.process.hf_sync import HFSyncConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WinrateRunResult:
    """Captures win rate outputs for CLI display."""

    output_path: Path
    result: _win.ModelCentricResult
    datasets: Sequence[tuple[str, Path]]


def discover_datasets(processed_dir: Path) -> list[tuple[str, Path]]:
    """Locate env parquet outputs under a processed directory."""
    processed_dir = processed_dir.expanduser()
    index_path = processed_dir / "env_index.json"
    datasets: list[tuple[str, Path]] = []

    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            for item in payload.get("environments", []) or []:
                env_id = str(item.get("env_id") or item.get("base_env_id") or "").strip()
                path_str = item.get("path")
                if not env_id or not path_str:
                    continue
                datasets.append((env_id, _resolve_dataset_path(path_str, processed_dir)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read env_index.json at %s: %s; falling back to globbing.", index_path, exc)

    if not datasets:
        for path in sorted(processed_dir.glob("*.parquet")):
            env_id = path.stem
            datasets.append((env_id, path))

    return datasets


def _resolve_dataset_path(path_str: str, processed_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (processed_dir / candidate).resolve()


def run_winrate(
    *,
    processed_dir: Path,
    output_path: Path | None,
    config: _win.WinrateConfig,
    processed_at: str | None = None,
    hf_config: HFSyncConfig | None = None,
) -> WinrateRunResult:
    if hf_config and hf_config.repo_id:
        local_dir = _download_hf_repo(hf_config)
        datasets = discover_datasets(local_dir)
        source_desc = f"HF repo {hf_config.repo_id}"
    else:
        datasets = discover_datasets(processed_dir)
        source_desc = f"processed dir {processed_dir}"
    if not datasets:
        raise ValueError(f"No datasets found from {source_desc}.")

    resolved_output = output_path or _default_winrate_path(processed_dir, processed_at=processed_at)
    result = _win.compute_winrates(datasets, config)
    _win.write_json(_win.to_json(result), resolved_output)
    return WinrateRunResult(output_path=resolved_output, result=result, datasets=datasets)


def _default_winrate_path(processed_dir: Path, *, processed_at: str | None) -> Path:
    timestamp = _format_timestamp_for_filename(processed_at)
    root = _winrate_root(processed_dir)
    return root / "winrate" / f"winrates-{timestamp}.json"


def _format_timestamp_for_filename(processed_at: str | None) -> str:
    if not processed_at:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        ts = processed_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return processed_at.replace(":", "-").replace(" ", "_")
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _winrate_root(processed_dir: Path) -> Path:
    parent = processed_dir.parent
    if parent == processed_dir:
        return processed_dir
    return parent


def _download_hf_repo(config: HFSyncConfig) -> Path:
    """Download a HF dataset repo snapshot to a temp dir and return the path."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("huggingface_hub is required for HF-backed winrate downloads.") from exc

    temp_root = Path(tempfile.mkdtemp(prefix="winrate-hf-"))
    snapshot_download(
        repo_id=config.repo_id,
        repo_type="dataset",
        revision=config.branch,
        token=config.token,
        allow_patterns="*.parquet",
        local_dir=temp_root,
        local_dir_use_symlinks=False,
    )
    return temp_root


def print_winrate_summary_markdown(result: _win.ModelCentricResult) -> None:
    """Print a compact markdown table of mean win rate per model."""
    try:
        models = result.models  # dict[str, dict]
    except Exception:
        return
    scoreboard: list[tuple[str, float | None, float | None, int]] = []
    for model, payload in models.items():
        mean_wr = payload.get("mean_winrate", {}) if isinstance(payload, dict) else {}
        simple = mean_wr.get("simple_mean")
        weighted = mean_wr.get("weighted_mean")
        n_ds = int(mean_wr.get("n_datasets", 0) or 0)
        scoreboard.append((str(model), simple, weighted, n_ds))

    def _key(item: tuple[str, float | None, float | None, int]) -> float:
        _, sm, lw, _ = item
        return float(lw if lw is not None else (sm if sm is not None else float("-inf")))

    scoreboard.sort(key=_key, reverse=True)
    rows: list[dict[str, str]] = []
    for model, sm, lw, n_ds in scoreboard:
        sm_str = f"{sm:.4f}" if isinstance(sm, (int, float)) and sm is not None else "-"
        lw_str = f"{lw:.4f}" if isinstance(lw, (int, float)) and lw is not None else "-"
        rows.append({"Model": model, "SimpleAvg": sm_str, "LnWeighted": lw_str, "Datasets": str(n_ds)})

    try:
        from tabulate import tabulate  # type: ignore[import-not-found]

        md_table = tabulate(rows, headers="keys", tablefmt="github")
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    try:
        import polars as pl  # type: ignore[import-not-found]

        import pandas as pd  # type: ignore[import-not-found]  # noqa: F401

        df = pl.DataFrame(rows).to_pandas()
        md_table = df.to_markdown(index=False)  # type: ignore[attr-defined]
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    lines: list[str] = [
        "",
        "Mean win rate per model (HELM-style):",
        "",
        "| Model | SimpleAvg | LnWeighted | Datasets |",
        "|-------|----------:|-----------:|---------:|",
    ]
    for row in rows:
        lines.append(f"| {row['Model']} | {row['SimpleAvg']} | {row['LnWeighted']} | {row['Datasets']} |")
    _emit_markdown_table("\n".join(lines))


def _emit_markdown_table(md_text: str) -> None:
    header = "Mean win rate per model (HELM-style):"
    try:
        from rich.console import Console
    except Exception:
        print("\n" + header + "\n")
        print(md_text)
        return
    console = Console()
    console.print("\n" + header + "\n")
    console.print(md_text)


__all__ = [
    "WinrateRunResult",
    "discover_datasets",
    "run_winrate",
    "print_winrate_summary_markdown",
]
