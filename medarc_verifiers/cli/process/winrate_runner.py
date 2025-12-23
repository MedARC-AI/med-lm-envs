"""Helpers for computing win rates from processed parquet outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from medarc_verifiers.cli.process import winrate as _win
from medarc_verifiers.cli.process.hf_sync import HFSyncConfig, download_hf_repo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WinrateRunResult:
    """Captures win rate outputs for CLI display."""

    output_path: Path
    result: _win.ModelCentricResult
    datasets: Sequence[tuple[str, Sequence[_win.PLDataFrame]]]


def discover_datasets(processed_dir: Path) -> list[tuple[str, list[_win.PLDataFrame]]]:
    """Load env splits via datasets metadata; requires dataset_infos.json."""
    datasets = _load_with_datasets(processed_dir)
    if not datasets:
        raise ValueError(
            f"No dataset_infos.json found under {processed_dir}. Regenerate with medarc-new process before winrate."
        )
    return datasets


def run_winrate(
    *,
    processed_dir: Path,
    output_path: Path | None,
    output_name: str | None = None,
    config: _win.WinrateConfig,
    processed_at: str | None = None,
    hf_config: HFSyncConfig | None = None,
) -> WinrateRunResult:
    local_dir, datasets, source_desc = _resolve_source(processed_dir, hf_config)
    if not datasets:
        raise ValueError(f"No datasets found from {source_desc}.")

    resolved_output = output_path or _default_winrate_path(
        processed_dir, processed_at=processed_at, base_name=output_name
    )
    result = _win.compute_winrates(datasets, config)
    _win.write_json(_win.to_json(result), resolved_output)
    return WinrateRunResult(output_path=resolved_output, result=result, datasets=datasets)


def _default_winrate_path(processed_dir: Path, *, processed_at: str | None, base_name: str | None) -> Path:
    timestamp = _format_timestamp_for_filename(processed_at)
    root = _winrate_root(processed_dir)
    base = (base_name.strip() if base_name else "winrates") or "winrates"
    return root / "winrate" / f"{base}-{timestamp}.json"


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


def _resolve_source(
    processed_dir: Path,
    hf_config: HFSyncConfig | None,
) -> tuple[Path, list[tuple[str, list[_win.PLDataFrame]]], str]:
    if hf_config and hf_config.repo_id:
        local_dir = download_hf_repo(
            repo_id=hf_config.repo_id,
            branch=hf_config.branch,
            token=hf_config.token,
            local_dir=None,
            local_only=False,
        )
        datasets = discover_datasets(local_dir)
        source_desc = f"HF repo {hf_config.repo_id}"
        return local_dir, datasets, source_desc
    datasets = discover_datasets(processed_dir)
    source_desc = f"processed dir {processed_dir}"
    return processed_dir, datasets, source_desc


def _load_with_datasets(processed_dir: Path) -> list[tuple[str, list[_win.PLDataFrame]]]:
    """Load all splits via Hugging Face datasets; requires dataset_infos.json."""
    dataset_infos = processed_dir / "dataset_infos.json"
    if not dataset_infos.exists():
        raise ValueError(f"dataset_infos.json missing under {processed_dir}; run medarc-new process first.")
    try:
        from datasets import DatasetDict, DownloadConfig, disable_progress_bar, load_dataset  # type: ignore[import-not-found]
        from datasets.utils.logging import set_verbosity_error  # type: ignore[import-not-found]
    except Exception:
        raise

    try:
        disable_progress_bar()
        set_verbosity_error()
        cache_root = processed_dir.parent if processed_dir.parent != processed_dir else processed_dir
        # Prefer runs/.hf_cache when processed outputs live under runs/processed/<...>
        if processed_dir.name == "processed" and cache_root.name == "runs":
            cache_root = cache_root
        elif processed_dir.parent.name == "processed" and processed_dir.parent.parent.name == "runs":
            cache_root = processed_dir.parent.parent
        cache_dir = cache_root / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(dataset_infos.read_text(encoding="utf-8"))
        config_key = next(iter(payload.keys()))
        config_payload = payload.get(config_key) or {}
        env_id_map = config_payload.get("extras", {}).get("env_id_map", {})
        data_files_raw = config_payload.get("data_files") or {}
        data_files: dict[str, list[str]] = {}
        for split, paths in data_files_raw.items():
            data_files[split] = [str(processed_dir / path) for path in paths]
        download_config = DownloadConfig(disable_tqdm=True)
        ds_dict = load_dataset(
            "parquet",
            data_files=data_files,
            split=None,
            cache_dir=str(cache_dir),
            download_config=download_config,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to load datasets from {processed_dir} via dataset_infos: {exc}") from exc

    if not isinstance(ds_dict, DatasetDict):
        logger.debug("datasets.load_dataset returned non-DatasetDict; skipping HF path.")
        return []

    datasets: list[tuple[str, list[_win.PLDataFrame]]] = []
    for split_name, split in ds_dict.items():
        if split_name == "train":
            continue
        try:
            # Convert to Polars DataFrame; fallback to pandas if unavailable.
            if hasattr(split, "to_polars"):
                df = split.to_polars()  # type: ignore[attr-defined]
            else:
                df = _win.pl.from_pandas(split.to_pandas())
            env_id = env_id_map.get(split_name, split_name)
            datasets.append((env_id, [df]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to convert split %s to Polars: %s", split_name, exc)
            continue
    return datasets


def list_models(datasets: Sequence[tuple[str, Sequence[_win.PLDataFrame]]]) -> list[str]:
    """List unique model_id values present across datasets."""
    models: set[str] = set()
    for _, splits in datasets:
        try:
            for split in splits:
                lf = _win.read_dataset_lazy(split)
                cols = lf.collect_schema().names()
                if "model_id" not in cols:
                    continue
                df = lf.select("model_id").collect()
                for value in df.get_column("model_id").unique():
                    if value is None:
                        continue
                    models.add(str(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping dataset while listing models: %s", exc)
            continue
    return sorted(models)


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
        rows.append({"Model": model, "Average": sm_str, "Weighted Avg": lw_str, "Datasets": str(n_ds)})

    try:
        from tabulate import tabulate  # type: ignore[import-not-found]

        md_table = tabulate(rows, headers="keys", tablefmt="github")
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore[import-not-found]  # noqa: F401

        md_table = pd.DataFrame(rows).to_markdown(index=False)  # type: ignore[attr-defined]
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    lines: list[str] = [
        "",
        "Mean win rate per model (HELM-style):",
        "",
        "| Model | Average | Weighted Avg | Datasets |",
        "|-------|----------:|-----------:|---------:|",
    ]
    for row in rows:
        lines.append(f"| {row['Model']} | {row['Average']} | {row['Weighted Avg']} | {row['Datasets']} |")
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
    "_resolve_source",
    "list_models",
    "run_winrate",
    "print_winrate_summary_markdown",
]
