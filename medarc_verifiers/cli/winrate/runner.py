"""Helpers for computing win rates from processed parquet outputs."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from medarc_verifiers.cli._constants import COMMAND, PROCESS_COMMAND
from medarc_verifiers.cli.process import workspace
from medarc_verifiers.cli.process.env_index import read_env_index_inventory, read_env_index_models
from medarc_verifiers.cli.winrate import api as _win
from medarc_verifiers.cli.hf import HFSyncConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WinrateRunResult:
    """Captures win rate outputs for CLI display."""

    output_path: Path
    result: _win.ModelCentricResult
    datasets: Sequence[tuple[str, Sequence[Path]]]
    output_paths: Sequence[Path] = ()


def discover_datasets(processed_dir: Path) -> list[tuple[str, list[Path]]]:
    """Load env datasets via env_index.json."""
    inventory = read_env_index_inventory(processed_dir)
    if not inventory.env_paths:
        raise ValueError(
            f"No env_index.json found under {processed_dir}. Regenerate with {COMMAND} {PROCESS_COMMAND} before winrate."
        )
    datasets = []
    for env_id in sorted(inventory.env_paths.keys()):
        datasets.append((env_id, sorted(inventory.env_paths[env_id])))
    return datasets


def run_winrate(
    *,
    processed_dir: Path,
    output_dir: Path,
    output_path: Path | None,
    output_name: str | None = None,
    config: _win.WinrateConfig,
    processed_at: str | None = None,
    hf_config: HFSyncConfig | None = None,
    hf_processed_pull: bool = False,
) -> WinrateRunResult:
    local_dir, datasets, source_desc = _resolve_source(
        processed_dir,
        hf_config=hf_config,
        hf_processed_pull=hf_processed_pull,
    )
    if not datasets:
        raise ValueError(f"No datasets found from {source_desc}.")

    resolved_output, output_paths, csv_paths = _resolve_output_paths(
        output_dir=output_dir,
        output_path=output_path,
        output_name=output_name,
        processed_at=processed_at,
    )
    known_models = read_env_index_models(local_dir)
    result = _win.compute_winrates(datasets, config, known_models=known_models or None)
    _win.write_json(_win.to_json(result), resolved_output)
    for extra_path in output_paths:
        if extra_path == resolved_output:
            continue
        _win.write_json(_win.to_json(result), extra_path)
    for csv_path in csv_paths:
        _write_model_csv(result, csv_path)
    return WinrateRunResult(
        output_path=resolved_output,
        result=result,
        datasets=datasets,
        output_paths=tuple(output_paths) + tuple(csv_paths) or (resolved_output,),
    )


def _default_winrate_path(output_dir: Path, *, processed_at: str | None, base_name: str | None) -> Path:
    timestamp = _format_timestamp_for_filename(processed_at)
    base = (base_name.strip() if base_name else "winrates") or "winrates"
    return output_dir / f"{base}-{timestamp}.json"


def _format_timestamp_for_filename(processed_at: str | None) -> str:
    if not processed_at:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        ts = processed_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return processed_at.replace(":", "-").replace(" ", "_")
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_output_paths(
    *,
    output_dir: Path,
    output_path: Path | None,
    output_name: str | None,
    processed_at: str | None,
) -> tuple[Path, Sequence[Path], Sequence[Path]]:
    if output_path is not None:
        return output_path, (output_path,), ()
    resolved = _default_winrate_path(output_dir, processed_at=processed_at, base_name=output_name)
    latest = output_dir / "latest.json"
    latest_csv = output_dir / "latest.csv"
    timestamp_csv = resolved.with_suffix(".csv")
    return resolved, (resolved, latest), (timestamp_csv, latest_csv)


def _resolve_source(
    processed_dir: Path,
    *,
    hf_config: HFSyncConfig | None,
    hf_processed_pull: bool,
) -> tuple[Path, list[tuple[str, list[Path]]], str]:
    if hf_config and hf_config.repo_id:
        local_dir = processed_dir
        should_pull = hf_processed_pull or not workspace.is_nonempty_dir(local_dir)
        if should_pull:
            workspace.prepare_hf_baseline(
                output_dir=local_dir,
                hf_config=hf_config,
                pull_policy="pull",
                is_tty=False,
                prompt_func=None,
            )
        datasets = discover_datasets(local_dir)
        source_desc = f"HF repo {hf_config.repo_id}"
        return local_dir, datasets, source_desc
    datasets = discover_datasets(processed_dir)
    source_desc = f"processed dir {processed_dir}"
    return processed_dir, datasets, source_desc


def _write_model_csv(result: _win.ModelCentricResult, path: Path) -> None:
    models = result.models or {}
    dataset_names: set[str] = set()
    for payload in models.values():
        if not isinstance(payload, dict):
            continue
        avg_rewards = payload.get("avg_reward_per_dataset")
        if isinstance(avg_rewards, dict):
            dataset_names.update(str(name) for name in avg_rewards.keys())
    ordered_datasets = sorted(dataset_names)
    headers = ["model", "weighted_winrate", "simple_winrate", *ordered_datasets, "num_datasets"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for model, payload in sorted(models.items(), key=lambda item: str(item[0])):
            if not isinstance(payload, dict):
                continue
            mean_payload = payload.get("mean_winrate", {}) if isinstance(payload, dict) else {}
            simple_mean = mean_payload.get("simple_mean")
            weighted_mean = mean_payload.get("weighted_mean")
            n_datasets = mean_payload.get("n_datasets", 0)
            avg_rewards = payload.get("avg_reward_per_dataset", {})
            row = [
                str(model),
                weighted_mean,
                simple_mean,
            ]
            for dataset in ordered_datasets:
                value = avg_rewards.get(dataset) if isinstance(avg_rewards, dict) else None
                row.append(value)
            row.append(n_datasets)
            writer.writerow(row)


def list_models(datasets: Sequence[tuple[str, Sequence[Path]]]) -> list[str]:
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
