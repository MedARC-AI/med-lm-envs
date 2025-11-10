"""Top-level pipeline wiring discovery, row loading, aggregation, and writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from medarc_verifiers.cli_new._schemas import EnvironmentExportConfig
from medarc_verifiers.cli_new.process import aggregate, discovery, hf_sync, metadata, rows, rollout, writer
from medarc_verifiers.cli_new.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli_new.process.hf_sync import HFMergeSummary, HFSyncConfig
from medarc_verifiers.cli_new.process.writer import EnvWriteSummary, WriterConfig


@dataclass(slots=True)
class ProcessOptions:
    """User-configurable knobs for the process pipeline."""

    runs_dir: Path
    output_dir: Path
    exporter_version: str = "dev"
    processed_at: str | None = None
    processed_with_args: Mapping[str, Any] = field(default_factory=dict)
    status_filter: Sequence[str] = field(default_factory=tuple)
    include_prompt_completion: bool | None = None
    keep_columns: Sequence[str] = field(default_factory=tuple)
    drop_columns: Sequence[str] = field(default_factory=tuple)
    combine_rollouts: bool | None = None
    deduplicate_latest: bool = True
    dry_run: bool = False
    overwrite: bool = False
    hf_config: HFSyncConfig | None = None

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.output_dir = Path(self.output_dir)
        if not self.processed_at:
            self.processed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.status_filter = tuple(str(status) for status in self.status_filter)
        self.keep_columns = tuple(str(column).strip() for column in self.keep_columns if str(column).strip())
        self.drop_columns = tuple(str(column).strip() for column in self.drop_columns if str(column).strip())


@dataclass(slots=True)
class ProcessResult:
    """Outcome of a process pipeline execution."""

    records_processed: int
    rows_processed: int
    env_groups: list[AggregatedEnvRows]
    env_summaries: list[EnvWriteSummary]
    hf_summary: HFMergeSummary | None


def run_process(
    options: ProcessOptions,
    *,
    env_export_map: Mapping[str, EnvironmentExportConfig] | None = None,
) -> ProcessResult:
    """Run the exporter pipeline from discovery through Parquet output (and HF sync)."""
    env_export_map = env_export_map or {}

    records = discovery.discover_run_records(
        options.runs_dir,
        filter_status=options.status_filter or None,
    )

    # Deduplicate to keep only latest runs per model+env
    if options.deduplicate_latest:
        records = discovery.deduplicate_records_by_latest(records)

    normalized_rows: list[dict[str, Any]] = []

    for record in records:
        env_export = _resolve_env_export(record.manifest_env_id, env_export_map)
        include_prompt = _resolve_include_prompt(options, env_export)
        keep_columns = _resolve_columns(options.keep_columns, env_export.keep_columns if env_export else ())
        drop_columns = _resolve_columns(options.drop_columns, env_export.drop_columns if env_export else ())
        combine_rollouts = _resolve_combine_rollouts(options, env_export)

        normalized = metadata.load_normalized_metadata(record, combine_rollouts=combine_rollouts)
        row_batch = rows.load_rows(
            normalized,
            include_prompt_completion=include_prompt,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
        )
        normalized_rows.extend(row_batch)

    env_groups = aggregate.aggregate_rows_by_env(normalized_rows)
    writer_config = WriterConfig(
        output_dir=options.output_dir,
        exporter_version=options.exporter_version,
        processed_at=options.processed_at or "",
        processed_with_args=options.processed_with_args,
        dry_run=options.dry_run,
        overwrite=options.overwrite,
    )
    env_summaries = writer.write_env_groups(env_groups, writer_config)

    hf_summary: HFMergeSummary | None = None
    if options.hf_config:
        hf_summary = hf_sync.sync_to_hub(env_summaries, options.hf_config)

    return ProcessResult(
        records_processed=len(records),
        rows_processed=len(normalized_rows),
        env_groups=env_groups,
        env_summaries=env_summaries,
        hf_summary=hf_summary,
    )


def _resolve_env_export(
    manifest_env_id: str | None,
    env_export_map: Mapping[str, EnvironmentExportConfig],
) -> EnvironmentExportConfig | None:
    if not manifest_env_id:
        return None
    if manifest_env_id in env_export_map:
        return env_export_map[manifest_env_id]
    base_env_id, _ = rollout.derive_base_env_id(manifest_env_id)
    if base_env_id and base_env_id in env_export_map:
        return env_export_map[base_env_id]
    return None


def _resolve_include_prompt(
    options: ProcessOptions,
    env_export: EnvironmentExportConfig | None,
) -> bool:
    if options.include_prompt_completion is not None:
        return options.include_prompt_completion
    if env_export and env_export.include_prompt_completion is not None:
        return env_export.include_prompt_completion
    return False


def _resolve_columns(
    override_columns: Sequence[str],
    env_columns: Sequence[str],
) -> Sequence[str]:
    if override_columns:
        return override_columns
    if env_columns:
        return env_columns
    return ()


def _resolve_combine_rollouts(
    options: ProcessOptions,
    env_export: EnvironmentExportConfig | None,
) -> bool:
    if options.combine_rollouts is not None:
        return options.combine_rollouts
    if env_export:
        return env_export.combine_rollouts
    return True


__all__ = ["ProcessOptions", "ProcessResult", "run_process"]
