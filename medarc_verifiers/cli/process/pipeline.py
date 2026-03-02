"""Top-level pipeline wiring discovery, selection, row loading, aggregation, and writing."""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

from medarc_verifiers.cli import hf as hf_sync
from medarc_verifiers.cli._schemas import EnvironmentExportConfig
from medarc_verifiers.cli.hf import HFSyncConfig, HFSyncSummary
from medarc_verifiers.cli.process import aggregate, discovery, env_index, metadata, rollout, rows, workspace, writer
from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.process.metadata import RunIdentity
from medarc_verifiers.cli.process.writer import EXPORTER_METADATA_KEY, EnvWriteSummary, WriterConfig
from medarc_verifiers.cli.utils.shared import (
    count_jsonl_rows,
    dataset_is_excluded,
    model_is_excluded,
    normalize_dataset_ids,
    normalize_model_ids,
)

logger = logging.getLogger(__name__)
PROCESS_DEFAULT_STATUS_FILTER: tuple[str, ...] = ("completed",)


@dataclass(slots=True)
class ProcessOptions:
    """User-configurable knobs for the process pipeline."""

    runs_dir: Path
    output_dir: Path
    max_results_missing_pct: float = 2.5
    exclude_datasets: Sequence[str] = field(default_factory=tuple)
    exclude_models: Sequence[str] = field(default_factory=tuple)
    replace_models: Sequence[str] = field(default_factory=tuple)
    replace_envs: Sequence[str] = field(default_factory=tuple)
    processed_at: str | None = None
    processed_with_args: Mapping[str, Any] = field(default_factory=dict)
    status_filter: Sequence[str] = field(default_factory=lambda: PROCESS_DEFAULT_STATUS_FILTER)
    dry_run: bool = False
    clean: bool = False
    assume_yes: bool = False
    hf_config: HFSyncConfig | None = None
    hf_pull_policy: str | None = None
    max_workers: int = 4

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.output_dir = Path(self.output_dir)
        self.max_results_missing_pct = float(self.max_results_missing_pct)
        self.max_workers = max(1, int(self.max_workers))
        if not self.processed_at:
            self.processed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.status_filter = tuple(str(status) for status in self.status_filter)
        self.exclude_datasets = tuple(str(value) for value in self.exclude_datasets if str(value).strip())
        self.exclude_models = tuple(str(value) for value in self.exclude_models if str(value).strip())
        self.replace_models = tuple(str(value) for value in self.replace_models if str(value).strip())
        self.replace_envs = tuple(str(value) for value in self.replace_envs if str(value).strip())


@dataclass(slots=True)
class ProcessResult:
    """Outcome of a process pipeline execution."""

    records_processed: int
    rows_processed: int
    env_groups: list[AggregatedEnvRows]
    env_summaries: list[EnvWriteSummary]
    hf_summary: HFSyncSummary | None


@dataclass(frozen=True, slots=True)
class PlannedRecord:
    """Per-record settings for row loading."""

    normalized: metadata.NormalizedMetadata
    extra_columns: Sequence[str]
    drop_columns: Sequence[str]
    answer_column: str | None


@dataclass(frozen=True, slots=True)
class PlannedWorkItem:
    """A single selected (model, env) output to process."""

    identity: RunIdentity
    records: list[PlannedRecord]


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """Selection-time record settings before full normalization."""

    record: discovery.RunRecord
    identity: metadata.ResolvedRunIdentity
    combine_rollouts: bool
    extra_columns: Sequence[str]
    drop_columns: Sequence[str]
    answer_column: str | None


@dataclass(frozen=True, slots=True)
class SelectionWorkItem:
    """A selected work item before metadata normalization."""

    identity: metadata.ResolvedRunIdentity
    records: list[SelectionRecord]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Complete output of the selection phase."""

    work_items: list[PlannedWorkItem]
    skipped_by_delta: int
    skipped_by_exclusion: int
    total_discovered: int


def run_process(
    options: ProcessOptions,
    *,
    env_export_map: Mapping[str, EnvironmentExportConfig] | None = None,
) -> ProcessResult:
    """Run the exporter pipeline from discovery through Parquet output (and HF sync)."""
    env_export_map = env_export_map or {}

    def _run_pipeline() -> ProcessResult:
        if not options.dry_run:
            workspace.prepare_output_workspace(
                output_dir=options.output_dir,
                hf_config=options.hf_config,
                pull_policy=options.hf_pull_policy,
                clean=options.clean,
                assume_yes=options.assume_yes,
                is_tty=sys.stdin.isatty(),
                prompt_func=input,
            )

        index_files = {} if options.clean else env_index.read_env_index_files(options.output_dir)
        discovered = discovery.discover_run_records(
            options.runs_dir,
            filter_status=options.status_filter or None,
        )
        selection = select_work_items(
            discovered,
            options=options,
            env_export_map=env_export_map,
            index_files=index_files,
        )
        selected_records = [planned.normalized.record for item in selection.work_items for planned in item.records]
        _print_records_table(
            discovered,
            selected_records,
            options.max_results_missing_pct,
            exclude_datasets=options.exclude_datasets,
            exclude_models=options.exclude_models,
            skipped_by_delta=selection.skipped_by_delta,
            skipped_by_exclusion=selection.skipped_by_exclusion,
        )

        run_metadata: dict[str, dict[str, Any]] = {}
        for item in selection.work_items:
            for planned in item.records:
                record = planned.normalized.record
                run_metadata.setdefault(
                    record.manifest.job_run_id,
                    {
                        "created_at": record.manifest.created_at,
                        "updated_at": _source_updated_at(record),
                        "config_checksum": record.manifest.config_checksum,
                    },
                )

        writer_config = WriterConfig(
            output_dir=options.output_dir,
            processed_at=options.processed_at or "",
            processed_with_args=options.processed_with_args,
            dry_run=options.dry_run,
        )

        env_groups: list[AggregatedEnvRows] = []
        env_summaries: list[EnvWriteSummary] = []
        rows_processed = 0
        work_items = sorted(
            selection.work_items, key=lambda item: (item.identity.model_id, item.identity.output_env_id)
        )

        try:
            if options.max_workers <= 1 or len(work_items) <= 1:
                work_iter: Iterable[PlannedWorkItem] = work_items
                try:
                    from rich.progress import track

                    work_iter = track(work_items, description="Processing datasets", transient=True)
                except Exception:
                    work_iter = work_items

                for item in work_iter:
                    aggregated, row_count = _process_env_group(item)
                    rows_processed += row_count
                    env_groups.extend(aggregated)
                    summaries = writer.write_env_groups(aggregated, writer_config, write_index=False)
                    env_summaries.extend(summaries)
                    if not options.dry_run:
                        for group in aggregated:
                            group.rows.clear()
            else:
                executor: ProcessPoolExecutor | None = None
                futures = []
                try:
                    executor = ProcessPoolExecutor(max_workers=options.max_workers)
                    for item in work_items:
                        futures.append(executor.submit(_process_env_group, item))

                    future_iter: Iterable[Any] = as_completed(futures)
                    try:
                        from rich.progress import track

                        future_iter = track(
                            future_iter, total=len(futures), description="Processing datasets", transient=True
                        )
                    except Exception:
                        future_iter = as_completed(futures)

                    for future in future_iter:
                        aggregated, row_count = future.result()
                        rows_processed += row_count
                        env_groups.extend(aggregated)
                        summaries = writer.write_env_groups(aggregated, writer_config, write_index=False)
                        env_summaries.extend(summaries)
                        if not options.dry_run:
                            for group in aggregated:
                                group.rows.clear()
                except KeyboardInterrupt:
                    logger.warning("Processing cancelled by user; shutting down workers.")
                    for future in futures:
                        future.cancel()
                    if executor is not None:
                        executor.shutdown(cancel_futures=True)
                    raise
                finally:
                    if executor is not None:
                        try:
                            executor.shutdown(wait=True, cancel_futures=False)
                        except Exception:
                            pass
        except KeyboardInterrupt:
            logger.warning("Processing cancelled by user; partial outputs may exist.")
            raise

        metadata_paths: list[Path] = []
        if writer.write_hf_dataset_config(env_summaries, writer_config):
            metadata_paths.append(Path("dataset_infos.json"))
        if writer.write_env_index(env_summaries, writer_config, run_metadata=run_metadata):
            metadata_paths.append(Path("env_index.json"))

        hf_summary: HFSyncSummary | None = None
        if options.hf_config:
            hf_summary = hf_sync.sync_to_hub(
                env_summaries,
                options.hf_config,
                output_dir=options.output_dir,
                metadata_paths=metadata_paths,
                is_tty=sys.stdin.isatty(),
                assume_yes=options.assume_yes,
                prompt_func=input,
            )

        if options.dry_run:
            env_groups = [_strip_env_group_rows(group) for group in env_groups]

        return ProcessResult(
            records_processed=len(selected_records),
            rows_processed=rows_processed,
            env_groups=env_groups,
            env_summaries=env_summaries,
            hf_summary=hf_summary,
        )

    if options.dry_run:
        return _run_pipeline()
    workspace.ensure_output_dir(options.output_dir)
    return _run_pipeline()


def select_work_items(
    discovered: Sequence[discovery.RunRecord],
    *,
    options: ProcessOptions,
    env_export_map: Mapping[str, EnvironmentExportConfig],
    index_files: Mapping[str, Mapping[str, Any]],
) -> SelectionResult:
    """Filter discovered runs down to selected work items before row loading begins."""
    planned_records = [_plan_selection_record(record, env_export_map) for record in discovered]
    _raise_for_latest_invalid_selection(planned_records)
    work_items = _materialize_work_items(
        _select_latest_work_items([record for record in planned_records if record.identity.model_id])
    )

    work_items, skipped_by_exclusion = _apply_exclusions(
        work_items,
        exclude_datasets=options.exclude_datasets,
        exclude_models=options.exclude_models,
    )
    _validate_replace_targets(work_items, options)
    work_items, skipped_by_delta = _apply_additive_delta(work_items, options=options, index_files=index_files)
    _validate_selected_results_completeness(work_items, max_results_missing_pct=options.max_results_missing_pct)

    return SelectionResult(
        work_items=work_items,
        skipped_by_delta=skipped_by_delta,
        skipped_by_exclusion=skipped_by_exclusion,
        total_discovered=len(discovered),
    )


def _resolve_env_export(
    manifest_env_id: str | None,
    env_export_map: Mapping[str, EnvironmentExportConfig],
) -> EnvironmentExportConfig:
    if not manifest_env_id:
        return EnvironmentExportConfig()
    if manifest_env_id in env_export_map:
        return env_export_map[manifest_env_id]
    base_env_id, _ = rollout.derive_base_env_id(manifest_env_id)
    if base_env_id and base_env_id in env_export_map:
        return env_export_map[base_env_id]
    return EnvironmentExportConfig()


def _resolve_columns(env_columns: Sequence[str]) -> Sequence[str]:
    return tuple(str(column).strip() for column in env_columns if str(column).strip())


def _plan_selection_record(
    record: discovery.RunRecord,
    env_export_map: Mapping[str, EnvironmentExportConfig],
) -> SelectionRecord:
    env_export = _resolve_env_export(record.manifest_env_id, env_export_map)
    combine_rollouts = bool(env_export.combine_rollouts)
    identity = metadata.resolve_run_identity(record, combine_rollouts=combine_rollouts)
    return SelectionRecord(
        record=record,
        identity=identity,
        combine_rollouts=combine_rollouts,
        extra_columns=_resolve_columns(env_export.extra_columns),
        drop_columns=_resolve_columns(env_export.drop_columns),
        answer_column=env_export.answer_column,
    )


def _raise_for_latest_invalid_selection(records: Sequence[SelectionRecord]) -> None:
    latest_by_target: dict[tuple[str, str], SelectionRecord] = {}
    for planned in records:
        selection_key = (planned.identity.output_env_id, planned.record.job_id)
        current = latest_by_target.get(selection_key)
        if current is None or _run_sort_key(
            _source_updated_at(planned.record),
            planned.record.manifest.job_run_id,
        ) > _run_sort_key(_source_updated_at(current.record), current.record.manifest.job_run_id):
            latest_by_target[selection_key] = planned

    invalid_latest = [planned for planned in latest_by_target.values() if not planned.identity.model_id]
    if not invalid_latest:
        return

    failing = sorted(
        invalid_latest,
        key=lambda planned: (
            planned.identity.output_env_id,
            _run_sort_key(_source_updated_at(planned.record), planned.record.manifest.job_run_id),
        ),
    )[-1]
    raise RuntimeError(metadata.format_missing_model_id_error(failing.record))


def _select_latest_work_items(records: Sequence[SelectionRecord]) -> list[SelectionWorkItem]:
    grouped: dict[tuple[str, str], dict[str, list[SelectionRecord]]] = {}
    run_timestamps: dict[str, str] = {}

    for planned in records:
        identity = planned.identity
        if not identity.model_id:
            continue
        group_key = (identity.model_id, identity.output_env_id)
        grouped.setdefault(group_key, {}).setdefault(identity.job_run_id, []).append(planned)
        run_timestamps.setdefault(identity.job_run_id, _source_updated_at(planned.record))

    selected: list[SelectionWorkItem] = []
    for _, run_groups in grouped.items():
        latest_run_id = max(run_groups.keys(), key=lambda run_id: _run_sort_key(run_timestamps.get(run_id, ""), run_id))
        latest_records = run_groups[latest_run_id]
        representative = latest_records[0]
        selected.append(
            SelectionWorkItem(
                identity=representative.identity,
                records=list(latest_records),
            )
        )
    return selected


def _materialize_work_items(items: Sequence[SelectionWorkItem]) -> list[PlannedWorkItem]:
    materialized: list[PlannedWorkItem] = []
    for item in items:
        records: list[PlannedRecord] = []
        for selected in item.records:
            normalized = metadata.load_normalized_metadata(
                selected.record,
                combine_rollouts=selected.combine_rollouts,
            )
            records.append(
                PlannedRecord(
                    normalized=normalized,
                    extra_columns=selected.extra_columns,
                    drop_columns=selected.drop_columns,
                    answer_column=selected.answer_column,
                )
            )
        materialized.append(PlannedWorkItem(identity=records[0].normalized.identity, records=records))
    return materialized


def _apply_exclusions(
    work_items: Sequence[PlannedWorkItem],
    *,
    exclude_datasets: Sequence[str],
    exclude_models: Sequence[str],
) -> tuple[list[PlannedWorkItem], int]:
    exclude_dataset_set = normalize_dataset_ids(exclude_datasets, label="process exclude dataset")
    exclude_model_set = normalize_model_ids(exclude_models, label="process exclude model")
    filtered: list[PlannedWorkItem] = []
    skipped = 0
    for item in work_items:
        if exclude_dataset_set and _env_is_excluded(item.identity.output_env_id, exclude_dataset_set):
            skipped += 1
            continue
        if exclude_model_set and model_is_excluded(item.identity.model_id, exclude_model_set):
            skipped += 1
            continue
        filtered.append(item)
    return filtered, skipped


def _validate_replace_targets(work_items: Sequence[PlannedWorkItem], options: ProcessOptions) -> None:
    if not options.replace_models and not options.replace_envs:
        return

    if options.replace_models:
        matched_models = {
            item.identity.model_id for item in work_items if item.identity.model_id in options.replace_models
        }
        if not matched_models:
            raise RuntimeError(
                "No selected processed outputs match --replace-model values: "
                f"{', '.join(sorted(options.replace_models))}."
            )
    if options.replace_envs:
        matched_envs = {
            item.identity.output_env_id for item in work_items if item.identity.output_env_id in options.replace_envs
        }
        if not matched_envs:
            raise RuntimeError(
                f"No selected processed outputs match --replace-env values: {', '.join(sorted(options.replace_envs))}."
            )
    if options.replace_models and options.replace_envs:
        intersection = [
            item
            for item in work_items
            if item.identity.model_id in options.replace_models and item.identity.output_env_id in options.replace_envs
        ]
        if not intersection:
            raise RuntimeError(
                "No selected processed outputs match the intersection of --replace-model and --replace-env."
            )


def _apply_additive_delta(
    work_items: Sequence[PlannedWorkItem],
    *,
    options: ProcessOptions,
    index_files: Mapping[str, Mapping[str, Any]],
) -> tuple[list[PlannedWorkItem], int]:
    if options.clean:
        return list(work_items), 0

    filtered: list[PlannedWorkItem] = []
    skipped = 0
    for item in work_items:
        output_path = writer.build_output_path(
            options.output_dir,
            model_id=item.identity.model_id,
            env_id=item.identity.output_env_id,
        )
        if not output_path.exists():
            filtered.append(item)
            continue
        if _should_replace_existing_output(item.identity, options):
            filtered.append(item)
            continue
        parquet_metadata = _read_existing_output_metadata(output_path)
        _validate_existing_output_integrity(
            output_path,
            output_dir=options.output_dir,
            index_files=index_files,
            parquet_metadata=parquet_metadata,
        )
        if not _existing_output_matches_selected_runs(item, parquet_metadata):
            filtered.append(item)
            continue
        skipped += 1
    return filtered, skipped


def _should_replace_existing_output(identity: RunIdentity, options: ProcessOptions) -> bool:
    if options.clean:
        return True
    has_model_filter = bool(options.replace_models)
    has_env_filter = bool(options.replace_envs)
    if not has_model_filter and not has_env_filter:
        return False
    if has_model_filter and has_env_filter:
        return identity.model_id in options.replace_models and identity.output_env_id in options.replace_envs
    if has_model_filter:
        return identity.model_id in options.replace_models
    return identity.output_env_id in options.replace_envs


def _read_existing_output_metadata(output_path: Path) -> pq.FileMetaData:
    try:
        metadata_obj = pq.ParquetFile(output_path).metadata
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Existing processed output {output_path} is unreadable. "
            "Rebuild it with --replace-model/--replace-env or re-run with --clean."
        ) from exc

    if metadata_obj is None:
        raise RuntimeError(
            f"Existing processed output {output_path} is missing parquet footer metadata. "
            "Rebuild it with --replace-model/--replace-env or re-run with --clean."
        )
    return metadata_obj


def _validate_existing_output_integrity(
    output_path: Path,
    *,
    output_dir: Path,
    index_files: Mapping[str, Mapping[str, Any]],
    parquet_metadata: pq.FileMetaData | None = None,
) -> None:
    metadata_obj = parquet_metadata or _read_existing_output_metadata(output_path)

    rel_key = output_path.relative_to(output_dir).as_posix()
    index_entry = index_files.get(rel_key)
    if not isinstance(index_entry, Mapping):
        return
    expected_row_count = index_entry.get("row_count")
    if expected_row_count is None:
        return
    try:
        expected = int(expected_row_count)
    except (TypeError, ValueError):
        return
    actual = int(metadata_obj.num_rows)
    if actual != expected:
        raise RuntimeError(
            f"Existing processed output {output_path} has {actual} parquet rows but env_index.json records {expected}. "
            "Rebuild it with --replace-model/--replace-env or re-run with --clean."
        )


def _existing_output_matches_selected_runs(item: PlannedWorkItem, parquet_metadata: pq.FileMetaData) -> bool:
    existing_run_ids = _extract_exporter_source_runs(parquet_metadata)
    if existing_run_ids is None:
        return False
    selected_run_ids = {planned.normalized.record.manifest.job_run_id for planned in item.records}
    return existing_run_ids == selected_run_ids


def _extract_exporter_source_runs(parquet_metadata: pq.FileMetaData) -> set[str] | None:
    metadata_map = parquet_metadata.metadata
    if not metadata_map:
        return None
    payload = metadata_map.get(EXPORTER_METADATA_KEY)
    if not payload:
        return None
    try:
        exporter_metadata = json.loads(payload.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    source_runs = exporter_metadata.get("source_runs")
    if not isinstance(source_runs, list):
        return None
    run_ids = {str(run_id).strip() for run_id in source_runs if str(run_id).strip()}
    return run_ids or None


def _print_records_table(
    discovered: Sequence[discovery.RunRecord],
    selected: Sequence[discovery.RunRecord],
    max_results_missing_pct: float,
    *,
    exclude_datasets: Sequence[str] = (),
    exclude_models: Sequence[str] = (),
    skipped_by_delta: int = 0,
    skipped_by_exclusion: int = 0,
) -> None:
    """Pretty-print job discovery vs planned processing."""
    exclude_set = normalize_dataset_ids(exclude_datasets, label="process exclude dataset")
    exclude_model_set = normalize_model_ids(exclude_models, label="process exclude model")
    eligible_discovered = [
        rec
        for rec in discovered
        if not (exclude_set and _record_is_excluded(rec, exclude_set))
        and not (exclude_model_set and _record_model_is_excluded(rec, exclude_model_set))
    ]
    total_by_model: dict[str, int] = {}
    completed_by_model: dict[str, int] = {}
    selected_by_model: dict[str, int] = {}
    for rec in eligible_discovered:
        model_id = rec.model_id or "unknown"
        total_by_model[model_id] = total_by_model.get(model_id, 0) + 1
        if (rec.status or "").lower() in PROCESS_DEFAULT_STATUS_FILTER:
            completed_by_model[model_id] = completed_by_model.get(model_id, 0) + 1
    for rec in selected:
        model_id = rec.model_id or "unknown"
        selected_by_model[model_id] = selected_by_model.get(model_id, 0) + 1

    models = sorted(set(total_by_model.keys()) | set(selected_by_model.keys()))
    selected_models = sorted(model_id for model_id, count in selected_by_model.items() if count > 0)
    discovered_jobs_total = sum(total_by_model.get(model_id, 0) for model_id in models)
    selected_jobs_total = sum(selected_by_model.get(model_id, 0) for model_id in models)

    try:
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table
    except Exception:
        logger.info(
            "Processing %d job(s) across %d model(s) (max_results_missing_pct=%s; found %d eligible job(s) across %d model(s)); "
            "excluded=%d existing=%d.",
            selected_jobs_total,
            len(selected_models),
            _format_missing_pct(max_results_missing_pct),
            discovered_jobs_total,
            len(models),
            skipped_by_exclusion,
            skipped_by_delta,
        )
        for model_id in models:
            completed = completed_by_model.get(model_id, 0)
            total = total_by_model.get(model_id, 0)
            selected_count = selected_by_model.get(model_id, 0)
            logger.info("  - %s: selected=%d; %d/%d completed", model_id, selected_count, completed, total)
        return

    console = Console()
    title = (
        f"Processing {selected_jobs_total} job(s) across {len(selected_models)} model(s) "
        f"[dim](max_results_missing_pct={_format_missing_pct(max_results_missing_pct)})[/dim]"
    )
    title += (
        f" [dim](found {discovered_jobs_total} eligible job(s); excluded={skipped_by_exclusion}, "
        f"existing={skipped_by_delta})[/dim]"
    )
    table = Table(title=title, show_header=True, header_style="bold cyan", caption=None)
    table.add_column("Model", style="magenta")
    table.add_column("Jobs (completed/total)", style="green", justify="right")
    table.add_column("Selected", style="cyan", justify="right")

    for model_id in models:
        completed = completed_by_model.get(model_id, 0)
        total = total_by_model.get(model_id, 0)
        selected_count = selected_by_model.get(model_id, 0)
        table.add_row(escape(str(model_id)), f"{completed}/{total}", str(selected_count))

    console.print(table)


def _format_missing_pct(value: float) -> str:
    return f"{float(value):g}"


def _record_is_excluded(record: discovery.RunRecord, exclude_set: set[str]) -> bool:
    env_identifier: str | None = None
    if record.env_config and isinstance(record.env_config, Mapping):
        raw = record.env_config.get("id")
        if raw is not None:
            env_identifier = str(raw)
    if not env_identifier:
        env_identifier = str(record.manifest_env_id or "")
    return _env_is_excluded(env_identifier, exclude_set)


def _record_model_is_excluded(record: discovery.RunRecord, exclude_model_set: set[str]) -> bool:
    return model_is_excluded(str(record.model_id or "").strip(), exclude_model_set)


def _validate_selected_results_completeness(
    work_items: Sequence[PlannedWorkItem],
    *,
    max_results_missing_pct: float,
) -> None:
    missing_files: list[str] = []
    violations: list[str] = []
    ungateable = 0

    for item in work_items:
        for planned in item.records:
            normalized = planned.normalized
            record = normalized.record
            if not record.results_path.exists():
                missing_files.append(
                    "model_id={model_id} output_env_id={output_env_id} manifest_env_id={manifest_env_id} "
                    "job_run_id={job_run_id} job_id={job_id} results_path={results_path}".format(
                        model_id=item.identity.model_id,
                        output_env_id=item.identity.output_env_id,
                        manifest_env_id=normalized.manifest_env_id,
                        job_run_id=record.manifest.job_run_id,
                        job_id=record.job_id,
                        results_path=record.results_path,
                    )
                )
                continue

            expected_rows = _expected_results_rows(normalized)
            observed_rows = _completeness_observed_rows(record, expected_rows=expected_rows, threshold=max_results_missing_pct)
            if expected_rows is None or observed_rows is None:
                ungateable += 1
                continue

            missing_pct = _results_missing_pct(expected_rows=expected_rows, observed_rows=observed_rows)
            if missing_pct > max_results_missing_pct:
                violations.append(
                    "model_id={model_id} output_env_id={output_env_id} manifest_env_id={manifest_env_id} "
                    "job_run_id={job_run_id} job_id={job_id} expected_rows={expected_rows} "
                    "observed_rows={observed_rows} missing_pct={missing_pct:.2f} threshold={threshold:g}".format(
                        model_id=item.identity.model_id,
                        output_env_id=item.identity.output_env_id,
                        manifest_env_id=normalized.manifest_env_id,
                        job_run_id=record.manifest.job_run_id,
                        job_id=record.job_id,
                        expected_rows=expected_rows,
                        observed_rows=observed_rows,
                        missing_pct=missing_pct,
                        threshold=float(max_results_missing_pct),
                    )
                )

    if ungateable:
        logger.warning(
            "Results row completeness gate could not be applied to %d selected record(s) because expected_rows "
            "(num_examples * rollouts_per_example) or manifest row_count was unknown.",
            ungateable,
        )

    if not missing_files and not violations:
        return

    message_parts: list[str] = []
    if missing_files:
        missing_lines = "\n".join(f"  - {line}" for line in missing_files)
        message_parts.append("Selected records are missing results.jsonl files:\n" + missing_lines)
    if violations:
        violation_lines = "\n".join(f"  - {line}" for line in violations)
        message_parts.append(
            "Selected records exceeded --max-results-missing-pct based on manifest row_count and expected rows:\n"
            + violation_lines
        )
    raise RuntimeError("\n\n".join(message_parts))


def _expected_results_rows(normalized: metadata.NormalizedMetadata) -> int | None:
    num_examples = normalized.num_examples
    rollouts_per_example = normalized.rollouts_per_example
    if num_examples is None or rollouts_per_example is None:
        return None
    if num_examples == -1:
        return None
    if num_examples <= 0 or rollouts_per_example <= 0:
        return None
    return int(num_examples) * int(rollouts_per_example)


def _results_missing_pct(*, expected_rows: int, observed_rows: int) -> float:
    if expected_rows <= 0:
        return 0.0
    missing_rows = max(int(expected_rows) - max(int(observed_rows), 0), 0)
    return 100.0 * missing_rows / int(expected_rows)


def _completeness_observed_rows(
    record: discovery.RunRecord,
    *,
    expected_rows: int | None,
    threshold: float,
) -> int | None:
    observed_rows = record.row_count
    if expected_rows is None or observed_rows is None:
        return observed_rows

    missing_pct = _results_missing_pct(expected_rows=expected_rows, observed_rows=observed_rows)
    if missing_pct <= threshold:
        return observed_rows

    actual_rows = count_jsonl_rows(record.results_path)
    if actual_rows is None or actual_rows == observed_rows:
        return observed_rows

    logger.warning(
        "Manifest row_count mismatch for process input "
        "(job_run_id=%s, job_id=%s, results_path=%s): manifest row_count=%s actual_rows=%s. "
        "Using actual_rows for completeness validation.",
        record.manifest.job_run_id,
        record.job_id,
        record.results_path,
        observed_rows,
        actual_rows,
    )
    return actual_rows


def _process_env_group(item: PlannedWorkItem) -> tuple[list[AggregatedEnvRows], int]:
    """Load and aggregate all rows for a single selected dataset."""
    row_buffer: list[dict[str, Any]] = []
    identities: list[RunIdentity] = []
    for planned in item.records:
        row_batch = rows.load_rows(
            planned.normalized,
            extra_columns=planned.extra_columns,
            drop_columns=planned.drop_columns,
            answer_column=planned.answer_column,
        )
        row_buffer.extend(row_batch)
        identities.append(planned.normalized.identity)
    aggregated = aggregate.aggregate_rows_by_env(row_buffer, identities=identities)
    return aggregated, len(row_buffer)


def _source_updated_at(record: discovery.RunRecord) -> str:
    return record.manifest.updated_at or record.manifest.created_at or ""


def _env_is_excluded(env_id: str, exclude_set: set[str]) -> bool:
    env_identifier = str(env_id or "").strip()
    base_env_id, _ = rollout.derive_base_env_id(env_identifier)
    return dataset_is_excluded(env_identifier, exclude_set, base_dataset_id=base_env_id)


def _strip_env_group_rows(group: AggregatedEnvRows) -> AggregatedEnvRows:
    return AggregatedEnvRows(
        env_id=group.env_id,
        base_env_id=group.base_env_id,
        model_id=group.model_id,
        rows=[],
        column_names=group.column_names,
        job_run_ids=group.job_run_ids,
    )


def _run_sort_key(timestamp: str, job_run_id: str) -> tuple[int, datetime, str]:
    if not timestamp:
        return (0, datetime.min.replace(tzinfo=UTC), job_run_id)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return (1, parsed, job_run_id)
    except Exception:
        return (0, datetime.min.replace(tzinfo=UTC), job_run_id)


__all__ = [
    "PROCESS_DEFAULT_STATUS_FILTER",
    "PlannedRecord",
    "PlannedWorkItem",
    "ProcessOptions",
    "ProcessResult",
    "SelectionResult",
    "run_process",
    "select_work_items",
]
