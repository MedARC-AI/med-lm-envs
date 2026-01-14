"""Top-level pipeline wiring discovery, row loading, aggregation, and writing."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from medarc_verifiers.cli._schemas import EnvironmentExportConfig
from medarc_verifiers.cli import hf as hf_sync
from medarc_verifiers.cli.process import (
    aggregate,
    discovery,
    env_index,
    metadata,
    rows,
    rollout,
    writer,
    workspace,
)
from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.hf import HFSyncConfig, HFSyncSummary
from medarc_verifiers.cli.process.writer import EnvWriteSummary, WriterConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessOptions:
    """User-configurable knobs for the process pipeline."""

    runs_dir: Path
    output_dir: Path
    only_complete_runs: bool = True
    processed_at: str | None = None
    processed_with_args: Mapping[str, Any] = field(default_factory=dict)
    status_filter: Sequence[str] = field(default_factory=tuple)
    dry_run: bool = False
    clean: bool = False
    assume_yes: bool = False
    hf_config: HFSyncConfig | None = None
    hf_pull_policy: str | None = None
    max_workers: int = 4

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.output_dir = Path(self.output_dir)
        self.max_workers = max(1, int(self.max_workers))
        if not self.processed_at:
            self.processed_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.status_filter = tuple(str(status) for status in self.status_filter)


@dataclass(slots=True)
class ProcessResult:
    """Outcome of a process pipeline execution."""

    records_processed: int
    rows_processed: int
    env_groups: list[AggregatedEnvRows]
    env_summaries: list[EnvWriteSummary]
    hf_summary: HFSyncSummary | None


@dataclass(slots=True)
class _RecordWork:
    """Per-record settings for row loading."""

    normalized: metadata.NormalizedMetadata
    extra_columns: Sequence[str]
    drop_columns: Sequence[str]
    answer_column: str | None


@dataclass(slots=True)
class _NormalizedRecord:
    record: discovery.RunRecord
    normalized: metadata.NormalizedMetadata
    extra_columns: Sequence[str]
    drop_columns: Sequence[str]
    answer_column: str | None
    model_key: str
    env_key: str
    job_run_id: str
    run_timestamp: str


@dataclass(slots=True)
class _EnvGroupSelection:
    model_key: str
    env_key: str
    job_run_id: str
    run_timestamp: str
    records: list[_NormalizedRecord]


def run_process(
    options: ProcessOptions,
    *,
    env_export_map: Mapping[str, EnvironmentExportConfig] | None = None,
) -> ProcessResult:
    """Run the exporter pipeline from discovery through Parquet output (and HF sync)."""
    env_export_map = env_export_map or {}

    def _run_pipeline() -> ProcessResult:
        if not options.dry_run and options.clean:
            _confirm_clean_process(
                options.output_dir,
                assume_yes=options.assume_yes,
                is_tty=sys.stdin.isatty(),
                prompt_func=input,
            )
            workspace.clear_output_dir(options.output_dir)
        if not options.dry_run and options.hf_config and options.hf_config.repo_id and not options.clean:
            workspace.prepare_hf_baseline(
                output_dir=options.output_dir,
                hf_config=options.hf_config,
                pull_policy=options.hf_pull_policy,
                is_tty=sys.stdin.isatty(),
                prompt_func=input,
            )

        index_version, index_runs = env_index.read_env_index_runs(options.output_dir)
        index_files = env_index.read_env_index_files(options.output_dir)
        if options.clean:
            index_version = 0
            index_runs = {}
            index_files = {}

        discovered = discovery.discover_run_records(
            options.runs_dir,
            filter_status=options.status_filter or None,
            only_complete_runs=False,
        )

        use_delta = index_version == 2 and not options.clean
        if index_version != 2 and not options.clean:
            logger.info("Delta processing disabled: missing or legacy env_index.json; running full reprocess.")
        records: list[discovery.RunRecord] = list(discovered)
        if options.only_complete_runs:
            records = [
                record
                for record in records
                if not (
                    record.manifest.summary_total_known
                    and record.manifest.summary_completed != record.manifest.summary_total
                )
            ]
        normalized_records = _normalize_records(records, env_export_map)
        env_groups = _select_latest_env_groups(normalized_records)
        if use_delta:
            env_groups = _filter_env_groups_by_delta(
                env_groups,
                index_runs,
                index_files,
                output_dir=options.output_dir,
            )
        records = [item.record for group in env_groups for item in group.records]

        _print_records_table(discovered, records, options.only_complete_runs)

        grouped: dict[tuple[str, str], list[_RecordWork]] = {}
        run_metadata: dict[str, dict[str, Any]] = {}
        record_items = [item for group in env_groups for item in group.records]
        record_iter: Iterable[_NormalizedRecord] = record_items
        try:
            from rich.progress import track

            record_iter = track(record_items, description="Reading run outputs", transient=True)
        except Exception:
            pass

        for record in record_iter:
            normalized = record.normalized
            grouped.setdefault((record.model_key, record.env_key), []).append(
                _RecordWork(
                    normalized=normalized,
                    extra_columns=record.extra_columns,
                    drop_columns=record.drop_columns,
                    answer_column=record.answer_column,
                )
            )
            run_metadata.setdefault(
                record.job_run_id,
                {
                    "created_at": record.record.manifest.created_at,
                    "updated_at": _source_updated_at(record.record),
                    "config_checksum": record.record.manifest.config_checksum,
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

        env_items = sorted(grouped.items())
        try:
            if options.max_workers <= 1 or len(env_items) <= 1:
                env_iter: Iterable[tuple[tuple[str, str], list[_RecordWork]]] = env_items
                try:
                    from rich.progress import track

                    env_iter = track(env_items, description="Processing datasets", transient=True)
                except Exception:
                    env_iter = env_items

                for _, work_items in env_iter:
                    aggregated, row_count = _process_env_group(work_items)
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
                    for _, work_items in env_items:
                        futures.append(executor.submit(_process_env_group, work_items))

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
                    for f in futures:
                        f.cancel()
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
            )

        if options.dry_run:
            env_groups = [_strip_env_group_rows(group) for group in env_groups]

        return ProcessResult(
            records_processed=len(records),
            rows_processed=rows_processed,
            env_groups=env_groups,
            env_summaries=env_summaries,
            hf_summary=hf_summary,
        )

    if options.dry_run:
        return _run_pipeline()
    workspace.ensure_output_dir(options.output_dir)
    return _run_pipeline()


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


def _resolve_columns(env_columns: Sequence[str]) -> Sequence[str]:
    return tuple(str(column).strip() for column in env_columns if str(column).strip())


def _print_records_table(
    discovered: Sequence[discovery.RunRecord],
    selected: Sequence[discovery.RunRecord],
    only_complete_runs: bool,
) -> None:
    """Pretty-print job discovery vs planned processing."""
    total_by_model: dict[str, int] = {}
    completed_by_model: dict[str, int] = {}
    selected_by_model: dict[str, int] = {}
    completed_statuses = {"completed", "succeeded", "success"}
    for rec in discovered:
        model_id = rec.model_id or "unknown"
        total_by_model[model_id] = total_by_model.get(model_id, 0) + 1
        if (rec.status or "").lower() in completed_statuses:
            completed_by_model[model_id] = completed_by_model.get(model_id, 0) + 1
    for rec in selected:
        model_id = rec.model_id or "unknown"
        selected_by_model[model_id] = selected_by_model.get(model_id, 0) + 1

    models = sorted(total_by_model.keys())
    selected_models = sorted(m for m, c in selected_by_model.items() if c > 0)
    discovered_jobs_total = sum(total_by_model.get(m, 0) for m in models)
    selected_jobs_total = sum(selected_by_model.get(m, 0) for m in models)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.markup import escape
    except Exception:
        suffix = " (complete runs only)" if only_complete_runs else ""
        logger.info(
            "Processing %d job(s) across %d model(s)%s (found %d job(s) across %d model(s)).",
            selected_jobs_total,
            len(selected_models),
            suffix,
            discovered_jobs_total,
            len(models),
        )
        for model_id in models:
            comp = completed_by_model.get(model_id, 0)
            tot = total_by_model.get(model_id, 0)
            sel = selected_by_model.get(model_id, 0)
            processing = "yes" if sel > 0 else "no"
            logger.info("  - %s: processing=%s; %d/%d completed", model_id, processing, comp, tot)
        return

    console = Console()
    title = f"Processing {selected_jobs_total} job(s) across {len(selected_models)} model(s)"
    if only_complete_runs:
        title += " (complete runs only)"
    title += f" [dim](found {discovered_jobs_total} job(s) across {len(models)} model(s); pre-aggregation)[/dim]"
    table = Table(title=title, show_header=True, header_style="bold cyan", caption=None)
    table.add_column("Model", style="magenta")
    table.add_column("Jobs (completed/total)", style="green", justify="right")
    table.add_column("Processing", style="cyan", justify="center")

    for model_id in models:
        comp = completed_by_model.get(model_id, 0)
        tot = total_by_model.get(model_id, 0)
        sel = selected_by_model.get(model_id, 0)
        processing = "yes" if sel > 0 else "no"
        table.add_row(escape(str(model_id)), f"{comp}/{tot}", processing)

    console.print(table)


__all__ = ["ProcessOptions", "ProcessResult", "run_process"]


def _process_env_group(
    work_items: Sequence[_RecordWork],
) -> tuple[list[AggregatedEnvRows], int]:
    """Load and aggregate all rows for a single environment."""
    row_buffer: list[dict[str, Any]] = []
    for work in work_items:
        row_batch = rows.load_rows(
            work.normalized,
            extra_columns=work.extra_columns,
            drop_columns=work.drop_columns,
            answer_column=work.answer_column,
        )
        row_buffer.extend(row_batch)
    aggregated = aggregate.aggregate_rows_by_env(
        row_buffer,
    )
    return aggregated, len(row_buffer)


def _source_updated_at(record: discovery.RunRecord) -> str:
    return record.manifest.updated_at or record.manifest.created_at or ""


def _filter_env_groups_by_delta(
    env_groups: Sequence[_EnvGroupSelection],
    index_runs: Mapping[str, Mapping[str, Any]],
    index_files: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: Path,
) -> list[_EnvGroupSelection]:
    filtered: list[_EnvGroupSelection] = []
    for group in env_groups:
        expected_path = writer.build_output_path(output_dir, model_id=group.model_key, env_id=group.env_key)
        expected_rel = expected_path.relative_to(output_dir).as_posix()
        prior_file = index_files.get(expected_rel, {})
        if not prior_file:
            filtered.append(group)
            continue
        prior_updated_at = str(prior_file.get("updated_at") or prior_file.get("created_at") or "")
        if group.job_run_id not in index_runs:
            filtered.append(group)
            continue
        if _is_newer_timestamp(group.run_timestamp, prior_updated_at):
            filtered.append(group)
            continue
    return filtered


def _is_newer_timestamp(current: str, prior: str) -> bool:
    if not prior:
        return True if current else False
    if not current:
        return False
    try:
        current_dt = datetime.fromisoformat(current.replace("Z", "+00:00"))
        prior_dt = datetime.fromisoformat(prior.replace("Z", "+00:00"))
    except Exception:
        return current != prior
    return current_dt > prior_dt


def _strip_env_group_rows(group: AggregatedEnvRows) -> AggregatedEnvRows:
    return AggregatedEnvRows(
        env_id=group.env_id,
        base_env_id=group.base_env_id,
        model_id=group.model_id,
        rows=[],
        column_names=group.column_names,
        job_run_ids=group.job_run_ids,
    )


def _normalize_records(
    records: Sequence[discovery.RunRecord],
    env_export_map: Mapping[str, EnvironmentExportConfig],
) -> list[_NormalizedRecord]:
    normalized_records: list[_NormalizedRecord] = []
    for record in records:
        env_export = _resolve_env_export(record.manifest_env_id, env_export_map)
        extra_columns = _resolve_columns(env_export.extra_columns if env_export else ())
        drop_columns = _resolve_columns(env_export.drop_columns if env_export else ())
        answer_column = env_export.answer_column if env_export else None

        normalized = metadata.load_normalized_metadata(record)
        model_id = normalized.model_id
        if not model_id:
            raise RuntimeError(
                "Missing model_id for run "
                f"(job_run_id={record.manifest.job_run_id}, job_id={record.job_id}, "
                f"results_dir={record.results_dir}, manifest={record.manifest.manifest_path})"
            )

        env_key = normalized.base_env_id or normalized.manifest_env_id or record.manifest_env_id or record.job_id
        normalized_records.append(
            _NormalizedRecord(
                record=record,
                normalized=normalized,
                extra_columns=extra_columns,
                drop_columns=drop_columns,
                answer_column=answer_column,
                model_key=model_id,
                env_key=env_key,
                job_run_id=record.manifest.job_run_id,
                run_timestamp=_source_updated_at(record),
            )
        )
    return normalized_records


def _select_latest_env_groups(
    records: Sequence[_NormalizedRecord],
) -> list[_EnvGroupSelection]:
    env_groups: dict[tuple[str, str], dict[str, list[_NormalizedRecord]]] = {}
    run_timestamps: dict[str, str] = {}
    for record in records:
        env_groups.setdefault((record.model_key, record.env_key), {}).setdefault(record.job_run_id, []).append(record)
        run_timestamps.setdefault(record.job_run_id, record.run_timestamp)

    selected: list[_EnvGroupSelection] = []
    for (model_key, env_key), run_groups in env_groups.items():
        if not run_groups:
            continue
        latest_run_id = max(
            run_groups.keys(),
            key=lambda run_id: _run_sort_key(run_timestamps.get(run_id, ""), run_id),
        )
        selected.append(
            _EnvGroupSelection(
                model_key=model_key,
                env_key=env_key,
                job_run_id=latest_run_id,
                run_timestamp=run_timestamps.get(latest_run_id, ""),
                records=run_groups[latest_run_id],
            )
        )
    return selected


def _run_sort_key(timestamp: str, job_run_id: str) -> tuple[int, datetime, str]:
    if not timestamp:
        return (0, datetime.min.replace(tzinfo=UTC), job_run_id)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return (1, parsed, job_run_id)
    except Exception:
        return (0, datetime.min.replace(tzinfo=UTC), job_run_id)


def _confirm_clean_process(
    output_dir: Path,
    *,
    assume_yes: bool,
    is_tty: bool,
    prompt_func: Callable[[str], str] | None,
) -> None:
    if assume_yes:
        return
    if not is_tty or prompt_func is None:
        raise RuntimeError("Refusing to clean processed outputs without confirmation. Re-run with --yes to confirm.")
    prompt = f"--clean will delete all contents of {output_dir} and rebuild from runs. Type 'clean' to continue: "
    try:
        response = prompt_func(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):  # noqa: PERF203
        raise RuntimeError("Aborted clean process.") from None
    if response != "clean":
        raise RuntimeError("Aborted clean process.")
