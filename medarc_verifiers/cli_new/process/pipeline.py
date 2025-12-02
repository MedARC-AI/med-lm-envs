"""Top-level pipeline wiring discovery, row loading, aggregation, and writing."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from medarc_verifiers.cli_new._schemas import EnvironmentExportConfig
from medarc_verifiers.cli_new.process import aggregate, discovery, hf_sync, metadata, rows, rollout, writer
from medarc_verifiers.cli_new.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli_new.process.hf_sync import HFMergeSummary, HFSyncConfig
from medarc_verifiers.cli_new.process.writer import EnvWriteSummary, WriterConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessOptions:
    """User-configurable knobs for the process pipeline."""

    runs_dir: Path
    output_dir: Path
    only_complete_runs: bool = True
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
    append: bool = (
        True  # when True, merge into existing parquet files; when False, treat existing file + overwrite=False as error
    )
    max_workers: int = 4

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.output_dir = Path(self.output_dir)
        self.max_workers = max(1, int(self.max_workers))
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


@dataclass(slots=True)
class _RecordWork:
    """Per-record settings for row loading."""

    normalized: metadata.NormalizedMetadata
    include_prompt: bool
    keep_columns: Sequence[str]
    drop_columns: Sequence[str]


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
        only_complete_runs=bool(options.only_complete_runs),
    )

    # Deduplicate to keep only latest runs per model+env
    if options.deduplicate_latest:
        records = discovery.deduplicate_records_by_latest(records)

    _print_records_table(records, options.only_complete_runs, options.runs_dir)

    grouped: dict[str, list[_RecordWork]] = {}
    record_iter: Iterable[Any] = records
    try:
        from rich.progress import track

        record_iter = track(records, description="Reading run outputs", transient=True)
    except Exception:
        record_iter = records

    for record in record_iter:
        env_export = _resolve_env_export(record.manifest_env_id, env_export_map)
        include_prompt = _resolve_include_prompt(options, env_export)
        keep_columns = _resolve_columns(options.keep_columns, env_export.keep_columns if env_export else ())
        drop_columns = _resolve_columns(options.drop_columns, env_export.drop_columns if env_export else ())
        combine_rollouts = _resolve_combine_rollouts(options, env_export)

        normalized = metadata.load_normalized_metadata(record, combine_rollouts=combine_rollouts)
        env_key = normalized.base_env_id or normalized.manifest_env_id or record.manifest_env_id or record.job_id
        grouped.setdefault(env_key, []).append(
            _RecordWork(
                normalized=normalized,
                include_prompt=include_prompt,
                keep_columns=keep_columns,
                drop_columns=drop_columns,
            )
        )

    writer_config = WriterConfig(
        output_dir=options.output_dir,
        exporter_version=options.exporter_version,
        processed_at=options.processed_at or "",
        processed_with_args=options.processed_with_args,
        dry_run=options.dry_run,
        overwrite=options.overwrite,
        append=options.append,
    )

    env_groups: list[AggregatedEnvRows] = []
    env_summaries: list[EnvWriteSummary] = []
    rows_processed = 0

    env_items = sorted(grouped.items())
    try:
        if options.max_workers <= 1 or len(env_items) <= 1:
            env_iter: Iterable[tuple[str, list[_RecordWork]]] = env_items
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

                    future_iter = track(future_iter, total=len(futures), description="Processing datasets", transient=True)
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

    writer.write_env_index(env_summaries, writer_config)

    hf_summary: HFMergeSummary | None = None
    if options.hf_config:
        hf_summary = hf_sync.sync_to_hub(env_summaries, options.hf_config)

    return ProcessResult(
        records_processed=len(records),
        rows_processed=rows_processed,
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


def _print_records_table(
    records: Sequence[discovery.RunRecord], only_complete_runs: bool, runs_dir: Path | str
) -> None:
    """Pretty-print which models will be processed and how many jobs per model (pre-combining rollouts).

    Also indicates inclusion status and lists models present in runs but excluded by filters.
    """
    # Discover all records (deduped) to compute completed vs total per model
    try:
        all_records_raw = discovery.discover_run_records(runs_dir, filter_status=None, only_complete_runs=False)
        all_records = discovery.deduplicate_records_by_latest(all_records_raw)
    except Exception:  # pragma: no cover - best-effort
        all_records = list(records)

    # Compute totals
    total_by_model: dict[str, int] = {}
    completed_by_model: dict[str, int] = {}
    for rec in all_records:
        model_id = rec.model_id or "unknown"
        total_by_model[model_id] = total_by_model.get(model_id, 0) + 1
        if (rec.status or "").lower() == "completed":
            completed_by_model[model_id] = completed_by_model.get(model_id, 0) + 1

    # Inclusion: completed == total (>0)
    included_models: list[str] = []
    excluded_models: list[str] = []
    for model_id in sorted(total_by_model.keys()):
        tot = total_by_model.get(model_id, 0)
        comp = completed_by_model.get(model_id, 0)
        if tot > 0 and comp == tot:
            included_models.append(model_id)
        else:
            excluded_models.append(model_id)

    # For title, sum completed jobs among included models
    included_jobs_total = sum(completed_by_model.get(m, 0) for m in included_models)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.markup import escape
    except Exception:
        suffix = " (complete runs only)" if only_complete_runs else ""
        logger.info(
            "Processing %d job(s) across %d model(s)%s (pre-combining rollouts).",
            included_jobs_total,
            len(included_models),
            suffix,
        )
        for model_id in included_models:
            comp = completed_by_model.get(model_id, 0)
            tot = total_by_model.get(model_id, 0)
            logger.info("  - %s: %d/%d job(s) (included)", model_id, comp, tot)
        if excluded_models:
            logger.info("Excluded model(s):")
            for model_id in excluded_models:
                comp = completed_by_model.get(model_id, 0)
                tot = total_by_model.get(model_id, 0)
                logger.info("  - %s: %d/%d job(s) (excluded)", model_id, comp, tot)
        return

    console = Console()
    title = f"Processing {included_jobs_total} job(s) across {len(included_models)} model(s)"
    if only_complete_runs:
        title += " (complete runs only)"
    title += " [dim](pre-combining rollouts)[/dim]"
    table = Table(title=title, show_header=True, header_style="bold cyan", caption=None)
    table.add_column("Model", style="magenta")
    table.add_column("Jobs (completed/total)", style="green", justify="right")
    table.add_column("Included", style="yellow")

    for model_id in included_models:
        comp = completed_by_model.get(model_id, 0)
        tot = total_by_model.get(model_id, 0)
        table.add_row(escape(str(model_id)), f"{comp}/{tot}", "yes")
    # Append excluded models with their completed/total
    for model_id in excluded_models:
        comp = completed_by_model.get(model_id, 0)
        tot = total_by_model.get(model_id, 0)
        table.add_row(escape(str(model_id)), f"{comp}/{tot}", "no")

    console.print(table)


__all__ = ["ProcessOptions", "ProcessResult", "run_process"]


def _process_env_group(work_items: Sequence[_RecordWork]) -> tuple[list[AggregatedEnvRows], int]:
    """Load and aggregate all rows for a single environment."""
    row_buffer: list[dict[str, Any]] = []
    for work in work_items:
        row_batch = rows.load_rows(
            work.normalized,
            include_prompt_completion=work.include_prompt,
            keep_columns=work.keep_columns,
            drop_columns=work.drop_columns,
        )
        row_buffer.extend(row_batch)
    aggregated = aggregate.aggregate_rows_by_env(row_buffer)
    return aggregated, len(row_buffer)
