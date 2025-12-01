"""Top-level pipeline wiring discovery, row loading, aggregation, and writing."""

from __future__ import annotations

import logging
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
    compute_winrates: bool = True
    winrate_output: Path | None = None
    missing_policy: str = "neg-inf"
    epsilon: float = 1e-9
    min_common: int = 0
    weight_policy: str = "ln"
    weight_cap: int = 0
    append: bool = (
        True  # when True, merge into existing parquet files; when False, treat existing file + overwrite=False as error
    )

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.output_dir = Path(self.output_dir)
        if self.winrate_output is not None:
            self.winrate_output = Path(self.winrate_output)
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

    env_iter: Iterable[tuple[str, list[_RecordWork]]] = grouped.items()
    try:
        from rich.progress import track

        env_iter = track(sorted(grouped.items()), description="Processing datasets", transient=True)
    except Exception:
        env_iter = grouped.items()

    try:
        for _, work_items in env_iter:
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
            rows_processed += len(row_buffer)

            env_groups.extend(aggregated)
            summaries = writer.write_env_groups(aggregated, writer_config, write_index=False)
            env_summaries.extend(summaries)

            if not options.dry_run:
                for group in aggregated:
                    group.rows.clear()
    except KeyboardInterrupt:
        logger.warning("Processing cancelled by user; partial outputs may exist.")
        raise

    writer.write_env_index(env_summaries, writer_config)

    if options.compute_winrates:
        if options.dry_run:
            logger.info("Skipping win rate computation because --dry-run is enabled.")
        else:
            from . import winrate as _win

            dataset_inputs = [(summary.env_id, summary.output_path) for summary in env_summaries]
            cfg = _win.WinrateConfig(
                missing_policy=options.missing_policy,
                epsilon=options.epsilon,
                min_common=options.min_common,
                weight_policy=options.weight_policy,
                weight_cap=options.weight_cap,
            )
            result = _win.compute_winrates(dataset_inputs, cfg)
            winrate_path = options.winrate_output or _default_winrate_path(options)
            _win.write_json(_win.to_json(result), winrate_path)
            _print_winrate_summary_markdown(result)

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


def _default_winrate_path(options: ProcessOptions) -> Path:
    timestamp = _format_timestamp_for_filename(options.processed_at)
    root = _winrate_root(options)
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


def _winrate_root(options: ProcessOptions) -> Path:
    parent = options.output_dir.parent
    if parent == options.output_dir:
        return options.output_dir
    return parent


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


def _print_winrate_summary_markdown(result: Any) -> None:
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

    # Sort by weighted desc, falling back to simple desc
    def _key(item: tuple[str, float | None, float | None, int]) -> float:
        _, sm, lw, _ = item
        return float(lw if lw is not None else (sm if sm is not None else float("-inf")))

    scoreboard.sort(key=_key, reverse=True)
    # Prepare rows for table rendering
    rows: list[dict[str, str]] = []
    for model, sm, lw, n_ds in scoreboard:
        sm_str = f"{sm:.4f}" if isinstance(sm, (int, float)) and sm is not None else "-"
        lw_str = f"{lw:.4f}" if isinstance(lw, (int, float)) and lw is not None else "-"
        rows.append({"Model": model, "SimpleAvg": sm_str, "LnWeighted": lw_str, "Datasets": str(n_ds)})

    # Prefer tabulate for clean GitHub markdown if available
    try:
        from tabulate import tabulate  # type: ignore[import-not-found]

        md_table = tabulate(rows, headers="keys", tablefmt="github")
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    # Fallback: try pandas via polars for to_markdown
    try:
        import polars as pl  # type: ignore[import-not-found]

        # pandas may not be present; import inside try
        import pandas as pd  # type: ignore[import-not-found]  # noqa: F401

        df = pl.DataFrame(rows).to_pandas()
        md_table = df.to_markdown(index=False)  # type: ignore[attr-defined]
        _emit_markdown_table(md_table)
        return
    except Exception:
        pass

    # Final fallback: simple manual markdown
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
    """Emit the markdown table to stdout using Rich Console if available, else print."""
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
