from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from verifiers import setup_logging

from medarc_verifiers.export.parquet.aggregation import aggregate_rows_by_env
from medarc_verifiers.export.parquet.discovery import discover_run_records
from medarc_verifiers.export.parquet.metadata import load_normalized_metadata
from medarc_verifiers.export.parquet.rows import load_enriched_rows
from medarc_verifiers.export.parquet.writer import write_parquet_datasets

logger = logging.getLogger(__name__)

PROGRAM_NAME = "medarc-export"


@dataclass(frozen=True)
class ExportCLIOptions:
    """Container for normalized CLI arguments."""

    runs_dir: Path
    output_dir: Path
    dry_run: bool
    validate: bool
    strict: bool
    filter_status: tuple[str, ...]
    partition_by: tuple[str, ...]
    overwrite: bool
    schema_only: bool
    include_io: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the export CLI."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert evaluation run artifacts into environment-level Parquet datasets.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        metavar="PATH",
        help="Directory containing evaluation run outputs (run_manifest.json, metadata.json, results.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        metavar="PATH",
        help="Destination directory for Parquet exports and manifest files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the export and print a summary without writing Parquet files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable additional validation of manifests and results before exporting.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings and writer errors as fatal.",
    )
    parser.add_argument(
        "--filter-status",
        action="append",
        metavar="STATUS[,STATUS...]",
        help="Include only runs whose manifest status matches any of the provided values. Repeat to add more statuses.",
    )
    parser.add_argument(
        "--partition-by",
        action="append",
        metavar="COLUMN[,COLUMN...]",
        help="Partition Parquet outputs by the specified column names. Repeat to add more columns.",
    )
    parser.add_argument(
        "--include-io",
        action="store_true",
        help="Preserve prompt and completion payloads in the exported datasets.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Inspect manifests and rows to report schema information without writing Parquet files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing environment export directories.",
    )
    return parser


def _normalize_multi_values(raw_values: Sequence[str] | None) -> tuple[str, ...]:
    """Split comma-delimited inputs and remove duplicates while preserving order."""
    if not raw_values:
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for chunk in raw_values:
        for part in chunk.split(","):
            value = part.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def parse_args(argv: Sequence[str] | None = None) -> ExportCLIOptions:
    """Parse CLI arguments and return a normalized options container."""
    parser = build_parser()
    args = parser.parse_args(argv)

    return ExportCLIOptions(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        validate=args.validate,
        strict=args.strict,
        filter_status=_normalize_multi_values(args.filter_status),
        partition_by=_normalize_multi_values(args.partition_by),
        overwrite=args.overwrite,
        schema_only=args.schema_only,
        include_io=args.include_io,
    )


def execute_export(options: ExportCLIOptions) -> int:
    """Execute the export workflow (stub)."""
    records = discover_run_records(
        options.runs_dir,
        filter_status=options.filter_status,
    )
    drop_fields: tuple[str, ...] | set[str] = () if options.include_io else {"prompt", "completion"}
    enriched_rows: list[dict] = []
    metadata_count = 0
    warnings: list[str] = []
    for record in records:
        metadata = load_normalized_metadata(record)
        if metadata is None:
            continue
        metadata_count += 1
        rows = load_enriched_rows(metadata, drop_fields=drop_fields)
        if options.validate:
            expected: int | None = None
            if metadata.num_examples:
                rollouts = metadata.rollouts_per_example or 1
                expected = metadata.num_examples * rollouts
            if not rows:
                warnings.append(
                    f"{metadata.record.job_id} produced no rows (env={metadata.base_env_id or metadata.record.manifest_env_id})."
                )
            elif metadata.num_examples and len(rows) % metadata.num_examples != 0:
                warnings.append(
                    f"{metadata.record.job_id} row count {len(rows)} is not aligned with num_examples={metadata.num_examples}."
                )
            elif expected and len(rows) != expected:
                warnings.append(
                    f"{metadata.record.job_id} row count {len(rows)} does not match expected {expected} from metadata."
                )
        enriched_rows.extend(rows)

    unique_runs = {record.manifest.job_run_id for record in records}
    missing_artifacts = [
        record
        for record in records
        if not (record.has_metadata and record.has_results)
    ]
    logger.info(
        "Discovered %d job results across %d run directories (metadata=%d, rows=%d, missing-artifacts=%d, dry-run=%s).",
        len(records),
        len(unique_runs),
        metadata_count,
        len(enriched_rows),
        len(missing_artifacts),
        options.dry_run,
    )
    if missing_artifacts:
        logger.debug(
            "Runs missing artifacts: %s",
            ", ".join(sorted({record.job_id for record in missing_artifacts})),
        )
    for warning in warnings:
        logger.warning("Validation warning: %s", warning)

    warnings: list[str] = []
    if options.validate:
        for metadata, rows in metadata_with_rows:
            expected: int | None = None
            if metadata.num_examples:
                rollouts = metadata.rollouts_per_example or 1
                expected = metadata.num_examples * rollouts
            if not rows:
                warnings.append(
                    f"{metadata.record.job_id} produced no rows (env={metadata.base_env_id or metadata.record.manifest_env_id})."
                )
            elif metadata.num_examples and len(rows) % metadata.num_examples != 0:
                warnings.append(
                    f"{metadata.record.job_id} row count {len(rows)} is not aligned with num_examples={metadata.num_examples}."
                )
            elif expected and len(rows) != expected:
                warnings.append(
                    f"{metadata.record.job_id} row count {len(rows)} does not match expected {expected} from metadata."
                )
        for warning in warnings:
            logger.warning("Validation warning: %s", warning)

    aggregated = aggregate_rows_by_env(
        enriched_rows,
        partition_by=options.partition_by,
    )
    logger.info(
        "Aggregated rows into %d environment datasets (partitioned=%s).",
        len(aggregated),
        bool(options.partition_by),
    )
    if aggregated and (options.schema_only or options.dry_run):
        for group in aggregated:
            sample_columns = ", ".join(sorted(group.column_names)[:5])
            logger.info(
                "Schema preview [%s]: %d rows, %d columns%s%s",
                group.env_id,
                len(group.rows),
                len(group.column_names),
                f", partitions={group.partition_columns}" if group.partition_columns else "",
                f", sample_columns=[{sample_columns}]" if sample_columns else "",
            )

    summaries: list = []
    write_errors: list[str] = []
    if not options.schema_only:
        if options.dry_run:
            logger.info("Dry-run enabled; Parquet files will not be written.")
        summaries, write_errors = write_parquet_datasets(
            aggregated,
            options.output_dir,
            dry_run=options.dry_run,
            overwrite=options.overwrite,
        )
        for error in write_errors:
            logger.error("Writer error: %s", error)

        if summaries:
            logger.info(
                "Prepared %d datasets with a total of %d rows.",
                len(summaries),
                sum(summary.row_count for summary in summaries),
            )
            if not options.dry_run:
                logger.info("Export index written to %s", options.output_dir / "env_index.json")
        else:
            logger.info("No datasets were produced.")
    else:
        logger.info("Schema-only mode active; skipping Parquet writes.")

    if options.strict and (warnings or write_errors):
        return 1
    if write_errors:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the medarc-export CLI."""
    setup_logging("INFO")
    options = parse_args(argv)
    try:
        return execute_export(options)
    except KeyboardInterrupt:  # pragma: no cover - ensures graceful termination once implemented
        logger.error("Export interrupted by user.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
