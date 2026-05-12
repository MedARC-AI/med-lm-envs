"""Convert retired YAML-runner raw outputs into current eval-output directories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from medarc_verifiers.cli.eval_identity import BASE_VARIANT_ID, slug_component

MANIFEST_FILENAME = "run_manifest.json"
RESULTS_FILENAME = "results.jsonl"
METADATA_FILENAME = "metadata.json"
SUPPORTED_MANIFEST_VERSION = 3
MAX_VARIANT_LENGTH = 160


@dataclass(frozen=True, slots=True)
class ConversionEntry:
    run_id: str
    job_id: str | None
    status: str
    reason: str
    source_results: str | None = None
    target_dir: str | None = None


@dataclass(frozen=True, slots=True)
class ConversionReport:
    entries: tuple[ConversionEntry, ...]
    dry_run: bool

    @property
    def converted(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "converted")

    @property
    def would_convert(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "would_convert")

    @property
    def skipped(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "summary": {
                "converted": self.converted,
                "would_convert": self.would_convert,
                "skipped": self.skipped,
                "failed": self.failed,
            },
            "entries": [
                {
                    "run_id": entry.run_id,
                    "job_id": entry.job_id,
                    "status": entry.status,
                    "reason": entry.reason,
                    "source_results": entry.source_results,
                    "target_dir": entry.target_dir,
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class _PlannedConversion:
    run_id: str
    job: Mapping[str, Any]
    source_results: Path
    source_metadata: Path | None
    source_metadata_payload: Mapping[str, Any]
    target_dir: Path
    env_id: str
    model_id: str
    variant_id: str
    manifest: Mapping[str, Any]


def convert_legacy_raw_runs(
    *,
    raw_dir: Path | str,
    output_dir: Path | str,
    dry_run: bool = True,
) -> ConversionReport:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    entries: list[ConversionEntry] = []
    plans: list[_PlannedConversion] = []

    if not raw_path.exists():
        return ConversionReport(
            entries=(
                ConversionEntry(
                    run_id=str(raw_path),
                    job_id=None,
                    status="failed",
                    reason="raw directory does not exist",
                ),
            ),
            dry_run=dry_run,
        )

    for manifest_path in sorted(raw_path.glob(f"*/{MANIFEST_FILENAME}")):
        run_dir = manifest_path.parent
        try:
            manifest = _read_json_object(manifest_path)
        except ValueError as exc:
            entries.append(
                ConversionEntry(
                    run_id=run_dir.name,
                    job_id=None,
                    status="failed",
                    reason=str(exc),
                )
            )
            continue

        if manifest.get("version") != SUPPORTED_MANIFEST_VERSION:
            entries.append(
                ConversionEntry(
                    run_id=_run_id(manifest, run_dir),
                    job_id=None,
                    status="failed",
                    reason=f"unsupported manifest version {manifest.get('version')!r}; expected 3",
                )
            )
            continue

        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            entries.append(
                ConversionEntry(
                    run_id=_run_id(manifest, run_dir),
                    job_id=None,
                    status="failed",
                    reason="manifest jobs must be a list",
                )
            )
            continue

        for job in jobs:
            if not isinstance(job, Mapping):
                entries.append(
                    ConversionEntry(
                        run_id=_run_id(manifest, run_dir),
                        job_id=None,
                        status="skipped",
                        reason="job entry is not an object",
                    )
                )
                continue
            planned = _plan_job(run_dir, manifest, job, output_path)
            if isinstance(planned, ConversionEntry):
                entries.append(planned)
            else:
                plans.append(planned)

    entries.extend(_collision_entries(plans, existing_targets_fail=not dry_run))
    failed_targets = {
        entry.target_dir
        for entry in entries
        if entry.status == "failed" and entry.target_dir is not None and "collision" in entry.reason
    }
    failed_targets.update(
        entry.target_dir
        for entry in entries
        if entry.status == "failed" and entry.target_dir is not None and "already exists" in entry.reason
    )
    runnable_plans = [plan for plan in plans if str(plan.target_dir) not in failed_targets]

    for plan in runnable_plans:
        if dry_run:
            entries.append(_entry_for_plan(plan, status="would_convert", reason="dry run"))
            continue
        try:
            _write_conversion(plan)
        except (OSError, ValueError) as exc:
            entries.append(_entry_for_plan(plan, status="failed", reason=f"write failed: {exc}"))
            continue
        entries.append(_entry_for_plan(plan, status="converted", reason="converted"))

    return ConversionReport(entries=tuple(entries), dry_run=dry_run)


def _plan_job(
    run_dir: Path,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    output_dir: Path,
) -> _PlannedConversion | ConversionEntry:
    run_id = _run_id(manifest, run_dir)
    job_id = _string_or_none(job.get("job_id"))
    if not job_id:
        return ConversionEntry(run_id=run_id, job_id=None, status="skipped", reason="missing job_id")

    status = (_string_or_none(job.get("status")) or "pending").lower()
    if status != "completed":
        return ConversionEntry(run_id=run_id, job_id=job_id, status="skipped", reason=f"job status is {status!r}")

    model_id = _string_or_none(job.get("model_id"))
    env_id = _string_or_none(job.get("env_id"))
    if not model_id or not env_id:
        return ConversionEntry(run_id=run_id, job_id=job_id, status="skipped", reason="missing model_id or env_id")

    variant = _resolve_variant(job, env_id)
    if isinstance(variant, str):
        variant_id = variant
    else:
        return ConversionEntry(run_id=run_id, job_id=job_id, status="skipped", reason=variant["reason"])

    results_path = _resolve_results_path(run_dir, manifest, job, job_id)
    if not results_path.exists():
        return ConversionEntry(
            run_id=run_id,
            job_id=job_id,
            status="skipped",
            reason="missing results.jsonl",
            source_results=str(results_path),
        )

    source_metadata = _resolve_metadata_path(run_dir, manifest, job, results_path)
    source_metadata_payload: Mapping[str, Any] = {}
    if source_metadata is not None and not source_metadata.exists():
        source_metadata = None
    if source_metadata is not None:
        try:
            source_metadata_payload = _read_json_object(source_metadata)
        except ValueError as exc:
            return ConversionEntry(
                run_id=run_id,
                job_id=job_id,
                status="skipped",
                reason=f"invalid metadata.json: {exc}",
                source_results=str(results_path),
            )

    target_dir = output_dir / slug_component(model_id) / slug_component(env_id) / variant_id
    return _PlannedConversion(
        run_id=run_id,
        job=job,
        source_results=results_path,
        source_metadata=source_metadata,
        source_metadata_payload=source_metadata_payload,
        target_dir=target_dir,
        env_id=env_id,
        model_id=model_id,
        variant_id=variant_id,
        manifest=manifest,
    )


def _resolve_variant(job: Mapping[str, Any], env_id: str) -> str | dict[str, str]:
    raw = _string_or_none(job.get("env_variant_id"))
    if raw is None or raw == env_id:
        return BASE_VARIANT_ID

    split_variant = _resolve_split_variant(job, env_id, raw)
    if split_variant is not None:
        return split_variant

    prefix_colon = f"{env_id}::"
    prefix_slash = f"{env_id}/"
    if raw.startswith(prefix_colon):
        variant_id = raw[len(prefix_colon) :]
    elif raw.startswith(prefix_slash):
        variant_id = raw[len(prefix_slash) :]
    elif raw.startswith(f"{env_id}-") or raw.startswith(f"{env_id}_"):
        variant_id = raw[len(env_id) + 1 :]
    else:
        return {"reason": f"ambiguous env_variant_id {raw!r} for env_id {env_id!r}"}

    if not variant_id:
        return {"reason": f"empty parsed variant from env_variant_id {raw!r}"}
    if variant_id == BASE_VARIANT_ID:
        return {"reason": "variant identity conflict: source variant maps to reserved base"}
    if "/" in variant_id or "\\" in variant_id:
        return {"reason": f"path-unsafe variant {variant_id!r}"}
    if slug_component(variant_id, max_length=MAX_VARIANT_LENGTH) != variant_id:
        return {"reason": f"path-unsafe variant {variant_id!r}"}
    return variant_id


def _resolve_split_variant(job: Mapping[str, Any], env_id: str, raw: str) -> str | None:
    env_args = job.get("env_args")
    split = _string_or_none(env_args.get("split")) if isinstance(env_args, Mapping) else None
    if split != "en":
        return None

    for delimiter in ("_", "-"):
        split_prefix = f"{env_id}{delimiter}{split}"
        if raw == split_prefix:
            return BASE_VARIANT_ID
        rollout_prefix = f"{split_prefix}-"
        if raw.startswith(rollout_prefix):
            return raw[len(rollout_prefix) :]
    return None


def _resolve_results_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    job_id: str,
) -> Path:
    artifacts_root = _string_or_none(manifest.get("artifacts_root")) or "."
    base = run_dir / artifacts_root
    relpath = _string_or_none(job.get("results_relpath")) or _string_or_none(job.get("results_dir"))
    if relpath:
        candidate = base / relpath
        if candidate.name == RESULTS_FILENAME:
            return candidate
        return candidate / RESULTS_FILENAME
    return run_dir / job_id / RESULTS_FILENAME


def _resolve_metadata_path(
    run_dir: Path,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    results_path: Path,
) -> Path | None:
    artifacts_root = _string_or_none(manifest.get("artifacts_root")) or "."
    relpath = _string_or_none(job.get("metadata_relpath"))
    if relpath:
        return run_dir / artifacts_root / relpath
    candidate = results_path.parent / METADATA_FILENAME
    return candidate if candidate.exists() else None


def _collision_entries(
    plans: Sequence[_PlannedConversion],
    *,
    existing_targets_fail: bool,
) -> list[ConversionEntry]:
    entries: list[ConversionEntry] = []
    by_target: dict[Path, list[_PlannedConversion]] = {}
    for plan in plans:
        by_target.setdefault(plan.target_dir, []).append(plan)

    for target, target_plans in sorted(by_target.items(), key=lambda item: str(item[0])):
        if len(target_plans) > 1:
            for plan in target_plans:
                entries.append(_entry_for_plan(plan, status="failed", reason="planned output path collision"))
        elif existing_targets_fail and target.exists():
            entries.append(_entry_for_plan(target_plans[0], status="failed", reason="target path already exists"))
    return entries


def _write_conversion(plan: _PlannedConversion) -> None:
    plan.target_dir.mkdir(parents=True, exist_ok=False)
    row_stats = _write_converted_results(plan)
    metadata = _converted_metadata(plan, row_stats=row_stats)
    (plan.target_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_converted_results(plan: _PlannedConversion) -> dict[str, Any]:
    stats = _RowStats()
    with (
        plan.source_results.open("r", encoding="utf-8") as source,
        (plan.target_dir / RESULTS_FILENAME).open("w", encoding="utf-8") as target,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSON in {plan.source_results} line {line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected JSON object in {plan.source_results} line {line_number}")
            converted = _converted_result_row(payload)
            stats.add(converted)
            target.write(json.dumps(converted, sort_keys=True) + "\n")
    return stats.to_metadata()


def _converted_result_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(payload)
    converted["timing"] = _converted_timing(payload)
    converted.pop("generation_ms", None)
    converted.pop("scoring_ms", None)
    converted.pop("total_ms", None)

    converted["is_completed"] = bool(payload.get("is_completed", payload.get("error") is None))
    converted["is_truncated"] = bool(payload.get("is_truncated", False))
    converted["stop_condition"] = payload.get("stop_condition", "max_turns_reached")
    converted["metrics"] = _converted_metrics(payload)
    converted["tool_defs"] = payload.get("tool_defs", [])

    usage = _converted_token_usage(payload.get("token_usage"))
    if usage is not None:
        converted["token_usage"] = usage
    else:
        converted.pop("token_usage", None)

    return converted


def _converted_timing(payload: Mapping[str, Any]) -> dict[str, Any]:
    timing = payload.get("timing")
    if isinstance(timing, Mapping):
        return dict(timing)

    generation = _milliseconds_to_seconds(payload.get("generation_ms"))
    scoring = _milliseconds_to_seconds(payload.get("scoring_ms"))
    total = _milliseconds_to_seconds(payload.get("total_ms"))
    return {
        "setup": {"duration": 0.0, "spans": []},
        "generation": {"duration": generation, "start": 0.0, "end": generation},
        "scoring": {"duration": scoring, "start": generation, "end": generation + scoring},
        "model": {"duration": generation, "spans": [{"duration": generation, "start": 0.0, "end": generation}]},
        "env": {"duration": 0.0, "spans": []},
        "total": total,
        "overhead": max(0.0, total - generation - scoring),
    }


def _converted_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return {str(key): float(value) for key, value in metrics.items() if _float_or_none(value) is not None}

    converted: dict[str, float] = {}
    for key in ("accuracy", "num_turns"):
        value = _float_or_none(payload.get(key))
        if value is not None:
            converted[key] = value
    return converted


def _converted_token_usage(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    if "input_tokens" in value or "output_tokens" in value:
        input_tokens = _float_or_none(value.get("input_tokens")) or 0.0
        output_tokens = _float_or_none(value.get("output_tokens")) or 0.0
        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        final_input = _float_or_none(value.get("final_input_tokens"))
        final_output = _float_or_none(value.get("final_output_tokens"))
        if final_input is not None and final_output is not None:
            usage["final_input_tokens"] = final_input
            usage["final_output_tokens"] = final_output
        return usage

    model_usage = value.get("model")
    if not isinstance(model_usage, Mapping):
        return None
    input_tokens = _float_or_none(model_usage.get("prompt")) or 0.0
    output_tokens = _float_or_none(model_usage.get("completion")) or 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_input_tokens": input_tokens,
        "final_output_tokens": output_tokens,
    }


@dataclass(slots=True)
class _RowStats:
    count: int = 0
    reward_total: float = 0.0
    reward_count: int = 0
    error_count: int = 0
    metric_totals: dict[str, float] | None = None
    metric_counts: dict[str, int] | None = None
    input_tokens_total: float = 0.0
    output_tokens_total: float = 0.0
    usage_count: int = 0
    final_input_tokens_total: float = 0.0
    final_output_tokens_total: float = 0.0
    final_usage_count: int = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.count += 1
        reward = _float_or_none(row.get("reward"))
        if reward is not None:
            self.reward_total += reward
            self.reward_count += 1
        if row.get("error") is not None:
            self.error_count += 1

        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            if self.metric_totals is None:
                self.metric_totals = {}
                self.metric_counts = {}
            assert self.metric_counts is not None
            for key, value in metrics.items():
                numeric = _float_or_none(value)
                if numeric is None:
                    continue
                metric_key = str(key)
                self.metric_totals[metric_key] = self.metric_totals.get(metric_key, 0.0) + numeric
                self.metric_counts[metric_key] = self.metric_counts.get(metric_key, 0) + 1

        usage = row.get("token_usage")
        if isinstance(usage, Mapping):
            input_tokens = _float_or_none(usage.get("input_tokens"))
            output_tokens = _float_or_none(usage.get("output_tokens"))
            if input_tokens is not None or output_tokens is not None:
                self.input_tokens_total += input_tokens or 0.0
                self.output_tokens_total += output_tokens or 0.0
                self.usage_count += 1
            final_input = _float_or_none(usage.get("final_input_tokens"))
            final_output = _float_or_none(usage.get("final_output_tokens"))
            if final_input is not None and final_output is not None:
                self.final_input_tokens_total += final_input
                self.final_output_tokens_total += final_output
                self.final_usage_count += 1

    def to_metadata(self) -> dict[str, Any]:
        avg_metrics: dict[str, float] = {}
        if self.metric_totals and self.metric_counts:
            avg_metrics = {
                key: total / self.metric_counts[key]
                for key, total in sorted(self.metric_totals.items())
                if self.metric_counts.get(key)
            }

        usage: dict[str, float] | None = None
        if self.usage_count:
            usage = {
                "input_tokens": self.input_tokens_total / self.usage_count,
                "output_tokens": self.output_tokens_total / self.usage_count,
            }
            if self.final_usage_count:
                usage["final_input_tokens"] = self.final_input_tokens_total / self.final_usage_count
                usage["final_output_tokens"] = self.final_output_tokens_total / self.final_usage_count

        return {
            "row_count": self.count,
            "avg_reward": self.reward_total / self.reward_count if self.reward_count else 0.0,
            "avg_error": self.error_count / self.count if self.count else 0.0,
            "avg_metrics": avg_metrics,
            "usage": usage,
        }


def _converted_metadata(plan: _PlannedConversion, *, row_stats: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if plan.source_metadata_payload:
        source = plan.source_metadata_payload
        for key in (
            "env_args",
            "sampling_args",
            "num_examples",
            "rollouts_per_example",
            "avg_reward",
            "avg_metrics",
            "avg_error",
            "base_url",
            "state_columns",
            "tools",
            "usage",
            "version_info",
        ):
            if key in source:
                metadata[key] = source[key]
        if "time" in source:
            metadata["time"] = source["time"]
        elif "time_ms" in source:
            metadata["time"] = _milliseconds_to_seconds(source["time_ms"])

    model_table = plan.manifest.get("models")
    model_config = model_table.get(plan.model_id) if isinstance(model_table, Mapping) else None
    if "sampling_args" not in metadata and isinstance(model_config, Mapping):
        sampling_args = model_config.get("sampling_args")
        if isinstance(sampling_args, Mapping):
            metadata["sampling_args"] = dict(sampling_args)

    for key in ("env_args", "sampling_args"):
        job_value = plan.job.get(key)
        if key not in metadata and isinstance(job_value, Mapping):
            metadata[key] = dict(job_value)

    for key in ("num_examples", "rollouts_per_example", "avg_reward"):
        if key not in metadata and plan.job.get(key) is not None:
            metadata[key] = plan.job[key]

    metadata.setdefault("env_args", {})
    metadata.setdefault("sampling_args", {})
    metadata.setdefault("base_url", "")
    metadata.setdefault("time", plan.job.get("duration_seconds", 0.0))
    metadata["avg_reward"] = row_stats.get("avg_reward", metadata.get("avg_reward", 0.0))
    metadata["avg_metrics"] = row_stats.get("avg_metrics") or metadata.get("avg_metrics") or _job_metrics(plan.job)
    metadata["avg_error"] = row_stats.get("avg_error", metadata.get("avg_error", 0.0))
    metadata.setdefault("pass_at_k", {})
    metadata.setdefault("pass_all_k", {})
    metadata.setdefault("pass_threshold", 0.5)
    metadata["usage"] = row_stats.get("usage", metadata.get("usage"))
    metadata.setdefault("version_info", {})
    metadata.setdefault("state_columns", [])
    metadata.setdefault("tools", None)
    metadata["env_id"] = plan.env_id
    metadata["model"] = plan.model_id
    metadata["num_examples"] = int(metadata.get("num_examples") or row_stats.get("row_count") or 0)
    metadata["rollouts_per_example"] = int(metadata.get("rollouts_per_example") or 1)
    return metadata


def _job_metrics(job: Mapping[str, Any]) -> dict[str, float]:
    metrics = job.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    return {str(key): float(value) for key, value in metrics.items() if _float_or_none(value) is not None}


def _milliseconds_to_seconds(value: Any) -> float:
    numeric = _float_or_none(value)
    return 0.0 if numeric is None else numeric / 1000.0


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_for_plan(plan: _PlannedConversion, *, status: str, reason: str) -> ConversionEntry:
    return ConversionEntry(
        run_id=plan.run_id,
        job_id=_string_or_none(plan.job.get("job_id")),
        status=status,
        reason=reason,
        source_results=str(plan.source_results),
        target_dir=str(plan.target_dir),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _run_id(manifest: Mapping[str, Any], run_dir: Path) -> str:
    return _string_or_none(manifest.get("run_id")) or run_dir.name


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _run_conversion_cli(*, raw_dir: Path, output_dir: Path, dry_run: bool, report_path: Path | None) -> int:
    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=dry_run)
    encoded = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if report.failed else 0


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  python scripts/convert_legacy_raw_runs.py
      Preview conversion from runs/raw to runs/evals.

  python scripts/convert_legacy_raw_runs.py --no-dry-run --report-path report.json
      Write converted eval-output directories and save the JSON report.

  python scripts/convert_legacy_raw_runs.py --raw-dir old/runs/raw --output-dir runs/evals
      Preview conversion from a custom legacy raw-run directory.
""",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("runs") / "raw",
        help="legacy raw-run root directory containing */run_manifest.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "evals",
        help="converted eval-output root directory",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="plan conversion without writing files; use --no-dry-run to write converted outputs",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="optional path for a JSON copy of the conversion report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return _run_conversion_cli(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    sys.exit(main())
