import pyarrow.parquet as pq

from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.process import writer


def test_process_writer_emits_stable_schema_with_all_null_values(tmp_path) -> None:
    group = AggregatedEnvRows(
        env_id="medcalc_bench",
        base_env_id="medcalc_bench",
        model_id="test-model",
        variant_id=None,
        variant_payload=None,
        rows=[
            {
                "env_id": "medcalc_bench",
                "example_id": 1,
                "answer": None,
                "extras": None,
                "generation_ms": None,
                "job_run_id": "job-1",
                "judge_cost": None,
                "judge_token_completion": None,
                "judge_token_prompt": None,
                "judge_token_total": None,
                "model_cost": None,
                "model_id": "test-model",
                "model_token_completion": None,
                "model_token_prompt": None,
                "model_token_total": None,
                "reward": None,
                "rollout_index": 0,
                "run_id": "run-1",
                "scoring_ms": None,
                "status": "completed",
                "task": "medcalc_bench",
                "total_ms": None,
                "error": None,
            }
        ],
        column_names=tuple(writer.ALLOWED_COLUMNS),
        job_run_ids=("job-1",),
    )

    config = writer.WriterConfig(output_dir=tmp_path, processed_at="2026-01-01T00:00:00Z", dry_run=False)
    summaries = writer.write_env_groups([group], config, write_index=False)
    schema = pq.ParquetFile(summaries[0].output_path).schema_arrow

    assert str(schema.field("example_id").type) == "large_string"
    assert str(schema.field("extras").type) == "large_string"
    assert str(schema.field("answer").type) == "large_string"
    assert str(schema.field("error").type) == "large_string"
    assert str(schema.field("judge_cost").type) == "double"
    assert str(schema.field("model_cost").type) == "double"


def test_process_writer_emits_stable_schema_for_empty_groups(tmp_path) -> None:
    group = AggregatedEnvRows(
        env_id="empty_env",
        base_env_id="empty_env",
        model_id="test-model",
        variant_id=None,
        variant_payload=None,
        rows=[],
        column_names=(),
        job_run_ids=(),
    )
    config = writer.WriterConfig(output_dir=tmp_path, processed_at="2026-01-01T00:00:00Z", dry_run=False)
    summaries = writer.write_env_groups([group], config, write_index=False)
    schema = pq.ParquetFile(summaries[0].output_path).schema_arrow

    assert str(schema.field("example_id").type) == "large_string"
    assert str(schema.field("extras").type) == "large_string"
    assert str(schema.field("answer").type) == "large_string"
    assert str(schema.field("error").type) == "large_string"
