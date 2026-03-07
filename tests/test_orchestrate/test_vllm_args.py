import pytest

from medarc_verifiers.orchestrate.vllm_args import build_container_args, normalize_volume_mounts


def test_build_container_args_rejects_unknown_serve_keys() -> None:
    with pytest.raises(ValueError):
        build_container_args(
            "some/model",
            tensor_parallel_size=None,
            serve={"dtype": "bfloat16", "unknown_flag": True},
        )


def test_build_container_args_rejects_unknown_limit_mm_subkeys() -> None:
    with pytest.raises(ValueError):
        build_container_args(
            "some/model",
            tensor_parallel_size=None,
            serve={"dtype": "bfloat16", "limit_mm_per_prompt": {"audio": 0}},
        )


def test_build_container_args_renders_tensor_parallel_and_flags() -> None:
    args = build_container_args(
        "some/model",
        tensor_parallel_size=2,
        data_parallel_size=4,
        serve={"dtype": "bfloat16", "enable_prefix_caching": True, "max_model_len": 8192},
    )

    assert args == [
        "--model",
        "some/model",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enable-prefix-caching",
    ]


def test_normalize_volume_mounts_parses_mount_strings() -> None:
    mounts = normalize_volume_mounts(["/host/cache:/root/.cache/huggingface:ro", "/host/data:/data"])

    assert mounts == ["/host/cache:/root/.cache/huggingface:ro", "/host/data:/data:rw"]


def test_normalize_volume_mounts_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError):
        normalize_volume_mounts(["/host/only"])
