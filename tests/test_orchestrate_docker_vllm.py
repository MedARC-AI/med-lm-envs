import pytest

from medarc_verifiers.orchestrate.docker_vllm import DockerLaunchError, build_container_args, normalize_volumes


def test_normalize_volumes_parses_mount_strings():
    volumes = normalize_volumes(["/host/cache:/root/.cache/huggingface:ro", "/host/data:/data"])
    assert volumes["/host/cache"]["bind"] == "/root/.cache/huggingface"
    assert volumes["/host/cache"]["mode"] == "ro"
    assert volumes["/host/data"]["bind"] == "/data"
    assert volumes["/host/data"]["mode"] == "rw"


def test_normalize_volumes_rejects_bad_mount_string():
    with pytest.raises(DockerLaunchError):
        normalize_volumes(["/host/only"])


def test_build_container_args_rejects_unknown_serve_keys():
    with pytest.raises(ValueError):
        build_container_args(
            "some/model",
            tensor_parallel_size=None,
            serve={"dtype": "bfloat16", "unknown_flag": True},
        )


def test_build_container_args_rejects_unknown_limit_mm_subkeys():
    with pytest.raises(ValueError):
        build_container_args(
            "some/model",
            tensor_parallel_size=None,
            serve={"dtype": "bfloat16", "limit_mm_per_prompt": {"audio": 0}},
        )

