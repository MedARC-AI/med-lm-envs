import queue
import time

import pytest

from medarc_verifiers.orchestrate.docker_vllm import (
    ContainerLogStreamer,
    DockerLaunchError,
    build_container_args,
    normalize_volumes,
)


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


def test_container_log_streamer_stop_does_not_hang(tmp_path):
    class BlockingStream:
        def __init__(self):
            self._queue = queue.Queue()

        def __iter__(self):
            return self

        def __next__(self):
            item = self._queue.get()
            if item is None:
                raise StopIteration
            return item

        def send(self, payload: bytes) -> None:
            self._queue.put(payload)

        def close(self) -> None:
            self._queue.put(None)

    class FakeContainer:
        def __init__(self, stream):
            self._stream = stream

        def logs(self, stream: bool, follow: bool):
            return self._stream

    stream = BlockingStream()
    container = FakeContainer(stream)
    sink_path = tmp_path / "logs.txt"
    streamer = ContainerLogStreamer(container, str(sink_path))
    streamer.start()
    stream.send(b"hello\n")
    time.sleep(0.1)
    streamer.stop(timeout=1.0)

    assert streamer.is_alive() is False
    assert sink_path.read_text(encoding="utf-8") == "hello\n"
