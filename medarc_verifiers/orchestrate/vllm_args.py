"""Shared vLLM serve argument rendering for orchestrator runtimes."""

from __future__ import annotations

from typing import Mapping


def normalize_volume_mounts(volumes: object) -> list[str]:
    if volumes is None:
        return []
    if isinstance(volumes, Mapping):
        mounts: list[str] = []
        for host, target in volumes.items():
            if not isinstance(host, str) or not isinstance(target, Mapping):
                raise ValueError("orchestrate.vllm-container.volumes mapping entries must be host -> mapping.")
            bind = str(target.get("bind", "")).strip()
            mode = str(target.get("mode", "rw")).strip()
            if not bind:
                raise ValueError(f"Invalid volume mount for host {host!r}: missing bind path.")
            if mode not in {"ro", "rw"}:
                raise ValueError(f"Invalid volume mount mode for host {host!r}: expected ro/rw.")
            mounts.append(f"{host}:{bind}:{mode}")
        return mounts
    if not isinstance(volumes, list):
        raise ValueError("orchestrate.vllm-container.volumes must be a list of mount strings or a mapping.")
    mounts: list[str] = []
    for entry in volumes:
        if not entry:
            continue
        if not isinstance(entry, str):
            raise ValueError("orchestrate.vllm-container.volumes entries must be strings like host:container[:mode].")
        parts = entry.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(f"Invalid volume mount: {entry!r} (expected host:container[:mode])")
        host = parts[0].strip()
        container_path = parts[1].strip()
        mode = parts[2].strip() if len(parts) == 3 else "rw"
        if not host or not container_path:
            raise ValueError(f"Invalid volume mount: {entry!r} (host and container path required)")
        if mode not in {"ro", "rw"}:
            raise ValueError(f"Invalid volume mount mode: {entry!r} (expected ro/rw)")
        mounts.append(f"{host}:{container_path}:{mode}")
    return mounts


def build_container_args(
    model_id: str,
    *,
    tensor_parallel_size: int | None,
    data_parallel_size: int | None = None,
    serve: Mapping[str, object],
) -> list[str]:
    _validate_serve_config(serve)
    args = ["--model", model_id]
    if tensor_parallel_size and tensor_parallel_size > 1:
        args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if data_parallel_size and data_parallel_size > 1:
        args.extend(["--data-parallel-size", str(data_parallel_size)])
    args.extend(_render_serve_flags(serve))
    return args


def _render_serve_flags(serve: Mapping[str, object]) -> list[str]:
    flags: list[str] = []
    scalar_map = {
        "dtype": "--dtype",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "tokenizer_mode": "--tokenizer_mode",
        "config_format": "--config_format",
        "load_format": "--load_format",
        "reasoning_parser": "--reasoning-parser",
        "reasoning_parser_plugin": "--reasoning-parser-plugin",
        "tool_call_parser": "--tool-call-parser",
        "tool_parser_plugin": "--tool-parser-plugin",
        "mamba_ssm_cache_dtype": "--mamba_ssm_cache_dtype",
        "quantization": "--quantization",
        "chat_template": "--chat-template",
    }
    for key, flag in scalar_map.items():
        if key in serve and serve[key] is not None:
            flags.extend([flag, str(serve[key])])
    bool_map = {
        "async_scheduling": "--async-scheduling",
        "enable_prefix_caching": "--enable-prefix-caching",
        "enable_chunked_prefill": "--enable-chunked-prefill",
        "trust_remote_code": "--trust-remote-code",
        "enable_expert_parallel": "--enable-expert-parallel",
        "enable_auto_tool_choice": "--enable-auto-tool-choice",
    }
    for key, flag in bool_map.items():
        if serve.get(key) is True:
            flags.append(flag)
    limit_mm = serve.get("limit_mm_per_prompt")
    if isinstance(limit_mm, Mapping):
        for sub_key in ("image", "video"):
            if sub_key in limit_mm and limit_mm[sub_key] is not None:
                flags.extend([f"--limit-mm-per-prompt.{sub_key}", str(limit_mm[sub_key])])
    return flags


def _validate_serve_config(serve: Mapping[str, object]) -> None:
    scalar_keys = {
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tokenizer_mode",
        "config_format",
        "load_format",
        "reasoning_parser",
        "reasoning_parser_plugin",
        "tool_call_parser",
        "tool_parser_plugin",
        "mamba_ssm_cache_dtype",
        "quantization",
        "chat_template",
    }
    bool_keys = {
        "async_scheduling",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "trust_remote_code",
        "enable_expert_parallel",
        "enable_auto_tool_choice",
    }
    allowed = scalar_keys | bool_keys | {"limit_mm_per_prompt"}
    unknown = sorted(set(serve.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown vLLM serve keys: {unknown}")
    limit_mm = serve.get("limit_mm_per_prompt")
    if limit_mm is None:
        return
    if not isinstance(limit_mm, Mapping):
        raise ValueError("limit_mm_per_prompt must be a mapping.")
    unknown_subkeys = sorted(set(limit_mm.keys()) - {"image", "video"})
    if unknown_subkeys:
        raise ValueError(f"Unknown limit_mm_per_prompt keys: {unknown_subkeys}")


__all__ = ["build_container_args", "normalize_volume_mounts"]
