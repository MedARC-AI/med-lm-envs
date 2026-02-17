"""Shared helpers for building client and eval configs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from verifiers.types import ClientConfig, EndpointClientConfig, EvalConfig

from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, ModelConfigSchema
from medarc_verifiers.cli.utils.endpoint_utils import (
    EndpointRegistry,
    EnvMetadataCache,
    load_env_metadata,
    resolve_model_endpoint,
)
from medarc_verifiers.cli.utils.env_args import merge_env_args
from medarc_verifiers.cli.utils.shared import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    merge_sampling_overrides,
    normalize_headers,
    resolve_env_identifier,
    resolve_max_concurrent,
)
from medarc_verifiers.utils.prime_inference import prime_inference_overrides

logger = logging.getLogger(__name__)


def build_client_config(
    model_cfg: ModelConfigSchema,
    *,
    endpoints: EndpointRegistry,
    default_api_key_var: str,
    default_api_base_url: str,
    api_base_url_override: str | None,
    http_max_retries_override: int | None,
    timeout_override: float | None,
    headers: list[str] | dict[str, str] | None,
) -> tuple[str, ClientConfig, dict[str, Any]]:
    """Resolve model alias + endpoint settings into a ClientConfig.

    Returns:
        A tuple of (resolved_model, client_config, sampling_overrides).
        - resolved_model: The resolved model identifier
        - client_config: The ClientConfig for API calls
        - sampling_overrides: Prime Inference sampling args to merge (e.g., usage reporting)
    """
    normalized_headers = normalize_headers(headers if headers is not None else model_cfg.headers)
    model_alias = model_cfg.model or model_cfg.id
    if not model_alias:
        raise ValueError("Model entries must define 'id' or 'model'.")

    default_key_var = model_cfg.api_key_var or default_api_key_var
    default_base_url = model_cfg.api_base_url or default_api_base_url
    endpoint_group = endpoints.get(model_alias, [])
    resolved_model, api_key_var, api_base_url = resolve_model_endpoint(
        model_alias,
        endpoints,
        default_key_var=default_key_var,
        default_base_url=default_base_url,
    )
    if api_base_url_override is not None:
        logger.debug("Forcing api_base_url override for model '%s'.", model_alias)
        api_base_url = api_base_url_override

    # Get Prime Inference-specific overrides (headers, sampling args, api_key_var)
    prime_headers, sampling_overrides, prime_api_key_var = prime_inference_overrides(api_base_url)

    # Use Prime API key if auto-detected and user didn't explicitly override
    effective_api_key_var = prime_api_key_var if prime_api_key_var else api_key_var

    # Merge headers: user-provided headers take precedence over Prime auto-detected
    merged_headers = {**prime_headers, **(normalized_headers or {})}

    endpoint_configs: list[EndpointClientConfig] = []
    if api_base_url_override is None and len(endpoint_group) > 1:
        first_entry = endpoint_group[0]
        expected_model = first_entry.get("model", model_alias)
        expected_key = first_entry.get("key", default_key_var)
        for idx, endpoint in enumerate(endpoint_group[1:], start=1):
            entry_model = endpoint.get("model", model_alias)
            entry_key = endpoint.get("key", default_key_var)
            if entry_model != expected_model or entry_key != expected_key:
                raise ValueError(
                    "Endpoint replicas for "
                    f"'{model_alias}' must agree on 'model' and 'key'; "
                    f"variant 0 has model={expected_model!r}, key={expected_key!r}, "
                    f"variant {idx} has model={entry_model!r}, key={entry_key!r}."
                )
        endpoint_configs = [
            EndpointClientConfig(
                api_key_var=effective_api_key_var,
                api_base_url=endpoint["url"],
                extra_headers=merged_headers,
            )
            for endpoint in endpoint_group
        ]

    client_kwargs: dict[str, Any] = {
        "api_key_var": effective_api_key_var,
        "api_base_url": api_base_url,
        "endpoint_configs": endpoint_configs,
        "extra_headers": merged_headers,
    }
    timeout = timeout_override if timeout_override is not None else model_cfg.timeout
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if model_cfg.max_connections is not None:
        client_kwargs["max_connections"] = model_cfg.max_connections
    if model_cfg.max_keepalive_connections is not None:
        client_kwargs["max_keepalive_connections"] = model_cfg.max_keepalive_connections
    if http_max_retries_override is not None:
        client_kwargs["max_retries"] = http_max_retries_override
    elif model_cfg.max_retries is not None:
        client_kwargs["max_retries"] = model_cfg.max_retries

    return resolved_model, ClientConfig(**client_kwargs), sampling_overrides


def build_eval_config(
    *,
    job_label: str | None,
    model_cfg: ModelConfigSchema,
    env_cfg: EnvironmentConfigSchema,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
    cli_env_args: Mapping[str, Any] | None,
    cli_sampling_args: Mapping[str, Any] | None,
    resolved_model: str,
    client_config: ClientConfig,
    env_dir: Path,
    max_concurrent_override: int | None,
    max_concurrent_generation: int | None,
    max_concurrent_scoring: int | None,
    rollout_max_retries: int = 0,
    default_max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT,
    save_results: bool = True,
    save_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
    verbose: bool = False,
    env_metadata_cache: EnvMetadataCache | None = None,
    env_metadata_loader: Callable[..., Any] = load_env_metadata,
    enforce_required_env_args: bool = True,
    allow_unknown_env_args: bool = False,
) -> EvalConfig:
    """Assemble EvalConfig with shared env/sampling override handling."""
    env_id = resolve_env_identifier(env_cfg)
    try:
        metadata = _call_env_metadata_loader(env_metadata_loader, env_id, env_metadata_cache)
    except ImportError as exc:
        logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
        metadata = None

    merged_env_args = merge_env_args(
        env_id,
        sources=[env_args, cli_env_args or {}],
        metadata=metadata,
        metadata_cache=env_metadata_cache,
        allow_unknown=allow_unknown_env_args,
        enforce_required=enforce_required_env_args,
        verbose=verbose,
    )

    merged_sampling = dict(sampling_args)
    merged_sampling = merge_sampling_overrides(merged_sampling, cli_sampling_args)

    _warn_deprecated_eval_knobs(
        env_cfg=env_cfg,
        env_id=env_id,
        job_label=job_label,
        max_concurrent_generation=max_concurrent_generation,
        max_concurrent_scoring=max_concurrent_scoring,
    )

    max_concurrent = resolve_max_concurrent(
        cli_override=max_concurrent_override,
        model_max=model_cfg.max_concurrent,
        env_max=env_cfg.max_concurrent,
        default_max=default_max_concurrent,
    )
    verbose_flag = env_cfg.verbose if env_cfg.verbose is not None else verbose
    state_columns = list(env_cfg.state_columns) if env_cfg.state_columns else None
    eval_config_fields = _pydantic_field_names(EvalConfig)

    eval_kwargs: dict[str, Any] = {
        "env_id": env_id,
        "env_args": merged_env_args,
        "env_dir_path": str(env_dir),
        "model": resolved_model,
        "client_config": client_config,
        "sampling_args": merged_sampling,
        "num_examples": env_cfg.num_examples,
        "rollouts_per_example": env_cfg.rollouts_per_example,
        "max_concurrent": max_concurrent,
        "verbose": verbose_flag,
        "state_columns": state_columns,
        "save_results": save_results,
        "save_to_hf_hub": save_to_hf_hub,
        "hf_hub_dataset_name": hf_hub_dataset_name,
    }
    if "max_retries" in eval_config_fields:
        eval_kwargs["max_retries"] = rollout_max_retries

    independent_scoring = getattr(env_cfg, "independent_scoring", None)
    interleave_scoring = getattr(env_cfg, "interleave_scoring", None)

    if interleave_scoring is not None:
        raise ValueError(
            f"Environment '{env_id}' uses interleave_scoring, which is no longer supported; use independent_scoring."
        )

    if "independent_scoring" in eval_config_fields:
        if independent_scoring is None:
            independent_scoring = True
        eval_kwargs["independent_scoring"] = bool(independent_scoring)
    elif independent_scoring is not None:
        logger.warning(
            "Environment '%s' set independent_scoring=%s, but installed verifiers does not support it; ignoring.",
            env_id,
            independent_scoring,
        )

    if "extra_env_kwargs" in eval_config_fields:
        extra_env_kwargs = getattr(env_cfg, "extra_env_kwargs", None)
        if extra_env_kwargs is not None:
            eval_kwargs["extra_env_kwargs"] = dict(extra_env_kwargs)

    return EvalConfig(**eval_kwargs)


__all__ = ["build_client_config", "build_eval_config"]


def _call_env_metadata_loader(loader: Callable[..., Any], env_id: str, cache: EnvMetadataCache | None) -> Any:
    """Invoke env metadata loader tolerant of positional-only stubs used in tests."""
    try:
        return loader(env_id, cache=cache)
    except TypeError:
        return loader(env_id)


def _pydantic_field_names(model_type: type[Any]) -> set[str]:
    fields = getattr(model_type, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    fields = getattr(model_type, "__fields__", None)
    if isinstance(fields, dict):
        return set(fields.keys())
    return set()


def _warn_deprecated_eval_knobs(
    *,
    env_cfg: Any,
    env_id: str,
    job_label: str | None,
    max_concurrent_generation: int | None,
    max_concurrent_scoring: int | None,
) -> None:
    env_fields_set = set(getattr(env_cfg, "model_fields_set", set()))

    deprecated_env_knobs: list[str] = []
    if "save_every" in env_fields_set and getattr(env_cfg, "save_every", None) is not None:
        deprecated_env_knobs.append("save_every")
    if "print_results" in env_fields_set:
        deprecated_env_knobs.append("print_results")
    if deprecated_env_knobs:
        logger.warning(
            "Environment '%s' sets deprecated eval knob(s): %s. These options are ignored.",
            env_id,
            ", ".join(sorted(deprecated_env_knobs)),
        )

    deprecated_concurrency_knobs: list[str] = []
    if max_concurrent_generation is not None:
        deprecated_concurrency_knobs.append("max_concurrent_generation")
    if max_concurrent_scoring is not None:
        deprecated_concurrency_knobs.append("max_concurrent_scoring")
    if deprecated_concurrency_knobs:
        label = job_label or env_id
        logger.warning(
            "Job '%s' sets deprecated eval knob(s): %s. These options are ignored.",
            label,
            ", ".join(sorted(deprecated_concurrency_knobs)),
        )
