"""Shared helpers for building client and eval configs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from verifiers.types import ClientConfig, EvalConfig

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

    client_kwargs: dict[str, Any] = {
        "api_key_var": effective_api_key_var,
        "api_base_url": api_base_url,
        "extra_headers": merged_headers or None,
    }
    timeout = timeout_override if timeout_override is not None else model_cfg.timeout
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if model_cfg.max_connections is not None:
        client_kwargs["max_connections"] = model_cfg.max_connections
    if model_cfg.max_keepalive_connections is not None:
        client_kwargs["max_keepalive_connections"] = model_cfg.max_keepalive_connections
    if model_cfg.max_retries is not None:
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

    max_concurrent = resolve_max_concurrent(
        cli_override=max_concurrent_override,
        model_max=model_cfg.max_concurrent,
        env_max=env_cfg.max_concurrent,
        default_max=default_max_concurrent,
    )
    verbose_flag = env_cfg.verbose if env_cfg.verbose is not None else verbose
    save_every = env_cfg.save_every if env_cfg.save_every is not None else -1
    state_columns = list(env_cfg.state_columns) if env_cfg.state_columns else None

    return EvalConfig(
        env_id=env_id,
        env_args=merged_env_args,
        env_dir_path=str(env_dir),
        model=resolved_model,
        client_config=client_config,
        sampling_args=merged_sampling,
        num_examples=env_cfg.num_examples,
        rollouts_per_example=env_cfg.rollouts_per_example,
        max_concurrent=max_concurrent,
        max_concurrent_generation=max_concurrent_generation,
        max_concurrent_scoring=max_concurrent_scoring,
        interleave_scoring=env_cfg.interleave_scoring,
        print_results=env_cfg.print_results,
        verbose=verbose_flag,
        state_columns=state_columns,
        save_results=save_results,
        save_every=save_every,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
    )


__all__ = ["build_client_config", "build_eval_config"]


def _call_env_metadata_loader(loader: Callable[..., Any], env_id: str, cache: EnvMetadataCache | None) -> Any:
    """Invoke env metadata loader tolerant of positional-only stubs used in tests."""
    try:
        return loader(env_id, cache=cache)
    except TypeError:
        return loader(env_id)
