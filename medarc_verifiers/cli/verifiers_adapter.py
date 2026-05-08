"""Small adapter for upstream ``verifiers`` eval configuration.

Upstream ``verifiers`` owns TOML loading and eval execution, but in 0.1.12 the
``EvalConfig`` builder lives inside ``verifiers.scripts.eval.main()`` and cannot
be imported directly. Keep this module deliberately narrow until upstream exposes
a public builder.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from verifiers.types import (
    ClientConfig,
    ClientType,
    Endpoint,
    EndpointClientConfig,
    EvalConfig,
)
from verifiers.utils.eval_utils import load_endpoints, load_toml_config, resolve_endpoints_file
from verifiers.utils.import_utils import load_toml

from medarc_verifiers.cli.utils.endpoint_utils import load_endpoint_sampling_profiles
from medarc_verifiers.utils.prime_inference import prime_inference_overrides
from medarc_verifiers.utils.sampling_args import sanitize_sampling_args_for_openai

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_CLIENT_TYPE = "openai_chat_completions"
DEFAULT_PROVIDER = "prime"
ADAPTER_TOML_FIELDS = {"debug", "header_from_state", "headers_from_state", "timeout"}
MEDARC_TOML_METADATA_FIELD = "medarc"
MEDARC_TOML_IDENTITY_FIELDS = {"name", "variant_id"}

PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "prime": {
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic_messages",
    },
    "minimax": {
        "url": "https://api.minimax.chat/v1",
        "key": "MINIMAX_API_KEY",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1",
        "key": "DEEPSEEK_API_KEY",
    },
    "glm": {
        "url": "https://open.bigmodel.cn/api/paas/v4",
        "key": "GLM_API_KEY",
    },
    "local": {
        "url": "http://localhost:8000/v1",
        "key": "VLLM_API_KEY",
    },
    "vllm": {
        "url": "http://localhost:8000/v1",
        "key": "VLLM_API_KEY",
    },
}


@dataclass(frozen=True)
class EvalConfigOverrides:
    """CLI-level overrides applied after TOML globals and per-eval fields."""

    model: str | None = None
    provider: str | None = None
    api_base_url: str | None = None
    api_key_var: str | None = None
    api_client_type: str | None = None
    endpoints_path: str | Path | None = None
    max_concurrent: int | None = None
    env_args: Mapping[str, Any] | None = None
    sampling_args: Mapping[str, Any] | None = None


def load_toml_eval_configs(path: str | Path, *, extra_valid_fields: set[str] | None = None) -> list[dict[str, Any]]:
    """Load upstream TOML eval configs, including ``[[ablation]]`` expansion."""

    valid_fields = (
        ADAPTER_TOML_FIELDS | {MEDARC_TOML_METADATA_FIELD} | MEDARC_TOML_IDENTITY_FIELDS | (extra_valid_fields or set())
    )
    return [_strip_medarc_metadata(raw) for raw in load_toml_config(Path(path), extra_valid_fields=valid_fields)]


def _strip_medarc_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(raw)
    cleaned.pop(MEDARC_TOML_METADATA_FIELD, None)
    return cleaned


def build_eval_config(raw: Mapping[str, Any], *, overrides: EvalConfigOverrides | None = None) -> EvalConfig:
    """Build an upstream ``EvalConfig`` from one loaded TOML/CLI eval mapping."""

    merged_raw = _apply_overrides(dict(raw), overrides)
    env_id = merged_raw["env_id"]

    env_defaults = get_env_eval_defaults(env_id)
    raw_num_examples = merged_raw.get("num_examples")
    raw_rollouts = merged_raw.get("rollouts_per_example")
    num_examples = (
        raw_num_examples if raw_num_examples is not None else env_defaults.get("num_examples", DEFAULT_NUM_EXAMPLES)
    )
    rollouts_per_example = (
        raw_rollouts
        if raw_rollouts is not None
        else env_defaults.get("rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE)
    )

    endpoints_path = str(merged_raw.get("endpoints_path", DEFAULT_ENDPOINTS_PATH))
    endpoints = load_endpoints(endpoints_path)
    model, resolved_endpoint_id, client_config = _build_client_config(merged_raw, endpoints, endpoints_path)

    endpoint_sampling_profiles = load_endpoint_sampling_profiles(endpoints_path)
    endpoint_sampling_args = _resolve_endpoint_sampling_args(endpoint_sampling_profiles, resolved_endpoint_id)
    cli_sampling_args = overrides.sampling_args if overrides is not None else None
    sampling_args = _build_sampling_args(
        merged_raw,
        client_config.api_base_url,
        endpoint_sampling_args=endpoint_sampling_args,
        cli_sampling_args=cli_sampling_args,
    )

    extra_env_kwargs = dict(merged_raw.get("extra_env_kwargs", {}))
    if merged_raw.get("timeout") is not None:
        extra_env_kwargs["timeout_seconds"] = merged_raw["timeout"]

    return EvalConfig(
        env_id=env_id,
        env_args=merged_raw.get("env_args", {}),
        env_dir_path=merged_raw.get("env_dir_path", DEFAULT_ENV_DIR_PATH),
        output_dir=merged_raw.get("output_dir"),
        extra_env_kwargs=extra_env_kwargs,
        endpoint_id=resolved_endpoint_id,
        model=model,
        client_config=client_config,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=merged_raw.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
        max_retries=merged_raw.get("max_retries", 0),
        num_workers=merged_raw.get("num_workers", "auto"),
        disable_env_server=merged_raw.get("disable_env_server", False),
        debug=merged_raw.get("debug", False),
        verbose=merged_raw.get("verbose", False),
        state_columns=merged_raw.get("state_columns", []),
        save_results=merged_raw.get("save_results", False),
        resume_path=None,
        independent_scoring=merged_raw.get("independent_scoring", False),
        save_to_hf_hub=merged_raw.get("save_to_hf_hub", False),
        hf_hub_dataset_name=merged_raw.get("hf_hub_dataset_name", ""),
    )


def get_env_eval_defaults(env_id: str) -> dict[str, Any]:
    """Read ``[tool.verifiers.eval]`` defaults from an installed env package."""

    defaults: dict[str, Any] = {}
    module_name = env_id.replace("-", "_").split("/")[-1]

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ModuleNotFoundError(module_name)

        if spec.submodule_search_locations:
            base_dir = Path(next(iter(spec.submodule_search_locations)))
        elif spec.origin:
            base_dir = Path(spec.origin).parent
        else:
            logger.debug("Could not determine module path for %s; skipping eval defaults", module_name)
            return defaults

        pyproject_file = _find_env_pyproject(base_dir)
        if not pyproject_file.is_file():
            logger.debug("pyproject.toml not found for installed module %s", module_name)
            return defaults

        with pyproject_file.open("rb") as handle:
            pyproject_data = load_toml(handle)

        eval_config = pyproject_data.get("tool", {}).get("verifiers", {}).get("eval", {})
        if "num_examples" in eval_config:
            defaults["num_examples"] = eval_config["num_examples"]
        if "rollouts_per_example" in eval_config:
            defaults["rollouts_per_example"] = eval_config["rollouts_per_example"]
    except ModuleNotFoundError:
        logger.debug("Module %s not installed", module_name)
    except Exception as exc:
        logger.debug("Could not load eval defaults from %s pyproject.toml: %s", module_name, exc)

    return defaults


def _find_env_pyproject(base_dir: Path) -> Path:
    candidates = [base_dir / "pyproject.toml", base_dir.parent / "pyproject.toml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _apply_overrides(raw: dict[str, Any], overrides: EvalConfigOverrides | None) -> dict[str, Any]:
    if overrides is None:
        return raw

    for field in ("provider", "api_base_url", "api_key_var", "api_client_type", "max_concurrent"):
        value = getattr(overrides, field)
        if value is not None:
            raw[field] = value

    if overrides.endpoints_path is not None:
        raw["endpoints_path"] = str(overrides.endpoints_path)

    if overrides.model is not None:
        raw["model"] = overrides.model
        raw.pop("endpoint_id", None)

    if overrides.env_args:
        raw["env_args"] = {**dict(raw.get("env_args", {})), **dict(overrides.env_args)}
    return raw


def _build_client_config(
    raw: Mapping[str, Any], endpoints: Mapping[str, list[Endpoint]], endpoints_path: str
) -> tuple[str, str | None, ClientConfig]:
    raw_endpoint_id = raw.get("endpoint_id")
    raw_model_field = raw.get("model")
    if raw_endpoint_id is not None and raw_model_field is not None:
        raise ValueError("Cannot set both 'endpoint_id' and 'model' in eval config; choose one.")
    if raw_endpoint_id is not None and not isinstance(raw_endpoint_id, str):
        raise ValueError("'endpoint_id' must be a string when provided.")
    if isinstance(raw_endpoint_id, str) and not raw_endpoint_id:
        raise ValueError("'endpoint_id' must be a non-empty string when provided.")

    resolved_endpoints_file = resolve_endpoints_file(endpoints_path)
    if raw_endpoint_id is not None and (resolved_endpoints_file is None or resolved_endpoints_file.suffix != ".toml"):
        raise ValueError(
            "'endpoint_id' is only supported with TOML endpoint registries. Set endpoints_path to an endpoints.toml file."
        )

    raw_model = raw_model_field if raw_model_field is not None else DEFAULT_MODEL
    endpoint_lookup_id = raw_endpoint_id if raw_endpoint_id is not None else raw_model
    raw_api_base_url = raw.get("api_base_url")
    if isinstance(raw_api_base_url, list):
        raise ValueError(
            "api_base_url lists are no longer supported. Use endpoint_id + endpoints.toml for multi-endpoint configuration."
        )

    raw_provider = raw.get("provider")
    if raw_provider is not None and raw_provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider '{raw_provider}'. Valid providers are: {sorted(PROVIDER_CONFIGS)}")

    api_key_override = raw.get("api_key_var") is not None
    api_base_url_override = raw_api_base_url is not None
    client_type_override = raw.get("api_client_type") is not None
    endpoint_group: list[Endpoint] | None = None
    resolved_endpoint_id: str | None = None

    if endpoint_lookup_id in endpoints:
        endpoint_group = list(endpoints[endpoint_lookup_id])
        resolved_endpoint_id = cast(str, endpoint_lookup_id)
        endpoint = endpoint_group[0]

        api_key_var = endpoint["key"]
        api_base_url = endpoint["url"]
        client_type = endpoint.get("api_client_type", DEFAULT_CLIENT_TYPE)

        endpoint_models = {entry["model"] for entry in endpoint_group}
        if len(endpoint_models) > 1:
            raise ValueError(
                f"Endpoint alias '{endpoint_lookup_id}' maps to multiple model ids {sorted(endpoint_models)}, "
                "which is not yet supported by EvalConfig."
            )
        model = endpoint["model"]

        if raw_provider is not None:
            provider_cfg = PROVIDER_CONFIGS[raw_provider]
            api_key_var = provider_cfg["key"]
            api_base_url = provider_cfg["url"]
            client_type = provider_cfg.get("client_type", client_type)
        if api_key_override:
            api_key_var = raw["api_key_var"]
        if api_base_url_override:
            api_base_url = raw_api_base_url
        if client_type_override:
            client_type = raw["api_client_type"]
    else:
        if raw_endpoint_id is not None:
            raise ValueError(f"Endpoint id '{raw_endpoint_id}' not found in endpoint registry at {endpoints_path}")
        provider_cfg = PROVIDER_CONFIGS[raw_provider or DEFAULT_PROVIDER]
        model = raw_model
        api_key_var = raw["api_key_var"] if api_key_override else raw.get("default_api_key_var", provider_cfg["key"])
        api_base_url = (
            raw_api_base_url if api_base_url_override else raw.get("default_api_base_url", provider_cfg["url"])
        )
        client_type = (
            raw["api_client_type"] if client_type_override else provider_cfg.get("client_type", DEFAULT_CLIENT_TYPE)
        )

    if not isinstance(api_base_url, str):
        raise ValueError("api_base_url must be a single string URL")
    if not isinstance(api_key_var, str):
        raise ValueError("api_key_var must be a string")

    eval_headers_merged = _build_extra_headers(raw)
    prime_headers, _ = prime_inference_overrides(api_base_url)
    eval_headers_from_state = {"X-Session-ID": "example_id", **_build_extra_headers_from_state(raw)}

    registry_headers_base: dict[str, str] = {}
    if endpoint_group is not None:
        registry_headers_base = dict(endpoint_group[0].get("extra_headers", {}))
    merged_headers = {**prime_headers, **registry_headers_base, **eval_headers_merged}

    endpoint_configs: list[EndpointClientConfig] = []
    if endpoint_group is not None and not api_base_url_override and raw_provider is None and len(endpoint_group) > 1:
        endpoint_configs = [
            EndpointClientConfig(
                api_key_var=api_key_var if api_key_override else endpoint["key"],
                api_base_url=endpoint["url"],
                extra_headers={**prime_headers, **dict(endpoint.get("extra_headers", {})), **eval_headers_merged},
            )
            for endpoint in endpoint_group
        ]

    client_kwargs: dict[str, Any] = {
        "client_type": cast(ClientType, client_type),
        "api_key_var": api_key_var,
        "api_base_url": api_base_url,
        "endpoint_configs": endpoint_configs,
        "extra_headers": merged_headers,
        "extra_headers_from_state": eval_headers_from_state,
    }
    if raw.get("client_timeout") is not None:
        client_kwargs["timeout"] = raw["client_timeout"]
    if raw.get("http_max_retries") is not None:
        client_kwargs["max_retries"] = raw["http_max_retries"]

    client_config = ClientConfig(**client_kwargs)
    return cast(str, model), resolved_endpoint_id, client_config


def _resolve_endpoint_sampling_args(
    endpoint_sampling_profiles: Mapping[str, list[dict[str, Any]]], endpoint_id: str | None
) -> dict[str, Any]:
    if endpoint_id is None:
        return {}

    profiles = endpoint_sampling_profiles.get(endpoint_id, [])
    if not profiles:
        return {}

    first = profiles[0]
    for profile in profiles[1:]:
        if profile != first:
            raise ValueError(
                f"Endpoint alias '{endpoint_id}' has conflicting sampling_args across replica entries. "
                "Use identical sampling_args for every replica or omit them from every replica."
            )
    return dict(first)


def _build_sampling_args(
    raw: Mapping[str, Any],
    api_base_url: str,
    *,
    endpoint_sampling_args: Mapping[str, Any] | None = None,
    cli_sampling_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _, prime_sampling_overrides = prime_inference_overrides(api_base_url)
    endpoint_sampling = _validate_sampling_mapping(endpoint_sampling_args, "endpoint sampling_args")
    include_none_max_tokens = raw.get("include_none_max_tokens", True) and (
        "max_tokens" in raw or "max_tokens" not in endpoint_sampling
    )
    scalar_sampling_args = _merge_sampling_args(
        None,
        max_tokens=raw.get("max_tokens"),
        temperature=raw.get("temperature"),
        include_none_max_tokens=include_none_max_tokens,
    )
    merged = _deep_merge(prime_sampling_overrides, endpoint_sampling)
    merged = _deep_merge(merged, scalar_sampling_args)
    merged = _deep_merge(merged, _validate_sampling_mapping(raw.get("sampling_args"), "sampling_args"))
    merged = _deep_merge(merged, _validate_sampling_mapping(cli_sampling_args, "CLI sampling_args"))
    return sanitize_sampling_args_for_openai(merged)


def _merge_sampling_args(
    sampling_args: Mapping[str, Any] | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    prefer_existing_keys: bool = True,
    include_none_max_tokens: bool = False,
) -> dict[str, Any]:
    merged_sampling_args = dict(sampling_args or {})
    if (not prefer_existing_keys or "max_tokens" not in merged_sampling_args) and (
        include_none_max_tokens or max_tokens is not None
    ):
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and (not prefer_existing_keys or "temperature" not in merged_sampling_args):
        merged_sampling_args["temperature"] = temperature
    return merged_sampling_args


def _build_extra_headers(raw: Mapping[str, Any]) -> dict[str, str]:
    eval_headers_table: dict[str, str] = {}
    raw_headers = raw.get("headers")
    if raw_headers is not None:
        eval_headers_table = _validate_header_mapping(raw_headers)

    raw_header_values = raw.get("header") or []
    if not isinstance(raw_header_values, list):
        raise ValueError("'header' must be a list of 'Name: Value' strings")

    eval_headers_from_list: dict[str, str] = {}
    for header_value in raw_header_values:
        if not isinstance(header_value, str):
            raise ValueError(f"Each 'header' entry must be a string 'Name: Value', got: {header_value!r}")
        if ":" not in header_value:
            raise ValueError(f"--header must be 'Name: Value', got: {header_value!r}")
        key, value = header_value.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError("--header name cannot be empty")
        eval_headers_from_list[key] = value

    return {**eval_headers_table, **eval_headers_from_list}


def _build_extra_headers_from_state(raw: Mapping[str, Any]) -> dict[str, str]:
    table: dict[str, str] = {}
    raw_table = raw.get("headers_from_state")
    if raw_table is not None:
        table = _validate_header_mapping(raw_table)

    raw_list = raw.get("header_from_state") or []
    if not isinstance(raw_list, list):
        raise ValueError("'header_from_state' must be a list of 'Name: state_key' strings")

    from_list: dict[str, str] = {}
    for entry in raw_list:
        if not isinstance(entry, str):
            raise ValueError(f"Each 'header_from_state' entry must be a string 'Name: state_key', got: {entry!r}")
        if ":" not in entry:
            raise ValueError(f"--header-from-state must be 'Name: state_key', got: {entry!r}")
        key, value = entry.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError("--header-from-state name cannot be empty")
        if not value:
            raise ValueError("--header-from-state state_key cannot be empty")
        from_list[key] = value

    return {**table, **from_list}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(cast(Mapping[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def _validate_sampling_mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a dict")
    return dict(cast(Mapping[str, Any], value))


def _validate_header_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("headers must be a dict")

    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("headers keys must be non-empty strings")
        if not isinstance(header_value, str):
            raise ValueError("headers values must be strings")
        headers[key] = header_value
    return headers


__all__ = [
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_NUM_EXAMPLES",
    "DEFAULT_ROLLOUTS_PER_EXAMPLE",
    "EvalConfigOverrides",
    "build_eval_config",
    "get_env_eval_defaults",
    "load_toml_eval_configs",
]
