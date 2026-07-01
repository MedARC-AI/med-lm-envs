"""AWS Bedrock provider for medmarks.

Call `register()` at CLI startup to enable api_client_type = "bedrock_converse".
"""

from __future__ import annotations


def register() -> None:
    """Monkeypatch verifiers to support bedrock_converse client type."""
    from typing import Literal

    import verifiers.types as vt
    import verifiers.clients as vc
    from verifiers.types import ClientConfig

    # 1. Extend ClientType Literal and rebuild Pydantic model
    vt.ClientType = Literal[
        "openai_completions",
        "openai_chat_completions",
        "openai_chat_completions_token",
        "openai_responses",
        "renderer",
        "anthropic_messages",
        "nemorl_chat_completions",
        "bedrock_converse",
    ]
    ClientConfig.model_fields["client_type"].annotation = vt.ClientType
    ClientConfig.model_rebuild(force=True)

    # Rebuild serve types that embed ClientConfig so worker deserialization accepts bedrock_converse
    try:
        from verifiers.serve.types import RunRolloutRequest, RunGroupRequest

        RunRolloutRequest.model_rebuild(force=True)
        RunGroupRequest.model_rebuild(force=True)
    except ImportError:
        pass

    # 2. Patch resolve_client to handle bedrock_converse
    _original_resolve = vc.resolve_client

    def _patched_resolve(client_or_config):
        if isinstance(client_or_config, ClientConfig) and client_or_config.client_type == "bedrock_converse":
            from medarc_verifiers.bedrock.client import BedrockConverseClient
            return BedrockConverseClient(client_or_config)
        return _original_resolve(client_or_config)

    vc.resolve_client = _patched_resolve

    # Also patch in the env_worker module since it uses `from verifiers.clients import resolve_client`
    try:
        import verifiers.serve.server.env_worker as _ew
        _ew.resolve_client = _patched_resolve
    except (ImportError, AttributeError):
        pass
