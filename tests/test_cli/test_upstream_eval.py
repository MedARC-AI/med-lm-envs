from __future__ import annotations

import verifiers.scripts.eval as upstream_eval_script

from medarc_verifiers.cli import upstream_eval
from medarc_verifiers.cli import verifiers_adapter


def test_upstream_eval_boundary_uses_temporary_adapter_until_public_builder_exists() -> None:
    assert not hasattr(upstream_eval_script, "build_eval_config")
    assert upstream_eval.build_eval_config is verifiers_adapter.build_eval_config
    assert upstream_eval.load_toml_eval_configs is verifiers_adapter.load_toml_eval_configs


def test_temporary_adapter_provider_constants_match_upstream() -> None:
    assert verifiers_adapter.DEFAULT_MODEL == upstream_eval_script.DEFAULT_MODEL
    assert verifiers_adapter.DEFAULT_ENV_DIR_PATH == upstream_eval_script.DEFAULT_ENV_DIR_PATH
    assert verifiers_adapter.DEFAULT_ENDPOINTS_PATH == upstream_eval_script.DEFAULT_ENDPOINTS_PATH
    assert verifiers_adapter.DEFAULT_NUM_EXAMPLES == upstream_eval_script.DEFAULT_NUM_EXAMPLES
    assert verifiers_adapter.DEFAULT_ROLLOUTS_PER_EXAMPLE == upstream_eval_script.DEFAULT_ROLLOUTS_PER_EXAMPLE
    assert verifiers_adapter.DEFAULT_MAX_CONCURRENT == upstream_eval_script.DEFAULT_MAX_CONCURRENT
    assert verifiers_adapter.DEFAULT_CLIENT_TYPE == upstream_eval_script.DEFAULT_CLIENT_TYPE
    assert verifiers_adapter.DEFAULT_PROVIDER == upstream_eval_script.DEFAULT_PROVIDER
    assert verifiers_adapter.PROVIDER_CONFIGS == upstream_eval_script.PROVIDER_CONFIGS
