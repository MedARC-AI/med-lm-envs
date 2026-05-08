"""Boundary for upstream ``verifiers`` eval configuration.

``verifiers==0.1.14`` keeps full ``EvalConfig`` construction nested inside
``verifiers.scripts.eval.main()``, so MedARC still uses a temporary adapter.
Import eval config behavior through this module so callers do not depend on the
adapter directly and the deletion point is isolated when upstream exposes a
public builder.
"""

from __future__ import annotations

from medarc_verifiers.cli.verifiers_adapter import EvalConfigOverrides, build_eval_config, load_toml_eval_configs

__all__ = [
    "EvalConfigOverrides",
    "build_eval_config",
    "load_toml_eval_configs",
]
