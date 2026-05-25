import logging

__version__ = "0.2.0"

# Always install judge cache namespacing.
try:
    from medarc_verifiers.judging.judge_cache_fix import install_cache_patch

    _CACHE_PATCH_INSTALLED = install_cache_patch()
    if _CACHE_PATCH_INSTALLED:
        logging.getLogger(__name__).debug("Judge cache namespacing enabled")
    else:
        logging.getLogger(__name__).warning(
            "Judge cache namespacing failed to initialize. Multi-judge runs may share cache entries."
        )
except ImportError as e:
    logging.getLogger(__name__).warning(f"Could not import judge_cache_fix: {e}")

# Honor MedARC endpoint reasoning_field overrides without modifying upstream
# Verifiers' source tree.
try:
    from medarc_verifiers.utils.reasoning_field_patch import install_reasoning_field_patch

    _REASONING_FIELD_PATCH_INSTALLED = install_reasoning_field_patch()
    if _REASONING_FIELD_PATCH_INSTALLED:
        logging.getLogger(__name__).debug("OpenAI chat reasoning_field overrides enabled")
    else:
        logging.getLogger(__name__).warning(
            "OpenAI chat reasoning_field patch failed to initialize. "
            "Endpoint reasoning_field values will be ignored."
        )
except ImportError as e:
    logging.getLogger(__name__).warning(f"Could not import reasoning_field_patch: {e}")
