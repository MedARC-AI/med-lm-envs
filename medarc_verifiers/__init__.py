import logging

__version__ = "0.1.0"

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

# Token tracking monkey patches are intentionally not installed at import time.
# See plans/verifiers-0.1.10-support-findings.md Step 0.
logging.getLogger(__name__).debug("Token tracking import-time patching disabled")
