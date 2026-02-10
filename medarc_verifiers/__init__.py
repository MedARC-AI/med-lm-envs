import logging
import os

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

# Auto-enable token tracking (unless disabled)
if os.getenv("MEDARC_DISABLE_TOKEN_TRACKING", "false").lower() != "true":
    try:
        from medarc_verifiers.utils.token_tracker import install_patches

        _PATCHES_INSTALLED = install_patches()

        if _PATCHES_INSTALLED:
            logging.getLogger(__name__).debug("Token tracking enabled")
        else:
            logging.getLogger(__name__).warning(
                "Token tracking failed to initialize. Evaluations will continue without token tracking."
            )
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Could not import token_tracker: {e}")
else:
    logging.getLogger(__name__).debug("Token tracking disabled via MEDARC_DISABLE_TOKEN_TRACKING")
