"""
Token tracking for OpenAI-compatible API responses.
Tracks model and judge token usage separately via monkey-patching.
Automatically enabled on import unless MEDARC_DISABLE_TOKEN_TRACKING=true.
"""

import logging

logger = logging.getLogger(__name__)


class TokenTracker:
    """
    Tracks token usage from OpenAI-compatible API responses.
    Stores in state["token_usage"] as nested dict.
    """

    STATE_KEY = "token_usage"

    @staticmethod
    def _safe_get(obj, key, default=None):
        """Get attribute or dict item safely."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _get_usage_field(usage, field, default=0):
        """Return a usage field from dicts or objects."""
        if usage is None:
            return default
        if isinstance(usage, dict):
            return usage.get(field, default)
        return getattr(usage, field, default)

    @staticmethod
    def _update_usage_stats(stats: dict, usage) -> None:
        """
        Update a stats dict in-place with information from a usage object.
        Only adds optional keys (reasoning_tokens, cost) if present.
        """
        # Basic token counts
        stats["prompt"] = stats.get("prompt", 0) + (TokenTracker._get_usage_field(usage, "prompt_tokens", 0) or 0)
        stats["completion"] = stats.get("completion", 0) + (
            TokenTracker._get_usage_field(usage, "completion_tokens", 0) or 0
        )
        stats["total"] = stats.get("total", 0) + (TokenTracker._get_usage_field(usage, "total_tokens", 0) or 0)

        # Reasoning tokens (if available)
        completion_details = TokenTracker._get_usage_field(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = TokenTracker._get_usage_field(completion_details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                stats["reasoning_tokens"] = stats.get("reasoning_tokens", 0) + (reasoning_tokens or 0)

        # Cost (if available)
        cost = TokenTracker._get_usage_field(usage, "cost", None)
        if cost is not None:
            stats["cost"] = stats.get("cost", 0.0) + cost

    @staticmethod
    def init_tracking(state: dict) -> None:
        """Initialize token tracking structure in state."""
        if TokenTracker.STATE_KEY not in state:
            state[TokenTracker.STATE_KEY] = {
                "model": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                    "cost": 0.0,
                },
                "judge": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                    "cost": 0.0,
                },
            }

    @staticmethod
    def track_judge_tokens(state: dict, response) -> None:
        """
        Track judge tokens from ChatCompletion response.
        Args:
            state: Rollout state dict
            response: ChatCompletion object (before conversion to string)
        """
        TokenTracker.init_tracking(state)

        usage = TokenTracker._safe_get(response, "usage", None)
        if usage:
            TokenTracker._update_usage_stats(state[TokenTracker.STATE_KEY]["judge"], usage)


def install_patches() -> bool:
    """
    Monkey-patch verifiers for token tracking.
    Patches:
    1. JudgeRubric.judge() - Track judge tokens before text extraction
    2. eval_utils.make_dataset() - Extract model + judge tokens, add to results
    Returns:
        bool: True on success, False on failure (with warning)
    """
    try:
        from verifiers.rubrics.judge_rubric import JudgeRubric
        from verifiers.utils import eval_utils
        from verifiers.utils.async_utils import maybe_await
        from openai import APIError, APITimeoutError, RateLimitError

        # ===== PATCH 1: JudgeRubric.judge() =====
        # Store original as a class attribute to ensure all instances see it
        if not hasattr(JudgeRubric, "_original_judge_unpatched"):
            JudgeRubric._original_judge_unpatched = JudgeRubric.judge

        async def patched_judge(self, prompt, completion, answer, state, **kwargs):
            """Patched judge() that tracks token usage before text extraction."""

            # Replicate cache check logic from original
            if isinstance(prompt, list):
                last_msg = prompt[-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    question = str(last_msg["content"])
                else:
                    question = ""
            else:
                question = str(prompt)

            response_text = self.parser.parse_answer(completion)
            judge_prompt = self.judge_prompt.format(question=question, answer=answer, response=response_text)

            # Check cache
            cached = state.get("judge_response")
            if isinstance(cached, dict) and judge_prompt in cached:
                return cached[judge_prompt]  # Cache hit, no API call

            # Normalize judge sampling args for chat API
            judge_args = dict(self.judge_sampling_args or {})
            if "max_tokens" in judge_args:
                if judge_args["max_tokens"] is None:
                    judge_args.pop("max_tokens")
                else:
                    judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
            if "max_completion_tokens" in judge_args and judge_args["max_completion_tokens"] is None:
                judge_args.pop("max_completion_tokens")
            judge_args = {k: v for k, v in judge_args.items() if v is not None}

            # Make API call with error handling
            try:
                judge_response_obj = await maybe_await(
                    self.judge_client.chat.completions.create,
                    model=self.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    **judge_args,
                )

                # *** TRACK TOKENS BEFORE DISCARDING, not implemented in verifiers judgerubric ***
                TokenTracker.track_judge_tokens(state, judge_response_obj)

                # Extract text (original behavior)
                judge_response_text = str(judge_response_obj.choices[0].message.content)
            except RateLimitError as e:
                self.logger.warning(
                    f"Rate limit exceeded when calling judge model '{self.judge_model}'. "
                    f"Try reducing concurrency or waiting before retrying. Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Judge model rate limit exceeded. Try reducing concurrency or waiting before retrying. "
                    f"Model: {self.judge_model}, Error: {str(e)}"
                ) from e
            except APITimeoutError as e:
                self.logger.warning(
                    f"Timeout when calling judge model '{self.judge_model}'. "
                    f"Increase timeout in judge_sampling_args or check model responsiveness. Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Judge model timeout. Increase timeout in judge_sampling_args or check model responsiveness. "
                    f"Model: {self.judge_model}, Error: {str(e)}"
                ) from e
            except APIError as e:
                self.logger.warning(
                    f"API error when calling judge model '{self.judge_model}'. "
                    f"Check model availability and API key. Error: {str(e)}"
                )
                raise RuntimeError(
                    f"Judge model API error. Check model availability and API key. "
                    f"Model: {self.judge_model}, Error: {str(e)}"
                ) from e
            except Exception as e:
                self.logger.warning(f"Unexpected error when calling judge model '{self.judge_model}'. Error: {str(e)}")
                raise RuntimeError(
                    f"Unexpected error when calling judge model '{self.judge_model}'. Error: {str(e)}"
                ) from e

            # Cache and return
            if not isinstance(cached, dict):
                cached = {}
            cached[judge_prompt] = judge_response_text
            state["judge_response"] = cached

            return judge_response_text

        JudgeRubric.judge = patched_judge

        # ===== PATCH 2: eval_utils.make_dataset() =====
        original_make_dataset = eval_utils.make_dataset

        def patched_make_dataset(results, push_to_hf_hub=False, hf_hub_dataset_name=None, **kwargs):
            """Patched make_dataset() that adds token_usage column."""

            try:
                # Upstream make_dataset currently accepts only (results, **kwargs).
                # Do NOT pass extra positional args to preserve compatibility across versions.
                dataset = original_make_dataset(results, **kwargs)

                # Build token_usage dict for each rollout
                states = TokenTracker._safe_get(results, "state", []) or []
                token_data = []
                for state in states:
                    # Extract model tokens from existing state["responses"]
                    model_tokens = {
                        "prompt": 0,
                        "completion": 0,
                        "total": 0,
                        "cost": 0.0,
                    }

                    # Legacy path: old verifiers stored raw responses under state["responses"]
                    for response in TokenTracker._safe_get(state, "responses", []) or []:
                        usage = TokenTracker._safe_get(response, "usage", None)
                        if usage:
                            TokenTracker._update_usage_stats(model_tokens, usage)
                    # Current path: responses live inside trajectory steps
                    for step in TokenTracker._safe_get(state, "trajectory", []) or []:
                        response = TokenTracker._safe_get(step, "response", None)
                        usage = TokenTracker._safe_get(response, "usage", None)
                        if usage:
                            TokenTracker._update_usage_stats(model_tokens, usage)

                    # Get judge tokens from our patch
                    judge_tokens = TokenTracker._safe_get(state, TokenTracker.STATE_KEY, {}) or {}
                    judge_tokens = judge_tokens.get(
                        "judge",
                        {
                            "prompt": 0,
                            "completion": 0,
                            "total": 0,
                            "cost": 0.0,
                        },
                    )

                    # Calculate totals
                    total_tokens = {
                        "prompt": model_tokens.get("prompt", 0) + judge_tokens.get("prompt", 0),
                        "completion": model_tokens.get("completion", 0) + judge_tokens.get("completion", 0),
                        "total": model_tokens.get("total", 0) + judge_tokens.get("total", 0),
                    }

                    # Reasoning tokens if either side has them
                    if "reasoning_tokens" in model_tokens or "reasoning_tokens" in judge_tokens:
                        total_tokens["reasoning_tokens"] = model_tokens.get("reasoning_tokens", 0) + judge_tokens.get(
                            "reasoning_tokens", 0
                        )

                    # Cost if either side has it
                    if "cost" in model_tokens or "cost" in judge_tokens:
                        total_tokens["cost"] = float(model_tokens.get("cost", 0.0)) + float(
                            judge_tokens.get("cost", 0.0)
                        )

                    # Single dict with all token data
                    token_data.append(
                        {
                            "model": model_tokens,
                            "judge": judge_tokens,
                            "total": total_tokens,
                            "cost": total_tokens.get("cost", 0.0),
                        }
                    )

                # Add single column with dict
                dataset = dataset.add_column("token_usage", token_data)

                return dataset
            except Exception as e:
                logger.error(f"Error adding token_usage column: {e}", exc_info=True)
                # Fallback to original dataset without token_usage if our augmentation fails
                try:
                    return original_make_dataset(results, **kwargs)
                except Exception:
                    # If even the original fails, re-raise to preserve upstream behavior
                    raise

        eval_utils.make_dataset = patched_make_dataset

        logger.debug("Token tracking patches installed successfully")
        return True

    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to install token tracking patches: {e}. "
            f"Token tracking will be disabled. This may indicate a verifiers version mismatch."
        )
        return False
