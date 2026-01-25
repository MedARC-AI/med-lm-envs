import importlib
import io
import json
import math
import operator
import re
from contextlib import redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy
import verifiers as vf
from RestrictedPython import limited_builtins, safe_builtins, utility_builtins
from RestrictedPython.compile import CompileResult, compile_restricted_exec
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector
from simpleeval import SimpleEval

# Per-async-task storage to isolate parallel REPL sessions
CURRENT_SESSION: ContextVar["ReplSession"] = ContextVar("python_repl_session")

_INPLACE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "|=": operator.ior,
    "^=": operator.ixor,
    "&=": operator.iand,
    "//=": operator.ifloordiv,
    "@=": operator.imatmul,
}


def _inplacevar_(op: str, lhs: Any, rhs: Any) -> Any:
    """RestrictedPython hook for augmented assignment on names (e.g. `x += 1`)."""
    func = _INPLACE_OPS.get(op)
    if func is None:
        raise TypeError(f"Unsupported in-place operator: {op!r}")
    return func(lhs, rhs)


def _restricted_import(
    name: str, globals_: dict | None = None, locals_: dict | None = None, fromlist: tuple = (), level: int = 0
):
    if level != 0:
        raise ImportError("Relative imports are not allowed")
    base = name.split(".", 1)[0]
    if base == "math":
        return math
    if base == "numpy":
        return importlib.import_module(name)
    if base == "scipy":
        return importlib.import_module(name)
    raise ImportError("Only 'math', 'numpy', and 'scipy' imports are allowed")


def _build_restricted_globals() -> dict[str, Any]:
    """Build a restricted globals dict for sandboxed Python execution.

    Allows: math, numpy, scipy, basic builtins (print, len, range, etc.)
    Denies: file I/O, network, os, subprocess, eval/exec
    """
    # Combine RestrictedPython's builtin sets:
    # - safe_builtins: core safe builtins (None, True, False, etc.)
    # - limited_builtins: range, min, max, sum, enumerate, zip, etc.
    # - utility_builtins: math, string, random, set, frozenset
    restricted_builtins = {**safe_builtins, **limited_builtins, **utility_builtins}
    # RestrictedPython's builtin sets are intentionally conservative and may omit
    # common pure functions like min/max/sum/enumerate. Add a small allowlist of
    # "boring" helpers that don't increase capability (no I/O, no introspection).
    restricted_builtins.update(
        {
            "min": min,
            "max": max,
            "sum": sum,
            "enumerate": enumerate,
            "any": any,
            "all": all,
            "reversed": reversed,
            "map": map,
            "filter": filter,
        }
    )
    restricted_builtins["__import__"] = _restricted_import

    return {
        "__builtins__": restricted_builtins,
        # RestrictedPython guard functions
        "_getattr_": safer_getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_inplacevar_": _inplacevar_,
        "_write_": lambda x: x,  # Allow writes to safe objects
        # RestrictedPython transforms `print(...)` to calls on a per-exec `_print`
        # object that is created by calling the `_print_` factory with `_getattr_`.
        # PrintCollector captures printed output; we surface it in `ReplSession.run()`.
        "_print_": PrintCollector,
        # Pre-imported safe modules (no import statement needed)
        "math": math,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "sp": scipy,
    }


@dataclass
class ReplSession:
    """Sandboxed Python REPL session using RestrictedPython."""

    ns: dict[str, Any] = field(default_factory=dict)
    _globals: dict[str, Any] = field(default_factory=_build_restricted_globals)

    def run(self, code: str) -> str:
        """
        REPL-like behavior with RestrictedPython sandboxing:
        - exec all statements in a restricted environment
        - if the last statement is an expression, eval it and print repr(val)
        - persist '_' like the Python REPL
        - capture stdout/stderr and return combined text

        Security: Uses RestrictedPython to prevent:
        - File I/O operations
        - Network access
        - OS/subprocess calls
        - Dangerous imports
        - Attribute access to private members
        """
        code = (code or "").rstrip()
        if not code:
            return ""

        out = io.StringIO()
        err = io.StringIO()
        prior_print_obj = self.ns.get("_print", None)

        try:
            # Compile with RestrictedPython
            result: CompileResult = compile_restricted_exec(code, "<repl>")

            errors = result.errors
            warnings = result.warnings
            byte_code = result.code

            if errors:
                # RestrictedPython found policy violations
                err.write("Execution error:\n")
                for error in errors:
                    err.write(f"  {error}\n")
            elif byte_code is None:
                err.write("Compilation failed (no bytecode produced)\n")
            else:
                # Merge session namespace with restricted globals for execution
                exec_globals = {**self._globals, **self.ns}

                with redirect_stdout(out), redirect_stderr(err):
                    exec(byte_code, exec_globals, self.ns)
                if warnings:
                    filtered = [w for w in warnings if "printed" not in str(w).lower()]
                    if filtered:
                        err.write("Execution warning:\n")
                        for warning in filtered:
                            err.write(f"  {warning}\n")

        except ImportError as e:
            err.write(f"ImportError: {e}\n")
            err.write("Hint: Only 'math', 'numpy', and 'scipy' imports are allowed.\n")
        except SyntaxError as e:
            err.write(f"SyntaxError: {e}\n")
        except NameError as e:
            err.write(f"Error: NameError: {e}\n")
            err.write(
                "Hint: this may be a typo, or the name may be unavailable in this restricted Python sandbox. "
                "Allowed imports: math, numpy (np), scipy.\n"
            )
        except Exception as e:
            err.write(f"Error: {e.__class__.__name__}: {e}\n")

        # RestrictedPython prints are collected via a per-exec `_print` object.
        # Append its captured text to stdout output so callers see print() output,
        # even if execution later raises.
        print_obj = self.ns.get("_print", None)
        if print_obj is not None and print_obj is not prior_print_obj and callable(print_obj):
            try:
                out.write(print_obj())
            except Exception:
                # If user code clobbered `_print`, don't fail the whole run.
                pass

        stdout = out.getvalue()
        stderr = err.getvalue()

        if stdout and stderr and not stdout.endswith("\n"):
            stdout += "\n"

        # propagate output and errors back to llm
        return stdout + stderr


def code_interpreter(*, code: str) -> str:
    """Execute Python code in a sandboxed REPL environment. Variables persist across calls.

    This is a restricted environment - file I/O and network access are disabled.
    Allowed imports: math, numpy, scipy. Pre-loaded: numpy (as np), math, scipy. Use print() to display results.

    Args:
        code: A block of Python code to execute.

    Returns:
        The output (stdout) or error message.

    Examples:
        {"code": "print(np.array([1, 2, 3]) + np.array([4, 5, 6]))"} -> "[5 7 9]"
        {"code": "x = 5\\ny = 10\\nprint(x + y)"} -> "15"
        {"code": "result = 2 ** 10\\nprint(result)"} -> "1024"
        {"code": "area = 3.14159 * 5 ** 2\\nprint(f'Area: {area:.2f}')"} -> "Area: 78.54"
    """
    session = CURRENT_SESSION.get(None)
    if session is None:
        return "Internal error: no REPL session bound"
    return session.run(code)


# SimpleEval-based calculator implementation
_ALLOWED_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}
_ALLOWED_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def safe_simpleeval(expr: str) -> float:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Empty expression")
    if len(expr) > 200:
        raise ValueError("Expression too long")
    # Normalize common LLM/user math notation
    expr = expr.replace("^", "**")
    expr = expr.replace("×", "*").replace("÷", "/")

    # Basic character allowlist
    if not re.fullmatch(r"[0-9\.\+\-\*\/\%\(\)\,\s\*\^a-zA-Z_×÷]+", expr):
        raise ValueError("Invalid characters")

    s = SimpleEval(functions=_ALLOWED_FUNCS, names=_ALLOWED_NAMES)
    # Extra paranoia: disallow attribute access / indexing
    s.ATTR_INDEX_FALLBACK = None
    return float(s.eval(expr))


def calculator(*, expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic (+, -, *, /, %, **), parentheses, and common math functions. Use ^ or ** for exponentiation.

    Available functions: sqrt, sin, cos, tan, log, log10, exp, abs, round
    Available constants: pi, e

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The numeric result as a string, or an error message.

    Examples:
        {"expression": "(140 - 87) * 48 * 0.85 / 1.4"} -> "1544.57"
        {"expression": "sqrt(16) + 2^3"} -> "12.0"
        {"expression": "log10(1000)"} -> "3.0"
        {"expression": "2 * pi * 5"} -> "31.42"
    """
    expression = (expression or "").strip()
    if not expression:
        return "Error: Empty expression"
    try:
        result = safe_simpleeval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"


class SimpleToolEnv(vf.StatefulToolEnv):
    """
    Python REPL environment with persistent state across calls within a single rollout (one dataset row).

    Supports configurable tools:
    - python: Full Python execution with persistent state
    - calculator: Safe mathematical expression evaluator
    """

    SESSION_KEY = "python_repl_session"

    def __init__(
        self,
        use_python: bool = True,
        use_calculator: bool = False,
        tools: list[Callable] | None = None,
        **kwargs: Any,
    ):
        """Initialize the environment with configurable tools.

        Args:
            use_python: Include the python_repl tool (default: True)
            use_calculator: Include the calculator tool (default: False)
            tools: Override with a custom list of tools (ignores use_python/use_calculator)
            **kwargs: Additional arguments passed to StatefulToolEnv
        """
        if tools is not None:
            # Custom tools provided, use them directly
            selected_tools = tools
        else:
            # Build tool list from flags
            selected_tools = []
            if use_calculator:
                selected_tools.append(calculator)
            if use_python:
                selected_tools.append(code_interpreter)

            if not selected_tools:
                raise ValueError("At least one tool must be enabled (use_python or use_calculator)")

        super().__init__(tools=selected_tools, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        """Called before each tool invocation to inject the REPL session.

        Each dataset row gets a new state dict, as vf.StatefulToolEnv is derived from vf.MultiTurnEnv
        https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/envs/multiturn_env.py#L103

        This ensures REPL isolation: variables persist within a question but reset
        between questions, since each new episode gets a new state dict and thus
        a new ReplSession.
        """
        allowed_args: dict[str, set[str]] = {"code_interpreter": {"code"}, "calculator": {"expression"}}
        if tool_name in allowed_args:
            tool_args = {key: value for key, value in tool_args.items() if key in allowed_args[tool_name]}

        # Only the python tool needs session management; calculator is stateless
        if tool_name not in ("code_interpreter"):
            return tool_args

        session = state.get(self.SESSION_KEY)
        if session is None:
            session = ReplSession()
            state[self.SESSION_KEY] = session

        # Bind session for this tool call
        CURRENT_SESSION.set(session)
        return tool_args

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> tuple[vf.Messages, vf.State]:
        """Like StatefulToolEnv.env_response, but tolerate malformed tool-call arguments.

        Some models occasionally emit non-JSON `tool_call.function.arguments` strings; the upstream
        implementation `json.loads(...)` will raise and crash the rollout. Here we convert those into
        a tool error message instead.
        """
        assert isinstance(messages, list)
        tool_calls = messages[-1].get("tool_calls", [])

        tool_messages: list[vf.Message] = []
        for tool_call in tool_calls:
            # Some providers/models emit tool calls as JSON-encoded strings. Normalize those.
            if isinstance(tool_call, str):
                try:
                    tool_call = json.loads(tool_call)
                except json.JSONDecodeError:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "content": "Error: Invalid tool formatting.",
                            "tool_call_id": "",
                        }
                    )
                    continue

            # Support both OpenAI tool-call objects and plain dicts.
            tool_call_id = getattr(tool_call, "id", None) or (
                tool_call.get("id", "") if isinstance(tool_call, dict) else ""
            )
            func = getattr(tool_call, "function", None) or (
                tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            )
            tool_name = getattr(func, "name", None) or (func.get("name", "") if isinstance(func, dict) else "")
            raw_args = getattr(func, "arguments", None) or (func.get("arguments", "") if isinstance(func, dict) else "")

            try:
                # Empty/None arguments occur in the wild; treat them as {} so we can return a normal
                # tool error (e.g. missing required params) rather than crashing on JSONDecodeError.
                if raw_args is None or str(raw_args).strip() == "":
                    tool_args = {}
                else:
                    tool_args = json.loads(raw_args)
            except json.JSONDecodeError:
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": (f"Error: Invalid tool formatting for '{tool_name}'."),
                        "tool_call_id": tool_call_id or "",
                    }
                )
                continue

            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            tool_message = await self.call_tool(tool_name, tool_args, tool_call_id or "")
            tool_messages.append(tool_message)

        return tool_messages, state
