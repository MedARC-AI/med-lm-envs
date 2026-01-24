# based on https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/utilities/python.py
import ast
import io
import math
import traceback
import re
from simpleeval import SimpleEval
from contextlib import redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable
import verifiers as vf

# Per-async-task storage to isolate parallel REPL sessions
CURRENT_SESSION: ContextVar["ReplSession"] = ContextVar("python_repl_session")


@dataclass
class ReplSession:
    ns: dict[str, Any] = field(default_factory=dict)

    def run(self, code: str) -> str:
        """
        REPL-like behavior:
        - exec all statements
        - if the last statement is an expression, eval it and print repr(val)
        - persist '_' like the Python REPL
        - capture stdout/stderr and return combined text
        """
        code = (code or "").rstrip()
        if not code:
            return ""

        out = io.StringIO()
        err = io.StringIO()

        try:
            tree = ast.parse(code, mode="exec")
            last_is_expr = bool(tree.body) and isinstance(tree.body[-1], ast.Expr)

            with redirect_stdout(out), redirect_stderr(err):
                if last_is_expr:
                    prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                    if prefix.body:
                        exec(compile(prefix, "<repl>", "exec"), self.ns, self.ns)

                    expr = ast.Expression(tree.body[-1].value)
                    val = eval(compile(expr, "<repl>", "eval"), self.ns, self.ns)
                    self.ns["_"] = val
                    if val is not None:
                        print(repr(val))
                else:
                    exec(compile(tree, "<repl>", "exec"), self.ns, self.ns)

        except Exception:
            err.write(traceback.format_exc())

        stdout = out.getvalue()
        stderr = err.getvalue()

        if stdout and stderr and not stdout.endswith("\n"):
            stdout += "\n"

        # propagate output and errors back to llm
        return stdout + stderr


def python(*, code: str) -> str:
    """Execute Python code in a persistent REPL environment. Variables persist across calls.

    Packages available: numpy (as np), math, and Python standard library.

    Args:
        code: A block of Python code to execute.

    Returns:
        The output (stdout + last expression value if not None) or error message.

    Examples:
        {"code": "import numpy as np\\nnp.array([1, 2, 3]) + np.array([4, 5, 6])"} -> "array([5, 7, 9])"
        {"code": "x = 5\\ny = 10\\nx + y"} -> "15"
        {"code": "a, b = 3, 4\\na, b"} -> "(3, 4)"
        {"code": "result = 2 ** 10"} -> ""
        {"code": "result"} -> "1024"
        {"code": "area = 3.14159 * 5 ** 2\\nprint(f'Area: {area:.2f}')"} -> "Area: 78.54\\n"
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

    Supports basic arithmetic (+, -, *, /, %, **), parentheses, and common math functions.
    Use ^ or ** for exponentiation.

    Args:
        expression: A mathematical expression to evaluate.

    Available functions: sqrt, sin, cos, tan, log, log10, exp, abs, round
    Available constants: pi, e

    Returns:
        The numeric result as a string, or an error message.

    Examples:
        {"expression": "(140 - 87) * 48 * 0.85 / 1.4"} -> "1544.5714285714284"
        {"expression": "sqrt(16) + 2^3"} -> "12.0"
        {"expression": "round(3.14159, 2)"} -> "3.14"
        {"expression": "log10(1000)"} -> "3.0"
        {"expression": "2 * pi * 5"} -> "31.41592653589793"
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
                selected_tools.append(python)

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
        # Only the python tool needs session management; calculator is stateless
        if tool_name not in ("python"):
            return tool_args

        session = state.get(self.SESSION_KEY)
        if session is None:
            session = ReplSession()
            state[self.SESSION_KEY] = session

        # Bind session for this tool call
        CURRENT_SESSION.set(session)
        return tool_args
