# based on https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/utilities/python.py
import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
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


def python_repl(*, code: str) -> str:
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


class PyREPLEnv(vf.StatefulToolEnv):
    """
    Python REPL environment with persistent state across calls within a single rollout (one dataset row).
    """
    SESSION_KEY = "python_repl_session"

    def __init__(self, **kwargs: Any):
        # only python_repl tool available for this env
        super().__init__(tools=[python_repl], **kwargs)

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
        if tool_name != "python_repl":
            return tool_args

        session = state.get(self.SESSION_KEY)
        if session is None:
            session = ReplSession()
            state[self.SESSION_KEY] = session

        # Bind session for this tool call
        CURRENT_SESSION.set(session)
        return tool_args