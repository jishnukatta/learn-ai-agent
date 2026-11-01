"""tools.py — the "hands": small local functions the agent can choose to call.

Two pieces live here:
  1. The actual Python functions (get_current_time, calculate).
  2. TOOL_SPECS — a description of those tools in the shape Bedrock expects, so
     the model knows they exist and what arguments they take.

Keeping both together makes it easy to add a new tool: write the function,
add its spec, and register it in TOOL_FUNCTIONS at the bottom.
"""

from datetime import datetime


# ── The tool functions (the real work) ───────────────────────────────

def get_current_time() -> str:
    """Return the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '12 * (3 + 4)'.

    We only allow digits and basic math characters, so a model can't trick us
    into running arbitrary code. (A tiny but real guardrail — the tool itself
    enforces safety, not the model.)
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        # eval is dangerous in general; safe here only because we filtered the
        # input to math characters above.
        return str(eval(expression))
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# ── Tool specifications (what we tell the model) ──────────────────────
# Bedrock's converse API needs each tool described with a name, a plain-English
# description (THIS is how the model decides when to use it), and a JSON-schema
# of its inputs.

TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use when the user asks what time or date it is.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression. Use for any math the user asks.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A math expression, e.g. '12 * (3 + 4)'.",
                        }
                    },
                    "required": ["expression"],
                }
            },
        }
    },
]


# Map tool name -> the Python function that implements it. The loop uses this
# to actually run whatever tool the model asks for.
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}
