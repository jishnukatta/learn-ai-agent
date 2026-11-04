"""agent_lc.py — the SAME agent as agent.py, but built with LangChain + LangGraph.

Compare this with agent.py (the hand-written loop). Everything you wrote by hand
there — tracking messages, checking stopReason, parsing toolUse, running tools,
feeding back toolResult, the MAX_STEPS loop — is done FOR you here by
`create_react_agent`. You only: define tools, pick a model, and invoke.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


# ── Tools ─────────────────────────────────────────────────────────────
# The @tool decorator turns a plain function into a LangChain tool. It reads
# the type hints + docstring to build the schema automatically — no TOOL_SPECS
# to hand-write like we did in Stage 2. The docstring IS the description the
# model uses to decide when to call it.

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression like '12 * (3 + 4)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# ── The agent ─────────────────────────────────────────────────────────
# ChatBedrockConverse = LangChain's wrapper around Bedrock's converse API
# (the same API we called by hand). It speaks LangChain's message format.
_model = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
)

# create_react_agent builds the ENTIRE agent loop for us: it wires the tools to
# the model, and when the model asks for a tool it runs it and feeds the result
# back, repeating until the model gives a final answer. This one line replaces
# the whole for-loop in agent.py.
_agent = create_react_agent(_model, [get_current_time, calculate])


def run_agent(user_message: str) -> str:
    """Run the LangChain agent on a question and return the final text answer."""
    # We hand it the user's message; it returns the full conversation. The last
    # message is the agent's final answer.
    result = _agent.invoke({"messages": [("user", user_message)]})
    return result["messages"][-1].content
