"""agent_graph.py — the SAME agent, built as an EXPLICIT LangGraph graph.

Stage 5 hid the loop inside create_react_agent. Here we build the graph by hand
so you can SEE the structure: two nodes (call the model, run tools) connected by
a conditional edge (branch: did the model ask for a tool, or give an answer?).

The Stage 2 loop, drawn as a flowchart:

    START -> model -> (wants tool?) -> tools -> (loop back) -> model
                      (no tool)      -> END (final answer)
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from assistant.rag import search

load_dotenv()


# ── Tools (same as before) ────────────────────────────────────────────
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


@tool
def search_notes(question: str) -> str:
    """Search the user's personal notes about their own AWS project: EC2, IAM roles,
    security groups, deployment, and pgvector. Use this for any question about the
    user's own setup. Returns the most relevant notes."""
    try:
        rows = search(question, k=3)
    except Exception as e:  # noqa: BLE001
        # Return the error as text so the model can tell the user, instead of crashing the graph.
        return f"Error searching notes: {e}"
    return "\n\n".join(f"[{source}] {content}" for source, content, _distance in rows)


_tools = [get_current_time, calculate, search_notes]

# The model, with the tools "bound" so it knows they exist.
_model = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
).bind_tools(_tools)


# ── State ─────────────────────────────────────────────────────────────
# The shared data that flows through the graph. Here it's just the message
# list. `add_messages` tells LangGraph "APPEND returned messages to the list"
# (rather than replace) — the growing memory you managed by hand in Stage 2.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ── Nodes (the steps) ─────────────────────────────────────────────────
def call_model(state: State) -> dict:
    """Node: send the conversation to the model, return its reply."""
    reply = _model.invoke(state["messages"])
    return {"messages": [reply]}   # add_messages appends this to state


# ToolNode is a prebuilt node that runs whatever tool the model asked for.
_tool_node = ToolNode(_tools)


# ── Build the graph ───────────────────────────────────────────────────
def _build():
    graph = StateGraph(State)

    # register the two nodes
    graph.add_node("model", call_model)
    graph.add_node("tools", _tool_node)

    # edges (the wiring):
    graph.add_edge(START, "model")               # start -> model
    # conditional edge: after "model", branch on whether it asked for a tool.
    # tools_condition (prebuilt) returns "tools" if a tool was requested, else END.
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")             # after tools, loop back to model

    return graph.compile()


_app = _build()


def run_agent(user_message: str) -> str:
    """Run the graph on a question and return the final answer."""
    result = _app.invoke({"messages": [("user", user_message)]})
    return result["messages"][-1].content
