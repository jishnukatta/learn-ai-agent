"""agent.py — the agent LOOP.

This is the core of agentic AI. Instead of asking the model once, we let it
decide to CALL A TOOL. We run the tool, feed the result back, and ask again —
repeating until the model gives a final text answer.

The cycle:
    1. send the conversation (so far) + the tool list to the model
    2. the model replies with EITHER:
         - a final text answer   -> we're done, return it
         - a "toolUse" request   -> it wants us to run a tool
    3. if a tool was requested: run it, append the result to the conversation
    4. go back to step 1
"""

import os

import boto3
from dotenv import load_dotenv

from assistant.tools import TOOL_FUNCTIONS, TOOL_SPECS

load_dotenv()

MODEL_ID = "amazon.nova-micro-v1:0"
MAX_STEPS = 5  # safety: never loop forever, even if the model misbehaves


def _client():
    """Create the Bedrock client (region from AWS_REGION)."""
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
    )


def run_agent(user_message: str, verbose: bool = True) -> str:
    """Run the agent loop for a single user request and return the final answer.

    Args:
        user_message: what the user asked.
        verbose: if True, print each step so you can WATCH the loop work.
    """
    client = _client()

    # The conversation is a running list of messages. It grows as the model and
    # the tools talk back and forth. This is the agent's short-term memory.
    messages = [
        {"role": "user", "content": [{"text": user_message}]},
    ]

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n── step {step}: asking the model ──")

        # Send the whole conversation + the tool list. toolConfig is what tells
        # the model "you may use these tools".
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig={"tools": TOOL_SPECS},
        )

        # The model's reply message — append it to the conversation immediately
        # (the model expects to see its own prior turn next time).
        reply = response["output"]["message"]
        messages.append(reply)

        # Why did the model stop? "tool_use" = it wants a tool; "end_turn" = done.
        stop_reason = response.get("stopReason")

        if stop_reason != "tool_use":
            # No tool requested -> this is the final answer. Pull out the text.
            final_text = "".join(
                block.get("text", "") for block in reply["content"]
            )
            if verbose:
                print("── model gave a final answer ──")
            return final_text.strip()

        # Otherwise the model asked for one or more tools. Run each, collect
        # results, and send them back as a "tool result" user message.
        tool_results = []
        for block in reply["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            name = tool_use["name"]
            args = tool_use.get("input", {}) or {}

            if verbose:
                print(f"   model wants tool: {name}({args})")

            # Look up and run the real Python function.
            func = TOOL_FUNCTIONS.get(name)
            result = func(**args) if func else f"Unknown tool: {name}"

            if verbose:
                print(f"   tool result: {result}")

            # Package the result in the shape Bedrock expects, tying it back to
            # the model's request via toolUseId.
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": str(result)}],
                    }
                }
            )

        # Feed the tool results back into the conversation as the next turn.
        messages.append({"role": "user", "content": tool_results})

    # If we fell out of the loop, we hit the step limit without a final answer.
    return "(stopped: reached the maximum number of steps)"
