"""llm.py — the "brain": one function that sends a prompt to an LLM and returns its reply.

We use Amazon Bedrock (AWS's service for calling foundation models) with the
Amazon Nova Micro model. The only library we need is boto3 (the AWS SDK).

The important idea: everything else we build later (the agent loop, the API,
LangChain) will call THIS one function. If we ever switch models, we change
only this file.
"""

import os

import boto3
from dotenv import load_dotenv

# Read the AWS keys from the local .env file into environment variables.
# boto3 automatically picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION.
load_dotenv()

# The Bedrock model id we enabled. "nova-micro" is Amazon's small, cheap, fast model.
MODEL_ID = "amazon.nova-micro-v1:0"


def ask(prompt: str) -> str:
    """Send `prompt` to the model and return its text reply.

    Args:
        prompt: the question / instruction for the model.

    Returns:
        The model's reply as a plain string.
    """
    # A "client" is boto3's handle to a specific AWS service — here, Bedrock's
    # runtime (the part that actually runs models). Region comes from AWS_REGION.
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
    )

    # Bedrock's "converse" API is the modern, model-agnostic way to chat.
    # We send a list of messages; each has a role ("user") and content.
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {"role": "user", "content": [{"text": prompt}]},
        ],
    )

    # Dig the reply text out of the response structure.
    return response["output"]["message"]["content"][0]["text"]
