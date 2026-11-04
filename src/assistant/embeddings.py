"""embeddings.py — turn text into a vector using Amazon Titan Embeddings V2 on Bedrock.

An embedding model is a different kind of model from the chat model: you give it
text and it returns a list of numbers (a vector) that captures the text's meaning.
Texts with similar meaning get vectors that are close together.

Everything that needs a vector (indexing documents, embedding a question) calls
embed() from this one file.
"""

import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024  # Titan V2 supports 256, 512 or 1024. This must match the DB column.


def embed(text: str) -> list[float]:
    """Return the embedding vector (a list of DIMENSIONS floats) for `text`."""
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
    )

    # Embedding models use invoke_model (raw JSON in and out), not the chat
    # `converse` API.
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(
            {
                "inputText": text,
                "dimensions": DIMENSIONS,
                "normalize": True,  # scale to length 1, which works well with cosine search
            }
        ),
    )
    return json.loads(response["body"].read())["embedding"]
