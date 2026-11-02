"""api.py — expose the agent over HTTP using FastAPI.

A "web API" means: other programs send an HTTP request to a URL, and get a
response back. Here we expose one endpoint:

    POST /ask   with body { "question": "..." }   ->   { "answer": "..." }

Run the server with:
    uv run uvicorn assistant.api:app --reload

Then send a request (in another terminal):
    curl -X POST http://127.0.0.1:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "what is 17 * 23?"}'
"""

from fastapi import FastAPI
from pydantic import BaseModel

from assistant.agent import run_agent

# The FastAPI application object. `app` is what uvicorn runs.
app = FastAPI(title="Learn AI Agent", version="0.1.0")


# ── Request / response shapes ─────────────────────────────────────────
# Pydantic models describe the JSON we expect in and send out. FastAPI uses
# them to validate incoming data automatically (bad input -> clear error).

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """A simple check that the server is up. GET http://.../health -> {"status": "ok"}"""
    return {"status": "ok", "deployed_via": "github-actions"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Run the agent on the user's question and return the answer.

    FastAPI parses the incoming JSON into an AskRequest automatically, so
    `req.question` is the user's text. We run the agent and wrap the result.
    """
    answer = run_agent(req.question, verbose=False)
    return AskResponse(answer=answer)
