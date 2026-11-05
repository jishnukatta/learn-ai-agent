# learn-ai-agent

A hands-on journey that grows a single Python script into a **deployed, RAG-powered AI agent with its own MCP server** — learning one concept at a time.

This README is written to be **re-readable as a course**. Each stage explains the idea in plain words, shows the real code, and points out the AWS or Python concept it taught. If you read it top to bottom, you should understand the whole system and be able to rebuild it.

---

## What this project is

A **personal knowledge assistant**: an AI agent that can do small tasks (math, current time) and answer questions **from your own notes**, exposed both as a web API and as an MCP server that tools like Kiro can call.

The point was never the app itself — it was to learn, by building, the full stack of modern **AI-native engineering**:

- **Python** project structure with `uv`
- **LLM** calls (Amazon Bedrock)
- the **agent loop** (tool-calling) — first by hand, then with frameworks
- **FastAPI** web service
- **AWS**: EC2, IAM (users vs roles), SSM, security groups, public/private IP, RDS
- **CI/CD**: systemd + GitHub Actions
- **LangChain** and **LangGraph**
- **RAG**: vector embeddings + **pgvector** on PostgreSQL
- **FastMCP**: building your own MCP server (stdio + HTTP)

---

## The two mental models that explain everything

**1. Brain vs Hands.**
An LLM can only produce text. It cannot run code, read a file, or query a database. So every agent is split in two:

- **The brain** = the LLM + the loop that decides *what to do next*.
- **The hands** = *tools* (plain functions) that actually touch the world.

The brain never runs a tool itself — it *asks* for one, and the loop (the orchestrator) runs it and feeds the result back.

**2. The agent loop.**
```
loop:
    ask the model (with the conversation so far + the list of tools)
    if the model returned a final answer  -> done
    if the model asked for a tool         -> run it, add the result to the
                                             conversation, loop again
```
Everything else in this repo — LangChain, LangGraph, MCP, Kiro — is this loop with more structure or a network boundary around it.

---

## Final architecture

```
                         YOUR LAPTOP
   ┌───────────────────────────────────────────────────────┐
   │  Kiro (an MCP client)  ──stdio──►  mcp_server.py        │
   │                                     (calculate, time,   │
   │                                      search_notes)      │
   │  curl / browser ──HTTP──►  (to the EC2 below)           │
   └───────────────────────────────────────────────────────┘
                                   │ HTTPS / HTTP
                                   ▼
                         AWS (ap-southeast-2)
   ┌───────────────────────────────────────────────────────┐
   │  EC2 (t3.micro)                                         │
   │   ├─ FastAPI app (systemd service, port 8000)           │
   │   │    └─ LangGraph agent ─ tools: calculate, time,     │
   │   │         search_notes                                │
   │   └─ MCP server (optional, port 8001)                   │
   │   IAM role: SSM (connect) + Bedrock (call models)       │
   └──────────┬───────────────────────────┬─────────────────┘
              │ Bedrock API (HTTPS)        │ Postgres :5432 (private)
              ▼                            ▼
   ┌──────────────────────┐    ┌───────────────────────────┐
   │ Amazon Bedrock       │    │ RDS PostgreSQL + pgvector  │
   │  - Nova Micro (chat) │    │  documents(content,         │
   │  - Titan (embeddings)│    │   embedding vector(1024))   │
   └──────────────────────┘    └───────────────────────────┘

   Deploy path:  push to GitHub ─► GitHub Actions ─► AWS SSM ─►
                 git pull + uv sync + systemctl restart on the EC2
```

---

## Project layout

```
learn-ai-agent/
├── pyproject.toml              # project identity + dependencies (uv reads this)
├── uv.lock                     # exact pinned versions (reproducible installs)
├── .python-version             # pins Python 3.14 (laptop and server must match)
├── .env                        # secrets, git-ignored (AWS keys, DB settings)
├── .github/workflows/deploy.yml# CI/CD: auto-deploy to EC2 on push
├── notes/                      # the documents RAG searches over
│   ├── aws-ec2.md  deployment.md  iam-roles.md  pgvector.md  security-groups.md
└── src/assistant/
    ├── __init__.py
    ├── cli.py                  # entry point for `uv run assistant`
    ├── llm.py                  # Stage 1: one Bedrock chat call
    ├── tools.py                # Stage 2: tool functions + hand-written specs
    ├── agent.py                # Stage 2: the agent loop, BY HAND
    ├── agent_lc.py             # Stage 5: same agent via LangChain (prebuilt)
    ├── agent_graph.py          # Stage 6: same agent as an explicit LangGraph graph
    ├── embeddings.py           # Stage 7: text -> vector via Titan
    ├── db.py                   # Stage 7: connect to RDS PostgreSQL
    ├── rag.py                  # Stage 7: index notes + similarity search
    └── mcp_server.py           # Stage 8: expose the tools as an MCP server
```

---

## How to run it

```bash
# install deps + a matching Python into an isolated .venv
uv sync

# run the CLI agent (currently wired to the LangGraph version)
uv run assistant

# run the web API locally
uv run uvicorn assistant.api:app --host localhost --port 8000
#   then open http://localhost:8000/docs

# run the MCP server (stdio, how Kiro launches it)
uv run python -m assistant.mcp_server
```

Two commands are the whole workflow: **`uv sync`** (install) and **`uv run ...`** (run).

---

# The stages, explained

## Stage 0 — Project skeleton (Python + uv)

**Goal:** a properly structured Python project.

- **`uv`** is a fast package/Python manager. `uv sync` reads `pyproject.toml`, downloads a matching Python, creates an isolated `.venv/`, and installs dependencies. Your system Python is never touched.
- **`pyproject.toml`** is the project's single control file: name, Python version, dependency list, and the `assistant` command.
- A **package** is a folder with `__init__.py`. `src/assistant/` is importable as `assistant`.
- The entry point `[project.scripts] assistant = "assistant.cli:main"` creates the `uv run assistant` command.

Concept learned: **isolated, reproducible environments** and standard project layout.

---

## Stage 1 — First LLM call (Amazon Bedrock)

**Goal:** give the app a brain — one call to a model.

`src/assistant/llm.py`:
```python
import boto3
from dotenv import load_dotenv
load_dotenv()  # reads AWS keys from .env into the environment

MODEL_ID = "amazon.nova-micro-v1:0"

def ask(prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]
```

Concepts learned:
- **`boto3`** is AWS's Python SDK; `boto3.client("bedrock-runtime")` is a handle to the model service.
- **`converse`** is Bedrock's model-agnostic chat API: you send `{role, content}` messages, it returns a reply.
- **Secrets live in `.env`** (git-ignored), loaded by `load_dotenv()` — never hard-coded.
- AWS setup needed: an **IAM user** with `AmazonBedrockFullAccess` and **access keys** in `.env`, and the model enabled in Bedrock (we used **Nova Micro** because Anthropic wasn't available in the region).

---

## Stage 2 — The agent loop, by hand (the core of agentic AI)

**Goal:** let the model *decide to use tools*, run them, feed results back, repeat.

Tools are a function **plus a spec** that describes them to the model (`tools.py`):
```python
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:         # guardrail in the tool, not the model
        return "Error: disallowed characters."
    return str(eval(expression))

TOOL_SPECS = [{
    "toolSpec": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression. Use for any math.",
        "inputSchema": {"json": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}},
    }
}]
TOOL_FUNCTIONS = {"calculate": calculate, "get_current_time": get_current_time}
```

The loop (`agent.py`, trimmed):
```python
messages = [{"role": "user", "content": [{"text": user_message}]}]
for step in range(1, MAX_STEPS + 1):            # MAX_STEPS = safety cap
    response = client.converse(modelId=MODEL_ID, messages=messages,
                               toolConfig={"tools": TOOL_SPECS})
    reply = response["output"]["message"]
    messages.append(reply)                      # remember the model's turn

    if response["stopReason"] != "tool_use":    # final answer -> done
        return "".join(b.get("text","") for b in reply["content"]).strip()

    tool_results = []                           # otherwise run the requested tools
    for block in reply["content"]:
        if "toolUse" not in block: continue
        t = block["toolUse"]
        result = TOOL_FUNCTIONS[t["name"]](**(t.get("input") or {}))
        tool_results.append({"toolResult": {
            "toolUseId": t["toolUseId"],
            "content": [{"text": str(result)}]}})
    messages.append({"role": "user", "content": tool_results})  # feed results back
```

Concepts learned (the ones everything else builds on):
- **`messages` is the agent's memory.** The model is stateless; it "remembers" only because we re-send the whole conversation each call. The list grows: user → model(tool request) → tool results → model(answer).
- **`stopReason`** drives the loop: `tool_use` = keep going; anything else = final answer.
- **`toolUse` / `toolResult` / `toolUseId`** are how a request is matched to its result.
- **Guardrails belong in the tool code**, not in trusting the model (see `calculate`'s character filter, and `MAX_STEPS`).
- **Brain vs hands, concretely:** the model *chose* the tool and args; our code *ran* it.

---

## Stage 3 — FastAPI (expose the agent over HTTP)

**Goal:** make the agent callable by other programs via a URL.

`api.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from assistant.agent_graph import run_agent

app = FastAPI(title="Learn AI Agent")

class AskRequest(BaseModel):   # Pydantic model = typed, self-validating shape
    question: str
class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(answer=run_agent(req.question))
```

Run: `uv run uvicorn assistant.api:app --host localhost --port 8000`.

Concepts learned:
- **FastAPI** turns a function into a URL endpoint via `@app.post(...)`.
- **Pydantic** (`BaseModel`) declares the request/response shape; FastAPI auto-validates incoming JSON (bad input → clear 422 error) and serializes the response.
- **uvicorn** is the server that runs the app; `assistant.api:app` means "the `app` object in `assistant/api.py`".
- **`GET` vs `POST`**: a browser address bar sends GET; `/ask` is POST (data goes in the JSON body, not the URL).
- Free bonus: interactive docs at **`/docs`** (Swagger UI), generated from the code + Pydantic models.
- **`localhost`/`127.0.0.1`** = reachable only from this machine. (Contrast with `0.0.0.0` in Stage 4.)

---

## Stage 4 — Deploy to AWS EC2

**Goal:** run the app on an always-on cloud server, reachable from outside.

The steps and the concept each taught:

1. **Launch an EC2 instance** (`t3.micro`, Amazon Linux). EC2 = a virtual computer in AWS.
2. **IAM role for the machine.** Instead of putting access keys on the server, we attached an **IAM role** (`learn-ai-agent-ssm-role`) to the EC2.
   - **User vs Role:** a *user* (you, GitHub) proves identity with **access keys**; a *role* is worn by an AWS resource (the EC2) and gives it **automatic, keyless, temporary** credentials. Roles are preferred for anything inside AWS.
3. **SSM (Systems Manager) to connect.** No SSH key, no open port 22.
   - The EC2 runs an **SSM agent** that *dials out* to AWS. You give commands to AWS; AWS relays them down the line the EC2 already opened. **Direction: `YOU ↔ SSM ↔ EC2`** — nothing connects *into* the box. The role's `AmazonSSMManagedInstanceCore` policy is what *allows* the EC2 to use SSM.
4. **Install runtime + clone code.** `git` + `uv` on the box, then `git clone` the repo, `uv sync`.
   - For Python (interpreted), the **source code** lives on the server and `uv run` executes it directly. (Compiled languages ship a built artifact instead.)
5. **Bedrock permission via the same role.** Added `AmazonBedrockFullAccess` to the role, so the app calls Bedrock with **no `.env` keys on the server**.
   - Both SSM and Bedrock are the **EC2 calling out** to an AWS service, so both permissions live on the EC2's role.
6. **Run with `--host 0.0.0.0` + open the firewall.**
   - `--host 0.0.0.0` = listen on all addresses (reachable from outside), vs `localhost` (self only).
   - **Security group** = the firewall. Opened **port 8000**, source **My IP**.
     - *Security group ≠ public IP.* The security group controls *which ports + which source IPs* are allowed. The **public IP** controls *whether the internet can route to the box at all*. Both needed.
   - **Regional vs VPC services:** EC2/RDS live in your VPC (private IPs, security groups). Bedrock/SSM/S3 are **regional API endpoints** reached over HTTPS and governed mainly by IAM — no VPC wiring.

Result: `http://<public-ip>:8000/ask` answered from the laptop. Keyless Bedrock via the role, proven.

> **Gotchas learned here:** the public IP changes on stop/start; a `source = My IP` rule breaks when your network changes; and a corporate firewall can block plain HTTP to a raw IP (an "Application Blocked" page). Testing *on the EC2* avoids all three.

---

## Stage 4.5 — CI/CD (systemd + GitHub Actions)

**Goal:** `git push` → the EC2 updates itself. No manual server steps.

### systemd — run the app as a managed service

A unit file at `/etc/systemd/system/learn-ai-agent.service`:
```ini
[Service]
User=ssm-user
WorkingDirectory=/home/ssm-user/learn-ai-agent
ExecStart=/home/ssm-user/.local/bin/uv run uvicorn assistant.api:app --host 0.0.0.0 --port 8000
Restart=on-failure
Environment=HOME=/home/ssm-user
Environment=PATH=/home/ssm-user/.local/bin:/usr/bin:/bin
[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable learn-ai-agent   # start on boot
sudo systemctl start learn-ai-agent    # start now
sudo systemctl restart learn-ai-agent  # apply new code
```
Concept: **systemd** turns the app into a background service that survives logout, auto-restarts on crash, and is controlled by one command. That `restart` is the hook CI/CD uses.

### GitHub Actions — deploy on push

`.github/workflows/deploy.yml` (core idea):
```yaml
on:
  push: { branches: [ main ] }
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-southeast-2
      - name: Deploy on EC2 via SSM
        run: |
          aws ssm send-command \
            --instance-ids "${{ secrets.EC2_INSTANCE_ID }}" \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=[
              "set -e",
              "runuser -l ssm-user -c \"cd /home/ssm-user/learn-ai-agent && git pull origin main && /home/ssm-user/.local/bin/uv sync\"",
              "systemctl restart learn-ai-agent"
            ]'
          # ...then read Status and fail the job if it is not "Success"
```
Concepts learned:
- A **workflow** is YAML triggered by an event (`on: push`). It runs on a temporary GitHub **runner**.
- **GitHub Secrets** hold the deployer's AWS keys (a separate IAM **user**, `github-actions-deployer`), never in code.
- The runner doesn't touch the EC2's IP — it calls **SSM by instance ID**, so deploys are immune to IP changes.
- **The "fake green" lesson:** SSM reported "Success" for *delivering* a command even though the inner `git pull` did nothing. Two real bugs fixed: SSM runs as **root** (our app lives under `ssm-user`, so we wrap the real steps in `runuser -l ssm-user`), and we added `set -e` so an inner failure actually fails the job. **A pipeline is only as truthful as what you verify.**
- **Pin the Python version** (`.python-version` + `requires-python = ">=3.11,<3.15"`) so the server doesn't grab a newer Python lacking prebuilt wheels (that caused a `cc` compiler error on the micro instance).

---

## Stage 5 — LangChain (the loop, pre-packaged)

**Goal:** see the same agent in ~15 lines, and understand what a framework hides.

`agent_lc.py`:
```python
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression like '12 * (3 + 4)'."""
    ...

model = ChatBedrockConverse(model="amazon.nova-micro-v1:0", region_name="ap-southeast-2")
agent = create_react_agent(model, [get_current_time, calculate])

def run_agent(q: str) -> str:
    return agent.invoke({"messages": [("user", q)]})["messages"][-1].content
```

What the framework replaced (vs Stage 2, line by line):

| Did by hand (Stage 2) | LangChain does it |
|---|---|
| hand-wrote `TOOL_SPECS` | `@tool` generates the schema from type hints + docstring |
| managed `messages` list | handled internally |
| checked `stopReason` | handled internally |
| parsed `toolUse`, ran tools, built `toolResult` | handled internally |
| wrote the `for` loop | `create_react_agent` *is* the loop |

Concept learned: a framework is your loop pre-built. Because you wrote the raw version first, `create_react_agent` is not magic — you can map every line to the manual work it replaces.

---

## Stage 6 — LangGraph (build the loop as an explicit graph)

**Goal:** open the box — define the agent as **nodes + edges + state**, so you can branch and insert steps.

`agent_graph.py` (core):
```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # growing memory

def call_model(state):                       # NODE: ask the model
    return {"messages": [_model.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(_tools))    # NODE: run requested tools
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)  # BRANCH: tool? -> tools, else END
graph.add_edge("tools", "model")             # loop back
app = graph.compile()
```
Drawn:
```
START -> model -> (wants tool?) -> tools -> (loop back) -> model
                  (no tool)      -> END
```
Concepts learned:
- **State** = shared data flowing through the graph (`add_messages` = "append, don't replace" — same growing memory as Stage 2).
- **Nodes** = steps; **edges** = flow; a **conditional edge** = branching (what a plain loop can't do cleanly).
- This is the foundation for real workflows: inserting validation/approval steps, branching by topic, or making a node be *another agent* (multi-agent orchestration).

This is also the agent the FastAPI app and the CLI now use.

---

## Stage 7 — RAG: embeddings + pgvector on RDS

**Goal:** let the agent answer from **your own notes**, not just the model's training.

**RAG = Retrieval-Augmented Generation.** Before the model answers, we *retrieve* the most relevant notes and hand them to it as context.

### Embeddings — meaning as numbers

An **embedding** turns text into a vector (a list of numbers) where **similar meanings are close together**. A *different* model from the chat model.

`embeddings.py`:
```python
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024   # must match the DB column

def embed(text: str) -> list[float]:
    client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
    response = client.invoke_model(           # embeddings use invoke_model, not converse
        modelId=MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": DIMENSIONS, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]
```
Proof it matches on meaning (normalized dot product = similarity):
```
"reset password" vs "recover login credentials"  -> 0.38   (related)
"reset password" vs "pizza recipe"               -> 0.04   (unrelated)
```

### pgvector on PostgreSQL (RDS) — store + search vectors

RDS PostgreSQL with the `vector` extension. The table:
```sql
CREATE EXTENSION vector;
CREATE TABLE documents (
  id serial PRIMARY KEY,
  source text,
  content text,
  embedding vector(1024)                       -- the embedding
);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);  -- fast similarity index
```
- `<=>` is **cosine distance** (smaller = more similar). `<->` is L2; `<#>` inner product.
- **hnsw** is an index built for nearest-neighbor search (B-tree can't do "find the closest vector").

`rag.py` — the two halves of RAG:
```python
def index_notes():                             # INDEXING (once)
    for path in NOTES_DIR.glob("*.md"):
        conn.execute(
            "INSERT INTO documents (source, content, embedding) VALUES (%s,%s,%s::vector)",
            (path.name, text, _vector_literal(embed(text))))

def search(question, k=3):                      # RETRIEVAL (per query)
    q = _vector_literal(embed(question))
    return conn.execute(
        "SELECT source, content, embedding <=> %s::vector AS distance "
        "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
        (q, q, k)).fetchall()
```

`db.py` connects using env vars (`DB_HOST`, `DB_PASSWORD`, …) from `.env`.

### The AWS networking lesson

- The **database is private** (`Publicly accessible: No`). Only the EC2 can reach it.
- A dedicated security group `learn-ai-agent-db-sg` allows **port 5432 from the EC2's security group** (source = a security group, not an IP — so it survives IP changes).
- Therefore the RAG code **runs on the EC2**, not the laptop. On the EC2, `search_notes` hit Titan + the DB purely via the **IAM role** (`Found credentials from IAM Role: learn-ai-agent-ssm-role`).

### Who does what
- **Titan** understands meaning (text → vector).
- **Postgres** stores numbers and does distance math fast — it understands nothing.
- Intelligence is in the embedding model; the database is a fast filing cabinet.

### The search tool, wired into the agent (`agent_graph.py`)
```python
@tool
def search_notes(question: str) -> str:
    """Search the user's personal notes about their AWS project (EC2, IAM, security
    groups, deployment, pgvector). Use for questions about the user's own setup."""
    try:
        rows = search(question, k=3)
    except Exception as e:
        return f"Error searching notes: {e}"   # graceful: works even if DB unreachable
    return "\n\n".join(f"[{src}] {content}" for src, content, _ in rows)
```
Now the agent **chooses** `search_notes` for "how is my app deployed?" and `calculate` for math — tool selection is the agentic part.

---

## Stage 8 — FastMCP (your own MCP server)

**Goal:** expose the tools over the **Model Context Protocol** so *any* MCP client (Kiro, Claude Desktop, another agent) can use them — closing the loop with the wallet MCP you first explored.

`mcp_server.py`:
```python
from mcp.server.fastmcp import FastMCP     # mcp<2 (v1 API, like wallet-agent-mcp)
mcp = FastMCP("learn-ai-agent")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression like '12 * (3 + 4)'."""
    ...

@mcp.tool()
def search_notes(question: str) -> str:
    """Search the user's personal notes ..."""
    from assistant.rag import search          # lazy import: other tools still work w/o a DB
    ...

def main():
    if os.environ.get("TRANSPORT") == "http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT", "8001"))
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)   # needed when reached by IP/proxy
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")           # default: launched by a client over stdin/stdout
```

### Two transports — and the difference

- **stdio** (default): the client (Kiro) **launches the server as a subprocess** and talks over stdin/stdout. Configured in Kiro's `mcp.json` with a `command` + `args`. Simple, local, one user. **This is the recommended everyday mode.**
- **HTTP** (`TRANSPORT=http`): the server listens on a port; a client calls it over the network with MCP's JSON-RPC protocol. This is the pattern for a **shared/remote** MCP (like the hosted wallet MCP behind nginx).

### What "the orchestrator calls the MCP" actually looks like

We did the HTTP handshake by hand with `curl` to *see* what a client does under the hood:
```bash
# 1. open a session -> returns an `mcp-session-id` header
curl -sD - http://<ip>:8001/mcp -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'

# 2. confirm ready (notification, no response)
curl ... -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 3. call a tool
curl ... -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"calculate","arguments":{"expression":"144 * 17 + 9"}}}'
# -> data: {"result":{"content":[{"type":"text","text":"2457"}],"isError":false}}
```
That `initialize → notifications/initialized → tools/call` sequence is **exactly what Kiro (or any orchestrator) does automatically** every time it uses an MCP tool. The LLM never calls the MCP directly — the orchestrator does, over stdio or HTTP.

Concepts learned:
- `@mcp.tool()` turns a function into an MCP tool; the **schema is generated from type hints + docstring** (same as the wallet MCP).
- MCP cleanly separates **brain (client) and hands (server)** across a process/network boundary.
- MCP over HTTP is **JSON-RPC over streamable-HTTP (SSE)**, with a session id, not plain REST — you can't just `curl /ask`.
- DNS-rebinding protection rejects non-localhost `Host` headers with **HTTP 421**; disabling it (or fronting with an authenticated proxy) is required to reach it by IP — the exact issue the wallet deployment doc documents.

### Connect it to Kiro (stdio) — `~/.kiro/settings/mcp.json`
```json
"learn-ai-agent": {
  "command": "/path/to/uv",
  "args": ["run", "--directory", "/path/to/learn-ai-agent", "python", "-m", "assistant.mcp_server"],
  "disabled": false
}
```
Then Kiro can call `calculate`, `get_current_time`, and `search_notes` directly.

---

# Concept cheat-sheet (quick recall)

### The agent loop
```
ask model (history + tools) → final answer? return it : run tool, add result, repeat
```
`messages` = memory (model is stateless; you re-send everything). A step cap stops infinite loops.

### IAM: user vs role
| | User | Role |
|---|---|---|
| For | a person / external system (you, GitHub) | an AWS resource (EC2) |
| Credentials | long-lived **access keys** you manage | **automatic, temporary, keyless** |
| Here | `learn-ai-agent-dev` (laptop), `github-actions-deployer` (CI) | `learn-ai-agent-ssm-role` (EC2) |

### Network: three independent knobs for "can I reach the app?"
1. **Public IP** — is there an internet-routable address at all?
2. **Security group** — which **ports** are open + which **source IPs** allowed? (the firewall)
3. **`--host`** — is the app listening on `0.0.0.0` (all) or `localhost` (self only)?
All three must line up. A **timeout** = firewall dropping; **connection refused** = reached host, nothing listening on that port.

### SSM direction
`YOU ↔ SSM ↔ EC2`. The EC2 **dials out** and polls SSM; you leave commands with SSM; nothing connects *into* the box. The EC2 needs the SSM permission because *it* is the caller.

### Which AWS services need VPC/firewall?
- **In your VPC** (private IP, security groups): EC2, RDS.
- **Regional API endpoints** (HTTPS, governed by IAM): Bedrock, SSM, S3. No VPC wiring.

### converse vs invoke_model
- **Chat** (Nova Micro): `client.converse(...)` with `{role, content}` messages.
- **Embeddings** (Titan): `client.invoke_model(...)` with raw JSON.

### Vector search operators (pgvector)
`<=>` cosine · `<->` L2 · `<#>` inner product. Smaller distance = more similar. `hnsw` index makes it fast.

### MCP transports
- **stdio**: client launches the server as a subprocess (local, default).
- **HTTP**: server listens on a port; client speaks JSON-RPC (`initialize` → `notifications/initialized` → `tools/call`) with a session id. For shared/remote.

### Local terminal keys
- **Ctrl+C** stops a process. **Ctrl+Z** *suspends* it (resume with `fg`). A suspended server won't serve.
- Port already in use (Errno 98): `sudo ss -tlnp | grep <port>` → `sudo kill -9 <pid>`.

---

# AWS resources created (and what they cost)

| Resource | Name | Billing |
|---|---|---|
| IAM user (laptop) | `learn-ai-agent-dev` | free |
| IAM user (CI) | `github-actions-deployer` | free |
| IAM role (EC2) | `learn-ai-agent-ssm-role` | free |
| EC2 instance | `learn-ai-agent` (t3.micro) | per hour **while running**; EBS disk while stopped |
| Security groups | `launch-wizard-1` (app), `learn-ai-agent-db-sg` (db) | free |
| RDS PostgreSQL | `learn-ai-agent-db` (db.t4g.micro) | per hour **while running**; storage + backups while stopped |
| Bedrock | Nova Micro + Titan | per token (tiny) |

**Free meters:** IAM, security groups, SSM, VPC basics.
**Time meters to watch:** EC2 and RDS.

### Stop everything when done (save credits)
```bash
# on the EC2 (if running a manual server): Ctrl+C
# then in the console:
#   EC2  → Instance state → Stop instance
#   RDS  → Actions → Stop temporarily   (auto-restarts after 7 days!)
```
Set a **Budgets → Zero-spend / $5 budget** alert so a forgotten resource can't drain credits silently.

---

# How to resume after a stop

1. Start **EC2** and **RDS** in the console (RDS takes a few minutes).
2. The EC2's **public IP will have changed** — grab the new one; update the port-8000 security-group rule if you call it from the laptop.
3. The `learn-ai-agent` systemd service **auto-starts** on boot, so the API is already serving.
4. Push to `main` and GitHub Actions deploys automatically (it targets the instance by ID, immune to IP changes).

---

# Glossary

- **uv** — fast Python/package manager; `uv sync` installs, `uv run` runs.
- **Bedrock** — AWS service for calling foundation models (chat + embeddings).
- **boto3** — AWS SDK for Python.
- **agent loop** — decide → call tool → observe → repeat, until a final answer.
- **tool** — a function the model can ask to run; its docstring/type hints tell the model what it does.
- **FastAPI / uvicorn** — web framework / the server that runs it.
- **Pydantic** — typed, self-validating data models (request/response shapes).
- **EC2** — a virtual machine in AWS.
- **IAM user / role / policy** — identity with keys / identity worn by a resource / a permission list.
- **SSM** — keyless, portless way to run commands on an EC2 (the EC2 dials out).
- **security group** — a firewall: open ports + allowed source IPs.
- **systemd** — Linux service manager (auto-start, auto-restart, `systemctl`).
- **GitHub Actions** — CI/CD; workflows triggered by repo events.
- **LangChain / LangGraph** — agent framework / graph-based agent builder (nodes, edges, state).
- **embedding** — text turned into a vector capturing meaning.
- **RAG** — retrieve relevant documents, then let the LLM answer from them.
- **pgvector** — PostgreSQL extension adding a `vector` type + similarity search.
- **RDS** — AWS managed relational database.
- **MCP** — Model Context Protocol; a standard way to expose tools to AI clients.
- **FastMCP** — the library for building an MCP server with `@mcp.tool()`.
- **stdio / HTTP transport** — client launches server as a subprocess / client calls it over the network.

---

*Built stage by stage, every layer by hand, from a one-line script to a deployed RAG agent with its own MCP server. Re-read any stage above to refresh the concept and the code behind it.*
